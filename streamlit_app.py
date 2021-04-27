import streamlit as st
import numpy as np
import pandas as pd
import base64 
import altair as alt

st.title('Gravity Model')

def gz(xp, zp, x1,x2,z1,z2, dens):
        #: Conversion factor from SI units to mGal: :math:`1\ m/s^2 = 10^5\ mGal`
        SI2MGAL = 100000.0
        #: The gravitational constant in :math:`m^3 kg^{-1} s^{-1}`
        G = 0.00000000006673

        density = dens
        x = np.array([x1,x1,x2,x2,x1])
        z = np.array([z1,z2,z2,z1,z1])
        size = len(xp)
        res = np.zeros(size, dtype=np.float)
        poly_z = np.array([z1,z2,z2,z1,z1])
        poly_x = np.array([x1,x1,x2,x2,x1])

        nverts = np.size(poly_x)  #5
        
        for v in range(nverts):
            # Change the coordinates of this vertice
            xv = x[v] - xp
            zv = z[v] - zp
            # The last vertice pairs with the first one
            if v == nverts - 1:
                xvp1 = x[0] - xp
                zvp1 = z[0] - zp
            else:
                xvp1 = x[v + 1] - xp
                zvp1 = z[v + 1] - zp
            #st.write("Break")

            #print(np.c_[xvp1,zvp1])
            # Temporary fix. The analytical conditions for these limits don't
            # work. So if the conditions are breached, sum 0.01 meters to the
            # coodinates and be happy
            xv[xv == 0.] += 0.01
            xv[xv == xvp1] += 0.01
            zv[xv == zv] = 0.
            zv[zv == 0.] += 0.01
            zv[zv == zvp1] += 0.01
            zvp1[xvp1 == zvp1] = 0.
            zvp1[zvp1 == 0.] += 0.01
            xvp1[xvp1 == 0.] += 0.01
            #st.write("Break2")

            # End of fix
            phi_v = np.arctan2(zvp1 - zv, xvp1 - xv)
            ai = xvp1 + zvp1 * (xvp1 - xv) / (zv - zvp1)
            theta_v = np.arctan2(zv, xv)
            theta_vp1 = np.arctan2(zvp1, xvp1)
            theta_v[theta_v < 0] += np.pi
            theta_vp1[theta_vp1 < 0] += np.pi
            tmp = ai * np.sin(phi_v) * np.cos(phi_v) * (
                theta_v - theta_vp1 + np.tan(phi_v) * np.log(
                    (np.cos(theta_v) * (np.tan(theta_v) - np.tan(phi_v))) /
                    (np.cos(theta_vp1) * (np.tan(theta_vp1) - np.tan(phi_v)))))
            tmp[theta_v == theta_vp1] = 0.
            res = res + tmp * density
        res = res * SI2MGAL * 2.0 * G
        return res

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    return href


xp = np.linspace(0,100,100)
zp = np.zeros_like(xp)

st.sidebar.title('Select rectangle extent')
x1=st.sidebar.text_input('Input X1 here',30)
x2=st.sidebar.text_input('Input X2 here',40)
y1=st.sidebar.text_input('Input Y1 here',10)
y2=st.sidebar.text_input('Input Y2 here',20)
density=st.sidebar.text_input('Input Model Density contrast here',2000)


x1=float(x1)
x2=float(x2)
y1=float(y1)
y2=float(y2)
density=float(density)
line = gz(xp,zp,x1,x2,y1,y2,density)
#line=zp

st.write("Box extent, X:",x1,"-",x2," and Y:",y1,"-",y2, "Box density:",density)
#if st.button("Calculate"):
#    line = gz(xp,zp,x1,x2,y1,y2,density)
#    line=zp

x = np.array([x1,x1,x2,x2,x1])
y = np.array([y1,y2,y2,y1,y1])

#box = pd.DataFrame(data=np.column_stack((x,y)),columns=['X','Y'])
#c=alt.Chart(box).mark_line().encode(x='X',y="Y",color=alt.Color('X', scale=alt.Scale(scheme='dark2'))
#        scale=alt.Scale(domain=(0, 100))
#        )
box = pd.DataFrame({'x1': [x1], 'x2': [x2], 'y1': [y1], 'y2': [y2]})

c = alt.Chart(box,width=600,height=400).mark_rect(fill='red', stroke='red').encode(
    alt.X('x1', scale=alt.Scale(domain=(0, 100)),title="Distance (m)"),
    alt.Y('y1', scale=alt.Scale(domain=(60, 0)),title="Depth (m)"),
    x2='x2',
    y2='y2').configure_axis(
                labelFontSize=18,
                titleFontSize=18
)

st.write("Box position")

st.altair_chart(c)

chart_data = pd.DataFrame(data=np.column_stack((xp,line)),columns=['Distance (m)','Gravity (mGal)'])
#st.line_chart(chart_data.rename(columns={'Distance (m)':'index'}).set_index('index'),width=600)

domain = ['Distance (m)']
range_ = ['blue']
line_chart = alt.Chart(chart_data,width=600,height=400).mark_line(interpolate='basis').encode(
    alt.X('Distance (m)', title='Distance (m)'),
    alt.Y('Gravity (mGal)', title='Gravity (mGal)'),
    color=alt.value('blue')).properties(
    title='Gravity of a box').configure_axis(
                labelFontSize=18,
                titleFontSize=18
)

# scale=alt.Scale(scheme='blueorange', domain=[50, 100])
st.altair_chart(line_chart)


#alt.Chart(source).mark_line().encode(
#    x='x',
#    y='f(x)'
#)

#st.line_chart(line)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(data=np.column_stack((xp,line)),columns=['Distance (m)','Gravity (mGal)'])
    #chart_data1 = pd.DataFrame(data=xp,columns=['Distance (m)'])
    #chart_data2 = pd.DataFrame(data=line,columns=['Gravity (mGal)'])
    #st.write("Distance (m)\n")
    #st.write(np.c_[xp])
    #st.dataframe(chart_data1,510,2600)
    #st.write("Gravity (mGal)\n")
    #st.dataframe(chart_data2,610,2600)
    st.dataframe(chart_data,610,2600)

    #st.write(chart_data2)
    #st.write(np.c_[line])

st.markdown(get_table_download_link(chart_data), unsafe_allow_html=True)
    

