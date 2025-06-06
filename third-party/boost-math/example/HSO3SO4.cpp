// test file for HSO3.hpp and HSO4.hpp

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <iostream>

#include <boost/math/quaternion.hpp>

#include "HSO3.hpp"
#include "HSO4.hpp"


const int    number_of_intervals = 5;

const float    pi = ::std::atan(1.0f)*4;



void    test_SO3();
    
void    test_SO4();


int    main()

{
    test_SO3();
    
    test_SO4();
    
    ::std::cout << "That's all folks!" << ::std::endl;
}


//
//    Test of quaternion and R^3 rotation relationship
//

void    test_SO3_spherical()
{
    ::std::cout << "Testing spherical:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    const float    rho = 1.0f;
    
    float        theta;
    float        phi1;
    float        phi2;
    
    for    (int idxphi2 = 0; idxphi2 <= number_of_intervals; idxphi2++)
    {
        phi2 = (-pi/2)+(idxphi2*pi)/number_of_intervals;
        
        for    (int idxphi1 = 0; idxphi1 <= number_of_intervals; idxphi1++)
        {
            phi1 = (-pi/2)+(idxphi1*pi)/number_of_intervals;
            
            for    (int idxtheta = 0; idxtheta <= number_of_intervals; idxtheta++)
            {
                theta = -pi+(idxtheta*(2*pi))/number_of_intervals;
                
                ::std::cout << "theta = " << theta << " ; ";
                ::std::cout << "phi1 = " << phi1 << " ; ";
                ::std::cout << "phi2 = " << phi2;
                ::std::cout << ::std::endl;
                
                ::boost::math::quaternion<float>    q = ::boost::math::spherical(rho, theta, phi1, phi2);
                
                ::std::cout << "q = " << q << ::std::endl;
                
                R3_matrix<float>                    rot = quaternion_to_R3_rotation(q);
                
                ::std::cout << "rot = ";
                ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << ::std::endl;
                
                ::boost::math::quaternion<float>    p = R3_rotation_to_quaternion(rot, &q);
                
                ::std::cout << "p = " << p << ::std::endl;
                
                ::std::cout << "round trip discrepancy: " << ::boost::math::abs(q-p) << ::std::endl;
                
                ::std::cout << ::std::endl;
            }
        }
    }
    
    ::std::cout << ::std::endl;
}

    
void    test_SO3_semipolar()
{
    ::std::cout << "Testing semipolar:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    const float    rho = 1.0f;
    
    float        alpha;
    float        theta1;
    float        theta2;
    
    for    (int idxalpha = 0; idxalpha <= number_of_intervals; idxalpha++)
    {
        alpha = (idxalpha*(pi/2))/number_of_intervals;
        
        for    (int idxtheta1 = 0; idxtheta1 <= number_of_intervals; idxtheta1++)
        {
            theta1 = -pi+(idxtheta1*(2*pi))/number_of_intervals;
            
            for    (int idxtheta2 = 0; idxtheta2 <= number_of_intervals; idxtheta2++)
            {
                theta2 = -pi+(idxtheta2*(2*pi))/number_of_intervals;
                
                ::std::cout << "alpha = " << alpha << " ; ";
                ::std::cout << "theta1 = " << theta1 << " ; ";
                ::std::cout << "theta2 = " << theta2;
                ::std::cout << ::std::endl;
                
                ::boost::math::quaternion<float>    q = ::boost::math::semipolar(rho, alpha, theta1, theta2);
                
                ::std::cout << "q = " << q << ::std::endl;
                
                R3_matrix<float>                    rot = quaternion_to_R3_rotation(q);
                
                ::std::cout << "rot = ";
                ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << ::std::endl;
                
                ::boost::math::quaternion<float>    p = R3_rotation_to_quaternion(rot, &q);
                
                ::std::cout << "p = " << p << ::std::endl;
                
                ::std::cout << "round trip discrepancy: " << ::boost::math::abs(q-p) << ::std::endl;
                
                ::std::cout << ::std::endl;
            }
        }
    }
    
    ::std::cout << ::std::endl;
}

    
void    test_SO3_multipolar()
{
    ::std::cout << "Testing multipolar:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    float    rho1;
    float    rho2;
    
    float    theta1;
    float    theta2;
    
    for    (int idxrho = 0; idxrho <= number_of_intervals; idxrho++)
    {
        rho1 = (idxrho*1.0f)/number_of_intervals;
        rho2 = ::std::sqrt(1.0f-rho1*rho1);
        
        for    (int idxtheta1 = 0; idxtheta1 <= number_of_intervals; idxtheta1++)
        {
            theta1 = -pi+(idxtheta1*(2*pi))/number_of_intervals;
            
            for    (int idxtheta2 = 0; idxtheta2 <= number_of_intervals; idxtheta2++)
            {
                theta2 = -pi+(idxtheta2*(2*pi))/number_of_intervals;
                
                ::std::cout << "rho1 = " << rho1 << " ; ";
                ::std::cout << "theta1 = " << theta1 << " ; ";
                ::std::cout << "theta2 = " << theta2;
                ::std::cout << ::std::endl;
                
                ::boost::math::quaternion<float>    q = ::boost::math::multipolar(rho1, theta1, rho2, theta2);
                
                ::std::cout << "q = " << q << ::std::endl;
                
                R3_matrix<float>                    rot = quaternion_to_R3_rotation(q);
                
                ::std::cout << "rot = ";
                ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << ::std::endl;
                
                ::boost::math::quaternion<float>    p = R3_rotation_to_quaternion(rot, &q);
                
                ::std::cout << "p = " << p << ::std::endl;
                
                ::std::cout << "round trip discrepancy: " << ::boost::math::abs(q-p) << ::std::endl;
                
                ::std::cout << ::std::endl;
            }
        }
    }
    
    ::std::cout << ::std::endl;
}

    
void    test_SO3_cylindrospherical()
{
    ::std::cout << "Testing cylindrospherical:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    float    t;
    
    float    radius;
    float    longitude;
    float    latitude;
    
    for    (int idxt = 0; idxt <= number_of_intervals; idxt++)
    {
        t = -1.0f+(idxt*2.0f)/number_of_intervals;
        radius = ::std::sqrt(1.0f-t*t);
        
        for    (int idxlatitude = 0; idxlatitude <= number_of_intervals; idxlatitude++)
        {
            latitude = (-pi/2)+(idxlatitude*pi)/number_of_intervals;
            
            for    (int idxlongitude = 0; idxlongitude <= number_of_intervals; idxlongitude++)
            {
                longitude = -pi+(idxlongitude*(2*pi))/number_of_intervals;
                
                ::std::cout << "t = " << t << " ; ";
                ::std::cout << "longitude = " << longitude;
                ::std::cout << "latitude = " << latitude;
                ::std::cout << ::std::endl;
                
                ::boost::math::quaternion<float>    q = ::boost::math::cylindrospherical(t, radius, longitude, latitude);
                
                ::std::cout << "q = " << q << ::std::endl;
                
                R3_matrix<float>                    rot = quaternion_to_R3_rotation(q);
                
                ::std::cout << "rot = ";
                ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << ::std::endl;
                
                ::boost::math::quaternion<float>    p = R3_rotation_to_quaternion(rot, &q);
                
                ::std::cout << "p = " << p << ::std::endl;
                
                ::std::cout << "round trip discrepancy: " << ::boost::math::abs(q-p) << ::std::endl;
                
                ::std::cout << ::std::endl;
            }
        }
    }
    
    ::std::cout << ::std::endl;
}

    
void    test_SO3_cylindrical()
{
    ::std::cout << "Testing cylindrical:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    float    r;
    float    angle;
    
    float    h1;
    float    h2;
    
    for    (int idxh2 = 0; idxh2 <= number_of_intervals; idxh2++)
    {
        h2 = -1.0f+(idxh2*2.0f)/number_of_intervals;
        
        for    (int idxh1 = 0; idxh1 <= number_of_intervals; idxh1++)
        {
            h1 = ::std::sqrt(1.0f-h2*h2)*(-1.0f+(idxh2*2.0f)/number_of_intervals);
            r = ::std::sqrt(1.0f-h1*h1-h2*h2);
            
            for    (int idxangle = 0; idxangle <= number_of_intervals; idxangle++)
            {
                angle = -pi+(idxangle*(2*pi))/number_of_intervals;
                
                ::std::cout << "angle = " << angle << " ; ";
                ::std::cout << "h1 = " << h1;
                ::std::cout << "h2 = " << h2;
                ::std::cout << ::std::endl;
                
                ::boost::math::quaternion<float>    q = ::boost::math::cylindrical(r, angle, h1, h2);
                
                ::std::cout << "q = " << q << ::std::endl;
                
                R3_matrix<float>                    rot = quaternion_to_R3_rotation(q);
                
                ::std::cout << "rot = ";
                ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << ::std::endl;
                ::std::cout << "\t";
                ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << ::std::endl;
                
                ::boost::math::quaternion<float>    p = R3_rotation_to_quaternion(rot, &q);
                
                ::std::cout << "p = " << p << ::std::endl;
                
                ::std::cout << "round trip discrepancy: " << ::boost::math::abs(q-p) << ::std::endl;
                
                ::std::cout << ::std::endl;
            }
        }
    }
    
    ::std::cout << ::std::endl;
}


void    test_SO3()
{
    ::std::cout << "Testing SO3:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    test_SO3_spherical();
    
    test_SO3_semipolar();
    
    test_SO3_multipolar();
    
    test_SO3_cylindrospherical();
    
    test_SO3_cylindrical();
}


//
//    Test of quaternion and R^4 rotation relationship
//

void    test_SO4_spherical()
{
    ::std::cout << "Testing spherical:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    const float    rho1 = 1.0f;
    const float    rho2 = 1.0f;
    
    float        theta1;
    float        phi11;
    float        phi21;
    
    float        theta2;
    float        phi12;
    float        phi22;
    
    for    (int idxphi21 = 0; idxphi21 <= number_of_intervals; idxphi21++)
    {
        phi21 = (-pi/2)+(idxphi21*pi)/number_of_intervals;
        
        for    (int idxphi22 = 0; idxphi22 <= number_of_intervals; idxphi22++)
        {
            phi22 = (-pi/2)+(idxphi22*pi)/number_of_intervals;
            
            for    (int idxphi11 = 0; idxphi11 <= number_of_intervals; idxphi11++)
            {
                phi11 = (-pi/2)+(idxphi11*pi)/number_of_intervals;
                
                for    (int idxphi12 = 0; idxphi12 <= number_of_intervals; idxphi12++)
                {
                    phi12 = (-pi/2)+(idxphi12*pi)/number_of_intervals;
                    
                    for    (int idxtheta1 = 0; idxtheta1 <= number_of_intervals; idxtheta1++)
                    {
                        theta1 = -pi+(idxtheta1*(2*pi))/number_of_intervals;
                        
                        for    (int idxtheta2 = 0; idxtheta2 <= number_of_intervals; idxtheta2++)
                        {
                            theta2 = -pi+(idxtheta2*(2*pi))/number_of_intervals;
                            
                            ::std::cout << "theta1 = " << theta1 << " ; ";
                            ::std::cout << "phi11 = " << phi11 << " ; ";
                            ::std::cout << "phi21 = " << phi21;
                            ::std::cout << "theta2 = " << theta2 << " ; ";
                            ::std::cout << "phi12 = " << phi12 << " ; ";
                            ::std::cout << "phi22 = " << phi22;
                            ::std::cout << ::std::endl;
                            
                            ::boost::math::quaternion<float>    p1 = ::boost::math::spherical(rho1, theta1, phi11, phi21);
                            
                            ::std::cout << "p1 = " << p1 << ::std::endl;
                            
                            ::boost::math::quaternion<float>    q1 = ::boost::math::spherical(rho2, theta2, phi12, phi22);
                            
                            ::std::cout << "q1 = " << q1 << ::std::endl;
                            
                            ::std::pair< ::boost::math::quaternion<float> , ::boost::math::quaternion<float> >    pq1 =
                                ::std::make_pair(p1,q1);
                            
                            R4_matrix<float>                    rot = quaternions_to_R4_rotation(pq1);
                            
                            ::std::cout << "rot = ";
                            ::std::cout << "\t" << rot.a11 << "\t" << rot.a12 << "\t" << rot.a13 << "\t" << rot.a14 << ::std::endl;
                            ::std::cout << "\t";
                            ::std::cout << "\t" << rot.a21 << "\t" << rot.a22 << "\t" << rot.a23 << "\t" << rot.a24 << ::std::endl;
                            ::std::cout << "\t";
                            ::std::cout << "\t" << rot.a31 << "\t" << rot.a32 << "\t" << rot.a33 << "\t" << rot.a34 << ::std::endl;
                            ::std::cout << "\t";
                            ::std::cout << "\t" << rot.a41 << "\t" << rot.a42 << "\t" << rot.a43 << "\t" << rot.a44 << ::std::endl;
                            
                            ::std::pair< ::boost::math::quaternion<float> , ::boost::math::quaternion<float> >    pq2 =
                                R4_rotation_to_quaternions(rot, &pq1);
                            
                            ::std::cout << "p1 = " << pq2.first << ::std::endl;
                            ::std::cout << "p2 = " << pq2.second << ::std::endl;
                            
                            ::std::cout << "round trip discrepancy: " << ::std::sqrt(::boost::math::norm(pq1.first-pq2.first)+::boost::math::norm(pq1.second-pq2.second)) << ::std::endl;
                            
                            ::std::cout << ::std::endl;
                        }
                    }
                }
            }
        }
    }
    
    ::std::cout << ::std::endl;
}


void    test_SO4()
{
    ::std::cout << "Testing SO4:" << ::std::endl;
    ::std::cout << ::std::endl;
    
    test_SO4_spherical();
}


