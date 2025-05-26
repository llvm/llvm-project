
/********************************************************************************************/
/*                                                                                          */
/*                                HSO4.hpp header file                                      */
/*                                                                                          */
/* This file is not currently part of the Boost library. It is simply an example of the use */
/* quaternions can be put to. Hopefully it will be useful too.                             */
/*                                                                                          */
/* This file provides tools to convert between quaternions and R^4 rotation matrices.       */
/*                                                                                          */
/********************************************************************************************/

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef TEST_HSO4_HPP
#define TEST_HSO4_HPP

#include <utility>

#include "HSO3.hpp"


template<typename TYPE_FLOAT>
struct  R4_matrix
{
    TYPE_FLOAT a11, a12, a13, a14;
    TYPE_FLOAT a21, a22, a23, a24;
    TYPE_FLOAT a31, a32, a33, a34;
    TYPE_FLOAT a41, a42, a43, a44;
};


// Note:    the input quaternions need not be of norm 1 for the following function

template<typename TYPE_FLOAT>
R4_matrix<TYPE_FLOAT>    quaternions_to_R4_rotation(::std::pair< ::boost::math::quaternion<TYPE_FLOAT> , ::boost::math::quaternion<TYPE_FLOAT> > const & pq)
{
    using    ::std::numeric_limits;
    
    TYPE_FLOAT    a0 = pq.first.R_component_1();
    TYPE_FLOAT    b0 = pq.first.R_component_2();
    TYPE_FLOAT    c0 = pq.first.R_component_3();
    TYPE_FLOAT    d0 = pq.first.R_component_4();
    
    TYPE_FLOAT    norme_carre0 = a0*a0+b0*b0+c0*c0+d0*d0;
    
    if    (norme_carre0 <= numeric_limits<TYPE_FLOAT>::epsilon())
    {
        ::std::string            error_reporting("Argument to quaternions_to_R4_rotation is too small!");
        ::std::underflow_error   bad_argument(error_reporting);
        
        throw(bad_argument);
    }
    
    TYPE_FLOAT    a1 = pq.second.R_component_1();
    TYPE_FLOAT    b1 = pq.second.R_component_2();
    TYPE_FLOAT    c1 = pq.second.R_component_3();
    TYPE_FLOAT    d1 = pq.second.R_component_4();
    
    TYPE_FLOAT    norme_carre1 = a1*a1+b1*b1+c1*c1+d1*d1;
    
    if    (norme_carre1 <= numeric_limits<TYPE_FLOAT>::epsilon())
    {
        ::std::string            error_reporting("Argument to quaternions_to_R4_rotation is too small!");
        ::std::underflow_error   bad_argument(error_reporting);
        
        throw(bad_argument);
    }
    
    TYPE_FLOAT    prod_norm = norme_carre0*norme_carre1;
    
    TYPE_FLOAT    a0a1 = a0*a1;
    TYPE_FLOAT    a0b1 = a0*b1;
    TYPE_FLOAT    a0c1 = a0*c1;
    TYPE_FLOAT    a0d1 = a0*d1;
    TYPE_FLOAT    b0a1 = b0*a1;
    TYPE_FLOAT    b0b1 = b0*b1;
    TYPE_FLOAT    b0c1 = b0*c1;
    TYPE_FLOAT    b0d1 = b0*d1;
    TYPE_FLOAT    c0a1 = c0*a1;
    TYPE_FLOAT    c0b1 = c0*b1;
    TYPE_FLOAT    c0c1 = c0*c1;
    TYPE_FLOAT    c0d1 = c0*d1;
    TYPE_FLOAT    d0a1 = d0*a1;
    TYPE_FLOAT    d0b1 = d0*b1;
    TYPE_FLOAT    d0c1 = d0*c1;
    TYPE_FLOAT    d0d1 = d0*d1;
    
    R4_matrix<TYPE_FLOAT>    out_matrix;
    
    out_matrix.a11 = (+a0a1+b0b1+c0c1+d0d1)/prod_norm;
    out_matrix.a12 = (+a0b1-b0a1-c0d1+d0c1)/prod_norm;
    out_matrix.a13 = (+a0c1+b0d1-c0a1-d0b1)/prod_norm;
    out_matrix.a14 = (+a0d1-b0c1+c0b1-d0a1)/prod_norm;
    out_matrix.a21 = (-a0b1+b0a1-c0d1+d0c1)/prod_norm;
    out_matrix.a22 = (+a0a1+b0b1-c0c1-d0d1)/prod_norm;
    out_matrix.a23 = (-a0d1+b0c1+c0b1-d0a1)/prod_norm;
    out_matrix.a24 = (+a0c1+b0d1+c0a1+d0b1)/prod_norm;
    out_matrix.a31 = (-a0c1+b0d1+c0a1-d0b1)/prod_norm;
    out_matrix.a32 = (+a0d1+b0c1+c0b1+d0a1)/prod_norm;
    out_matrix.a33 = (+a0a1-b0b1+c0c1-d0d1)/prod_norm;
    out_matrix.a34 = (-a0b1-b0a1+c0d1+d0c1)/prod_norm;
    out_matrix.a41 = (-a0d1-b0c1+c0b1+d0a1)/prod_norm;
    out_matrix.a42 = (-a0c1+b0d1-c0a1+d0b1)/prod_norm;
    out_matrix.a43 = (+a0b1+b0a1+c0d1+d0c1)/prod_norm;
    out_matrix.a44 = (+a0a1-b0b1-c0c1+d0d1)/prod_norm;
    
    return(out_matrix);
}


template<typename TYPE_FLOAT>
inline bool                        is_R4_rotation_matrix(R4_matrix<TYPE_FLOAT> const & mat)
{
    using    ::std::abs;
    
    using    ::std::numeric_limits;
    
    return    (
                !(
                    (abs(mat.a11*mat.a11+mat.a21*mat.a21+mat.a31*mat.a31+mat.a41*mat.a41 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a11*mat.a12+mat.a21*mat.a22+mat.a31*mat.a32+mat.a41*mat.a42 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a11*mat.a13+mat.a21*mat.a23+mat.a31*mat.a33+mat.a41*mat.a43 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a11*mat.a14+mat.a21*mat.a24+mat.a31*mat.a34+mat.a41*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a11*mat.a12+mat.a21*mat.a22+mat.a31*mat.a32+mat.a41*mat.a42 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a12*mat.a12+mat.a22*mat.a22+mat.a32*mat.a32+mat.a42*mat.a42 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a12*mat.a13+mat.a22*mat.a23+mat.a32*mat.a33+mat.a42*mat.a43 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a12*mat.a14+mat.a22*mat.a24+mat.a32*mat.a34+mat.a42*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a11*mat.a13+mat.a21*mat.a23+mat.a31*mat.a33+mat.a41*mat.a43 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a12*mat.a13+mat.a22*mat.a23+mat.a32*mat.a33+mat.a42*mat.a43 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a13*mat.a13+mat.a23*mat.a23+mat.a33*mat.a33+mat.a43*mat.a43 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a13*mat.a14+mat.a23*mat.a24+mat.a33*mat.a34+mat.a43*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a11*mat.a14+mat.a21*mat.a24+mat.a31*mat.a34+mat.a41*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a12*mat.a14+mat.a22*mat.a24+mat.a32*mat.a34+mat.a42*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a13*mat.a14+mat.a23*mat.a24+mat.a33*mat.a34+mat.a43*mat.a44 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a14*mat.a14+mat.a24*mat.a24+mat.a34*mat.a34+mat.a44*mat.a44 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())
                )
            );
}


template<typename TYPE_FLOAT>
::std::pair< ::boost::math::quaternion<TYPE_FLOAT> , ::boost::math::quaternion<TYPE_FLOAT> >    R4_rotation_to_quaternions(    R4_matrix<TYPE_FLOAT> const & rot,
                                                                                                                            ::std::pair< ::boost::math::quaternion<TYPE_FLOAT> , ::boost::math::quaternion<TYPE_FLOAT> > const * hint = 0)
{
    if    (!is_R4_rotation_matrix(rot))
    {
        ::std::string        error_reporting("Argument to R4_rotation_to_quaternions is not an R^4 rotation matrix!");
        ::std::range_error   bad_argument(error_reporting);
        
        throw(bad_argument);
    }
    
    R3_matrix<TYPE_FLOAT>    mat;
    
    mat.a11 = -rot.a31*rot.a42+rot.a32*rot.a41+rot.a22*rot.a11-rot.a21*rot.a12;
    mat.a12 = -rot.a31*rot.a43+rot.a33*rot.a41+rot.a23*rot.a11-rot.a21*rot.a13;
    mat.a13 = -rot.a31*rot.a44+rot.a34*rot.a41+rot.a24*rot.a11-rot.a21*rot.a14;
    mat.a21 = -rot.a31*rot.a12-rot.a22*rot.a41+rot.a32*rot.a11+rot.a21*rot.a42;
    mat.a22 = -rot.a31*rot.a13-rot.a23*rot.a41+rot.a33*rot.a11+rot.a21*rot.a43;
    mat.a23 = -rot.a31*rot.a14-rot.a24*rot.a41+rot.a34*rot.a11+rot.a21*rot.a44;
    mat.a31 = +rot.a31*rot.a22-rot.a12*rot.a41+rot.a42*rot.a11-rot.a21*rot.a32;
    mat.a32 = +rot.a31*rot.a23-rot.a13*rot.a41+rot.a43*rot.a11-rot.a21*rot.a33;
    mat.a33 = +rot.a31*rot.a24-rot.a14*rot.a41+rot.a44*rot.a11-rot.a21*rot.a34;
    
    ::boost::math::quaternion<TYPE_FLOAT>    q = R3_rotation_to_quaternion(mat);
    
    ::boost::math::quaternion<TYPE_FLOAT>    p =
        ::boost::math::quaternion<TYPE_FLOAT>(rot.a11,rot.a12,rot.a13,rot.a14)*q;
    
    if    ((hint != 0) && (abs(hint->second+q) < abs(hint->second-q)))
    {
        return(::std::make_pair(-p,-q));
    }
    
    return(::std::make_pair(p,q));
}

#endif /* TEST_HSO4_HPP */

