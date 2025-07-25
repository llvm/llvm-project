
/********************************************************************************************/
/*                                                                                          */
/*                                HSO3.hpp header file                                      */
/*                                                                                          */
/* This file is not currently part of the Boost library. It is simply an example of the use */
/* quaternions can be put to. Hopefully it will be useful too.                              */
/*                                                                                          */
/* This file provides tools to convert between quaternions and R^3 rotation matrices.       */
/*                                                                                          */
/********************************************************************************************/

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef TEST_HSO3_HPP
#define TEST_HSO3_HPP

#include <algorithm>

#if    defined(__GNUC__) && (__GNUC__ < 3)
#include <boost/limits.hpp>
#else
#include <limits>
#endif

#include <stdexcept>
#include <string>

#include <boost/math/quaternion.hpp>


#if    defined(__GNUC__) && (__GNUC__ < 3)
// gcc 2.x ignores function scope using declarations, put them here instead:
using    namespace ::std;
using    namespace ::boost::math;
#endif

template<typename TYPE_FLOAT>
struct  R3_matrix
{
    TYPE_FLOAT a11, a12, a13;
    TYPE_FLOAT a21, a22, a23;
    TYPE_FLOAT a31, a32, a33;
};


// Note:    the input quaternion need not be of norm 1 for the following function

template<typename TYPE_FLOAT>
R3_matrix<TYPE_FLOAT>    quaternion_to_R3_rotation(::boost::math::quaternion<TYPE_FLOAT> const & q)
{
    using    ::std::numeric_limits;
    
    TYPE_FLOAT    a = q.R_component_1();
    TYPE_FLOAT    b = q.R_component_2();
    TYPE_FLOAT    c = q.R_component_3();
    TYPE_FLOAT    d = q.R_component_4();
    
    TYPE_FLOAT    aa = a*a;
    TYPE_FLOAT    ab = a*b;
    TYPE_FLOAT    ac = a*c;
    TYPE_FLOAT    ad = a*d;
    TYPE_FLOAT    bb = b*b;
    TYPE_FLOAT    bc = b*c;
    TYPE_FLOAT    bd = b*d;
    TYPE_FLOAT    cc = c*c;
    TYPE_FLOAT    cd = c*d;
    TYPE_FLOAT    dd = d*d;
    
    TYPE_FLOAT    norme_carre = aa+bb+cc+dd;
    
    if    (norme_carre <= numeric_limits<TYPE_FLOAT>::epsilon())
    {
        ::std::string            error_reporting("Argument to quaternion_to_R3_rotation is too small!");
        ::std::underflow_error   bad_argument(error_reporting);
        
        throw(bad_argument);
    }
    
    R3_matrix<TYPE_FLOAT>    out_matrix;
    
    out_matrix.a11 = (aa+bb-cc-dd)/norme_carre;
    out_matrix.a12 = 2*(-ad+bc)/norme_carre;
    out_matrix.a13 = 2*(ac+bd)/norme_carre;
    out_matrix.a21 = 2*(ad+bc)/norme_carre;
    out_matrix.a22 = (aa-bb+cc-dd)/norme_carre;
    out_matrix.a23 = 2*(-ab+cd)/norme_carre;
    out_matrix.a31 = 2*(-ac+bd)/norme_carre;
    out_matrix.a32 = 2*(ab+cd)/norme_carre;
    out_matrix.a33 = (aa-bb-cc+dd)/norme_carre;
    
    return(out_matrix);
}


    template<typename TYPE_FLOAT>
    void    find_invariant_vector(  R3_matrix<TYPE_FLOAT> const & rot,
                                    TYPE_FLOAT & x,
                                    TYPE_FLOAT & y,
                                    TYPE_FLOAT & z)
    {
        using    ::std::sqrt;
        
        using    ::std::numeric_limits;
        
        TYPE_FLOAT    b11 = rot.a11 - static_cast<TYPE_FLOAT>(1);
        TYPE_FLOAT    b12 = rot.a12;
        TYPE_FLOAT    b13 = rot.a13;
        TYPE_FLOAT    b21 = rot.a21;
        TYPE_FLOAT    b22 = rot.a22 - static_cast<TYPE_FLOAT>(1);
        TYPE_FLOAT    b23 = rot.a23;
        TYPE_FLOAT    b31 = rot.a31;
        TYPE_FLOAT    b32 = rot.a32;
        TYPE_FLOAT    b33 = rot.a33 - static_cast<TYPE_FLOAT>(1);
        
        TYPE_FLOAT    minors[9] =
        {
            b11*b22-b12*b21,
            b11*b23-b13*b21,
            b12*b23-b13*b22,
            b11*b32-b12*b31,
            b11*b33-b13*b31,
            b12*b33-b13*b32,
            b21*b32-b22*b31,
            b21*b33-b23*b31,
            b22*b33-b23*b32
        };
        
        TYPE_FLOAT *        where = ::std::max_element(minors, minors+9);
        
        TYPE_FLOAT          det = *where;
        
        if    (det <= numeric_limits<TYPE_FLOAT>::epsilon())
        {
            ::std::string            error_reporting("Underflow error in find_invariant_vector!");
            ::std::underflow_error   processing_error(error_reporting);
            
            throw(processing_error);
        }
        
        switch    (where-minors)
        {
            case 0:
                
                z = static_cast<TYPE_FLOAT>(1);
                
                x = (-b13*b22+b12*b23)/det;
                y = (-b11*b23+b13*b21)/det;
                
                break;
                
            case 1:
                
                y = static_cast<TYPE_FLOAT>(1);
                
                x = (-b12*b23+b13*b22)/det;
                z = (-b11*b22+b12*b21)/det;
                
                break;
                
            case 2:
                
                x = static_cast<TYPE_FLOAT>(1);
                
                y = (-b11*b23+b13*b21)/det;
                z = (-b12*b21+b11*b22)/det;
                
                break;
                
            case 3:
                
                z = static_cast<TYPE_FLOAT>(1);
                
                x = (-b13*b32+b12*b33)/det;
                y = (-b11*b33+b13*b31)/det;
                
                break;
                
            case 4:
                
                y = static_cast<TYPE_FLOAT>(1);
                
                x = (-b12*b33+b13*b32)/det;
                z = (-b11*b32+b12*b31)/det;
                
                break;
                
            case 5:
                
                x = static_cast<TYPE_FLOAT>(1);
                
                y = (-b11*b33+b13*b31)/det;
                z = (-b12*b31+b11*b32)/det;
                
                break;
                
            case 6:
                
                z = static_cast<TYPE_FLOAT>(1);
                
                x = (-b23*b32+b22*b33)/det;
                y = (-b21*b33+b23*b31)/det;
                
                break;
                
            case 7:
                
                y = static_cast<TYPE_FLOAT>(1);
                
                x = (-b22*b33+b23*b32)/det;
                z = (-b21*b32+b22*b31)/det;
                
                break;
                
            case 8:
                
                x = static_cast<TYPE_FLOAT>(1);
                
                y = (-b21*b33+b23*b31)/det;
                z = (-b22*b31+b21*b32)/det;
                
                break;
                
            default:
                
                ::std::string        error_reporting("Impossible condition in find_invariant_vector");
                ::std::logic_error   processing_error(error_reporting);
                
                throw(processing_error);
                
                break;
        }
        
        TYPE_FLOAT    vecnorm = sqrt(x*x+y*y+z*z);
        
        if    (vecnorm <= numeric_limits<TYPE_FLOAT>::epsilon())
        {
            ::std::string            error_reporting("Overflow error in find_invariant_vector!");
            ::std::overflow_error    processing_error(error_reporting);
            
            throw(processing_error);
        }
        
        x /= vecnorm;
        y /= vecnorm;
        z /= vecnorm;
    }
    
    
    template<typename TYPE_FLOAT>
    void    find_orthogonal_vector( TYPE_FLOAT x,
                                    TYPE_FLOAT y,
                                    TYPE_FLOAT z,
                                    TYPE_FLOAT & u,
                                    TYPE_FLOAT & v,
                                    TYPE_FLOAT & w)
    {
        using    ::std::abs;
        using    ::std::sqrt;
        
        using    ::std::numeric_limits;
        
        TYPE_FLOAT    vecnormsqr = x*x+y*y+z*z;
        
        if    (vecnormsqr <= numeric_limits<TYPE_FLOAT>::epsilon())
        {
            ::std::string            error_reporting("Underflow error in find_orthogonal_vector!");
            ::std::underflow_error   processing_error(error_reporting);
            
            throw(processing_error);
        }
        
        TYPE_FLOAT        lambda;
        
        TYPE_FLOAT        components[3] =
        {
            abs(x),
            abs(y),
            abs(z)
        };
        
        TYPE_FLOAT *    where = ::std::min_element(components, components+3);
        
        switch    (where-components)
        {
            case 0:
                
                if    (*where <= numeric_limits<TYPE_FLOAT>::epsilon())
                {
                    v =
                    w = static_cast<TYPE_FLOAT>(0);
                    u = static_cast<TYPE_FLOAT>(1);
                }
                else
                {
                    lambda = -x/vecnormsqr;
                    
                    u = static_cast<TYPE_FLOAT>(1) + lambda*x;
                    v = lambda*y;
                    w = lambda*z;
                }
                
                break;
                
            case 1:
                
                if    (*where <= numeric_limits<TYPE_FLOAT>::epsilon())
                {
                    u =
                    w = static_cast<TYPE_FLOAT>(0);
                    v = static_cast<TYPE_FLOAT>(1);
                }
                else
                {
                    lambda = -y/vecnormsqr;
                    
                    u = lambda*x;
                    v = static_cast<TYPE_FLOAT>(1) + lambda*y;
                    w = lambda*z;
                }
                
                break;
                
            case 2:
                
                if    (*where <= numeric_limits<TYPE_FLOAT>::epsilon())
                {
                    u =
                    v = static_cast<TYPE_FLOAT>(0);
                    w = static_cast<TYPE_FLOAT>(1);
                }
                else
                {
                    lambda = -z/vecnormsqr;
                    
                    u = lambda*x;
                    v = lambda*y;
                    w = static_cast<TYPE_FLOAT>(1) + lambda*z;
                }
                
                break;
                
            default:
                
                ::std::string        error_reporting("Impossible condition in find_invariant_vector");
                ::std::logic_error   processing_error(error_reporting);
                
                throw(processing_error);
                
                break;
        }
        
        TYPE_FLOAT    vecnorm = sqrt(u*u+v*v+w*w);
        
        if    (vecnorm <= numeric_limits<TYPE_FLOAT>::epsilon())
        {
            ::std::string            error_reporting("Underflow error in find_orthogonal_vector!");
            ::std::underflow_error   processing_error(error_reporting);
            
            throw(processing_error);
        }
        
        u /= vecnorm;
        v /= vecnorm;
        w /= vecnorm;
    }
    
    
    // Note:    we want [[v, v, w], [r, s, t], [x, y, z]] to be a direct orthogonal basis
    //            of R^3. It might not be orthonormal, however, and we do not check if the
    //            two input vectors are colinear or not.
    
    template<typename TYPE_FLOAT>
    void    find_vector_for_BOD(TYPE_FLOAT x,
                                TYPE_FLOAT y,
                                TYPE_FLOAT z,
                                TYPE_FLOAT u, 
                                TYPE_FLOAT v,
                                TYPE_FLOAT w,
                                TYPE_FLOAT & r,
                                TYPE_FLOAT & s,
                                TYPE_FLOAT & t)
    {
        r = +y*w-z*v;
        s = -x*w+z*u;
        t = +x*v-y*u;
    }



template<typename TYPE_FLOAT>
inline bool                                is_R3_rotation_matrix(R3_matrix<TYPE_FLOAT> const & mat)
{
    using    ::std::abs;
    
    using    ::std::numeric_limits;
    
    return    (
                !(
                    (abs(mat.a11*mat.a11+mat.a21*mat.a21+mat.a31*mat.a31 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a11*mat.a12+mat.a21*mat.a22+mat.a31*mat.a32 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a11*mat.a13+mat.a21*mat.a23+mat.a31*mat.a33 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a11*mat.a12+mat.a21*mat.a22+mat.a31*mat.a32 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a12*mat.a12+mat.a22*mat.a22+mat.a32*mat.a32 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a12*mat.a13+mat.a22*mat.a23+mat.a32*mat.a33 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a11*mat.a13+mat.a21*mat.a23+mat.a31*mat.a33 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    //(abs(mat.a12*mat.a13+mat.a22*mat.a23+mat.a32*mat.a33 - static_cast<TYPE_FLOAT>(0)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())||
                    (abs(mat.a13*mat.a13+mat.a23*mat.a23+mat.a33*mat.a33 - static_cast<TYPE_FLOAT>(1)) > static_cast<TYPE_FLOAT>(10)*numeric_limits<TYPE_FLOAT>::epsilon())
                )
            );
}


template<typename TYPE_FLOAT>
::boost::math::quaternion<TYPE_FLOAT>    R3_rotation_to_quaternion(    R3_matrix<TYPE_FLOAT> const & rot,
                                                                    ::boost::math::quaternion<TYPE_FLOAT> const * hint = 0)
{
    using    ::boost::math::abs;
    
    using    ::std::abs;
    using    ::std::sqrt;
    
    using    ::std::numeric_limits;
    
    if    (!is_R3_rotation_matrix(rot))
    {
        ::std::string        error_reporting("Argument to R3_rotation_to_quaternion is not an R^3 rotation matrix!");
        ::std::range_error   bad_argument(error_reporting);
        
        throw(bad_argument);
    }
    
    ::boost::math::quaternion<TYPE_FLOAT>    q;
    
    if    (
            (abs(rot.a11 - static_cast<TYPE_FLOAT>(1)) <= numeric_limits<TYPE_FLOAT>::epsilon())&&
            (abs(rot.a22 - static_cast<TYPE_FLOAT>(1)) <= numeric_limits<TYPE_FLOAT>::epsilon())&&
            (abs(rot.a33 - static_cast<TYPE_FLOAT>(1)) <= numeric_limits<TYPE_FLOAT>::epsilon())
        )
    {
        q = ::boost::math::quaternion<TYPE_FLOAT>(1);
    }
    else
    {
        TYPE_FLOAT    cos_theta = (rot.a11+rot.a22+rot.a33-static_cast<TYPE_FLOAT>(1))/static_cast<TYPE_FLOAT>(2);
        TYPE_FLOAT    stuff = (cos_theta+static_cast<TYPE_FLOAT>(1))/static_cast<TYPE_FLOAT>(2);
        TYPE_FLOAT    cos_theta_sur_2 = sqrt(stuff);
        TYPE_FLOAT    sin_theta_sur_2 = sqrt(1-stuff);
        
        TYPE_FLOAT    x;
        TYPE_FLOAT    y;
        TYPE_FLOAT    z;
        
        find_invariant_vector(rot, x, y, z);
        
        TYPE_FLOAT    u;
        TYPE_FLOAT    v;
        TYPE_FLOAT    w;
        
        find_orthogonal_vector(x, y, z, u, v, w);
        
        TYPE_FLOAT    r;
        TYPE_FLOAT    s;
        TYPE_FLOAT    t;
        
        find_vector_for_BOD(x, y, z, u, v, w, r, s, t);
        
        TYPE_FLOAT    ru = rot.a11*u+rot.a12*v+rot.a13*w;
        TYPE_FLOAT    rv = rot.a21*u+rot.a22*v+rot.a23*w;
        TYPE_FLOAT    rw = rot.a31*u+rot.a32*v+rot.a33*w;
        
        TYPE_FLOAT    angle_sign_determinator = r*ru+s*rv+t*rw;
        
        if        (angle_sign_determinator > +numeric_limits<TYPE_FLOAT>::epsilon())
        {
            q = ::boost::math::quaternion<TYPE_FLOAT>(cos_theta_sur_2, +x*sin_theta_sur_2, +y*sin_theta_sur_2, +z*sin_theta_sur_2);
        }
        else if    (angle_sign_determinator < -numeric_limits<TYPE_FLOAT>::epsilon())
        {
            q = ::boost::math::quaternion<TYPE_FLOAT>(cos_theta_sur_2, -x*sin_theta_sur_2, -y*sin_theta_sur_2, -z*sin_theta_sur_2);
        }
        else
        {
            TYPE_FLOAT    desambiguator = u*ru+v*rv+w*rw;
            
            if    (desambiguator >= static_cast<TYPE_FLOAT>(1))
            {
                q = ::boost::math::quaternion<TYPE_FLOAT>(0, +x, +y, +z);
            }
            else
            {
                q = ::boost::math::quaternion<TYPE_FLOAT>(0, -x, -y, -z);
            }
        }
    }
    
    if    ((hint != 0) && (abs(*hint+q) < abs(*hint-q)))
    {
        return(-q);
    }
    
    return(q);
}

#endif /* TEST_HSO3_HPP */

