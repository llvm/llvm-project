// test file for quaternion.hpp

//  (C) Copyright Hubert Holin 2001.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "quaternion_mi2.h"


#include <boost/math/quaternion.hpp>

void    quaternion_mi2()
{
    ::boost::math::quaternion<float>    q0;
    
    q0 *= q0;
}
