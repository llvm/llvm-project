//  Copyright John Maddock 2013.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern "C" _Quad __fabs(_Quad);

int main()
{
   _Quad f = -2.0Q;
   f = __fabsq(f);

   return 0;
}


