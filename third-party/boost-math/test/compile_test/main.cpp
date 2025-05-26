//  Copyright John Maddock 2009.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern void compile_and_link_test();

int main(int argc, char* /*argv*/ [])
{
   if(argc > 1000)
      compile_and_link_test(); // should never actually be called.
   return 0;
}
