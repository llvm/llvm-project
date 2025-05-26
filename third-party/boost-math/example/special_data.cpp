// Copyright John Maddock 2006.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[special_data_example

#ifndef BOOST_MATH_STANDALONE

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/tools/test_data.hpp>
#include <boost/test/included/prg_exec_monitor.hpp>
#include <fstream>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;
using namespace boost::multiprecision;

template <class T>
T my_special(T a, T b)
{
   // Implementation of my_special here...
   return a + b;
}

int cpp_main(int argc, char*argv [])
{
   //
   // We'll use so many digits of precision that any
   // calculation errors will still leave us with
   // 40-50 good digits.  We'll only run this program
   // once so it doesn't matter too much how long this takes!
   //
   typedef number<cpp_dec_float<500> > bignum;

   parameter_info<bignum> arg1, arg2;
   test_data<bignum> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   do{
      //
      // User interface which prompts for 
      // range of input parameters:
      //
      if(0 == get_user_parameter_info(arg1, "a"))
         return 1;
      if(0 == get_user_parameter_info(arg2, "b"))
         return 1;

      //
      // Get a pointer to the function and call
      // test_data::insert to actually generate
      // the values.
      //
      bignum (*fp)(bignum, bignum) = &my_special;
      data.insert(fp, arg2, arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);
   //
   // Just need to write the results to a file:
   //
   std::cout << "Enter name of test data file [default=my_special.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "my_special.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(50);
   write_code(ofs, data, line.c_str());

   return 0;
}

//]

#endif // BOOST_MATH_STANDALONE
