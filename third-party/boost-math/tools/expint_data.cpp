//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;

struct expint_data_generator
{
   mp_t operator()(mp_t a, mp_t b)
   {
      unsigned n = boost::math::tools::real_cast<unsigned>(a);
      std::cout << n << "  " << b << "  ";
      mp_t result = boost::math::expint(n, b);
      std::cout << result << std::endl;
      return result;
   }
};


int main()
{
   boost::math::expint(1, 0.06227754056453704833984375);
   std::cout << boost::math::expint(1, mp_t(0.5)) << std::endl;

   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the expint function:\n"
      "  expint(a, b)\n\n";

   bool cont;
   std::string line;

   do{
      get_user_parameter_info(arg1, "a");
      get_user_parameter_info(arg2, "b");
      data.insert(expint_data_generator(), arg1, arg2);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=expint_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "expint_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "expint_data");
   
   return 0;
}

