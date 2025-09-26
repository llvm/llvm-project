//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fstream>
#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>

using namespace boost::math::tools;
using namespace std;

struct data_generator
{
   boost::math::tuple<mp_t, mp_t> operator()(mp_t z)
   {
      return boost::math::make_tuple(boost::math::log1p(z), boost::math::expm1(z));
   }
};

int main(int argc, char* argv[])
{
   parameter_info<mp_t> arg1;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the log1p and expm1 functions:\n\n";

   bool cont;
   std::string line;

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=log1p_expm1_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "log1p_expm1_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "log1p_expm1_data");
   
   return 0;
}




