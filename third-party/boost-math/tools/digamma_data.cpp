//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/test/included/prg_exec_monitor.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;
using namespace std;

float external_f;
float force_truncate(const float* f)
{
   external_f = *f;
   return external_f;
}

float truncate_to_float(mp_t r)
{
   float f = boost::math::tools::real_cast<float>(r);
   return force_truncate(&f);
}

int cpp_main(int argc, char*argv [])
{
   parameter_info<mp_t> arg1;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the digamma function:\n"
      "  digamma(z)\n\n";

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(&boost::math::digamma<mp_t>, arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=digamma_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "digamma_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "digamma_data");
   
   return 0;
}



