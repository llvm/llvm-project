//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/test_data.hpp>
#include <fstream>

using namespace boost::math::tools;

struct beta_data_generator
{
   mp_t operator()(mp_t a, mp_t b)
   {
      if(a < b)
         throw std::domain_error("");
      // very naively calculate spots:
      mp_t g1, g2, g3;
      int s1, s2, s3;
      g1 = boost::math::lgamma(a, &s1);
      g2 = boost::math::lgamma(b, &s2);
      g3 = boost::math::lgamma(a+b, &s3);
      g1 += g2 - g3;
      g1 = exp(g1);
      g1 *= s1 * s2 * s3;
      return g1;
   }
};


int main()
{
   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the beta function:\n"
      "  beta(a, b)\n\n";

   bool cont;
   std::string line;

   do{
      get_user_parameter_info(arg1, "a");
      get_user_parameter_info(arg2, "b");
      data.insert(beta_data_generator(), arg1, arg2);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=beta_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "beta_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "beta_data");
   
   return 0;
}

