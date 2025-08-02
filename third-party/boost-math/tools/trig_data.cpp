//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/constants/constants.hpp>
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

struct trig_data_generator
{
   boost::math::tuple<mp_t, mp_t> operator()(mp_t z)
   {
      return boost::math::make_tuple(sin(z * boost::math::constants::pi<mp_t>()), cos(z * boost::math::constants::pi<mp_t>()));
   }
};


int main(int argc, char*argv [])
{
   parameter_info<mp_t> arg1;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the cos_pi and sin_pi functions:\n";

   do{
      if(0 == get_user_parameter_info(arg1, "a"))
         return 1;
      data.insert(trig_data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=trig_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "trig_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "trig_data");
   
   return 0;
}

/* Output for asymptotic limits:

Erf asymptotic limit for 24 bit numbers is 2.8 after approximately 6 terms.
Erfc asymptotic limit for 24 bit numbers is 4.12064 after approximately 17 terms.
Erf asymptotic limit for 53 bit numbers is 4.3 after approximately 11 terms.
Erfc asymptotic limit for 53 bit numbers is 6.19035 after approximately 29 terms.
Erf asymptotic limit for 64 bit numbers is 4.8 after approximately 12 terms.
Erfc asymptotic limit for 64 bit numbers is 7.06004 after approximately 29 terms.
Erf asymptotic limit for 106 bit numbers is 6.5 after approximately 14 terms.
Erfc asymptotic limit for 106 bit numbers is 11.6626 after approximately 29 terms.
Erf asymptotic limit for 113 bit numbers is 6.8 after approximately 14 terms.
Erfc asymptotic limit for 113 bit numbers is 12.6802 after approximately 29 terms.
*/

