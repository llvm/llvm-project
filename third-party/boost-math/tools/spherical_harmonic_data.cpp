//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

float extern_val;
// confuse the compilers optimiser, and force a truncation to float precision:
float truncate_to_float(float const * pf)
{
   extern_val = *pf;
   return *pf;
}



template<class T>
boost::math::tuple<T, T, T, T, T, T> spherical_harmonic_data(T i)
{
   static boost::mt19937 r;

   int n = real_cast<int>(floor(i));
   boost::uniform_int<> ui(0, (std::min)(n, 40));
   int m = ui(r);
   
   boost::uniform_real<float> ur(-2*constants::pi<float>(), 2*constants::pi<float>());
   float _theta = ur(r);
   float _phi = ur(r);
   T theta = truncate_to_float(&_theta);
   T phi = truncate_to_float(&_phi);

   T r1 = spherical_harmonic_r(n, m, theta, phi);
   T r2 = spherical_harmonic_i(n, m, theta, phi);
   return boost::math::make_tuple(n, m, theta, phi, r1, r2);
}

int main(int argc, char*argv [])
{
   using namespace boost::math::tools;

   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   do{
      if(0 == get_user_parameter_info(arg1, "n"))
         return 1;
      arg1.type |= dummy_param;
      arg2.type |= dummy_param;
      arg3 = arg2;

      data.insert(&spherical_harmonic_data<mp_t>, arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=spherical_harmonic.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "spherical_harmonic.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}

