//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;

//
// Force truncation to float precision of input values:
// we must ensure that the input values are exactly representable
// in whatever type we are testing, or the output values will all
// be thrown off:
//
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

struct gamma_inverse_generator_a
{
   boost::math::tuple<mp_t, mp_t> operator()(const mp_t x, const mp_t p)
   {
      mp_t x1 = boost::math::gamma_p_inva(x, p);
      mp_t x2 = boost::math::gamma_q_inva(x, p);
      std::cout << "Inverse for " << x << " " << p << std::endl;
      return boost::math::make_tuple(x1, x2);
   }
};


int main(int argc, char*argv [])
{
   bool cont;
   std::string line;

   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the inverse incomplete gamma function:\n"
      "  gamma_p_inva(a, p) and gamma_q_inva(a, q)\n\n";

   arg1 = make_power_param<mp_t>(mp_t(0), -4, 24);
   arg2 = make_random_param<mp_t>(mp_t(0), mp_t(1), 15);
   data.insert(gamma_inverse_generator_a(), arg1, arg2);
 
   line = "igamma_inva_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "igamma_inva_data");
   
   return 0;
}

