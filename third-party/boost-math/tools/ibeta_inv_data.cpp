//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

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

boost::mt19937 rnd;
boost::uniform_real<float> ur_a(1.0F, 5.0F);
boost::variate_generator<boost::mt19937, boost::uniform_real<float> > gen(rnd, ur_a);
boost::uniform_real<float> ur_a2(0.0F, 100.0F);
boost::variate_generator<boost::mt19937, boost::uniform_real<float> > gen2(rnd, ur_a2);

struct ibeta_inv_data_generator
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> operator()(mp_t ap, mp_t bp, mp_t x_)
   {
      float a = truncate_to_float(real_cast<float>(gen() * pow(mp_t(10), ap)));      
      float b = truncate_to_float(real_cast<float>(gen() * pow(mp_t(10), bp))); 
      float x = truncate_to_float(real_cast<float>(x_));
      std::cout << a << " " << b << " " << x << std::flush;
      mp_t inv = boost::math::ibeta_inv(mp_t(a), mp_t(b), mp_t(x));
      std::cout << " " << inv << std::flush;
      mp_t invc = boost::math::ibetac_inv(mp_t(a), mp_t(b), mp_t(x));
      std::cout << " " << invc << std::endl;
      return boost::math::make_tuple(a, b, x, inv, invc);
   }
};

int main(int argc, char*argv [])
{
   bool cont;
   std::string line;

   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the inverse incomplete beta function:\n"
      "  ibeta_inv(a, p) and ibetac_inv(a, q)\n\n";

   arg1 = make_periodic_param(mp_t(-5), mp_t(6), 11);
   arg2 = make_periodic_param(mp_t(-5), mp_t(6), 11);
   arg3 = make_random_param(mp_t(0.0001), mp_t(1), 10);

   arg1.type |= dummy_param;
   arg2.type |= dummy_param;
   arg3.type |= dummy_param;

   data.insert(ibeta_inv_data_generator(), arg1, arg2, arg3);

   line = "ibeta_inv_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "ibeta_inv_data");
   
   return 0;
}

