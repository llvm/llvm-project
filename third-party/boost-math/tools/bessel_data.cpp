//  Copyright (c) 2007 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Computes test data for the various bessel functions using
// archived - deliberately naive - version of the code.
// We'll rely on the high precision of mp_t to get us out of
// trouble and not worry about how long the calculations take.
// This provides a reasonably independent set of test data to
// compare against newly added asymptotic expansions etc.
//
#include <fstream>

#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>
#include <boost/math/special_functions/bessel.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace boost::math::detail;
using namespace std;

// Compute J(v, x) and Y(v, x) simultaneously by Steed's method, see
// Barnett et al, Computer Physics Communications, vol 8, 377 (1974)
template <typename T>
int bessel_jy_bare(T v, T x, T* J, T* Y, int kind = need_j|need_y)
{
    // Jv1 = J_(v+1), Yv1 = Y_(v+1), fv = J_(v+1) / J_v
    // Ju1 = J_(u+1), Yu1 = Y_(u+1), fu = J_(u+1) / J_u
    T u, Jv, Ju, Yv, Yv1, Yu, Yu1, fv, fu;
    T W, p, q, gamma, current, prev, next;
    bool reflect = false;
    int n, k, s;

    using namespace std;
    using namespace boost::math::tools;
    using namespace boost::math::constants;

    if (v < 0)
    {
        reflect = true;
        v = -v;                             // v is non-negative from here
        kind = need_j|need_y;               // need both for reflection formula
    }
    n = real_cast<int>(v + 0.5L);
    u = v - n;                              // -1/2 <= u < 1/2

    if (x < 0)
    {
       *J = *Y = policies::raise_domain_error<T>("",
          "Real argument x=%1% must be non-negative, complex number result not supported", x, policies::policy<>());
        return 1;
    }
    if (x == 0)
    {
       *J = *Y = policies::raise_overflow_error<T>(
          "", 0, policies::policy<>());
       return 1;
    }

    // x is positive until reflection
    W = T(2) / (x * pi<T>());               // Wronskian
    if (x <= 2)                           // x in (0, 2]
    {
       if(temme_jy(u, x, &Yu, &Yu1, policies::policy<>()))             // Temme series
        {
           // domain error:
           *J = *Y = Yu;
           return 1;
        }
        prev = Yu;
        current = Yu1;
        for (k = 1; k <= n; k++)            // forward recurrence for Y
        {
            next = 2 * (u + k) * current / x - prev;
            prev = current;
            current = next;
        }
        Yv = prev;
        Yv1 = current;
        CF1_jy(v, x, &fv, &s, policies::policy<>());                 // continued fraction CF1
        Jv = W / (Yv * fv - Yv1);           // Wronskian relation
    }
    else                                    // x in (2, \infty)
    {
        // Get Y(u, x):
        CF1_jy(v, x, &fv, &s, policies::policy<>());
        // tiny initial value to prevent overflow
        T init = sqrt(tools::min_value<T>());
        prev = fv * s * init;
        current = s * init;
        for (k = n; k > 0; k--)             // backward recurrence for J
        {
            next = 2 * (u + k) * current / x - prev;
            prev = current;
            current = next;
        }
        T ratio = (s * init) / current;     // scaling ratio
        // can also call CF1() to get fu, not much difference in precision
        fu = prev / current;
        CF2_jy(u, x, &p, &q, policies::policy<>());                  // continued fraction CF2
        T t = u / x - fu;                   // t = J'/J
        gamma = (p - t) / q;
        Ju = sign(current) * sqrt(W / (q + gamma * (p - t)));

        Jv = Ju * ratio;                    // normalization

        Yu = gamma * Ju;
        Yu1 = Yu * (u/x - p - q/gamma);

        // compute Y:
        prev = Yu;
        current = Yu1;
        for (k = 1; k <= n; k++)            // forward recurrence for Y
        {
            next = 2 * (u + k) * current / x - prev;
            prev = current;
            current = next;
        }
        Yv = prev;
    }

    if (reflect)
    {
        T z = (u + n % 2) * pi<T>();
        *J = cos(z) * Jv - sin(z) * Yv;     // reflection formula
        *Y = sin(z) * Jv + cos(z) * Yv;
    }
    else
    {
        *J = Jv;
        *Y = Yv;
    }

    return 0;
}

int progress = 0;

template <class T>
T cyl_bessel_j_bare(T v, T x)
{
   T j, y;
   bessel_jy_bare(v, x, &j, &y);

   std::cout << progress++ << ":   J(" << v << ", " << x << ") = " << j << std::endl;

   if(fabs(j) > 1e30)
      throw std::domain_error("");

   return j;
}

template <class T>
T cyl_bessel_i_bare(T v, T x)
{
   using namespace std;
   if(x < 0)
   {
      // better have integer v:
      if(floor(v) == v)
      {
         T r = cyl_bessel_i_bare(v, -x);
         if(tools::real_cast<int>(v) & 1)
            r = -r;
         return r;
      }
      else
         return policies::raise_domain_error<T>(
            "",
            "Got x = %1%, but we need x >= 0", x, policies::policy<>());
   }
   if(x == 0)
   {
      return (v == 0) ? 1 : 0;
   }
   T I, K;
   boost::math::detail::bessel_ik(v, x, &I, &K, 0xffff, policies::policy<>());

   std::cout << progress++ << ":   I(" << v << ", " << x << ") = " << I << std::endl;

   if(fabs(I) > 1e30)
      throw std::domain_error("");

   return I;
}

template <class T>
T cyl_bessel_k_bare(T v, T x)
{
   using namespace std;
   if(x < 0)
   {
      return policies::raise_domain_error<T>(
         "",
         "Got x = %1%, but we need x > 0", x, policies::policy<>());
   }
   if(x == 0)
   {
      return (v == 0) ? policies::raise_overflow_error<T>("", 0, policies::policy<>())
         : policies::raise_domain_error<T>(
         "",
         "Got x = %1%, but we need x > 0", x, policies::policy<>());
   }
   T I, K;
   bessel_ik(v, x, &I, &K, 0xFFFF, policies::policy<>());

   std::cout << progress++ << ":   K(" << v << ", " << x << ") = " << K << std::endl;

   if(fabs(K) > 1e30)
      throw std::domain_error("");

   return K;
}

template <class T>
T cyl_neumann_bare(T v, T x)
{
   T j, y;
   bessel_jy(v, x, &j, &y, 0xFFFF, policies::policy<>());

   std::cout << progress++ << ":   Y(" << v << ", " << x << ") = " << y << std::endl;

   if(fabs(y) > 1e30)
      throw std::domain_error("");

   return y;
}

template <class T>
T sph_bessel_j_bare(T v, T x)
{
   std::cout << progress++ << ":   j(" << v << ", " << x << ") = ";
   if((v < 0) || (floor(v) != v))
      throw std::domain_error("");
   T r = sqrt(constants::pi<T>() / (2 * x)) * cyl_bessel_j_bare(v+0.5, x);
   std::cout << r << std::endl;
   return r;
}

template <class T>
T sph_bessel_y_bare(T v, T x)
{
   std::cout << progress++ << ":   y(" << v << ", " << x << ") = ";
   if((v < 0) || (floor(v) != v))
      throw std::domain_error("");
   T r = sqrt(constants::pi<T>() / (2 * x)) * cyl_neumann_bare(v+0.5, x);
   std::cout << r << std::endl;
   return r;
}

enum
{
   func_J = 0,
   func_Y,
   func_I,
   func_K,
   func_j,
   func_y
};

int main(int argc, char* argv[])
{
   std::cout << std::setprecision(17) << std::scientific;
   std::cout << sph_bessel_j_bare(0., 0.1185395751953125e4) << std::endl;
   std::cout << sph_bessel_j_bare(22., 0.6540834903717041015625) << std::endl;

   std::cout << std::setprecision(40) << std::scientific;

   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   int functype = 0;
   std::string letter = "J";

   if(argc == 2)
   {
      if(std::strcmp(argv[1], "--Y") == 0)
      {
         functype = func_Y;
         letter = "Y";
      }
      else if(std::strcmp(argv[1], "--I") == 0)
      {
         functype = func_I;
         letter = "I";
      }
      else if(std::strcmp(argv[1], "--K") == 0)
      {
         functype = func_K;
         letter = "K";
      }
      else if(std::strcmp(argv[1], "--j") == 0)
      {
         functype = func_j;
         letter = "j";
      }
      else if(std::strcmp(argv[1], "--y") == 0)
      {
         functype = func_y;
         letter = "y";
      }
      else
         BOOST_MATH_ASSERT(0);
   }

   bool cont;
   std::string line;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the Bessel " << letter << " function\n\n";
   do{
      get_user_parameter_info(arg1, "v");
      get_user_parameter_info(arg2, "x");
      mp_t (*fp)(mp_t, mp_t);
      if(functype == func_J)
         fp = cyl_bessel_j_bare;
      else if(functype == func_I)
         fp = cyl_bessel_i_bare;
      else if(functype == func_K)
         fp = cyl_bessel_k_bare;
      else if(functype == func_Y)
         fp = cyl_neumann_bare;
      else if(functype == func_j)
         fp = sph_bessel_j_bare;
      else if(functype == func_y)
         fp = sph_bessel_y_bare;
      else
         BOOST_MATH_ASSERT(0);

      data.insert(fp, arg1, arg2);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=bessel_j_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "bessel_j_data.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}




