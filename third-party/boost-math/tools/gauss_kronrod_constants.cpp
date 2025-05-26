//  Copyright (c) 2017 John Maddock
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

template <class T>
void print_gauss_constants(const char* suffix, int prec, int tag)
{
   auto ab = T::abscissa();
   auto w = T::weights();
   std::cout << std::setprecision(prec) << std::scientific;
   std::size_t order = (ab[0] == 0) ? (ab.size() * 2) - 1 : ab.size() * 2;
   std::cout <<
      "template <class T>\n"
      "class gauss_detail<T, " << order << ", " << tag << ">\n"
      "   {\n"
      "   public:\n"
      "      static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << ab.size() << "> const & abscissa()\n"
      "      {\n"
      "         static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << ab.size() << "> data = {\n";
   for (unsigned i = 0; i < ab.size(); ++i)
      std::cout << "            " << (prec > 40 ? "BOOST_MATH_HUGE_CONSTANT(T, 0, " : "") << ab[i] << (prec > 40 ? ")" : suffix) << ",\n";
   std::cout <<
      "};\n"
      "         return data;\n"
      "      }\n"
      "      static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << w.size() << "> const & weights()\n"
      "      {\n"
      "         static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << w.size() << "> data = {\n";
   for (unsigned i = 0; i < w.size(); ++i)
      std::cout << "            " << (prec > 40 ? "BOOST_MATH_HUGE_CONSTANT(T, 0, " : "") << w[i] << (prec > 40 ? ")" : suffix) << ",\n";

   std::cout << "         };\n"
      "         return data;\n"
      "      }\n"
      "   };\n\n";
}

template <class T>
void print_gauss_kronrod_constants(const char* suffix, int prec, int tag)
{
   auto ab = T::abscissa();
   auto w = T::weights();
   std::cout << std::setprecision(prec) << std::scientific;
   std::size_t order = (ab.size() * 2) - 1;
   std::cout <<
      "   template <class T>\n"
      "   class gauss_kronrod_detail<T, " << order << ", " << tag << ">\n"
      "   {\n"
      "   public:\n"
      "      static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << ab.size() << "> const & abscissa()\n"
      "      {\n"
      "         static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << ab.size() << "> data = {\n";

   for (unsigned i = 0; i < ab.size(); ++i)
      std::cout << "            " << (prec > 40 ? "BOOST_MATH_HUGE_CONSTANT(T, 0, " : "") << ab[i] << (prec > 40 ? ")" : suffix) << ",\n";

   std::cout << "         };\n"
      "         return data;\n"
      "      }\n"
      "      static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << w.size() << "> const & weights()\n"
      "      {\n"
      "         static " << (prec > 40 ? " " : "constexpr ") << "std::array<T, " << w.size() << "> data = {\n";

   for (unsigned i = 0; i < w.size(); ++i)
      std::cout << "            " << (prec > 40 ? "BOOST_MATH_HUGE_CONSTANT(T, 0, " : "") << w[i] << (prec > 40 ? ")" : suffix) << ",\n";

   std::cout << "         };\n"
      "         return data;\n"
      "      }\n"
      "   };\n\n";
}



int main()
{
   typedef boost::multiprecision::number<boost::multiprecision::cpp_bin_float<250> > mp_type;

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 7> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 7> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 7> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 7> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 7> >("", 115, 4);

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 10> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 10> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 10> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 10> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 10> >("", 115, 4);

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 15> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 15> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 15> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 15> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 15> >("", 115, 4);

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 20> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 20> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 20> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 20> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 20> >("", 115, 4);

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 25> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 25> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 25> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 25> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 25> >("", 115, 4);

   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 30> >("f", 9, 0);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 30> >("", 17, 1);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 30> >("L", 35, 2);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 30> >("Q", 35, 3);
   print_gauss_constants<boost::math::quadrature::gauss<mp_type, 30> >("", 115, 4);


   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 15> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 15> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 15> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 15> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 15> >("", 115, 4);

   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 21> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 21> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 21> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 21> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 21> >("", 115, 4);

   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 31> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 31> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 31> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 31> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 31> >("", 115, 4);

   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 41> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 41> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 41> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 41> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 41> >("", 115, 4);

   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 51> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 51> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 51> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 51> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 51> >("", 115, 4);

   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 61> >("f", 9, 0);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 61> >("", 17, 1);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 61> >("L", 35, 2);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 61> >("Q", 35, 3);
   print_gauss_kronrod_constants<boost::math::quadrature::gauss_kronrod<mp_type, 61> >("", 115, 4);

   return 0;
}

