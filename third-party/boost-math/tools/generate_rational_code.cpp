//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>

int max_order = 20;
const char* path_prefix = "..\\..\\..\\boost\\math\\tools\\detail\\polynomial_";
const char* path_prefix2 = "..\\..\\..\\boost\\math\\tools\\detail\\rational_";

const char* copyright_string = 
"//  (C) Copyright John Maddock 2007.\n"
"//  Use, modification and distribution are subject to the\n"
"//  Boost Software License, Version 1.0. (See accompanying file\n"
"//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
"//\n"
"//  This file is machine generated, do not edit by hand\n\n";


void print_polynomials(int max_order)
{
   for(int i = 2; i <= max_order; ++i)
   {
      std::stringstream filename;
      filename << path_prefix << "horner1_" << i << ".hpp";
      std::ofstream ofs(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Polynomial evaluation using Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]);\n"
         "}\n\n";

      for(int order = 2; order <= i; ++order)
      {
         ofs << 
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   return static_cast<V>(";
         
         for(int bracket = 2; bracket < order; ++bracket)
            ofs << "(";
         ofs << "a[" << order - 1 << "] * x + a[" << order - 2 << "]" ;
         for(int item = order - 3; item >= 0; --item)
         {
            ofs << ") * x + a[" << item << "]";
         }
         
         ofs << ");\n"
         "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";

      filename.str("");
      filename << path_prefix << "horner2_" << i << ".hpp";
      ofs.close();
      ofs.open(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Polynomial evaluation using second order Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 2>*)\n"
         "{\n"
         "   return static_cast<V>(a[1] * x + a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 3>*)\n"
         "{\n"
         "   return static_cast<V>((a[2] * x + a[1]) * x + a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 4>*)\n"
         "{\n"
         "   return static_cast<V>(((a[3] * x + a[2]) * x + a[1]) * x + a[0]);\n"
         "}\n\n";

      for(int order = 5; order <= i; ++order)
      {
         ofs << 
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   V x2 = x * x;\n"
         "   return static_cast<V>(";

         if(order & 1)
         {
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 1 << "] * x2 + a[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << " + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 2 << "] * x2 + a[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << ") * x";
         }
         else
         {
            for(int bracket = 0; bracket < (order - 1) / 2; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 1 << "] * x2 + a[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << ") * x + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 2 << "] * x2 + a[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
         }
         ofs << ");\n"
            "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";


      filename.str("");
      filename << path_prefix << "horner3_" << i << ".hpp";
      ofs.close();
      ofs.open(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Unrolled polynomial evaluation using second order Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_POLY_EVAL_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 2>*)\n"
         "{\n"
         "   return static_cast<V>(a[1] * x + a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 3>*)\n"
         "{\n"
         "   return static_cast<V>((a[2] * x + a[1]) * x + a[0]);\n"
         "}\n\n"
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, 4>*)\n"
         "{\n"
         "   return static_cast<V>(((a[3] * x + a[2]) * x + a[1]) * x + a[0]);\n"
         "}\n\n";

      for(int order = 5; order <= i; ++order)
      {
         ofs << 
         "template <class T, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_polynomial_c_imp(const T* a, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   V x2 = x * x;\n"
         "   V t[2];\n";

         if(order & 1)
         {
            ofs << "   t[0] = static_cast<V>(a[" << order - 1 << "] * x2 + a[" << order - 3 << "]);\n" ;
            ofs << "   t[1] = static_cast<V>(a[" << order - 2 << "] * x2 + a[" << order - 4 << "]);\n" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << "   t[0] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "   t[1] *= x2;\n";
               ofs << "   t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs << "   t[1] += static_cast<V>(a[" << item - 1 << "]);\n";
            }
            ofs << 
               "   t[1] *= x;\n"
               "   return t[0] + t[1];\n";
         }
         else
         {
            ofs << "   t[0] = a[" << order - 1 << "] * x2 + a[" << order - 3 << "];\n" ;
            ofs << "   t[1] = a[" << order - 2 << "] * x2 + a[" << order - 4 << "];\n" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << "   t[0] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "   t[1] *= x2;\n";
               ofs << "   t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs <<  "   t[1] += static_cast<V>(a[" << item - 1 << "]);\n";
            }
            ofs << "   t[0] *= x;\n";
            ofs << "   return t[0] + t[1];\n";
         }
         ofs << "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";
   }
}

void print_rationals(int max_order)
{
   for(int i = 2; i <= max_order; ++i)
   {
      std::stringstream filename;
      filename << path_prefix2 << "horner1_" << i << ".hpp";
      std::ofstream ofs(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Polynomial evaluation using Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_POLY_RAT_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_POLY_RAT_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T*, const U*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]) / static_cast<V>(b[0]);\n"
         "}\n\n";

      for(int order = 2; order <= i; ++order)
      {
         ofs << 
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   if((-1 <= x) && (x <= 1))\n"
         "     return static_cast<V>((";
         
         for(int bracket = 2; bracket < order; ++bracket)
            ofs << "(";
         ofs << "a[" << order - 1 << "] * x + a[" << order - 2 << "]" ;
         for(int item = order - 3; item >= 0; --item)
         {
            ofs << ") * x + a[" << item << "]";
         }

         ofs << ") / (";
         for(int bracket = 2; bracket < order; ++bracket)
            ofs << "(";
         ofs << "b[" << order - 1 << "] * x + b[" << order - 2 << "]" ;
         for(int item = order - 3; item >= 0; --item)
         {
            ofs << ") * x + b[" << item << "]";
         }
         
         ofs << "));\n   else\n   {\n      V z = 1 / x;\n      return static_cast<V>((";

         for(int bracket = order - 1; bracket > 1; --bracket)
            ofs << "(";
         ofs << "a[" << 0 << "] * z + a[" << 1 << "]" ;
         for(int item = 2; item <= order - 1; ++item)
         {
            ofs << ") * z + a[" << item << "]";
         }

         ofs << ") / (";
         for(int bracket = 2; bracket < order; ++bracket)
            ofs << "(";
         ofs << "b[" << 0 << "] * z + b[" << 1 << "]" ;
         for(int item = 2; item <= order - 1; ++item)
         {
            ofs << ") * z + b[" << item << "]";
         }
         
         ofs << "));\n   }\n";
         
         ofs << "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";

      filename.str("");
      filename << path_prefix2 << "horner2_" << i << ".hpp";
      ofs.close();
      ofs.open(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Polynomial evaluation using second order Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_RAT_EVAL_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_RAT_EVAL_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T*, const U*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]) / static_cast<V>(b[0]);\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 2>*)\n"
         "{\n"
         "   return static_cast<V>((a[1] * x + a[0]) / (b[1] * x + b[0]));\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 3>*)\n"
         "{\n"
         "   return static_cast<V>(((a[2] * x + a[1]) * x + a[0]) / ((b[2] * x + b[1]) * x + b[0]));\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 4>*)\n"
         "{\n"
         "   return static_cast<V>((((a[3] * x + a[2]) * x + a[1]) * x + a[0]) / (((b[3] * x + b[2]) * x + b[1]) * x + b[0]));\n"
         "}\n\n";

      for(int order = 5; order <= i; ++order)
      {
         ofs << 
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   if((-1 <= x) && (x <= 1))\n   {\n"
         "      V x2 = x * x;\n"
         "      return static_cast<V>((";

         if(order & 1)
         {
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 1 << "] * x2 + a[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << " + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 2 << "] * x2 + a[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << ") * x";
         }
         else
         {
            for(int bracket = 0; bracket < (order - 1) / 2; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 1 << "] * x2 + a[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
            ofs << ") * x + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << order - 2 << "] * x2 + a[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + a[" << item << "]";
            }
         }
         ofs << ") / (";
         if(order & 1)
         {
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << order - 1 << "] * x2 + b[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + b[" << item << "]";
            }
            ofs << " + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << order - 2 << "] * x2 + b[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + b[" << item << "]";
            }
            ofs << ") * x";
         }
         else
         {
            for(int bracket = 0; bracket < (order - 1) / 2; ++bracket)
               ofs << "(";
            ofs << "b[" << order - 1 << "] * x2 + b[" << order - 3 << "]" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << ") * x2 + b[" << item << "]";
            }
            ofs << ") * x + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << order - 2 << "] * x2 + b[" << order - 4 << "]" ;
            for(int item = order - 6; item >= 0; item -= 2)
            {
               ofs << ") * x2 + b[" << item << "]";
            }
         }

         ofs << "));\n   }\n   else\n   {\n      V z = 1 / x;\n      V z2 = 1 / (x * x);\n      return static_cast<V>((";

         if(order & 1)
         {
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << 0 << "] * z2 + a[" << 2 << "]" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << ") * z2 + a[" << item << "]";
            }
            ofs << " + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << 1 << "] * z2 + a[" << 3 << "]" ;
            for(int item = 5; item < order; item += 2)
            {
               ofs << ") * z2 + a[" << item << "]";
            }
            ofs << ") * z";
         }
         else
         {
            for(int bracket = 0; bracket < (order - 1) / 2; ++bracket)
               ofs << "(";
            ofs << "a[" << 0 << "] * z2 + a[" << 2 << "]" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << ") * z2 + a[" << item << "]";
            }
            ofs << ") * z + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "a[" << 1 << "] * z2 + a[" << 3 << "]" ;
            for(int item = 5; item < order; item += 2)
            {
               ofs << ") * z2 + a[" << item << "]";
            }
         }

         ofs << ") / (";

         if(order & 1)
         {
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << 0 << "] * z2 + b[" << 2 << "]" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << ") * z2 + b[" << item << "]";
            }
            ofs << " + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << 1 << "] * z2 + b[" << 3 << "]" ;
            for(int item = 5; item < order; item += 2)
            {
               ofs << ") * z2 + b[" << item << "]";
            }
            ofs << ") * z";
         }
         else
         {
            for(int bracket = 0; bracket < (order - 1) / 2; ++bracket)
               ofs << "(";
            ofs << "b[" << 0 << "] * z2 + b[" << 2 << "]" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << ") * z2 + b[" << item << "]";
            }
            ofs << ") * z + ";
            for(int bracket = 0; bracket < (order - 1) / 2 - 1; ++bracket)
               ofs << "(";
            ofs << "b[" << 1 << "] * z2 + b[" << 3 << "]" ;
            for(int item = 5; item < order; item += 2)
            {
               ofs << ") * z2 + b[" << item << "]";
            }
         }
         ofs << "));\n   }\n";

         ofs << "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";


      filename.str("");
      filename << path_prefix2 << "horner3_" << i << ".hpp";
      ofs.close();
      ofs.open(filename.str().c_str());
      if(ofs.bad())
         break;
      //
      // Output the boilerplate at the top of the header:
      //
      ofs << copyright_string <<
         "// Polynomial evaluation using second order Horners rule\n"
         "#ifndef BOOST_MATH_TOOLS_RAT_EVAL_" << i << "_HPP\n"
         "#define BOOST_MATH_TOOLS_RAT_EVAL_" << i << "_HPP\n\n"
         "namespace boost{ namespace math{ namespace tools{ namespace detail{\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T*, const U*, const V&, const boost::math::integral_constant<int, 0>*)\n"
         "{\n"
         "   return static_cast<V>(0);\n"
         "}\n"
         "\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V&, const boost::math::integral_constant<int, 1>*)\n"
         "{\n"
         "   return static_cast<V>(a[0]) / static_cast<V>(b[0]);\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 2>*)\n"
         "{\n"
         "   return static_cast<V>((a[1] * x + a[0]) / (b[1] * x + b[0]));\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 3>*)\n"
         "{\n"
         "   return static_cast<V>(((a[2] * x + a[1]) * x + a[0]) / ((b[2] * x + b[1]) * x + b[0]));\n"
         "}\n\n"
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, 4>*)\n"
         "{\n"
         "   return static_cast<V>((((a[3] * x + a[2]) * x + a[1]) * x + a[0]) / (((b[3] * x + b[2]) * x + b[1]) * x + b[0]));\n"
         "}\n\n";

      for(int order = 5; order <= i; ++order)
      {
         ofs << 
         "template <class T, class U, class V>\n"
         "BOOST_MATH_GPU_ENABLED inline V evaluate_rational_c_imp(const T* a, const U* b, const V& x, const boost::math::integral_constant<int, " << order << ">*)\n"
         "{\n"
         "   if((-1 <= x) && (x <= 1))\n   {\n"
         "      V x2 = x * x;\n"
         "      V t[4];\n";

         if(order & 1)
         {
            ofs << "      t[0] = a[" << order - 1 << "] * x2 + a[" << order - 3 << "];\n" ;
            ofs << "      t[1] = a[" << order - 2 << "] * x2 + a[" << order - 4 << "];\n" ;
            ofs << "      t[2] = b[" << order - 1 << "] * x2 + b[" << order - 3 << "];\n" ;
            ofs << "      t[3] = b[" << order - 2 << "] * x2 + b[" << order - 4 << "];\n" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << "      t[0] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "      t[1] *= x2;\n";
               ofs << "      t[2] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "      t[3] *= x2;\n";
               ofs << "      t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs << "      t[1] += static_cast<V>(a[" << item - 1 << "]);\n";
               ofs << "      t[2] += static_cast<V>(b[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs << "      t[3] += static_cast<V>(b[" << item - 1 << "]);\n";
            }
            ofs << "      t[1] *= x;\n";
            ofs << "      t[3] *= x;\n";
         }
         else
         {
            ofs << "      t[0] = a[" << order - 1 << "] * x2 + a[" << order - 3 << "];\n" ;
            ofs << "      t[1] = a[" << order - 2 << "] * x2 + a[" << order - 4 << "];\n" ;
            ofs << "      t[2] = b[" << order - 1 << "] * x2 + b[" << order - 3 << "];\n" ;
            ofs << "      t[3] = b[" << order - 2 << "] * x2 + b[" << order - 4 << "];\n" ;
            for(int item = order - 5; item >= 0; item -= 2)
            {
               ofs << "      t[0] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "      t[1] *= x2;\n";
               ofs << "      t[2] *= x2;\n";
               if(item - 1 >= 0)
                  ofs << "      t[3] *= x2;\n";
               ofs << "      t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs << "      t[1] += static_cast<V>(a[" << item - 1 << "]);\n";
               ofs << "      t[2] += static_cast<V>(b[" << item << "]);\n";
               if(item - 1 >= 0)
                  ofs << "      t[3] += static_cast<V>(b[" << item - 1 << "]);\n";
            }
            ofs << "      t[0] *= x;\n";
            ofs << "      t[2] *= x;\n";
         }
         ofs << "      return (t[0] + t[1]) / (t[2] + t[3]);\n";

         ofs << "   }\n   else\n   {\n      V z = 1 / x;\n      V z2 = 1 / (x * x);\n      V t[4];\n";

         if(order & 1)
         {
            ofs << "      t[0] = a[" << 0 << "] * z2 + a[" << 2 << "];\n" ;
            ofs << "      t[1] = a[" << 1 << "] * z2 + a[" << 3 << "];\n" ;
            ofs << "      t[2] = b[" << 0 << "] * z2 + b[" << 2 << "];\n" ;
            ofs << "      t[3] = b[" << 1 << "] * z2 + b[" << 3 << "];\n" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << "      t[0] *= z2;\n";
               if(item + 1 < order)
                  ofs << "      t[1] *= z2;\n";
               ofs << "      t[2] *= z2;\n";
               if(item + 1 < order)
                  ofs << "      t[3] *= z2;\n";
               ofs << "      t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item + 1 < order)
                  ofs << "      t[1] += static_cast<V>(a[" << item + 1 << "]);\n";
               ofs << "      t[2] += static_cast<V>(b[" << item << "]);\n";
               if(item + 1 < order)
                  ofs << "      t[3] += static_cast<V>(b[" << item + 1 << "]);\n";
            }
            ofs << "      t[1] *= z;\n";
            ofs << "      t[3] *= z;\n";
         }
         else
         {
            ofs << "      t[0] = a[" << 0 << "] * z2 + a[" << 2 << "];\n" ;
            ofs << "      t[1] = a[" << 1 << "] * z2 + a[" << 3 << "];\n" ;
            ofs << "      t[2] = b[" << 0 << "] * z2 + b[" << 2 << "];\n" ;
            ofs << "      t[3] = b[" << 1 << "] * z2 + b[" << 3 << "];\n" ;
            for(int item = 4; item < order; item += 2)
            {
               ofs << "      t[0] *= z2;\n";
               if(item + 1 < order)
                  ofs << "      t[1] *= z2;\n";
               ofs << "      t[2] *= z2;\n";
               if(item + 1 < order)
                  ofs << "      t[3] *= z2;\n";
               ofs << "      t[0] += static_cast<V>(a[" << item << "]);\n";
               if(item + 1 < order)
                  ofs << "      t[1] += static_cast<V>(a[" << item + 1 << "]);\n";
               ofs << "      t[2] += static_cast<V>(b[" << item << "]);\n";
               if(item + 1 < order)
                  ofs << "      t[3] += static_cast<V>(b[" << item + 1 << "]);\n";
            }
            ofs << "      t[0] *= z;\n";
            ofs << "      t[2] *= z;\n";
         }
         ofs << "      return (t[0] + t[1]) / (t[2] + t[3]);\n   }\n";

         ofs << "}\n\n";
      }
      //
      // And finally the boilerplate at the end of the header:
      //
      ofs << "\n}}}} // namespaces\n\n#endif // include guard\n\n";
   }
}

int main()
{
   for(int i = 2; i <= max_order; ++i)
   {
      print_polynomials(i);
      print_rationals(i);
   }
   return 0;
}



