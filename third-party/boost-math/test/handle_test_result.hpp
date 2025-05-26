//  (C) Copyright John Maddock 2006-7.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_HANDLE_TEST_RESULT
#define BOOST_MATH_HANDLE_TEST_RESULT

#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/regex.hpp>
#include <boost/test/test_tools.hpp>
#include <iostream>
#include <iomanip>

#if defined(__INTEL_COMPILER)
#  pragma warning(disable:239)
#  pragma warning(disable:264)
#endif

//
// Every client of this header has to define this function,
// and initialise the table of expected results:
//
void expected_results();

typedef std::pair<boost::regex, std::pair<std::uintmax_t, std::uintmax_t> > expected_data_type;
typedef std::list<expected_data_type> list_type;

inline list_type& 
   get_expected_data()
{
   static list_type data;
   return data;
}

inline void add_expected_result(
   const char* compiler,
   const char* library,
   const char* platform,
   const char* type_name,
   const char* test_name,
   const char* group_name, 
   std::uintmax_t max_peek_error, 
   std::uintmax_t max_mean_error)
{
   std::string re("(?:");
   re += compiler;
   re += ")";
   re += "\\|";
   re += "(?:";
   re += library;
   re += ")";
   re += "\\|";
   re += "(?:";
   re += platform;
   re += ")";
   re += "\\|";
   re += "(?:";
   re += type_name;
   re += ")";
   re += "\\|";
   re += "(?:";
   re += test_name;
   re += ")";
   re += "\\|";
   re += "(?:";
   re += group_name;
   re += ")";
   get_expected_data().push_back(
      std::make_pair(boost::regex(re, boost::regex::perl | boost::regex::icase), 
         std::make_pair(max_peek_error, max_mean_error)));
}

inline std::string build_test_name(const char* type_name, const char* test_name, const char* group_name)
{
   std::string result(BOOST_COMPILER);
   result += "|";
   result += BOOST_STDLIB;
   result += "|";
   result += BOOST_PLATFORM;
   result += "|";
   result += type_name;
   result += "|";
   result += group_name;
   result += "|";
   result += test_name;
   return result;
}

inline const std::pair<std::uintmax_t, std::uintmax_t>&
   get_max_errors(const char* type_name, const char* test_name, const char* group_name)
{
   static const std::pair<std::uintmax_t, std::uintmax_t> defaults(1, 1);
   std::string name = build_test_name(type_name, test_name, group_name);
   list_type& l = get_expected_data();
   list_type::const_iterator a(l.begin()), b(l.end());
   while(a != b)
   {
      if(regex_match(name, a->first))
      {
#if 0
         std::cout << name << std::endl;
         std::cout << a->first.str() << std::endl;
#endif
         return a->second;
      }
      ++a;
   }
   return defaults;
}

template <class T, class Seq>
void handle_test_result(const boost::math::tools::test_result<T>& result,
                       const Seq& worst, int row, 
                       const char* type_name, 
                       const char* test_name, 
                       const char* group_name)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif
   using namespace std; // To aid selection of the right pow.
   T eps = boost::math::tools::epsilon<T>();
   std::cout << std::setprecision(4);

   T max_error_found = (result.max)()/eps;
   T mean_error_found = result.rms()/eps;
   //
   // Begin by printing the main tag line with the results:
   //
   std::cout << test_name << "<" << type_name << "> Max = " << max_error_found
      << " RMS Mean=" << mean_error_found;
   //
   // If the max error is non-zero, give the row of the table that
   // produced the worst error:
   //
   if((result.max)() != 0)
   {
      std::cout << "\n    worst case at row: "
         << row << "\n    { ";
      boost::math::tools::set_output_precision(max_error_found, std::cout);
      for(unsigned i = 0; i < worst.size(); ++i)
      {
         if(i)
            std::cout << ", ";
#if defined(__SGI_STL_PORT)
         std::cout << boost::math::tools::real_cast<double>(worst[i]);
#else
         std::cout << worst[i];
#endif
      }
      std::cout << " }";
   }
   std::cout << std::endl;
   //
   // Now verify that the results are within our expected bounds:
   //
   std::pair<std::uintmax_t, std::uintmax_t> const& bounds = get_max_errors(type_name, test_name, group_name);
   if(bounds.first < max_error_found)
   {
      std::cerr << "Peak error greater than expected value of " << bounds.first << std::endl;
      BOOST_CHECK(bounds.first >= max_error_found);
   }
   if(bounds.second < mean_error_found)
   {
      std::cerr << "Mean error greater than expected value of " << bounds.second << std::endl;
      BOOST_CHECK(bounds.second >= mean_error_found);
   }
   std::cout << std::endl;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

template <class T, class Seq>
void print_test_result(const boost::math::tools::test_result<T>& result,
                       const Seq& worst, int row, const char* name, const char* test)
{
   using namespace std; // To aid selection of the right pow.
   T eps = boost::math::tools::epsilon<T>();
   std::cout << std::setprecision(4);

   T max_error_found = (result.max)()/eps;
   T mean_error_found = result.rms()/eps;
   //
   // Begin by printing the main tag line with the results:
   //
   std::cout << test << "(" << name << ") Max = " << max_error_found
      << " RMS Mean=" << mean_error_found;
   //
   // If the max error is non-zero, give the row of the table that
   // produced the worst error:
   //
   if((result.max)() != 0)
   {
      std::cout << "\n    worst case at row: "
         << row << "\n    { ";
      for(unsigned i = 0; i < worst.size(); ++i)
      {
         if(i)
            std::cout << ", ";
         std::cout << worst[i];
      }
      std::cout << " }";
   }
   std::cout << std::endl;
}

#endif // BOOST_MATH_HANDLE_TEST_RESULT

