//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_TEST_DATA_HPP
#define BOOST_MATH_TOOLS_TEST_DATA_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/assert.hpp>
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4127 4701 4512)
#  pragma warning(disable: 4130) // '==' : logical operation on address of string constant.
#endif
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/integral_constant.hpp>
#ifndef BOOST_NO_CXX11_HDR_RANDOM
#include <random>
namespace random_ns = std;
#else
#include <boost/random.hpp>
namespace random_ns = boost::random;
#endif
#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/real_cast.hpp>

#include <set>
#include <vector>
#include <iostream>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4130) // '==' : logical operation on address of string constant.
// Used as a warning with BOOST_MATH_ASSERT
#endif

namespace boost{ namespace math{ namespace tools{

enum parameter_type
{
   random_in_range = 0,
   periodic_in_range = 1,
   power_series = 2,
   single_value = 3,
   plus_minus_value = 4,
   dummy_param = 0x80
};

parameter_type operator | (parameter_type a, parameter_type b)
{
   return static_cast<parameter_type>((int)a|(int)b);
}
parameter_type& operator |= (parameter_type& a, parameter_type b)
{
   a = static_cast<parameter_type>(a|b);
   return a;
}

//
// If type == random_in_range then
// z1 and r2 are the endpoints of the half open range and n1 is the number of points.
//
// If type == periodic_in_range then
// z1 and r2 are the endpoints of the half open range and n1 is the number of points.
//
// If type == power_series then
// n1 and n2 are the endpoints of the exponents (closed range) and z1 is the basis.
//
// If type == single_value then z1 contains the single value to add.
//
// If type == plus_minus_value then test at +-z1
//
// If type & dummy_param then this data is ignored and not stored in the output, it
// is passed to the generator function however which can do with it as it sees fit.
//
template <class T>
struct parameter_info
{
   parameter_type type;
   T z1, z2;
   int n1, n2;
};

template <class T>
inline parameter_info<T> make_random_param(T start_range, T end_range, int n_points)
{
   parameter_info<T> result = { random_in_range, start_range, end_range, n_points, 0 };
   return result;
}

template <class T>
inline parameter_info<T> make_periodic_param(T start_range, T end_range, int n_points)
{
   parameter_info<T> result = { periodic_in_range, start_range, end_range, n_points, 0 };
   return result;
}

template <class T>
inline parameter_info<T> make_power_param(T basis, int start_exponent, int end_exponent)
{
   parameter_info<T> result = { power_series, basis, 0, start_exponent, end_exponent };
   return result;
}

template <class T>
inline parameter_info<T> make_single_param(T val)
{
   parameter_info<T> result = { single_value, val };
   return result;
}

template <class T>
inline parameter_info<T> make_plus_minus_param(T val)
{
   parameter_info<T> result = { plus_minus_value, val };
   return result;
}

namespace detail{

template <class Seq, class Item, int N>
inline void unpack_and_append_tuple(Seq&,
                                    const Item&,
                                    const std::integral_constant<int, N>&,
                                    const std::false_type&)
{
   // termination condition nothing to do here
}

template <class Seq, class Item, int N>
inline void unpack_and_append_tuple(Seq& s,
                                    const Item& data,
                                    const std::integral_constant<int, N>&,
                                    const std::true_type&)
{
   // extract the N'th element, append, and recurse:
   typedef typename Seq::value_type value_type;
   value_type val = boost::math::get<N>(data);
   s.push_back(val);

   typedef std::integral_constant<int, N+1> next_value;
   typedef std::integral_constant<bool, (boost::math::tuple_size<Item>::value > N+1)> terminate;

   unpack_and_append_tuple(s, data, next_value(), terminate());
}

template <class Seq, class Item>
inline void unpack_and_append(Seq& s, const Item& data, const std::true_type&)
{
   s.push_back(data);
}

template <class Seq, class Item>
inline void unpack_and_append(Seq& s, const Item& data, const std::false_type&)
{
   // Item had better be a tuple-like type or we've had it!!!!
   typedef std::integral_constant<int, 0> next_value;
   typedef std::integral_constant<bool, (boost::math::tuple_size<Item>::value > 0)> terminate;

   unpack_and_append_tuple(s, data, next_value(), terminate());
}

template <class Seq, class Item>
inline void unpack_and_append(Seq& s, const Item& data)
{
   typedef typename Seq::value_type value_type;
   unpack_and_append(s, data, ::std::is_convertible<Item, value_type>());
}

} // detail

template <class T>
class test_data
{
public:
   typedef std::vector<T> row_type;
   typedef row_type value_type;
private:
   typedef std::set<row_type> container_type;
public:
   typedef typename container_type::reference reference;
   typedef typename container_type::const_reference const_reference;
   typedef typename container_type::iterator iterator;
   typedef typename container_type::const_iterator const_iterator;
   typedef typename container_type::difference_type difference_type;
   typedef typename container_type::size_type size_type;

   // creation:
   test_data(){}
   template <class F>
   test_data(F func, const parameter_info<T>& arg1)
   {
      insert(func, arg1);
   }

   // insertion:
   template <class F>
   test_data& insert(F func, const parameter_info<T>& arg1)
   {
      // generate data for single argument functor F

      typedef typename std::set<T>::const_iterator it_type;

      std::set<T> points;
      create_test_points(points, arg1);
      it_type a = points.begin();
      it_type b = points.end();
      row_type row;
      while(a != b)
      {
         if((arg1.type & dummy_param) == 0)
            row.push_back(*a);
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            // domain_error exceptions from func are swallowed
            // and this data point is ignored:
            boost::math::tools::detail::unpack_and_append(row, func(*a));
            m_data.insert(row);
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const std::domain_error&){}
#endif
         row.clear();
         ++a;
      }
      return *this;
   }

   template <class F>
   test_data& insert(F func, const parameter_info<T>& arg1, const parameter_info<T>& arg2)
   {
      // generate data for 2-argument functor F

      typedef typename std::set<T>::const_iterator it_type;

      std::set<T> points1, points2;
      create_test_points(points1, arg1);
      create_test_points(points2, arg2);
      it_type a = points1.begin();
      it_type b = points1.end();
      row_type row;
      while(a != b)
      {
         it_type c = points2.begin();
         it_type d = points2.end();
         while(c != d)
         {
            if((arg1.type & dummy_param) == 0)
               row.push_back(*a);
            if((arg2.type & dummy_param) == 0)
               row.push_back(*c);
#ifndef BOOST_NO_EXCEPTIONS
            try{
#endif
               // domain_error exceptions from func are swallowed
               // and this data point is ignored:
               detail::unpack_and_append(row, func(*a, *c));
               m_data.insert(row);
#ifndef BOOST_NO_EXCEPTIONS
            }
            catch(const std::domain_error&){}
#endif
            row.clear();
            ++c;
         }
         ++a;
      }
      return *this;
   }

   template <class F>
   test_data& insert(F func, const parameter_info<T>& arg1, const parameter_info<T>& arg2, const parameter_info<T>& arg3)
   {
      // generate data for 3-argument functor F

      typedef typename std::set<T>::const_iterator it_type;

      std::set<T> points1, points2, points3;
      create_test_points(points1, arg1);
      create_test_points(points2, arg2);
      create_test_points(points3, arg3);
      it_type a = points1.begin();
      it_type b = points1.end();
      row_type row;
      while(a != b)
      {
         it_type c = points2.begin();
         it_type d = points2.end();
         while(c != d)
         {
            it_type e = points3.begin();
            it_type f = points3.end();
            while(e != f)
            {
               if((arg1.type & dummy_param) == 0)
                  row.push_back(*a);
               if((arg2.type & dummy_param) == 0)
                  row.push_back(*c);
               if((arg3.type & dummy_param) == 0)
                  row.push_back(*e);
#ifndef BOOST_NO_EXCEPTIONS
               try{
#endif
                  // domain_error exceptions from func are swallowed
                  // and this data point is ignored:
                  detail::unpack_and_append(row, func(*a, *c, *e));
                  m_data.insert(row);
#ifndef BOOST_NO_EXCEPTIONS
               }
               catch(const std::domain_error&){}
#endif
               row.clear();
               ++e;
            }
            ++c;
         }
         ++a;
      }
      return *this;
   }

   template <class F>
   test_data& insert(F func, const parameter_info<T>& arg1, const parameter_info<T>& arg2, const parameter_info<T>& arg3, const parameter_info<T>& arg4)
   {
      // generate data for 4-argument functor F

      typedef typename std::set<T>::const_iterator it_type;

      std::set<T> points1, points2, points3, points4;
      create_test_points(points1, arg1);
      create_test_points(points2, arg2);
      create_test_points(points3, arg3);
      create_test_points(points4, arg4);
      it_type a = points1.begin();
      it_type b = points1.end();
      row_type row;
      while(a != b)
      {
         it_type c = points2.begin();
         it_type d = points2.end();
         while(c != d)
         {
            it_type e = points3.begin();
            it_type f = points3.end();
            while(e != f)
            {
               it_type g = points4.begin();
               it_type h = points4.end();
               while (g != h)
               {
                  if ((arg1.type & dummy_param) == 0)
                     row.push_back(*a);
                  if ((arg2.type & dummy_param) == 0)
                     row.push_back(*c);
                  if ((arg3.type & dummy_param) == 0)
                     row.push_back(*e);
                  if ((arg4.type & dummy_param) == 0)
                     row.push_back(*g);
#ifndef BOOST_NO_EXCEPTIONS
                  try {
#endif
                     // domain_error exceptions from func are swallowed
                     // and this data point is ignored:
                     detail::unpack_and_append(row, func(*a, *c, *e, *g));
                     m_data.insert(row);
#ifndef BOOST_NO_EXCEPTIONS
                  }
                  catch (const std::domain_error&) {}
#endif
                  row.clear();
                  ++g;
               }
               ++e;
            }
            ++c;
         }
         ++a;
      }
      return *this;
   }

   template <class F>
   test_data& insert(F func, const parameter_info<T>& arg1, const parameter_info<T>& arg2, const parameter_info<T>& arg3, const parameter_info<T>& arg4, const parameter_info<T>& arg5)
   {
      // generate data for 5-argument functor F

      typedef typename std::set<T>::const_iterator it_type;

      std::set<T> points1, points2, points3, points4, points5;
      create_test_points(points1, arg1);
      create_test_points(points2, arg2);
      create_test_points(points3, arg3);
      create_test_points(points4, arg4);
      create_test_points(points5, arg5);
      it_type a = points1.begin();
      it_type b = points1.end();
      row_type row;
      while(a != b)
      {
         it_type c = points2.begin();
         it_type d = points2.end();
         while(c != d)
         {
            it_type e = points3.begin();
            it_type f = points3.end();
            while(e != f)
            {
               it_type g = points4.begin();
               it_type h = points4.end();
               while (g != h)
               {
                  it_type i = points5.begin();
                  it_type j = points5.end();
                  while (i != j)
                  {
                     if ((arg1.type & dummy_param) == 0)
                        row.push_back(*a);
                     if ((arg2.type & dummy_param) == 0)
                        row.push_back(*c);
                     if ((arg3.type & dummy_param) == 0)
                        row.push_back(*e);
                     if ((arg4.type & dummy_param) == 0)
                        row.push_back(*g);
                     if ((arg5.type & dummy_param) == 0)
                        row.push_back(*i);
#ifndef BOOST_NO_EXCEPTIONS
                     try {
#endif
                        // domain_error exceptions from func are swallowed
                        // and this data point is ignored:
                        detail::unpack_and_append(row, func(*a, *c, *e, *g, *i));
                        m_data.insert(row);
#ifndef BOOST_NO_EXCEPTIONS
                     }
                     catch (const std::domain_error&) {}
#endif
                     row.clear();
                     ++i;
                  }
                  ++g;
               }
               ++e;
            }
            ++c;
         }
         ++a;
      }
      return *this;
   }

   void clear(){ m_data.clear(); }

   // access:
   iterator begin() { return m_data.begin(); }
   iterator end() { return m_data.end(); }
   const_iterator begin()const { return m_data.begin(); }
   const_iterator end()const { return m_data.end(); }
   bool operator==(const test_data& d)const{ return m_data == d.m_data; }
   bool operator!=(const test_data& d)const{ return m_data != d.m_data; }
   void swap(test_data& other){ m_data.swap(other.m_data); }
   size_type size()const{ return m_data.size(); }
   size_type max_size()const{ return m_data.max_size(); }
   bool empty()const{ return m_data.empty(); }

   bool operator < (const test_data& dat)const{ return m_data < dat.m_data; }
   bool operator <= (const test_data& dat)const{ return m_data <= dat.m_data; }
   bool operator > (const test_data& dat)const{ return m_data > dat.m_data; }
   bool operator >= (const test_data& dat)const{ return m_data >= dat.m_data; }

private:
   void create_test_points(std::set<T>& points, const parameter_info<T>& arg1);
   std::set<row_type> m_data;

   static float extern_val;
   static float truncate_to_float(float const * pf);
   static float truncate_to_float(float c){ return truncate_to_float(&c); }
};

//
// This code exists to bemuse the compiler's optimizer and force a
// truncation to float-precision only:
//
template <class T>
inline float test_data<T>::truncate_to_float(float const * pf)
{
   BOOST_MATH_STD_USING
   int expon;
   float f = floor(ldexp(frexp(*pf, &expon), 22));
   f = ldexp(f, expon - 22);
   return f;

   //extern_val = *pf;
   //return *pf;
}

template <class T>
float test_data<T>::extern_val = 0;

template <class T>
void test_data<T>::create_test_points(std::set<T>& points, const parameter_info<T>& arg1)
{
   BOOST_MATH_STD_USING
   //
   // Generate a set of test points as requested, try and generate points
   // at only float precision: otherwise when testing float versions of functions
   // there will be a rounding error in our input values which throws off the results
   // (Garbage in garbage out etc).
   //
   switch(arg1.type & 0x7F)
   {
   case random_in_range:
      {
         BOOST_MATH_ASSERT(arg1.z1 < arg1.z2);
         BOOST_MATH_ASSERT(arg1.n1 > 0);
         typedef float random_type;

         random_ns::mt19937 rnd;
         random_ns::uniform_real_distribution<random_type> ur_a(real_cast<random_type>(arg1.z1), real_cast<random_type>(arg1.z2));

         for(int i = 0; i < arg1.n1; ++i)
         {
            random_type r = ur_a(rnd);
            points.insert(truncate_to_float(r));
         }
     }
      break;
   case periodic_in_range:
      {
         BOOST_MATH_ASSERT(arg1.z1 < arg1.z2);
         BOOST_MATH_ASSERT(arg1.n1 > 0);
         float interval = real_cast<float>((arg1.z2 - arg1.z1) / arg1.n1);
         T val = arg1.z1;
         while(val < arg1.z2)
         {
            points.insert(truncate_to_float(real_cast<float>(val)));
            val += interval;
         }
      }
      break;
   case power_series:
      {
         BOOST_MATH_ASSERT(arg1.n1 < arg1.n2);

         typedef float random_type;
         typedef typename boost::mpl::if_<
            ::boost::is_floating_point<T>,
            T, long double>::type power_type;

         random_ns::mt19937 rnd;
         random_ns::uniform_real_distribution<random_type> ur_a(1.0, 2.0);

         for(int power = arg1.n1; power <= arg1.n2; ++power)
         {
            random_type r = ur_a(rnd);
            power_type p = ldexp(static_cast<power_type>(r), power);
            points.insert(truncate_to_float(real_cast<float>(arg1.z1 + p)));
         }
      }
      break;
   case single_value:
   {
      points.insert(truncate_to_float(real_cast<float>(arg1.z1)));
      break;
   }
   case plus_minus_value:
   {
      points.insert(truncate_to_float(real_cast<float>(arg1.z1)));
      points.insert(truncate_to_float(-real_cast<float>(arg1.z1)));
      break;
   }
   default:
      BOOST_MATH_ASSERT(0 == "Invalid parameter_info object");
      // Assert will fail if get here.
      // Triggers warning 4130) // '==' : logical operation on address of string constant.
   }
}

//
// Prompt a user for information on a parameter range:
//
template <class T>
bool get_user_parameter_info(parameter_info<T>& info, const char* param_name)
{
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4127)
#endif
   std::string line;
   do{
      std::cout << "What kind of distribution do you require for parameter " << param_name << "?\n"
         "Choices are:\n"
         "  r     Random values in a half open range\n"
         "  p     Evenly spaced periodic values in a half open range\n"
         "  e     Exponential power series at a particular point: a + 2^b for some range of b\n"
         "[Default=r]";

      std::getline(std::cin, line);
      boost::algorithm::trim(line);

      if(line == "r")
      {
         info.type = random_in_range;
         break;
      }
      else if(line == "p")
      {
         info.type = periodic_in_range;
         break;
      }
      else if(line == "e")
      {
         info.type = power_series;
         break;
      }
      else if(line == "")
      {
         info.type = random_in_range;
         break;
      }
      //
      // Ooops, not a valid input....
      //
      std::cout << "Sorry don't recognise \"" << line << "\" as a valid input\n"
         "do you want to try again [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      if(line == "n")
         return false;
      else if(line == "y")
         continue;
      std::cout << "Sorry don't recognise that either, giving up...\n\n";
      return false;
   }while(true);

   switch(info.type & ~dummy_param)
   {
   case random_in_range:
   case periodic_in_range:
      // get start and end points of range:
      do{
         std::cout << "Data will be in the half open range a <= x < b,\n"
            "enter value for the start point fo the range [default=0]:";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         if(line == "")
         {
            info.z1 = 0;
            break;
         }
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            info.z1 = boost::lexical_cast<T>(line);
            break;
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const boost::bad_lexical_cast&)
         {
            std::cout << "Sorry, that was not valid input, try again [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
#endif
      }while(true);
      do{
         std::cout << "Enter value for the end point fo the range [default=1]:";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         if(line == "")
         {
            info.z2 = 1;
         }
         else
         {
#ifndef BOOST_NO_EXCEPTIONS
            try
            {
#endif
               info.z2 = boost::lexical_cast<T>(line);
#ifndef BOOST_NO_EXCEPTIONS
            }
            catch(const boost::bad_lexical_cast&)
            {
               std::cout << "Sorry, that was not valid input, try again [y/n]?";
               std::getline(std::cin, line);
               boost::algorithm::trim(line);
               if(line == "y")
                  continue;
               if(line == "n")
                  return false;
               std::cout << "Sorry don't recognise that either, giving up...\n\n";
               return false;
            }
#endif
         }
         if(info.z1 >= info.z2)
         {
            std::cout << "The end point of the range was <= the start point\n"
               "try a different value for the endpoint [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
         break;
      }while(true);
      do{
         // get the number of points:
         std::cout << "How many data points do you want?";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            info.n1 = boost::lexical_cast<int>(line);
            info.n2 = 0;
            if(info.n1 <= 0)
            {
               std::cout << "The number of points should be > 0\n"
                  "try again [y/n]?";
               std::getline(std::cin, line);
               boost::algorithm::trim(line);
               if(line == "y")
                  continue;
               if(line == "n")
                  return false;
               std::cout << "Sorry don't recognise that either, giving up...\n\n";
               return false;
            }
            break;
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const boost::bad_lexical_cast&)
         {
            std::cout << "Sorry, that was not valid input, try again [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
#endif
      }while(true);
      break;
   case power_series:
      // get start and end points of range:
      info.z2 = 0;
      do{
         std::cout << "Data will be in the form a + r*2^b\n"
            "for random value r,\n"
            "enter value for the point a [default=0]:";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         if(line == "")
         {
            info.z1 = 0;
            break;
         }
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            info.z1 = boost::lexical_cast<T>(line);
            break;
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const boost::bad_lexical_cast&)
         {
            std::cout << "Sorry, that was not valid input, try again [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
#endif
      }while(true);

      do{
         std::cout << "Data will be in the form a + r*2^b\n"
            "for random value r,\n"
            "enter value for the starting exponent b:";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            info.n1 = boost::lexical_cast<int>(line);
            break;
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const boost::bad_lexical_cast&)
         {
            std::cout << "Sorry, that was not valid input, try again [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
#endif
      }while(true);

      do{
         std::cout << "Data will be in the form a + r*2^b\n"
            "for random value r,\n"
            "enter value for the ending exponent b:";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
#ifndef BOOST_NO_EXCEPTIONS
         try{
#endif
            info.n2 = boost::lexical_cast<int>(line);
            break;
#ifndef BOOST_NO_EXCEPTIONS
         }
         catch(const boost::bad_lexical_cast&)
         {
            std::cout << "Sorry, that was not valid input, try again [y/n]?";
            std::getline(std::cin, line);
            boost::algorithm::trim(line);
            if(line == "y")
               continue;
            if(line == "n")
               return false;
            std::cout << "Sorry don't recognise that either, giving up...\n\n";
            return false;
         }
#endif
      }while(true);

      break;
   default:
      BOOST_MATH_ASSERT(0); // should never get here!!
   }

   return true;
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
}

template <class charT, class traits, class T>
inline std::basic_ostream<charT, traits>& write_csv(std::basic_ostream<charT, traits>& os,
                                             const test_data<T>& data)
{
   const charT defarg[] = { ',', ' ', '\0' };
   return write_csv(os, data, defarg);
}

template <class charT, class traits, class T>
std::basic_ostream<charT, traits>& write_csv(std::basic_ostream<charT, traits>& os,
                                             const test_data<T>& data,
                                             const charT* separator)
{
   typedef typename test_data<T>::const_iterator it_type;
   typedef typename test_data<T>::value_type value_type;
   typedef typename value_type::const_iterator value_type_iterator;
   it_type a, b;
   a = data.begin();
   b = data.end();
   while(a != b)
   {
      value_type_iterator x, y;
      bool sep = false;
      x = a->begin();
      y = a->end();
      while(x != y)
      {
         if(sep)
            os << separator;
         os << *x;
         sep = true;
         ++x;
      }
      os << std::endl;
      ++a;
   }
   return os;
}

template <class T>
std::ostream& write_code(std::ostream& os,
                         const test_data<T>& data,
                         const char* name)
{
   typedef typename test_data<T>::const_iterator it_type;
   typedef typename test_data<T>::value_type value_type;
   typedef typename value_type::const_iterator value_type_iterator;

   BOOST_MATH_ASSERT(os.good());

   it_type a, b;
   a = data.begin();
   b = data.end();
   if(a == b)
      return os;

   os << "#ifndef SC_\n#  define SC_(x) static_cast<T>(BOOST_JOIN(x, L))\n#endif\n"
   "   static const std::array<std::array<T, "
   << a->size() << ">, " << data.size() << "> " << name << " = {{\n";

   while(a != b)
   {
      if(a != data.begin())
         os << ", \n";

      value_type_iterator x, y;
      x = a->begin();
      y = a->end();
      os << "      { ";
      while(x != y)
      {
         if(x != a->begin())
            os << ", ";
         os << "SC_(" << *x << ")";
         ++x;
      }
      os << " }";
      ++a;
   }
   os << "\n   }};\n//#undef SC_\n\n";
   return os;
}

} // namespace tools
} // namespace math
} // namespace boost

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#endif // BOOST_MATH_TOOLS_TEST_DATA_HPP


