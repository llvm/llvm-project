//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PERFORMANCE_HPP
#define PERFORMANCE_HPP

#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/array.hpp>
#include <boost/chrono.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <iomanip>

extern std::vector<std::vector<double> > data;

template <class Array>
void add_data(const Array& a)
{
   //
   // This function is called multiple times to merge multiple data sets into one big table:
   //
   for(typename Array::const_iterator i = a.begin(); i != a.end(); ++i)
   {
      data.push_back(std::vector<double>());
      for(typename Array::value_type::const_iterator j = i->begin(); j != i->end(); ++j)
      {
         data.back().push_back(*j);
      }
   }
}

template <class Func, class Result>
void screen_data(Func f, Result r)
{
   //
   // If any of the implementations being tested produces garbage for one of our
   // test cases (or else if we test a domain they don't support), then we remove that
   // row from the table.  This allows us to only test a common supported sub-set for performance:
   //
   for(std::vector<std::vector<double> >::size_type row = 0; row < data.size(); ++row)
   {
      try
      {
         double computed = f(data[row]);
         double expected = r(data[row]);
         double err = boost::math::relative_difference(computed, expected);
         if(err > 1e-7)
         {
            std::cout << "Erasing row: ";
            for(unsigned i = 0; i < data[row].size(); ++i)
            {
               std::cout << data[row][i] << " ";
            }
            std::cout << "Error was " << err << std::endl;
            data.erase(data.begin() + row);
            --row;
         }
      }
      catch(const std::exception& e)
      {
         std::cout << "Erasing row: ";
         for(unsigned i = 0; i < data[row].size(); ++i)
         {
            std::cout << data[row][i] << " ";
         }
         std::cout << "due to thrown exception: " << e.what() << std::endl;
         data.erase(data.begin() + row);
         --row;
      }
   }
}

template <class Clock>
struct stopwatch
{
   typedef typename Clock::duration duration;
   stopwatch()
   {
      m_start = Clock::now();
   }
   duration elapsed()
   {
      return Clock::now() - m_start;
   }
   void reset()
   {
      m_start = Clock::now();
   }

private:
   typename Clock::time_point m_start;
};

double sum = 0;

template <class Func>
double exec_timed_test(Func f)
{
   double t = 0;
   unsigned repeats = 1;
   do{
      stopwatch<boost::chrono::high_resolution_clock> w;

      for(unsigned count = 0; count < repeats; ++count)
      {
         for(std::vector<std::vector<double> >::const_iterator i = data.begin(); i != data.end(); ++i)
            sum += f(*i);
      }

      t = boost::chrono::duration_cast<boost::chrono::duration<double>>(w.elapsed()).count();
      if(t < 0.5)
         repeats *= 2;
   } while(t < 0.5);
   return t / (repeats * data.size());
}

#endif // PERFORMANCE_HPP
