//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_REMEZ_HPP
#define BOOST_MATH_TOOLS_REMEZ_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include "solve.hpp"
#include <boost/math/tools/minima.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/function/function1.hpp>
#include <boost/scoped_array.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/policy.hpp>

namespace boost{ namespace math{ namespace tools{

namespace detail{

//
// The error function: the difference between F(x) and
// the current approximation.  This is the function
// for which we must find the extrema.
//
template <class T>
struct remez_error_function
{
   typedef boost::function1<T, T const &> function_type;
public:
   remez_error_function(
      function_type f_, 
      const polynomial<T>& n, 
      const polynomial<T>& d, 
      bool rel_err)
         : f(f_), numerator(n), denominator(d), rel_error(rel_err) {}

   T operator()(const T& z)const
   {
      T y = f(z);
      T abs = y - (numerator.evaluate(z) / denominator.evaluate(z));
      T err;
      if(rel_error)
      {
         if(y != 0)
            err = abs / fabs(y);
         else if(0 == abs)
         {
            // we must be at a root, or it's not recoverable:
            BOOST_MATH_ASSERT(0 == abs);
            err = 0;
         }
         else
         {
            // We have a divide by zero!
            // Lets assume that f(x) is zero as a result of
            // internal cancellation error that occurs as a result
            // of shifting a root at point z to the origin so that
            // the approximation can be "pinned" to pass through
            // the origin: in that case it really
            // won't matter what our approximation calculates here
            // as long as it's a small number, return the absolute error:
            err = abs;
         }
      }
      else
         err = abs;
      return err;
   }
private:
   function_type f;
   polynomial<T> numerator;
   polynomial<T> denominator;
   bool rel_error;
};
//
// This function adapts the error function so that it's minima
// are the extrema of the error function.  We can find the minima
// with standard techniques.
//
template <class T>
struct remez_max_error_function
{
   remez_max_error_function(const remez_error_function<T>& f)
      : func(f) {}

   T operator()(const T& x)
   {
      BOOST_MATH_STD_USING
      return -fabs(func(x));
   }
private:
   remez_error_function<T> func;
};

} // detail

template <class T>
class remez_minimax
{
public:
   typedef boost::function1<T, T const &> function_type;
   typedef boost::numeric::ublas::vector<T> vector_type;
   typedef boost::numeric::ublas::matrix<T> matrix_type;

   remez_minimax(function_type f, unsigned oN, unsigned oD, T a, T b, bool pin = true, bool rel_err = false, int sk = 0, int bits = 0);
   remez_minimax(function_type f, unsigned oN, unsigned oD, T a, T b, bool pin, bool rel_err, int sk, int bits, const vector_type& points);

   void reset(unsigned oN, unsigned oD, T a, T b, bool pin = true, bool rel_err = false, int sk = 0, int bits = 0);
   void reset(unsigned oN, unsigned oD, T a, T b, bool pin, bool rel_err, int sk, int bits, const vector_type& points);

   void set_brake(int b)
   {
      BOOST_MATH_ASSERT(b < 100);
      BOOST_MATH_ASSERT(b >= 0);
      m_brake = b;
   }

   T iterate();

   polynomial<T> denominator()const;
   polynomial<T> numerator()const;

   vector_type const& chebyshev_points()const
   {
      return control_points;
   }

   vector_type const& zero_points()const
   {
      return zeros;
   }

   T error_term()const
   {
      return solution[solution.size() - 1];
   }
   T max_error()const
   {
      return m_max_error;
   }
   T max_change()const
   {
      return m_max_change;
   }
   void rotate()
   {
      --orderN;
      ++orderD;
   }
   void rescale(T a, T b)
   {
      T scale = (b - a) / (max - min);
      for(unsigned i = 0; i < control_points.size(); ++i)
      {
         control_points[i] = (control_points[i] - min) * scale + a;
      }
      min = a;
      max = b;
   }
private:

   void init_chebyshev();

   function_type func;            // The function to approximate.
   vector_type control_points;    // Current control points to be used for the next iteration.
   vector_type solution;          // Solution from the last iteration contains all unknowns including the error term.
   vector_type zeros;             // Location of points of zero error from last iteration, plus the two end points.
   vector_type maxima;            // Location of maxima of the error function, actually contains the control points used for the last iteration.
   T m_max_error;                 // Maximum error found in last approximation.
   T m_max_change;                // Maximum change in location of control points after last iteration.
   unsigned orderN;               // Order of the numerator polynomial.
   unsigned orderD;               // Order of the denominator polynomial.
   T min, max;                    // End points of the range to optimise over.
   bool rel_error;                // If true optimise for relative not absolute error.
   bool pinned;                   // If true the approximation is "pinned" to go through the origin.
   unsigned unknowns;             // Total number of unknowns.
   int m_precision;               // Number of bits precision to which the zeros and maxima are found.
   T m_max_change_history[2];     // Past history of changes to control points.
   int m_brake;                     // amount to break by in percentage points.
   int m_skew;                      // amount to skew starting points by in percentage points: -100-100
};

#ifndef BRAKE
#define BRAKE 0
#endif
#ifndef SKEW
#define SKEW 0
#endif

template <class T>
void remez_minimax<T>::init_chebyshev()
{
   BOOST_MATH_STD_USING
   //
   // Fill in the zeros:
   //
   unsigned terms = pinned ? orderD + orderN : orderD + orderN + 1;

   for(unsigned i = 0; i < terms; ++i)
   {
      T cheb = cos((2 * terms - 1 - 2 * i) * constants::pi<T>() / (2 * terms));
      cheb += 1;
      cheb /= 2;
      if(m_skew != 0)
      {
         T p = static_cast<T>(200 + m_skew) / 200;
         cheb = pow(cheb, p);
      }
      cheb *= (max - min);
      cheb += min;
      zeros[i+1] = cheb;
   }
   zeros[0] = min;
   zeros[unknowns] = max;
   // perform a regular interpolation fit:
   matrix_type A(terms, terms);
   vector_type b(terms);
   // fill in the y values:
   for(unsigned i = 0; i < b.size(); ++i)
   {
      b[i] = func(zeros[i+1]);
   }
   // fill in powers of x evaluated at each of the control points:
   unsigned offsetN = pinned ? 0 : 1;
   unsigned offsetD = offsetN + orderN;
   unsigned maxorder = (std::max)(orderN, orderD);
   for(unsigned i = 0; i < b.size(); ++i)
   {
      T x0 = zeros[i+1];
      T x = x0;
      if(!pinned)
         A(i, 0) = 1;
      for(unsigned j = 0; j < maxorder; ++j)
      {
         if(j < orderN)
            A(i, j + offsetN) = x;
         if(j < orderD)
         {
            A(i, j + offsetD) = -x * b[i];
         }
         x *= x0;
      }
   }
   //
   // Now go ahead and solve the expression to get our solution:
   //
   vector_type l_solution = boost::math::tools::solve(A, b);
   // need to add a "fake" error term:
   l_solution.resize(unknowns);
   l_solution[unknowns-1] = 0;
   solution = l_solution;
   //
   // Now find all the extrema of the error function:
   //
   detail::remez_error_function<T> Err(func, this->numerator(), this->denominator(), rel_error);
   detail::remez_max_error_function<T> Ex(Err);
   m_max_error = 0;
   //int max_err_location = 0;
   for(unsigned i = 0; i < unknowns; ++i)
   {
      std::pair<T, T> r = brent_find_minima(Ex, zeros[i], zeros[i+1], m_precision);
      maxima[i] = r.first;
      T rel_err = fabs(r.second);
      if(rel_err > m_max_error)
      {
         m_max_error = fabs(r.second);
         //max_err_location = i;
      }
   }
   control_points = maxima;
}

template <class T>
void remez_minimax<T>::reset(
         unsigned oN, 
         unsigned oD, 
         T a, 
         T b, 
         bool pin, 
         bool rel_err, 
         int sk,
         int bits)
{
   control_points = vector_type(oN + oD + (pin ? 1 : 2));
   solution = control_points;
   zeros = vector_type(oN + oD + (pin ? 2 : 3));
   maxima = control_points;
   orderN = oN;
   orderD = oD;
   rel_error = rel_err;
   pinned = pin;
   m_skew = sk;
   min = a;
   max = b;
   m_max_error = 0;
   unknowns = orderN + orderD + (pinned ? 1 : 2);
   // guess our initial control points:
   control_points[0] = min;
   control_points[unknowns - 1] = max;
   T interval = (max - min) / (unknowns - 1);
   T spot = min + interval;
   for(unsigned i = 1; i < control_points.size(); ++i)
   {
      control_points[i] = spot;
      spot += interval;
   }
   solution[unknowns - 1] = 0;
   m_max_error = 0;
   if(bits == 0)
   {
      // don't bother about more than float precision:
      m_precision = (std::min)(24, (boost::math::policies::digits<T, boost::math::policies::policy<> >() / 2) - 2);
   }
   else
   {
      // can't be more accurate than half the bits of T:
      m_precision = (std::min)(bits, (boost::math::policies::digits<T, boost::math::policies::policy<> >() / 2) - 2);
   }
   m_max_change_history[0] = m_max_change_history[1] = 1;
   init_chebyshev();
   // do one iteration whatever:
   //iterate();
}

template <class T>
inline remez_minimax<T>::remez_minimax(
         typename remez_minimax<T>::function_type f, 
         unsigned oN, 
         unsigned oD, 
         T a, 
         T b, 
         bool pin, 
         bool rel_err, 
         int sk,
         int bits)
   : func(f) 
{
   m_brake = 0;
   reset(oN, oD, a, b, pin, rel_err, sk, bits);
}

template <class T>
void remez_minimax<T>::reset(
         unsigned oN, 
         unsigned oD, 
         T a, 
         T b, 
         bool pin, 
         bool rel_err, 
         int sk,
         int bits,
         const vector_type& points)
{
   control_points = vector_type(oN + oD + (pin ? 1 : 2));
   solution = control_points;
   zeros = vector_type(oN + oD + (pin ? 2 : 3));
   maxima = control_points;
   orderN = oN;
   orderD = oD;
   rel_error = rel_err;
   pinned = pin;
   m_skew = sk;
   min = a;
   max = b;
   m_max_error = 0;
   unknowns = orderN + orderD + (pinned ? 1 : 2);
   control_points = points;
   solution[unknowns - 1] = 0;
   m_max_error = 0;
   if(bits == 0)
   {
      // don't bother about more than float precision:
      m_precision = (std::min)(24, (boost::math::policies::digits<T, boost::math::policies::policy<> >() / 2) - 2);
   }
   else
   {
      // can't be more accurate than half the bits of T:
      m_precision = (std::min)(bits, (boost::math::policies::digits<T, boost::math::policies::policy<> >() / 2) - 2);
   }
   m_max_change_history[0] = m_max_change_history[1] = 1;
   // do one iteration whatever:
   //iterate();
}

template <class T>
inline remez_minimax<T>::remez_minimax(
         typename remez_minimax<T>::function_type f, 
         unsigned oN, 
         unsigned oD, 
         T a, 
         T b, 
         bool pin, 
         bool rel_err, 
         int sk,
         int bits,
         const vector_type& points)
   : func(f)
{
   m_brake = 0;
   reset(oN, oD, a, b, pin, rel_err, sk, bits, points);
}

template <class T>
T remez_minimax<T>::iterate()
{
   BOOST_MATH_STD_USING
   matrix_type A(unknowns, unknowns);
   vector_type b(unknowns);

   // fill in evaluation of f(x) at each of the control points:
   for(unsigned i = 0; i < b.size(); ++i)
   {
      // take care that none of our control points are at the origin:
      if(pinned && (control_points[i] == 0))
      {
         if(i)
            control_points[i] = control_points[i-1] / 3;
         else
            control_points[i] = control_points[i+1] / 3;
      }
      b[i] = func(control_points[i]);
   }

   T err_err;
   unsigned convergence_count = 0;
   do{
      // fill in powers of x evaluated at each of the control points:
      int sign = 1;
      unsigned offsetN = pinned ? 0 : 1;
      unsigned offsetD = offsetN + orderN;
      unsigned maxorder = (std::max)(orderN, orderD);
      T Elast = solution[unknowns - 1];

      for(unsigned i = 0; i < b.size(); ++i)
      {
         T x0 = control_points[i];
         T x = x0;
         if(!pinned)
            A(i, 0) = 1;
         for(unsigned j = 0; j < maxorder; ++j)
         {
            if(j < orderN)
               A(i, j + offsetN) = x;
            if(j < orderD)
            {
               T mult = rel_error ? T(b[i] - sign * fabs(b[i]) * Elast): T(b[i] - sign * Elast);
               A(i, j + offsetD) = -x * mult;
            }
            x *= x0;
         }
         // The last variable to be solved for is the error term, 
         // sign changes with each control point:
         T E = rel_error ? T(sign * fabs(b[i])) : T(sign);
         A(i, unknowns - 1) = E;
         sign = -sign;
      }

   #ifdef BOOST_MATH_INSTRUMENT
      for(unsigned i = 0; i < b.size(); ++i)
         std::cout << b[i] << " ";
      std::cout << "\n\n";
      for(unsigned i = 0; i < b.size(); ++i)
      {
         for(unsigned j = 0; j < b.size(); ++ j)
            std::cout << A(i, j) << " ";
         std::cout << "\n";
      }
      std::cout << std::endl;
   #endif
      //
      // Now go ahead and solve the expression to get our solution:
      //
      solution = boost::math::tools::solve(A, b);

      err_err = (Elast != 0) ? T(fabs((fabs(solution[unknowns-1]) - fabs(Elast)) / fabs(Elast))) : T(1);
   }while(orderD && (convergence_count++ < 80) && (err_err > 0.001));

   //
   // Perform a sanity check to verify that the solution to the equations
   // is not so much in error as to be useless.  The matrix inversion can
   // be very close to singular, so this can be a real problem.
   //
   vector_type sanity = prod(A, solution);
   for(unsigned i = 0; i < b.size(); ++i)
   {
      T err = fabs((b[i] - sanity[i]) / fabs(b[i]));
      if(err > sqrt(epsilon<T>()))
      {
         std::cerr << "Sanity check failed: more than half the digits in the found solution are in error." << std::endl;
      }
   }

   //
   // Next comes another sanity check, we want to verify that all the control
   // points do actually alternate in sign, in practice we may have 
   // additional roots in the error function that cause this to fail.
   // Failure here is always fatal: even though this code attempts to correct
   // the problem it usually only postpones the inevitable.
   //
   polynomial<T> num, denom;
   num = this->numerator();
   denom = this->denominator();
   T e1 = b[0] - num.evaluate(control_points[0]) / denom.evaluate(control_points[0]);
#ifdef BOOST_MATH_INSTRUMENT
   std::cout << e1;
#endif
   for(unsigned i = 1; i < b.size(); ++i)
   {
      T e2 = b[i] - num.evaluate(control_points[i]) / denom.evaluate(control_points[i]);
#ifdef BOOST_MATH_INSTRUMENT
      std::cout << " " << e2;
#endif
      if(e2 * e1 > 0)
      {
         std::cerr << std::flush << "Basic sanity check failed: Error term does not alternate in sign, non-recoverable error may follow..." << std::endl;
         T perturbation = 0.05;
         do{
            T point = control_points[i] * (1 - perturbation) + control_points[i-1] * perturbation;
            e2 = func(point) - num.evaluate(point) / denom.evaluate(point);
            if(e2 * e1 < 0)
            {
               control_points[i] = point;
               break;
            }
            perturbation += 0.05;
         }while(perturbation < 0.8);

         if((e2 * e1 > 0) && (i + 1 < b.size()))
         {
            perturbation = 0.05;
            do{
               T point = control_points[i] * (1 - perturbation) + control_points[i+1] * perturbation;
               e2 = func(point) - num.evaluate(point) / denom.evaluate(point);
               if(e2 * e1 < 0)
               {
                  control_points[i] = point;
                  break;
               }
               perturbation += 0.05;
            }while(perturbation < 0.8);
         }

      }
      e1 = e2;
   }

#ifdef BOOST_MATH_INSTRUMENT
   for(unsigned i = 0; i < solution.size(); ++i)
      std::cout << solution[i] << " ";
   std::cout << std::endl << this->numerator() << std::endl;
   std::cout << this->denominator() << std::endl;
   std::cout << std::endl;
#endif

   //
   // The next step is to find all the intervals in which our maxima
   // lie:
   //
   detail::remez_error_function<T> Err(func, this->numerator(), this->denominator(), rel_error);
   zeros[0] = min;
   zeros[unknowns] = max;
   for(unsigned i = 1; i < control_points.size(); ++i)
   {
      eps_tolerance<T> tol(m_precision);
      std::uintmax_t max_iter = 1000;
      std::pair<T, T> p = toms748_solve(
         Err, 
         control_points[i-1], 
         control_points[i], 
         tol, 
         max_iter);
      zeros[i] = (p.first + p.second) / 2;
      //zeros[i] = bisect(Err, control_points[i-1], control_points[i], m_precision);
   }
   //
   // Now find all the extrema of the error function:
   //
   detail::remez_max_error_function<T> Ex(Err);
   m_max_error = 0;
   //int max_err_location = 0;
   for(unsigned i = 0; i < unknowns; ++i)
   {
      std::pair<T, T> r = brent_find_minima(Ex, zeros[i], zeros[i+1], m_precision);
      maxima[i] = r.first;
      T rel_err = fabs(r.second);
      if(rel_err > m_max_error)
      {
         m_max_error = fabs(r.second);
         //max_err_location = i;
      }
   }
   //
   // Almost done now! we just need to set our control points
   // to the extrema, and calculate how much each point has changed
   // (this will be our termination condition):
   //
   swap(control_points, maxima);
   m_max_change = 0;
   //int max_change_location = 0;
   for(unsigned i = 0; i < unknowns; ++i)
   {
      control_points[i] = (control_points[i] * (100 - m_brake) + maxima[i] * m_brake) / 100;
      T change = fabs((control_points[i] - maxima[i]) / control_points[i]);
#if 0
      if(change > m_max_change_history[1])
      {
         // divergence!!! try capping the change:
         std::cerr << "Possible divergent step, change will be capped!!" << std::endl;
         change = m_max_change_history[1];
         if(control_points[i] < maxima[i])
            control_points[i] = maxima[i] - change * maxima[i];
         else
            control_points[i] = maxima[i] + change * maxima[i];
      }
#endif
      if(change > m_max_change)
      {
         m_max_change = change;
         //max_change_location = i;
      }
   }
   //
   // store max change information:
   //
   m_max_change_history[0] = m_max_change_history[1];
   m_max_change_history[1] = fabs(m_max_change);

   return m_max_change;
}

template <class T>
polynomial<T> remez_minimax<T>::numerator()const
{
   boost::scoped_array<T> a(new T[orderN + 1]);
   if(pinned)
      a[0] = 0;
   unsigned terms = pinned ? orderN : orderN + 1;
   for(unsigned i = 0; i < terms; ++i)
      a[pinned ? i+1 : i] = solution[i];
   return boost::math::tools::polynomial<T>(&a[0], orderN);
}

template <class T>
polynomial<T> remez_minimax<T>::denominator()const
{
   unsigned terms = orderD + 1;
   unsigned offsetD = pinned ? orderN : (orderN + 1);
   boost::scoped_array<T> a(new T[terms]);
   a[0] = 1;
   for(unsigned i = 0; i < orderD; ++i)
      a[i+1] = solution[i + offsetD];
   return boost::math::tools::polynomial<T>(&a[0], orderD);
}


}}} // namespaces

#endif // BOOST_MATH_TOOLS_REMEZ_HPP



