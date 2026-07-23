//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_SOLVE_HPP
#define BOOST_MATH_TOOLS_SOLVE_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/math/tools/assert.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4996 4267 4244)
#endif

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace boost{ namespace math{ namespace tools{

//
// Find x such that Ax = b
//
// Caution: this uses undocumented, and untested ublas code,
// however short of writing our own LU-decomposition code
// it's the only game in town.
//
template <class T>
boost::numeric::ublas::vector<T> solve(
          const boost::numeric::ublas::matrix<T>& A_,
          const boost::numeric::ublas::vector<T>& b_)
{
   //BOOST_MATH_ASSERT(A_.size() == b_.size());

   boost::numeric::ublas::matrix<T> A(A_);
   boost::numeric::ublas::vector<T> b(b_);
   boost::numeric::ublas::permutation_matrix<> piv(b.size());
   lu_factorize(A, piv);
   lu_substitute(A, piv, b);
   //
   // iterate to reduce error:
   //
   boost::numeric::ublas::vector<T> delta(b.size());
   for(unsigned k = 0; k < 1; ++k)
   {
      noalias(delta) = prod(A_, b);
      delta -= b_;
      lu_substitute(A, piv, delta);
      b -= delta;

      T max_error = 0;

      for(unsigned i = 0; i < delta.size(); ++i)
      {
         using std::abs;
         T err = abs(delta[i] / b[i]);
         if(err > max_error)
            max_error = err;
      }
      //std::cout << "Max change in LU error correction: " << max_error << std::endl;
   }

   return b;
}

}}} // namespaces

#endif // BOOST_MATH_TOOLS_SOLVE_HPP


