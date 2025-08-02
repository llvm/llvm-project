// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently,
// but are changed infrequently.

// Copyright John Maddock 2007.
// Copyright Paul A. Bristow 2007, 2009, 2010, 2012

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Laplace added Aug 2009 PAB, and several others Nov 2010, 
// added skew_normal 2012.

#ifdef _MSC_VER
#  pragma once
#  pragma warning (disable : 4127)
#endif

#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error

#include <boost/math/distributions/bernoulli.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/cauchy.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/extreme_value.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/geometric.hpp>
#include <boost/math/distributions/hypergeometric.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>
#include <boost/math/distributions/inverse_gaussian.hpp>
#include <boost/math/distributions/laplace.hpp>
#include <boost/math/distributions/logistic.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/non_central_beta.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/non_central_f.hpp>
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/pareto.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/rayleigh.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/distributions/triangular.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/weibull.hpp>
