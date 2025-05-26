// Copyright John Maddock 2007.
// Copyright Paul A. Bristow 2007, 2009, 2012.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// boost_math.cpp  This is the main DLL file.

//#define BOOST_MATH_OVERFLOW_ERROR_POLICY errno_on_error
//#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
// These are now defined in project properties
// to avoid complications with pre-compiled headers:
// "BOOST_MATH_ASSERT_UNDEFINED_POLICY=0"
// "BOOST_MATH_OVERFLOW_ERROR_POLICY="errno_on_error""
// so command line shows:
// /D "BOOST_MATH_ASSERT_UNDEFINED_POLICY=0"
// /D "BOOST_MATH_OVERFLOW_ERROR_POLICY="errno_on_error""

#include "stdafx.h"

#ifdef _MSC_VER
#  pragma warning(disable: 4400) // 'const boost_math::any_distribution ^' : const/volatile qualifiers on this type are not supported
#  pragma warning(disable: 4244) // 'argument' : conversion from 'double' to 'unsigned int', possible loss of data
#  pragma warning(disable: 4512) // assignment operator could not be generated
// hypergeometric expects integer parameters.
#  pragma warning(disable: 4127) // constant
#endif

#include "boost_math.h"

namespace boost_math
{

any_distribution::any_distribution(int t, double arg1, double arg2, double arg3)
{
   TRANSLATE_EXCEPTIONS_BEGIN
   // This is where all the work gets done:
   switch(t) // index of distribution to distribution_info distributions[]
   {  // New entries must match distribution names, parameter name(s) and defaults defined below.
   case 0:
      this->reset(new concrete_distribution<boost::math::bernoulli>(boost::math::bernoulli(arg1)));
      break;
   case 1:
      this->reset(new concrete_distribution<boost::math::beta_distribution<> >(boost::math::beta_distribution<>(arg1, arg2)));
      break; // Note - no typedef, so need explicit type <> but rely on default = double.
   case 2:
      this->reset(new concrete_distribution<boost::math::binomial_distribution<> >(boost::math::binomial_distribution<>(arg1, arg2)));
      break; // Note - no typedef, so need explicit type <> but rely on default = double.
   case 3:
      this->reset(new concrete_distribution<boost::math::cauchy>(boost::math::cauchy(arg1, arg2)));
      break;
   case 4:
      this->reset(new concrete_distribution<boost::math::chi_squared>(boost::math::chi_squared(arg1)));
      break;
   case 5:
      this->reset(new concrete_distribution<boost::math::exponential>(boost::math::exponential(arg1)));
      break;
   case 6:
      this->reset(new concrete_distribution<boost::math::extreme_value>(boost::math::extreme_value(arg1)));
      break;
   case 7:
      this->reset(new concrete_distribution<boost::math::fisher_f >(boost::math::fisher_f(arg1, arg2)));
      break;
   case 8:
      this->reset(new concrete_distribution<boost::math::gamma_distribution<> >(boost::math::gamma_distribution<>(arg1, arg2)));
      break;
   case 9:
      this->reset(new concrete_distribution<boost::math::geometric_distribution<> >(boost::math::geometric_distribution<>(arg1)));
      break;
   case 10:
      this->reset(new concrete_distribution<boost::math::hypergeometric_distribution<> >(boost::math::hypergeometric_distribution<>(arg1, arg2, arg3)));
      break;
   case 11:
      this->reset(new concrete_distribution<boost::math::inverse_chi_squared_distribution<> >(boost::math::inverse_chi_squared_distribution<>(arg1, arg2)));
      break;
   case 12:
      this->reset(new concrete_distribution<boost::math::inverse_gamma_distribution<> >(boost::math::inverse_gamma_distribution<>(arg1, arg2)));
      break;
   case 13:
      this->reset(new concrete_distribution<boost::math::inverse_gaussian_distribution<> >(boost::math::inverse_gaussian_distribution<>(arg1, arg2)));
      break;
   case 14:
      this->reset(new concrete_distribution<boost::math::laplace_distribution<> >(boost::math::laplace_distribution<>(arg1, arg2)));
      break;
   case 15:
      this->reset(new concrete_distribution<boost::math::logistic_distribution<> >(boost::math::logistic_distribution<>(arg1, arg2)));
      break;
   case 16:
      this->reset(new concrete_distribution<boost::math::lognormal_distribution<> >(boost::math::lognormal_distribution<>(arg1, arg2)));
      break;
   case 17:
      this->reset(new concrete_distribution<boost::math::negative_binomial_distribution<> >(boost::math::negative_binomial_distribution<>(arg1, arg2)));
      break;
   case 18:
      this->reset(new concrete_distribution<boost::math::non_central_beta_distribution<> >(boost::math::non_central_beta_distribution<>(arg1, arg2, arg3)));
      break;
   case 19:
      this->reset(new concrete_distribution<boost::math::non_central_chi_squared_distribution<> >(boost::math::non_central_chi_squared_distribution<>(arg1, arg2)));
      break;
   case 20:
      this->reset(new concrete_distribution<boost::math::non_central_f_distribution<> >(boost::math::non_central_f_distribution<>(arg1, arg2, arg3)));
      break;
   case 21:
      this->reset(new concrete_distribution<boost::math::non_central_t_distribution<> >(boost::math::non_central_t_distribution<>(arg1, arg2)));
      break;
   case 22:
      this->reset(new concrete_distribution<boost::math::normal_distribution<> >(boost::math::normal_distribution<>(arg1, arg2)));
      break;
   case 23:
      this->reset(new concrete_distribution<boost::math::pareto>(boost::math::pareto(arg1, arg2)));
      break;
   case 24:
      this->reset(new concrete_distribution<boost::math::poisson>(boost::math::poisson(arg1)));
      break;
   case 25:
      this->reset(new concrete_distribution<boost::math::rayleigh>(boost::math::rayleigh(arg1)));
      break;
   case 26:
      this->reset(new concrete_distribution<boost::math::skew_normal>(boost::math::skew_normal(arg1, arg2, arg3)));
      break;
   case 27:
      this->reset(new concrete_distribution<boost::math::students_t>(boost::math::students_t(arg1)));
      break;
   case 28:
      this->reset(new concrete_distribution<boost::math::triangular>(boost::math::triangular(arg1, arg2, arg3)));
      break;
   case 29:
      this->reset(new concrete_distribution<boost::math::uniform>(boost::math::uniform(arg1, arg2)));
      break;
   case 30:
      this->reset(new concrete_distribution<boost::math::weibull>(boost::math::weibull(arg1, arg2)));
      break;


   default:
      // TODO  Need some proper error handling here?
      BOOST_MATH_ASSERT(0);
   }
   TRANSLATE_EXCEPTIONS_END
} // any_distribution constructor.

struct distribution_info
{
   const char* name; // of distribution.
   const char* first_param; // Parameters' name like "degrees of freedom",
   const char* second_param; // if required, else "",
   const char* third_param; // if required, else "".
   // triangular and non-centrals need 3 parameters.
   // (Only the Bi-Weibull would need 5 parameters?)
   double first_default; // distribution parameter value, often 0, 0.5 or 1.
   double second_default; // 0 if there isn't a second argument.
   // Note that defaults below follow default argument in constructors,
   // if any, but need not be the same.
   double third_default; // 0 if there isn't a third argument.
};

distribution_info distributions[] =
{ // distribution name, parameter name(s) and default(s)
  // Order must match any_distribution constructor above!
  // Null string "" and zero default for un-used arguments.
   { "Bernoulli", "Probability", "", "",0.5, 0, 0}, // case 0
   { "Beta", "Alpha", "Beta", "", 1, 1, 0}, // case 1
   { "Binomial", "Trials", "Probability of success", "", 1, 0.5, 0}, // case 2
   { "Cauchy", "Location", "Scale", "", 0, 1, 0}, // case 3
   { "Chi_squared", "Degrees of freedom", "", "", 1, 0, 0}, // case 4
   { "Exponential", "lambda", "", "", 1, 0, 0}, // case 5
   { "Extreme value", "Location", "Scale", "", 0, 1, 0}, // case 6
   { "Fisher-F", "Degrees of freedom 1", "Degrees of freedom 2", "", 1, 1, 0}, // case 7
   { "Gamma (Erlang)", "Shape", "Scale", "", 1, 1, 0}, // case 8
   { "Geometric", "Probability", "", "", 1, 0, 0}, // case 9
   { "HyperGeometric", "Defects", "Samples", "Objects", 1, 0, 1}, // case 10
   { "InverseChiSq", "Degrees of Freedom", "Scale", "", 1, 1, 0}, // case 11
   { "InverseGamma", "Shape", "Scale", "", 1, 1, 0}, // case 12
   { "InverseGaussian", "Mean", "Scale", "", 1, 1, 0}, // case 13
   { "Laplace", "Location", "Scale", "", 0, 1, 0}, // case 14
   { "Logistic", "Location", "Scale", "", 0, 1, 0}, // case 15
   { "LogNormal", "Location", "Scale", "", 0, 1, 0}, // case 16
   { "Negative Binomial", "Successes", "Probability of success", "", 1, 0.5, 0}, // case 17
   { "Noncentral Beta", "Shape alpha", "Shape beta", "Non-centrality", 1, 1, 0}, // case 18
   { "Noncentral ChiSquare", "Degrees of Freedom", "Non-centrality", "", 1, 0, 0}, // case 19
   { "Noncentral F", "Degrees of Freedom 1", "Degrees of Freedom 2", "Non-centrality", 1, 1, 0}, // case 20
   { "Noncentral t", "Degrees of Freedom", "Non-centrality", "", 1, 0, 0}, // case 21
   { "Normal (Gaussian)", "Mean", "Standard Deviation", "", 0, 1, 0}, // case 22
   { "Pareto", "Location", "Shape","", 1, 1, 0}, // case 23
   { "Poisson", "Mean", "", "", 1, 0, 0}, // case 24
   { "Rayleigh", "Shape", "", "", 1, 0, 0}, // case 25
   { "Skew Normal", "Location", "Shape", "Skew", 0, 1, 0}, // case 27  (defaults to Gaussian).
   { "Student's t", "Degrees of Freedom", "", "", 1, 0, 0}, // case 28
   { "Triangular", "Lower", "Mode", "Upper", -1, 0, +1 }, // case 29 3rd parameter!
   // 0, 0.5, 1 also said to be 'standard' but this is most like an approximation to Gaussian distribution.
   { "Uniform", "Lower", "Upper", "", 0, 1, 0}, // case 30
   { "Weibull", "Shape", "Scale", "", 1, 1, 0}, // case 31
};

// How many distributions are supported:
int any_distribution::size()
{
   return sizeof(distributions) / sizeof(distributions[0]);
}

// Display name of i'th distribution:
System::String^ any_distribution::distribution_name(int i)
{
   if(i >= size())
      return "";
   return gcnew System::String(distributions[i].name);
}
// Name of first distribution parameter, or null if not supported:
System::String^ any_distribution::first_param_name(int i)
{
   if(i >= size())
      return "";
   return gcnew System::String(distributions[i].first_param);
}
// Name of second distribution parameter, or null if not supported:
System::String^ any_distribution::second_param_name(int i)
{
   if(i >= size())
      return "";
   return gcnew System::String(distributions[i].second_param);
}
// Name of third distribution parameter, or null if not supported:
System::String^ any_distribution::third_param_name(int i)
{
   if(i >= size())
      return "";
   return gcnew System::String(distributions[i].third_param);
}
// default value for first parameter:
double any_distribution::first_param_default(int i)
{
   if(i >= size())
      return 0;
   return distributions[i].first_default;
}
// default value for second parameter:
double any_distribution::second_param_default(int i)
{
   if(i >= size())
      return 0;
   return distributions[i].second_default;
}
// default value for third parameter:
double any_distribution::third_param_default(int i)
{
   if(i >= size())
      return 0;
   return distributions[i].third_default;
}

} // namespace boost_math


