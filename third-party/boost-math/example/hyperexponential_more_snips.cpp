// Copyright 2014 Marco Guazzone (marco.guazzone@gmail.com).

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Caution: this file contains Quickbook markup as well as code
// and comments, don't change any of the special comment markups!


//[hyperexponential_more_snip1
#include <boost/math/distributions.hpp>
#include <iostream>
#include <string>

struct ds_info
{
   std::string name;
   double iat_sample_mean;
   double iat_sample_sd;
   boost::math::hyperexponential iat_he;
   double multi_lt_sample_mean;
   double multi_lt_sample_sd;
   boost::math::hyperexponential multi_lt_he;
   double single_lt_sample_mean;
   double single_lt_sample_sd;
   boost::math::hyperexponential single_lt_he;
};

// DS1 dataset
ds_info make_ds1()
{
   ds_info ds;

   ds.name = "DS1";

   // VM interarrival time distribution
   const double iat_fit_probs[] = { 0.34561,0.08648,0.56791 };
   const double iat_fit_rates[] = { 0.0008,0.00005,0.02894 };
   ds.iat_sample_mean = 2202.1;
   ds.iat_sample_sd = 2.2e+4;
   ds.iat_he = boost::math::hyperexponential(iat_fit_probs, iat_fit_rates);

   // Multi-core VM lifetime distribution
   const double multi_lt_fit_probs[] = { 0.24667,0.37948,0.37385 };
   const double multi_lt_fit_rates[] = { 0.00004,0.000002,0.00059 };
   ds.multi_lt_sample_mean = 257173;
   ds.multi_lt_sample_sd = 4.6e+5;
   ds.multi_lt_he = boost::math::hyperexponential(multi_lt_fit_probs, multi_lt_fit_rates);

   // Single-core VM lifetime distribution
   const double single_lt_fit_probs[] = { 0.09325,0.22251,0.68424 };
   const double single_lt_fit_rates[] = { 0.000003,0.00109,0.00109 };
   ds.single_lt_sample_mean = 28754.4;
   ds.single_lt_sample_sd = 1.6e+5;
   ds.single_lt_he = boost::math::hyperexponential(single_lt_fit_probs, single_lt_fit_rates);

   return ds;
}

// DS2 dataset
ds_info make_ds2()
{
   ds_info ds;

   ds.name = "DS2";

   // VM interarrival time distribution
   const double iat_fit_probs[] = { 0.38881,0.18227,0.42892 };
   const double iat_fit_rates[] = { 0.000006,0.05228,0.00081 };
   ds.iat_sample_mean = 41285.7;
   ds.iat_sample_sd = 1.1e+05;
   ds.iat_he = boost::math::hyperexponential(iat_fit_probs, iat_fit_rates);

   // Multi-core VM lifetime distribution
   const double multi_lt_fit_probs[] = { 0.42093,0.43960,0.13947 };
   const double multi_lt_fit_rates[] = { 0.00186,0.00008,0.0000008 };
   ds.multi_lt_sample_mean = 144669.0;
   ds.multi_lt_sample_sd = 7.9e+05;
   ds.multi_lt_he = boost::math::hyperexponential(multi_lt_fit_probs, multi_lt_fit_rates);

   // Single-core VM lifetime distribution
   const double single_lt_fit_probs[] = { 0.44885,0.30675,0.2444 };
   const double single_lt_fit_rates[] = { 0.00143,0.00005,0.0000004 };
   ds.single_lt_sample_mean = 599815.0;
   ds.single_lt_sample_sd = 1.7e+06;
   ds.single_lt_he = boost::math::hyperexponential(single_lt_fit_probs, single_lt_fit_rates);

   return ds;
}

// DS3 dataset
ds_info make_ds3()
{
   ds_info ds;

   ds.name = "DS3";

   // VM interarrival time distribution
   const double iat_fit_probs[] = { 0.39442,0.24644,0.35914 };
   const double iat_fit_rates[] = { 0.00030,0.00003,0.00257 };
   ds.iat_sample_mean = 11238.8;
   ds.iat_sample_sd = 3.0e+04;
   ds.iat_he = boost::math::hyperexponential(iat_fit_probs, iat_fit_rates);

   // Multi-core VM lifetime distribution
   const double multi_lt_fit_probs[] = { 0.37621,0.14838,0.47541 };
   const double multi_lt_fit_rates[] = { 0.00498,0.000005,0.00022 };
   ds.multi_lt_sample_mean = 30739.2;
   ds.multi_lt_sample_sd = 1.6e+05;
   ds.multi_lt_he = boost::math::hyperexponential(multi_lt_fit_probs, multi_lt_fit_rates);

   // Single-core VM lifetime distribution
   const double single_lt_fit_probs[] = { 0.34131,0.12544,0.53325 };
   const double single_lt_fit_rates[] = { 0.000297,0.000003,0.00410 };
   ds.single_lt_sample_mean = 44447.8;
   ds.single_lt_sample_sd = 2.2e+05;
   ds.single_lt_he = boost::math::hyperexponential(single_lt_fit_probs, single_lt_fit_rates);

   return ds;
}

void print_fitted(ds_info const& ds)
{
   const double secs_in_a_hour = 3600;
   const double secs_in_a_month = 30 * 24 * secs_in_a_hour;

   std::cout << "### " << ds.name << std::endl;
   std::cout << "* Fitted Request Interarrival Time" << std::endl;
   std::cout << " - Mean (SD): " << boost::math::mean(ds.iat_he) << " (" << boost::math::standard_deviation(ds.iat_he) << ") seconds." << std::endl;
   std::cout << " - 99th Percentile: " << boost::math::quantile(ds.iat_he, 0.99) << " seconds." << std::endl;
   std::cout << " - Probability that a VM will arrive within 30 minutes: " << boost::math::cdf(ds.iat_he, secs_in_a_hour / 2.0) << std::endl;
   std::cout << " - Probability that a VM will arrive after 1 hour: " << boost::math::cdf(boost::math::complement(ds.iat_he, secs_in_a_hour)) << std::endl;
   std::cout << "* Fitted Multi-core VM Lifetime" << std::endl;
   std::cout << " - Mean (SD): " << boost::math::mean(ds.multi_lt_he) << " (" << boost::math::standard_deviation(ds.multi_lt_he) << ") seconds." << std::endl;
   std::cout << " - 99th Percentile: " << boost::math::quantile(ds.multi_lt_he, 0.99) << " seconds." << std::endl;
   std::cout << " - Probability that a VM will last for less than 1 month: " << boost::math::cdf(ds.multi_lt_he, secs_in_a_month) << std::endl;
   std::cout << " - Probability that a VM will last for more than 3 months: " << boost::math::cdf(boost::math::complement(ds.multi_lt_he, 3.0*secs_in_a_month)) << std::endl;
   std::cout << "* Fitted Single-core VM Lifetime" << std::endl;
   std::cout << " - Mean (SD): " << boost::math::mean(ds.single_lt_he) << " (" << boost::math::standard_deviation(ds.single_lt_he) << ") seconds." << std::endl;
   std::cout << " - 99th Percentile: " << boost::math::quantile(ds.single_lt_he, 0.99) << " seconds." << std::endl;
   std::cout << " - Probability that a VM will last for less than 1 month: " << boost::math::cdf(ds.single_lt_he, secs_in_a_month) << std::endl;
   std::cout << " - Probability that a VM will last for more than 3 months: " << boost::math::cdf(boost::math::complement(ds.single_lt_he, 3.0*secs_in_a_month)) << std::endl;
}

int main()
{
   print_fitted(make_ds1());

   print_fitted(make_ds2());

   print_fitted(make_ds3());
}
//]
