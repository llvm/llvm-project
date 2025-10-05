/*! \file dist_graphs.cpp
    \brief Produces Scalable Vector Graphic (.svg) files for all distributions.
    \details These files can be viewed using most browsers,
    though MS Internet Explorer requires a plugin from Adobe.
    These file can be converted to .png using Inkscape
    (see www.inkscape.org) Export Bit option which by default produces
    a Portable Network Graphic file with that same filename but .png suffix instead of .svg.
    Using Python, generate.sh does this conversion automatically for all .svg files in a folder.

    \author John Maddock and Paul A. Bristow
  */
//  Copyright John Maddock 2008.
//  Copyright Paul A. Bristow 2008, 2009, 2012, 2016
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4180) // qualifier applied to function type has no meaning; ignored
#  pragma warning (disable : 4503) // decorated name length exceeded, name was truncated
#  pragma warning (disable : 4512) // assignment operator could not be generated
#  pragma warning (disable : 4224) // nonstandard extension used : formal parameter 'function_ptr' was previously defined as a type
#  pragma warning (disable : 4127) // conditional expression is constant
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/distributions.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/svg_plot/svg_2d_plot.hpp>

#include <list>
#include <map>
#include <string>

template <class Dist>
struct is_discrete_distribution
   : public std::false_type{}; // Default is continuous distribution.

// Some discrete distributions.
template<class T, class P>
struct is_discrete_distribution<boost::math::bernoulli_distribution<T,P> >
   : public std::true_type{};
template<class T, class P>
struct is_discrete_distribution<boost::math::binomial_distribution<T,P> >
   : public std::true_type{};
template<class T, class P>
struct is_discrete_distribution<boost::math::negative_binomial_distribution<T,P> >
   : public std::true_type{};
template<class T, class P>
struct is_discrete_distribution<boost::math::poisson_distribution<T,P> >
   : public std::true_type{};
template<class T, class P>
struct is_discrete_distribution<boost::math::hypergeometric_distribution<T,P> >
   : public std::true_type{};


template <class Dist>
struct value_finder
{
   value_finder(Dist const& d, typename Dist::value_type v)
      : m_dist(d), m_value(v) {}

   inline typename Dist::value_type operator()(const typename Dist::value_type& x)
   {
      return pdf(m_dist, x) - m_value;
   }

private:
   Dist m_dist;
   typename Dist::value_type m_value;
}; // value_finder

template <class Dist>
class distribution_plotter
{
public:
   distribution_plotter() : m_pdf(true), m_min_x(0), m_max_x(0), m_min_y(0), m_max_y(0) {}
   distribution_plotter(bool pdf) : m_pdf(pdf), m_min_x(0), m_max_x(0), m_min_y(0), m_max_y(0) {}

   void add(const Dist& d, const std::string& name)
   {
      // Add name of distribution to our list for later:
      m_distributions.push_back(std::make_pair(name, d));
      //
      // Get the extent of the distribution from the support:
      std::pair<double, double> r = support(d);
      double a = r.first;
      double b = r.second;
      //
      // PDF maximum is at the mode (probably):
      double mod;
      try
      {
         mod = mode(d);
      }
      catch(const std::domain_error& )
      { // but if not use the lower limit of support.
         mod = a;
      }
      if((mod <= a) && !is_discrete_distribution<Dist>::value)
      { // Continuous distribution at or below lower limit of support.
        double margin = 1e-2; // Margin of 1% (say) to get lowest off the 'end stop'.
         if((a != 0) && (fabs(a) > margin))
         {  
            mod = a * (1 + ((a > 0) ? margin : -margin)); 
         }
         else
         { // Case of mod near zero?
            mod = margin;
         }
      }
      double peek_y = pdf(d, mod);
      double min_y = peek_y / 20;
      //
      // If the extent is "infinite" then find out how large it
      // has to be for the PDF to decay to min_y:
      //
      if(a <= -(std::numeric_limits<double>::max)())
      {
         std::uintmax_t max_iter = 500;
         double guess = mod;
         if((pdf(d, 0) > min_y) || (guess == 0))
            guess = -1e-3;
         a = boost::math::tools::bracket_and_solve_root(
            value_finder<Dist>(d, min_y),
            guess,
            8.0,
            true,
            boost::math::tools::eps_tolerance<double>(10),
            max_iter).first;
      }
      if(b >= (std::numeric_limits<double>::max)())
      {
         std::uintmax_t max_iter = 500;
         double guess = mod;
         if(a <= 0)
            if((pdf(d, 0) > min_y) || (guess == 0))
               guess = 1e-3;
         b = boost::math::tools::bracket_and_solve_root(
            value_finder<Dist>(d, min_y),
            guess,
            8.0,
            false,
            boost::math::tools::eps_tolerance<double>(10),
            max_iter).first;
      }
      //
      // Recalculate peek_y and location of mod so that
      // it's not too close to one end of the graph:
      // otherwise we may be shooting off to infinity.
      //
      if(!is_discrete_distribution<Dist>::value)
      {
         if(mod <= a + (b-a)/50)
         {
            mod = a + (b-a)/50;
         }
         if(mod >= b - (b-a)/50)
         {
            mod = b - (b-a)/50;
         }
         peek_y = pdf(d, mod);
      }
      //
      // Now set our limits:
      //
      if(peek_y > m_max_y)
         m_max_y = peek_y;
      if(m_max_x == m_min_x)
      {
         m_max_x = b;
         m_min_x = a;
      }
      else
      {
         if(a < m_min_x)
            m_min_x = a;
         if(b > m_max_x)
            m_max_x = b;
      }
   } // add

   void plot(const std::string& title, const std::string& file)
   {
      using namespace boost::svg;

      static const svg_color colors[5] =
      {
         darkblue,
         darkred,
         darkgreen,
         darkorange,
         chartreuse
      };

      if(m_pdf == false)
      {
         m_min_y = 0;
         m_max_y = 1;
      }

      std::cout << "Plotting " << title << " to " << file << std::endl;

      svg_2d_plot plot;
      plot.image_x_size(750);
      plot.image_y_size(400);
      plot.copyright_holder("John Maddock").copyright_date("2008").boost_license_on(true);
      plot.coord_precision(4); // Avoids any visible steps.
      plot.title_font_size(20);
      plot.legend_title_font_size(15);
      plot.title(title);
      if((m_distributions.size() == 1) && (m_distributions.begin()->first == ""))
         plot.legend_on(false);
      else
         plot.legend_on(true);
      plot.title_on(true);
      //plot.x_major_labels_on(true).y_major_labels_on(true);
      //double x_delta = (m_max_x - m_min_x) / 10;
      double y_delta = (m_max_y - m_min_y) / 10;
      if(is_discrete_distribution<Dist>::value)
         plot.x_range(m_min_x - 0.5, m_max_x + 0.5)
             .y_range(m_min_y, m_max_y + y_delta);
      else
         plot.x_range(m_min_x, m_max_x)
             .y_range(m_min_y, m_max_y + y_delta);
      plot.x_label_on(true).x_label("Random Variable");
      plot.y_label_on(true).y_label("Probability");
      plot.plot_border_color(lightslategray)
          .background_border_color(lightslategray)
          .legend_border_color(lightslategray)
          .legend_background_color(white);
      //
      // Work out axis tick intervals:
      //
      double l = std::floor(std::log10((m_max_x - m_min_x) / 10) + 0.5);
      double interval = std::pow(10.0, (int)l);
      if(((m_max_x - m_min_x) / interval) > 10)
         interval *= 5;
      if(is_discrete_distribution<Dist>::value)
      {
         interval = interval > 1 ? std::floor(interval) : 1;
         plot.x_num_minor_ticks(0);
      }
      plot.x_major_interval(interval);
      l = std::floor(std::log10((m_max_y - m_min_y) / 10) + 0.5);
      interval = std::pow(10.0, (int)l);
      if(((m_max_y - m_min_y) / interval) > 10)
         interval *= 5;
      plot.y_major_interval(interval);

      int color_index = 0;

      if(!is_discrete_distribution<Dist>::value)
      {
         // Continuous distribution:
         for(typename std::list<std::pair<std::string, Dist> >::const_iterator i = m_distributions.begin();
            i != m_distributions.end(); ++i)
         {
            double x = m_min_x;
            double continuous_interval = (m_max_x - m_min_x) / 200;
            std::map<double, double> data;
            while(x <= m_max_x)
            {
               data[x] = m_pdf ? pdf(i->second, x) : cdf(i->second, x);
               x += continuous_interval;
            }
            plot.plot(data, i->first)
               .line_on(true)
               .line_color(colors[color_index])
               .line_width(1.)
               .shape(none);

               //.bezier_on(true) // Bezier can't cope with badly behaved like uniform & triangular.
            ++color_index;
            color_index = color_index % (sizeof(colors)/sizeof(colors[0]));
         }
      }
      else
      {
         // Discrete distribution:
         double x_width = 0.75 / m_distributions.size();
         double x_off = -0.5 * 0.75;
         for(typename std::list<std::pair<std::string, Dist> >::const_iterator i = m_distributions.begin();
            i != m_distributions.end(); ++i)
         {
            double x = ceil(m_min_x);
            double discrete_interval = 1;
            std::map<double, double> data;
            while(x <= m_max_x)
            {
               double p;
               try{
                  p = m_pdf ? pdf(i->second, x) : cdf(i->second, x);
               }
               catch(const std::domain_error&)
               {
                  p = 0;
               }
               data[x + x_off] = 0;
               data[x + x_off + 0.00001] = p;
               data[x + x_off + x_width] = p;
               data[x + x_off + x_width + 0.00001] = 0;
               x += discrete_interval;
            }
            x_off += x_width;
            svg_2d_plot_series& s = plot.plot(data, i->first);
            s.line_on(true)
               .line_color(colors[color_index])
               .line_width(1.)
               .shape(none)
               .area_fill(colors[color_index]);
            ++color_index;
            color_index = color_index % (sizeof(colors)/sizeof(colors[0]));
         }
      } // discrete
      plot.write(file);
   } // void plot(const std::string& title, const std::string& file)

private:
   bool m_pdf;
   std::list<std::pair<std::string, Dist> > m_distributions;
   double m_min_x, m_max_x, m_min_y, m_max_y;
};

int main()
{
  try
  {
   std::cout << "Distribution Graphs" << std::endl;
   distribution_plotter<boost::math::gamma_distribution<> >
      gamma_plotter;
   gamma_plotter.add(boost::math::gamma_distribution<>(0.75), "shape = 0.75");
   gamma_plotter.add(boost::math::gamma_distribution<>(1), "shape = 1");
   gamma_plotter.add(boost::math::gamma_distribution<>(3), "shape = 3");
   gamma_plotter.plot("Gamma Distribution PDF With Scale = 1", "gamma1_pdf.svg");

   distribution_plotter<boost::math::gamma_distribution<> >
      gamma_plotter2;
   gamma_plotter2.add(boost::math::gamma_distribution<>(2, 0.5), "scale = 0.5");
   gamma_plotter2.add(boost::math::gamma_distribution<>(2, 1), "scale = 1");
   gamma_plotter2.add(boost::math::gamma_distribution<>(2, 2), "scale = 2");
   gamma_plotter2.plot("Gamma Distribution PDF With Shape = 2", "gamma2_pdf.svg");

   distribution_plotter<boost::math::normal>
      normal_plotter;
   normal_plotter.add(boost::math::normal(0, 1), "&#x3BC; = 0, &#x3C3; = 1");
   normal_plotter.add(boost::math::normal(0, 0.5), "&#x3BC; = 0, &#x3C3; = 0.5");
   normal_plotter.add(boost::math::normal(0, 2), "&#x3BC; = 0, &#x3C3; = 2");
   normal_plotter.add(boost::math::normal(-1, 1), "&#x3BC; = -1, &#x3C3; = 1");
   normal_plotter.add(boost::math::normal(1, 1), "&#x3BC; = 1, &#x3C3; = 1");
   normal_plotter.plot("Normal Distribution PDF", "normal_pdf.svg");

   distribution_plotter<boost::math::laplace>
      laplace_plotter;
   laplace_plotter.add(boost::math::laplace(0, 1), "&#x3BC; = 0, &#x3C3; = 1");
   laplace_plotter.add(boost::math::laplace(0, 0.5), "&#x3BC; = 0, &#x3C3; = 0.5");
   laplace_plotter.add(boost::math::laplace(0, 2), "&#x3BC; = 0, &#x3C3; = 2");
   laplace_plotter.add(boost::math::laplace(-1, 1), "&#x3BC; = -1, &#x3C3; = 1");
   laplace_plotter.add(boost::math::laplace(1, 1), "&#x3BC; = 1, &#x3C3; = 1");
   laplace_plotter.plot("Laplace Distribution PDF", "laplace_pdf.svg");

   distribution_plotter<boost::math::non_central_chi_squared>
      nc_cs_plotter;
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 0), "v=20, &#x3BB;=0");
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 1), "v=20, &#x3BB;=1");
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 5), "v=20, &#x3BB;=5");
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 10), "v=20, &#x3BB;=10");
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 20), "v=20, &#x3BB;=20");
   nc_cs_plotter.add(boost::math::non_central_chi_squared(20, 100), "v=20, &#x3BB;=100");
   nc_cs_plotter.plot("Non Central Chi Squared PDF", "nccs_pdf.svg");

   distribution_plotter<boost::math::non_central_beta>
      nc_beta_plotter;
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 0), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=0");
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 1), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=1");
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 5), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=5");
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 10), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=10");
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 40), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=40");
   nc_beta_plotter.add(boost::math::non_central_beta(10, 15, 100), "&#x3B1;=10, &#x3B2;=15, &#x3B4;=100");
   nc_beta_plotter.plot("Non Central Beta PDF", "nc_beta_pdf.svg");

   distribution_plotter<boost::math::non_central_f>
      nc_f_plotter;
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 0), "v1=10, v2=20, &#x3BB;=0");
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 1), "v1=10, v2=20, &#x3BB;=1");
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 5), "v1=10, v2=20, &#x3BB;=5");
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 10), "v1=10, v2=20, &#x3BB;=10");
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 40), "v1=10, v2=20, &#x3BB;=40");
   nc_f_plotter.add(boost::math::non_central_f(10, 20, 100), "v1=10, v2=20, &#x3BB;=100");
   nc_f_plotter.plot("Non Central F PDF", "nc_f_pdf.svg");

   distribution_plotter<boost::math::non_central_t>
      nc_t_plotter;
   nc_t_plotter.add(boost::math::non_central_t(10, -10), "v=10, &#x3B4;=-10");
   nc_t_plotter.add(boost::math::non_central_t(10, -5), "v=10, &#x3B4;=-5");
   nc_t_plotter.add(boost::math::non_central_t(10, 0), "v=10, &#x3B4;=0");
   nc_t_plotter.add(boost::math::non_central_t(10, 5), "v=10, &#x3B4;=5");
   nc_t_plotter.add(boost::math::non_central_t(10, 10), "v=10, &#x3B4;=10");
   nc_t_plotter.add(boost::math::non_central_t(std::numeric_limits<double>::infinity(), 15), "v=inf, &#x3B4;=15");
   nc_t_plotter.plot("Non Central T PDF", "nc_t_pdf.svg");

   distribution_plotter<boost::math::non_central_t>
     nc_t_CDF_plotter(false);
   nc_t_CDF_plotter.add(boost::math::non_central_t(10, -10), "v=10, &#x3B4;=-10");
   nc_t_CDF_plotter.add(boost::math::non_central_t(10, -5), "v=10, &#x3B4;=-5");
   nc_t_CDF_plotter.add(boost::math::non_central_t(10, 0), "v=10, &#x3B4;=0");
   nc_t_CDF_plotter.add(boost::math::non_central_t(10, 5), "v=10, &#x3B4;=5");
   nc_t_CDF_plotter.add(boost::math::non_central_t(10, 10), "v=10, &#x3B4;=10");
   nc_t_CDF_plotter.add(boost::math::non_central_t(std::numeric_limits<double>::infinity(), 15), "v=inf, &#x3B4;=15");
   nc_t_CDF_plotter.plot("Non Central T CDF", "nc_t_cdf.svg");

   distribution_plotter<boost::math::beta_distribution<> >
      beta_plotter;
   beta_plotter.add(boost::math::beta_distribution<>(0.5, 0.5), "alpha=0.5, beta=0.5");
   beta_plotter.add(boost::math::beta_distribution<>(5, 1), "alpha=5, beta=1");
   beta_plotter.add(boost::math::beta_distribution<>(1, 3), "alpha=1, beta=3");
   beta_plotter.add(boost::math::beta_distribution<>(2, 2), "alpha=2, beta=2");
   beta_plotter.add(boost::math::beta_distribution<>(2, 5), "alpha=2, beta=5");
   beta_plotter.plot("Beta Distribution PDF", "beta_pdf.svg");

   distribution_plotter<boost::math::cauchy_distribution<> >
      cauchy_plotter;
   cauchy_plotter.add(boost::math::cauchy_distribution<>(-5, 1), "location = -5");
   cauchy_plotter.add(boost::math::cauchy_distribution<>(0, 1), "location = 0");
   cauchy_plotter.add(boost::math::cauchy_distribution<>(5, 1), "location = 5");
   cauchy_plotter.plot("Cauchy Distribution PDF (scale = 1)", "cauchy_pdf1.svg");

   distribution_plotter<boost::math::cauchy_distribution<> >
      cauchy_plotter2;
   cauchy_plotter2.add(boost::math::cauchy_distribution<>(0, 0.5), "scale = 0.5");
   cauchy_plotter2.add(boost::math::cauchy_distribution<>(0, 1), "scale = 1");
   cauchy_plotter2.add(boost::math::cauchy_distribution<>(0, 2), "scale = 2");
   cauchy_plotter2.plot("Cauchy Distribution PDF (location = 0)", "cauchy_pdf2.svg");

   distribution_plotter<boost::math::chi_squared_distribution<> >
      chi_squared_plotter;
   //chi_squared_plotter.add(boost::math::chi_squared_distribution<>(1), "v=1");
   chi_squared_plotter.add(boost::math::chi_squared_distribution<>(2), "v=2");
   chi_squared_plotter.add(boost::math::chi_squared_distribution<>(5), "v=5");
   chi_squared_plotter.add(boost::math::chi_squared_distribution<>(10), "v=10");
   chi_squared_plotter.plot("Chi Squared Distribution PDF", "chi_squared_pdf.svg");

   distribution_plotter<boost::math::exponential_distribution<> >
      exponential_plotter;
   exponential_plotter.add(boost::math::exponential_distribution<>(0.5), "&#x3BB;=0.5");
   exponential_plotter.add(boost::math::exponential_distribution<>(1), "&#x3BB;=1");
   exponential_plotter.add(boost::math::exponential_distribution<>(2), "&#x3BB;=2");
   exponential_plotter.plot("Exponential Distribution PDF", "exponential_pdf.svg");

   distribution_plotter<boost::math::extreme_value_distribution<> >
      extreme_value_plotter;
   extreme_value_plotter.add(boost::math::extreme_value_distribution<>(-5), "location=-5");
   extreme_value_plotter.add(boost::math::extreme_value_distribution<>(0), "location=0");
   extreme_value_plotter.add(boost::math::extreme_value_distribution<>(5), "location=5");
   extreme_value_plotter.plot("Extreme Value Distribution PDF (shape=1)", "extreme_value_pdf1.svg");

   distribution_plotter<boost::math::extreme_value_distribution<> >
      extreme_value_plotter2;
   extreme_value_plotter2.add(boost::math::extreme_value_distribution<>(0, 0.5), "shape=0.5");
   extreme_value_plotter2.add(boost::math::extreme_value_distribution<>(0, 1), "shape=1");
   extreme_value_plotter2.add(boost::math::extreme_value_distribution<>(0, 2), "shape=2");
   extreme_value_plotter2.plot("Extreme Value Distribution PDF (location=0)", "extreme_value_pdf2.svg");

   distribution_plotter<boost::math::fisher_f_distribution<> >
      fisher_f_plotter;
   fisher_f_plotter.add(boost::math::fisher_f_distribution<>(4, 4), "n=4, m=4");
   fisher_f_plotter.add(boost::math::fisher_f_distribution<>(10, 4), "n=10, m=4");
   fisher_f_plotter.add(boost::math::fisher_f_distribution<>(10, 10), "n=10, m=10");
   fisher_f_plotter.add(boost::math::fisher_f_distribution<>(4, 10), "n=4, m=10");
   fisher_f_plotter.plot("F Distribution PDF", "fisher_f_pdf.svg");

   distribution_plotter<boost::math::kolmogorov_smirnov_distribution<> >
      kolmogorov_smirnov_cdf_plotter(false);
   kolmogorov_smirnov_cdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(1), "n=1");
   kolmogorov_smirnov_cdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(2), "n=2");
   kolmogorov_smirnov_cdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(5), "n=5");
   kolmogorov_smirnov_cdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(10), "n=10");
   kolmogorov_smirnov_cdf_plotter.plot("Kolmogorov-Smirnov Distribution CDF", "kolmogorov_smirnov_cdf.svg");

   distribution_plotter<boost::math::kolmogorov_smirnov_distribution<> >
      kolmogorov_smirnov_pdf_plotter;
   kolmogorov_smirnov_pdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(1), "n=1");
   kolmogorov_smirnov_pdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(2), "n=2");
   kolmogorov_smirnov_pdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(5), "n=5");
   kolmogorov_smirnov_pdf_plotter.add(boost::math::kolmogorov_smirnov_distribution<>(10), "n=10");
   kolmogorov_smirnov_pdf_plotter.plot("Kolmogorov-Smirnov Distribution PDF", "kolmogorov_smirnov_pdf.svg");

   distribution_plotter<boost::math::lognormal_distribution<> >
      lognormal_plotter;
   lognormal_plotter.add(boost::math::lognormal_distribution<>(-1), "location=-1");
   lognormal_plotter.add(boost::math::lognormal_distribution<>(0), "location=0");
   lognormal_plotter.add(boost::math::lognormal_distribution<>(1), "location=1");
   lognormal_plotter.plot("Lognormal Distribution PDF (scale=1)", "lognormal_pdf1.svg");

   distribution_plotter<boost::math::lognormal_distribution<> >
      lognormal_plotter2;
   lognormal_plotter2.add(boost::math::lognormal_distribution<>(0, 0.5), "scale=0.5");
   lognormal_plotter2.add(boost::math::lognormal_distribution<>(0, 1), "scale=1");
   lognormal_plotter2.add(boost::math::lognormal_distribution<>(0, 2), "scale=2");
   lognormal_plotter2.plot("Lognormal Distribution PDF (location=0)", "lognormal_pdf2.svg");

   distribution_plotter<boost::math::pareto_distribution<> >
      pareto_plotter; // Rely on 2nd parameter shape = 1 default.
   pareto_plotter.add(boost::math::pareto_distribution<>(1), "scale=1");
   pareto_plotter.add(boost::math::pareto_distribution<>(2), "scale=2");
   pareto_plotter.add(boost::math::pareto_distribution<>(3), "scale=3");
   pareto_plotter.plot("Pareto Distribution PDF (shape=1)", "pareto_pdf1.svg");

   distribution_plotter<boost::math::pareto_distribution<> >
      pareto_plotter2;
   pareto_plotter2.add(boost::math::pareto_distribution<>(1, 0.5), "shape=0.5");
   pareto_plotter2.add(boost::math::pareto_distribution<>(1, 1), "shape=1");
   pareto_plotter2.add(boost::math::pareto_distribution<>(1, 2), "shape=2");
   pareto_plotter2.plot("Pareto Distribution PDF (scale=1)", "pareto_pdf2.svg");

   distribution_plotter<boost::math::rayleigh_distribution<> >
      rayleigh_plotter;
   rayleigh_plotter.add(boost::math::rayleigh_distribution<>(0.5), "&#x3C3;=0.5");
   rayleigh_plotter.add(boost::math::rayleigh_distribution<>(1), "&#x3C3;=1");
   rayleigh_plotter.add(boost::math::rayleigh_distribution<>(2), "&#x3C3;=2");
   rayleigh_plotter.add(boost::math::rayleigh_distribution<>(4), "&#x3C3;=4");
   rayleigh_plotter.add(boost::math::rayleigh_distribution<>(10), "&#x3C3;=10");
   rayleigh_plotter.plot("Rayleigh Distribution PDF", "rayleigh_pdf.svg");

   distribution_plotter<boost::math::rayleigh_distribution<> >
      rayleigh_cdf_plotter(false);
   rayleigh_cdf_plotter.add(boost::math::rayleigh_distribution<>(0.5), "&#x3C3;=0.5");
   rayleigh_cdf_plotter.add(boost::math::rayleigh_distribution<>(1), "&#x3C3;=1");
   rayleigh_cdf_plotter.add(boost::math::rayleigh_distribution<>(2), "&#x3C3;=2");
   rayleigh_cdf_plotter.add(boost::math::rayleigh_distribution<>(4), "&#x3C3;=4");
   rayleigh_cdf_plotter.add(boost::math::rayleigh_distribution<>(10), "&#x3C3;=10");
   rayleigh_cdf_plotter.plot("Rayleigh Distribution CDF", "rayleigh_cdf.svg");

   distribution_plotter<boost::math::skew_normal_distribution<> >
      skew_normal_plotter;
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(0,1,0), "{0,1,0}");
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(0,1,1), "{0,1,1}");
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(0,1,4), "{0,1,4}");
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(0,1,20), "{0,1,20}");
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(0,1,-2), "{0,1,-2}");
   skew_normal_plotter.add(boost::math::skew_normal_distribution<>(-2,0.5,-1), "{-2,0.5,-1}");
   skew_normal_plotter.plot("Skew Normal Distribution PDF", "skew_normal_pdf.svg");

   distribution_plotter<boost::math::skew_normal_distribution<> >
      skew_normal_cdf_plotter(false);
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(0,1,0), "{0,1,0}");
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(0,1,1), "{0,1,1}");
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(0,1,4), "{0,1,4}");
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(0,1,20), "{0,1,20}");
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(0,1,-2), "{0,1,-2}");
   skew_normal_cdf_plotter.add(boost::math::skew_normal_distribution<>(-2,0.5,-1), "{-2,0.5,-1}");
   skew_normal_cdf_plotter.plot("Skew Normal Distribution CDF", "skew_normal_cdf.svg");

   distribution_plotter<boost::math::triangular_distribution<> >
      triangular_plotter;
   triangular_plotter.add(boost::math::triangular_distribution<>(-1,0,1), "{-1,0,1}");
   triangular_plotter.add(boost::math::triangular_distribution<>(0,1,1), "{0,1,1}");
   triangular_plotter.add(boost::math::triangular_distribution<>(0,1,3), "{0,1,3}");
   triangular_plotter.add(boost::math::triangular_distribution<>(0,0.5,1), "{0,0.5,1}");
   triangular_plotter.add(boost::math::triangular_distribution<>(-2,0,3), "{-2,0,3}");
   triangular_plotter.plot("Triangular Distribution PDF", "triangular_pdf.svg");

   distribution_plotter<boost::math::triangular_distribution<> >
      triangular_cdf_plotter(false);
   triangular_cdf_plotter.add(boost::math::triangular_distribution<>(-1,0,1), "{-1,0,1}");
   triangular_cdf_plotter.add(boost::math::triangular_distribution<>(0,1,1), "{0,1,1}");
   triangular_cdf_plotter.add(boost::math::triangular_distribution<>(0,1,3), "{0,1,3}");
   triangular_cdf_plotter.add(boost::math::triangular_distribution<>(0,0.5,1), "{0,0.5,1}");
   triangular_cdf_plotter.add(boost::math::triangular_distribution<>(-2,0,3), "{-2,0,3}");
   triangular_cdf_plotter.plot("Triangular Distribution CDF", "triangular_cdf.svg");

   distribution_plotter<boost::math::students_t_distribution<> >
      students_t_plotter;
   students_t_plotter.add(boost::math::students_t_distribution<>(1), "v=1");
   students_t_plotter.add(boost::math::students_t_distribution<>(5), "v=5");
   students_t_plotter.add(boost::math::students_t_distribution<>(30), "v=30");
   students_t_plotter.plot("Students T Distribution PDF", "students_t_pdf.svg");

   distribution_plotter<boost::math::weibull_distribution<> >
      weibull_plotter;
   weibull_plotter.add(boost::math::weibull_distribution<>(0.75), "shape=0.75");
   weibull_plotter.add(boost::math::weibull_distribution<>(1), "shape=1");
   weibull_plotter.add(boost::math::weibull_distribution<>(5), "shape=5");
   weibull_plotter.add(boost::math::weibull_distribution<>(10), "shape=10");
   weibull_plotter.plot("Weibull Distribution PDF (scale=1)", "weibull_pdf1.svg");

   distribution_plotter<boost::math::weibull_distribution<> >
      weibull_plotter2;
   weibull_plotter2.add(boost::math::weibull_distribution<>(3, 0.5), "scale=0.5");
   weibull_plotter2.add(boost::math::weibull_distribution<>(3, 1), "scale=1");
   weibull_plotter2.add(boost::math::weibull_distribution<>(3, 2), "scale=2");
   weibull_plotter2.plot("Weibull Distribution PDF (shape=3)", "weibull_pdf2.svg");

   distribution_plotter<boost::math::uniform_distribution<> >
      uniform_plotter;
   uniform_plotter.add(boost::math::uniform_distribution<>(0, 1), "{0,1}");
   uniform_plotter.add(boost::math::uniform_distribution<>(0, 3), "{0,3}");
   uniform_plotter.add(boost::math::uniform_distribution<>(-2, 3), "{-2,3}");
   uniform_plotter.add(boost::math::uniform_distribution<>(-1, 1), "{-1,1}");
   uniform_plotter.plot("Uniform Distribution PDF", "uniform_pdf.svg");

   distribution_plotter<boost::math::uniform_distribution<> >
      uniform_cdf_plotter(false);
   uniform_cdf_plotter.add(boost::math::uniform_distribution<>(0, 1), "{0,1}");
   uniform_cdf_plotter.add(boost::math::uniform_distribution<>(0, 3), "{0,3}");
   uniform_cdf_plotter.add(boost::math::uniform_distribution<>(-2, 3), "{-2,3}");
   uniform_cdf_plotter.add(boost::math::uniform_distribution<>(-1, 1), "{-1,1}");
   uniform_cdf_plotter.plot("Uniform Distribution CDF", "uniform_cdf.svg");

   distribution_plotter<boost::math::bernoulli_distribution<> >
      bernoulli_plotter;
   bernoulli_plotter.add(boost::math::bernoulli_distribution<>(0.25), "p=0.25");
   bernoulli_plotter.add(boost::math::bernoulli_distribution<>(0.5), "p=0.5");
   bernoulli_plotter.add(boost::math::bernoulli_distribution<>(0.75), "p=0.75");
   bernoulli_plotter.plot("Bernoulli Distribution PDF", "bernoulli_pdf.svg");

   distribution_plotter<boost::math::bernoulli_distribution<> >
      bernoulli_cdf_plotter(false);
   bernoulli_cdf_plotter.add(boost::math::bernoulli_distribution<>(0.25), "p=0.25");
   bernoulli_cdf_plotter.add(boost::math::bernoulli_distribution<>(0.5), "p=0.5");
   bernoulli_cdf_plotter.add(boost::math::bernoulli_distribution<>(0.75), "p=0.75");
   bernoulli_cdf_plotter.plot("Bernoulli Distribution CDF", "bernoulli_cdf.svg");

   distribution_plotter<boost::math::binomial_distribution<> >
      binomial_plotter;
   binomial_plotter.add(boost::math::binomial_distribution<>(5, 0.5), "n=5 p=0.5");
   binomial_plotter.add(boost::math::binomial_distribution<>(20, 0.5), "n=20 p=0.5");
   binomial_plotter.add(boost::math::binomial_distribution<>(50, 0.5), "n=50 p=0.5");
   binomial_plotter.plot("Binomial Distribution PDF", "binomial_pdf_1.svg");

   distribution_plotter<boost::math::binomial_distribution<> >
      binomial_plotter2;
   binomial_plotter2.add(boost::math::binomial_distribution<>(20, 0.1), "n=20 p=0.1");
   binomial_plotter2.add(boost::math::binomial_distribution<>(20, 0.5), "n=20 p=0.5");
   binomial_plotter2.add(boost::math::binomial_distribution<>(20, 0.9), "n=20 p=0.9");
   binomial_plotter2.plot("Binomial Distribution PDF", "binomial_pdf_2.svg");

   distribution_plotter<boost::math::negative_binomial_distribution<> >
      negative_binomial_plotter;
   negative_binomial_plotter.add(boost::math::negative_binomial_distribution<>(20, 0.25), "n=20 p=0.25");
   negative_binomial_plotter.add(boost::math::negative_binomial_distribution<>(20, 0.5), "n=20 p=0.5");
   negative_binomial_plotter.add(boost::math::negative_binomial_distribution<>(20, 0.75), "n=20 p=0.75");
   negative_binomial_plotter.plot("Negative Binomial Distribution PDF", "negative_binomial_pdf_1.svg");

   distribution_plotter<boost::math::negative_binomial_distribution<> >
      negative_binomial_plotter2;
   negative_binomial_plotter2.add(boost::math::negative_binomial_distribution<>(10, 0.5), "n=10 p=0.5");
   negative_binomial_plotter2.add(boost::math::negative_binomial_distribution<>(20, 0.5), "n=20 p=0.5");
   negative_binomial_plotter2.add(boost::math::negative_binomial_distribution<>(70, 0.5), "n=70 p=0.5");
   negative_binomial_plotter2.plot("Negative Binomial Distribution PDF", "negative_binomial_pdf_2.svg");

   distribution_plotter<boost::math::poisson_distribution<> >
      poisson_plotter;
   poisson_plotter.add(boost::math::poisson_distribution<>(5), "&#x3BB;=5");
   poisson_plotter.add(boost::math::poisson_distribution<>(10), "&#x3BB;=10");
   poisson_plotter.add(boost::math::poisson_distribution<>(20), "&#x3BB;=20");
   poisson_plotter.plot("Poisson Distribution PDF", "poisson_pdf_1.svg");

   distribution_plotter<boost::math::hypergeometric_distribution<> >
      hypergeometric_plotter;
   hypergeometric_plotter.add(boost::math::hypergeometric_distribution<>(30, 50, 500), "N=500, r=50, n=30");
   hypergeometric_plotter.add(boost::math::hypergeometric_distribution<>(30, 100, 500), "N=500, r=100, n=30");
   hypergeometric_plotter.add(boost::math::hypergeometric_distribution<>(30, 250, 500), "N=500, r=250, n=30");
   hypergeometric_plotter.add(boost::math::hypergeometric_distribution<>(30, 400, 500), "N=500, r=400, n=30");
   hypergeometric_plotter.add(boost::math::hypergeometric_distribution<>(30, 450, 500), "N=500, r=450, n=30");
   hypergeometric_plotter.plot("Hypergeometric Distribution PDF", "hypergeometric_pdf_1.svg");

   distribution_plotter<boost::math::hypergeometric_distribution<> >
      hypergeometric_plotter2;
   hypergeometric_plotter2.add(boost::math::hypergeometric_distribution<>(50, 50, 500), "N=500, r=50, n=50");
   hypergeometric_plotter2.add(boost::math::hypergeometric_distribution<>(100, 50, 500), "N=500, r=50, n=100");
   hypergeometric_plotter2.add(boost::math::hypergeometric_distribution<>(250, 50, 500), "N=500, r=50, n=250");
   hypergeometric_plotter2.add(boost::math::hypergeometric_distribution<>(400, 50, 500), "N=500, r=50, n=400");
   hypergeometric_plotter2.add(boost::math::hypergeometric_distribution<>(450, 50, 500), "N=500, r=50, n=450");
   hypergeometric_plotter2.plot("Hypergeometric Distribution PDF", "hypergeometric_pdf_2.svg");

  }
  catch (std::exception ex)
  {
    std::cout << ex.what() << std::endl;
  }



   /* these graphs for hyperexponential distribution not used.

   distribution_plotter<boost::math::hyperexponential_distribution<> >
      hyperexponential_plotter;
   {
       const double probs1_1[] = {1.0};
       const double rates1_1[] = {1.0};
       hyperexponential_plotter.add(boost::math::hyperexponential_distribution<>(probs1_1,rates1_1), "&#x3B1=(1.0), &#x3BB=(1.0)");
       const double probs2_1[] = {0.1,0.9};
       const double rates2_1[] = {0.5,1.5};
       hyperexponential_plotter.add(boost::math::hyperexponential_distribution<>(probs2_1,rates2_1), "&#x3B1=(0.1,0.9), &#x3BB=(0.5,1.5)");
       const double probs2_2[] = {0.9,0.1};
       const double rates2_2[] = {0.5,1.5};
       hyperexponential_plotter.add(boost::math::hyperexponential_distribution<>(probs2_2,rates2_2), "&#x3B1=(0.9,0.1), &#x3BB=(0.5,1.5)");
       const double probs3_1[] = {0.2,0.3,0.5};
       const double rates3_1[] = {0.5,1.0,1.5};
       hyperexponential_plotter.add(boost::math::hyperexponential_distribution<>(probs3_1,rates3_1), "&#x3B1=(0.2,0.3,0.5), &#x3BB=(0.5,1.0,1.5)");
       const double probs3_2[] = {0.5,0.3,0.2};
       const double rates3_2[] = {0.5,1.0,1.5};
       hyperexponential_plotter.add(boost::math::hyperexponential_distribution<>(probs3_1,rates3_1), "&#x3B1=(0.5,0.3,0.2), &#x3BB=(0.5,1.0,1.5)");
   }
   hyperexponential_plotter.plot("Hyperexponential Distribution PDF", "hyperexponential_pdf.svg");

   distribution_plotter<boost::math::hyperexponential_distribution<> >
      hyperexponential_plotter2;
   {
       const double rates[] = {0.5,1.5};
       const double probs1[] = {0.1,0.9};
       hyperexponential_plotter2.add(boost::math::hyperexponential_distribution<>(probs1,rates), "&#x3B1=(0.1,0.9), &#x3BB=(0.5,1.5)");
       const double probs2[] = {0.6,0.4};
       hyperexponential_plotter2.add(boost::math::hyperexponential_distribution<>(probs2,rates), "&#x3B1=(0.6,0.4), &#x3BB=(0.5,1.5)");
       const double probs3[] = {0.9,0.1};
       hyperexponential_plotter2.add(boost::math::hyperexponential_distribution<>(probs3,rates), "&#x3B1=(0.9,0.1), &#x3BB=(0.5,1.5)");
   }
   hyperexponential_plotter2.plot("Hyperexponential Distribution PDF (Different Probabilities, Same Rates)", "hyperexponential_pdf_samerate.svg");

   distribution_plotter<boost::math::hyperexponential_distribution<> >
      hyperexponential_plotter3;
   {
       const double probs1[] = {1.0};
       const double rates1[] = {2.0};
       hyperexponential_plotter3.add(boost::math::hyperexponential_distribution<>(probs1,rates1), "&#x3B1=(1.0), &#x3BB=(2.0)");
       const double probs2[] = {0.5,0.5};
       const double rates2[] = {0.3,1.5};
       hyperexponential_plotter3.add(boost::math::hyperexponential_distribution<>(probs2,rates2), "&#x3B1=(0.5,0.5), &#x3BB=(0.3,1.5)");
       const double probs3[] = {1.0/3.0,1.0/3.0,1.0/3.0};
       const double rates3[] = {0.2,1.5,3.0};
       hyperexponential_plotter3.add(boost::math::hyperexponential_distribution<>(probs2,rates2), "&#x3B1=(1.0/3.0,1.0/3.0,1.0/3.0), &#x3BB=(0.2,1.5,3.0)");
   }
   hyperexponential_plotter3.plot("Hyperexponential Distribution PDF (Different Number of Phases, Same Mean)", "hyperexponential_pdf_samemean.svg");
   */

} // int main()
