//  Copyright (c) 2014 Anton Bikineev
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Appends negative test cases to the *.ipp files.
//  Takes the next parameters:
//  -f <file> file where the negative values will be appended;
//  -x add minus to existing x values and append result;
//  -v, -xv like previous option.
//  Usage example:
//  ./bessel_derivative_append_negative -f "bessel_y_derivative_large_data.ipp" -x -v -xv

#include <fstream>
#include <utility>
#include <functional>
#include <map>
#include <vector>
#include <iterator>
#include <algorithm>

#include <boost/multiprecision/mpfr.hpp>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/math/special_functions/bessel.hpp>

template <class T>
T bessel_j_derivative_bare(T v, T x)
{
   return (v / x) * boost::math::cyl_bessel_j(v, x) - boost::math::cyl_bessel_j(v+1, x);
}

template <class T>
T bessel_y_derivative_bare(T v, T x)
{
   return (v / x) * boost::math::cyl_neumann(v, x) - boost::math::cyl_neumann(v+1, x);
}

template <class T>
T bessel_i_derivative_bare(T v, T x)
{
   return (v / x) * boost::math::cyl_bessel_i(v, x) + boost::math::cyl_bessel_i(v+1, x);
}

template <class T>
T bessel_k_derivative_bare(T v, T x)
{
   return (v / x) * boost::math::cyl_bessel_k(v, x) - boost::math::cyl_bessel_k(v+1, x);
}

template <class T>
T sph_bessel_j_derivative_bare(T v, T x)
{
   if((v < 0) || (floor(v) != v))
      throw std::domain_error("");
   if(v == 0)
      return -boost::math::sph_bessel(1, x);
   return boost::math::sph_bessel(itrunc(v-1), x) - ((v + 1) / x) * boost::math::sph_bessel(itrunc(v), x);
}

template <class T>
T sph_bessel_y_derivative_bare(T v, T x)
{
   if((v < 0) || (floor(v) != v))
      throw std::domain_error("");
   if(v == 0)
      return -boost::math::sph_neumann(1, x);
   return boost::math::sph_neumann(itrunc(v-1), x) - ((v + 1) / x) * boost::math::sph_neumann(itrunc(v), x);
}

namespace opt = boost::program_options;
using FloatType = boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<200u> >;
using Function = FloatType(*)(FloatType, FloatType);
using Lines = std::vector<std::string>;

enum class Negate: char
{
   x,
   v,
   xv
};

namespace
{

const unsigned kSignificand = 50u;

std::map<std::string, Function> kFileMapper = {
   {"bessel_j_derivative_data.ipp", ::bessel_j_derivative_bare},
   {"bessel_j_derivative_int_data.ipp", ::bessel_j_derivative_bare},
   {"bessel_j_derivative_large_data.ipp", ::bessel_j_derivative_bare},
   {"bessel_y01_derivative_data.ipp", ::bessel_y_derivative_bare},
   {"bessel_yn_derivative_data.ipp", ::bessel_y_derivative_bare},
   {"bessel_yv_derivative_data.ipp", ::bessel_y_derivative_bare},
   {"bessel_i_derivative_data.ipp", ::bessel_i_derivative_bare},
   {"bessel_i_derivative_int_data.ipp", ::bessel_i_derivative_bare},
   {"bessel_k_derivative_data.ipp", ::bessel_k_derivative_bare},
   {"bessel_k_derivative_int_data.ipp", ::bessel_k_derivative_bare},
   {"sph_bessel_derivative_data.ipp", ::sph_bessel_j_derivative_bare},
   {"sph_neumann_derivative_data.ipp", ::sph_bessel_y_derivative_bare}
};

Function fp = ::bessel_j_derivative_bare;

Lines getSourcePartOfFile(std::fstream& file)
{
   file.seekg(std::ios::beg);

   Lines lines;
   while (true)
   {
      auto line = std::string{};
      std::getline(file, line);
      if (line.find("}};") != std::string::npos)
         break;
      lines.push_back(line);
   }
   file.seekg(std::ios::beg);
   return lines;
}

std::pair<std::string, std::string::iterator> parseValue(std::string::iterator& iter)
{
   using std::isdigit;

   auto value = std::string{};
   auto iterator = std::string::iterator{};

   while (!isdigit(*iter) && *iter != '-')
      ++iter;
   iterator = iter;
   while (isdigit(*iter) || *iter == '.' || *iter == 'e' || *iter == '-' || *iter == '+')
   {
      value.push_back(*iter);
      ++iter;
   }
   return {value, iterator};
}

void addMinusToValue(std::string& line, Negate which)
{
   using std::isdigit;

   auto iter = line.begin();
   switch (which)
   {
      case Negate::x:
      {
         ::parseValue(iter);
         auto value_begin = ::parseValue(iter).second;
         if (*value_begin != '-')
            line.insert(value_begin, '-');
         break;
      }
      case Negate::v:
      {
         auto value_begin = ::parseValue(iter).second;
         if (*value_begin != '-')
            line.insert(value_begin, '-');
         break;
      }
      case Negate::xv:
      {
         auto v_value_begin = ::parseValue(iter).second;
         if (*v_value_begin != '-')
            line.insert(v_value_begin, '-');
         // iterator could get invalid
         iter = line.begin();
         ::parseValue(iter);
         auto x_value_begin = ::parseValue(iter).second;
         if (*x_value_begin != '-')
            line.insert(x_value_begin, '-');
         break;
      }
   }
}

void replaceResultInLine(std::string& line)
{
   using std::isdigit;

   auto iter = line.begin();

   // parse v and x values from line and convert them to FloatType
   auto v = FloatType{::parseValue(iter).first};
   auto x = FloatType{::parseValue(iter).first};
   auto result = fp(v, x).str(kSignificand);

   while (!isdigit(*iter) && *iter != '-')
      ++iter;
   const auto where_to_write = iter;
   while (isdigit(*iter) || *iter == '.' || *iter == 'e' || *iter == '-' || *iter == '+')
      line.erase(iter);

   line.insert(where_to_write, result.begin(), result.end());
}

Lines processValues(const Lines& source_lines, Negate which)
{
   using std::placeholders::_1;

   auto processed_lines = source_lines;
   std::for_each(std::begin(processed_lines), std::end(processed_lines), std::bind(&addMinusToValue, _1, which));
   std::for_each(std::begin(processed_lines), std::end(processed_lines), &replaceResultInLine);

   return processed_lines;
}

void updateTestCount(Lines& source_lines, std::size_t mult)
{
   using std::isdigit;

   const auto where = std::find_if(std::begin(source_lines), std::end(source_lines),
      [](const std::string& str){ return str.find("std::array") != std::string::npos; });
   auto& str = *where;
   const auto pos = str.find(">, ") + 3;
   auto digits_length = 0;

   auto k = pos;
   while (isdigit(str[k++]))
      ++digits_length;

   const auto new_value = mult * boost::lexical_cast<std::size_t>(str.substr(pos, digits_length));
   str.replace(pos, digits_length, boost::lexical_cast<std::string>(new_value));
}

} // namespace

int main(int argc, char*argv [])
{
   auto desc = opt::options_description{"All options"};
   desc.add_options()
      ("help", "produce help message")
      ("file", opt::value<std::string>()->default_value("bessel_j_derivative_data.ipp"))
      ("x", "append negative x")
      ("v", "append negative v")
      ("xv", "append negative x and v");
   opt::variables_map vm;
   opt::store(opt::command_line_parser(argc, argv).options(desc)
         .style(opt::command_line_style::default_style |
         opt::command_line_style::allow_long_disguise)
      .run(),vm);
   opt::notify(vm);

   if (vm.count("help"))
   {
      std::cout << desc;
      return 0;
   }

   auto filename = vm["file"].as<std::string>();
   fp = kFileMapper[filename];

   std::fstream file{filename.c_str()};
   if (!file.is_open())
      return -1;
   auto source_part = ::getSourcePartOfFile(file);
   source_part.back().push_back(',');

   auto cases_lines = Lines{};
   for (const auto& str: source_part)
   {
      if (str.find("SC_") != std::string::npos)
         cases_lines.push_back(str);
   }

   auto new_lines = Lines{};
   new_lines.reserve(cases_lines.size());

   std::size_t mult = 1;
   if (vm.count("x"))
   {
      std::cout << "process x..." << std::endl;
      const auto x_lines = ::processValues(cases_lines, Negate::x);
      new_lines.insert(std::end(new_lines), std::begin(x_lines), std::end(x_lines));
      ++mult;
   }
   if (vm.count("v"))
   {
      std::cout << "process v..." << std::endl;
      const auto v_lines = ::processValues(cases_lines, Negate::v);
      new_lines.insert(std::end(new_lines), std::begin(v_lines), std::end(v_lines));
      ++mult;
   }
   if (vm.count("xv"))
   {
      std::cout << "process xv..." << std::endl;
      const auto xv_lines = ::processValues(cases_lines, Negate::xv);
      new_lines.insert(std::end(new_lines), std::begin(xv_lines), std::end(xv_lines));
      ++mult;
   }

   source_part.insert(std::end(source_part), std::begin(new_lines), std::end(new_lines));
   ::updateTestCount(source_part, mult);

   file.close();
   file.open(filename, std::ios::out | std::ios::trunc);
   std::for_each(std::begin(source_part), std::end(source_part), [&file](const std::string& str)
      { file << str << std::endl; });
   file << "   }};";

   std::cout << "processed, ok\n";
   return 0;
}
