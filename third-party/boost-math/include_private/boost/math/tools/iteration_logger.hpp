//  Copyright John Maddock 2014.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_ITERATION_LOGGER
#define BOOST_MATH_TOOLS_ITERATION_LOGGER

#include <map>
#include <string>
#include <utility>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <boost/current_function.hpp>

namespace boost{ namespace math{ namespace detail{

struct logger
{
   std::map<std::pair<std::string, std::string>, unsigned long long> data;

   ~logger()
   {
      for(std::map<std::pair<std::string, std::string>, unsigned long long>::const_iterator i = data.begin(); i != data.end(); ++i)
      {
         std::cout << "Logging iteration data:\n  file: " << i->first.first
            << "\n  function: " << i->first.second
            << "\n  count: " << i->second << std::endl;
         //
         // Read in existing data:
         //
         std::map<std::string, unsigned long long> file_data;
         std::string filename = i->first.first + ".logfile";
         std::ifstream is(filename.c_str());
         while(is.good())
         {
            std::string f;
            unsigned long long c;
            is >> std::ws;
            std::getline(is, f);
            is >> c;
            if(f.size())
               file_data[f] = c;
         }
         is.close();
         if(file_data.find(i->first.second) != file_data.end())
            file_data[i->first.second] += i->second;
         else
            file_data[i->first.second] = i->second;
         //
         // Write it out again:
         //
         std::ofstream os(filename.c_str());
         for(std::map<std::string, unsigned long long>::const_iterator j = file_data.begin(); j != file_data.end(); ++j)
            os << j->first << "\n    " << j->second << std::endl;
         os.close();
      }
   }
};

inline void log_iterations(const char* file, const char* function, unsigned long long count)
{
   static logger l;
   std::pair<std::string, std::string> key(file, function);
   if(l.data.find(key) == l.data.end())
      l.data[key] = 0;
   l.data[key] += count;
}

#define BOOST_MATH_LOG_COUNT(count) boost::math::detail::log_iterations(__FILE__, BOOST_CURRENT_FUNCTION, count);


}}} // namespaces

#endif // BOOST_MATH_TOOLS_ITERATION_LOGGER

