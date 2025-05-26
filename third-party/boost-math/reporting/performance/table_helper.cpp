//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <vector>
#include <set>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "table_helper.hpp"

void add_cell(std::intmax_t val, const std::string& table_name, const std::string& row_name, const std::string& column_heading);
void add_to_all_sections(const std::string& id, std::string list_name = "performance_all_sections");

std::vector<std::vector<double> > data;
std::vector<std::tuple<double, std::string, std::string, std::string> > items_to_add;

inline std::string sanitize_string(const std::string& s)
{
   static const boost::regex e("[^a-zA-Z0-9]+");
   std::string result = boost::regex_replace(s, e, "_");
   while(result[0] == '_')
      result.erase(0);
   return result;
}

std::string format_precision(double val, int digits)
{
   std::stringstream ss;
   ss << std::setprecision(digits);
   ss << std::fixed;
   ss << val;
   return ss.str();
}

static std::string content;
boost::filesystem::path path_to_content;

struct content_loader
{
   content_loader(){}
   ~content_loader()
   {
      boost::interprocess::named_mutex mu(boost::interprocess::open_or_create, "handle_test_result");
      boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(mu);
      boost::filesystem::path p(__FILE__);
      p = p.parent_path();
      p /= "doc";
      p /= "performance_tables.qbk";
      path_to_content = p;
      if(boost::filesystem::exists(p))
      {
         boost::filesystem::ifstream is(p);
         if(is.good())
         {
            do
            {
               char c = static_cast<char>(is.get());
               if(c != EOF)
                  content.append(1, c);
            } while(is.good());
         }
      }
      //
      // Now iterate through results and add them one at a time:
      //
      for(auto i = items_to_add.begin(); i != items_to_add.end(); ++i)
      {
         add_cell(static_cast<std::uintmax_t>(std::get<0>(*i) / 1e-9), std::get<1>(*i), std::get<2>(*i), std::get<3>(*i));
      }
      //
      // Write out the results:
      //
      boost::filesystem::ofstream os(path_to_content);
      os << content;
   }
   void instantiate()const
   {
   }
};

static const content_loader loader;

void load_table(std::vector<std::vector<std::string> >& table, std::string::const_iterator begin, std::string::const_iterator end)
{
   static const boost::regex item_e(
      "\\["
      "([^\\[\\]]*(?0)?)*"
      "\\]"
      );

   boost::regex_token_iterator<std::string::const_iterator> i(begin, end, item_e), j;

   while(i != j)
   {
      // Add a row:
      table.push_back(std::vector<std::string>());
      boost::regex_token_iterator<std::string::const_iterator> k(i->first + 1, i->second - 1, item_e);
      while(k != j)
      {
         // Add a cell:
         table.back().push_back(std::string(k->first + 1, k->second - 1));
         ++k;
      }
      ++i;
   }
}

std::string save_table(std::vector<std::vector<std::string> >& table)
{
   std::string result;

   for(std::vector<std::vector<std::string> >::const_iterator i = table.begin(), j = table.end(); i != j; ++i)
   {
      result += "[";
      for(std::vector<std::string>::const_iterator k = i->begin(), l = i->end(); k != l; ++k)
      {
         result += "[";
         result += *k;
         result += "]";
      }
      result += "]\n";
   }
   return result;
}

void add_to_all_sections(const std::string& id, std::string list_name)
{
   std::string::size_type pos = content.find("[template " + list_name + "[]"), end_pos;
   if(pos == std::string::npos)
   {
      //
      // Just append to the end:
      //
      content.append("\n[template ").append(list_name).append("[]\n[").append(id).append("]\n]\n");
   }
   else
   {
      //
      // Read in the all list of sections, add our new one (in alphabetical order),
      // and then rewrite the whole thing:
      //
      static const boost::regex item_e(
         "\\["
         "((?=[^\\]])[^\\[\\]]*+(?0)?+)*+"
         "\\]|\\]"
         );
      boost::regex_token_iterator<std::string::const_iterator> i(content.begin() + pos + 12 + list_name.size(), content.end(), item_e), j;
      std::set<std::string> sections;
      while(i != j)
      {
         if(i->length() == 1)
         {
            end_pos = i->first - content.begin();
            break;
         }
         sections.insert(std::string(i->first + 1, i->second - 1));
         ++i;
      }
      sections.insert(id);
      std::string new_list = "\n";
      for(std::set<std::string>::const_iterator sec = sections.begin(); sec != sections.end(); ++sec)
      {
         new_list += "[" + *sec + "]\n";
      }
      content.replace(pos + 12 + list_name.size(), end_pos - pos - 12 - list_name.size(), new_list);
   }
}

std::string get_colour(std::uintmax_t val, std::uintmax_t best)
{
   if(val <= best * 1.2)
      return "green";
   if(val > best * 2)
      return "red";
   return "blue";
}

std::intmax_t get_value_from_cell(const std::string& cell)
{
   static const boost::regex time_e("(\\d+)ns");
   boost::smatch what;
   if(regex_search(cell, what, time_e))
   {
      return boost::lexical_cast<std::uintmax_t>(what.str(1));
   }
   return -1;
}

void add_cell(std::intmax_t val, const std::string& table_name, const std::string& row_name, const std::string& column_heading)
{
   //
   // Load the table, add our data, and re-write:
   //
   std::string table_id = "table_" + sanitize_string(table_name);
   boost::regex table_e("\\[table:" + table_id
      + "\\s[^\\[]++"
      "((\\["
      "([^\\[\\]]*+(?2)?+)*+"
      "\\]\\s*+)*+\\s*+)"
      "\\]"
      );

   boost::smatch table_location;
   if(regex_search(content, table_location, table_e))
   {
      std::vector<std::vector<std::string> > table_data;
      load_table(table_data, table_location[1].first, table_location[1].second);
      //
      // Figure out which column we're on:
      //
      unsigned column_id = 1001u;
      for(unsigned i = 0; i < table_data[0].size(); ++i)
      {
         if(table_data[0][i] == column_heading)
         {
            column_id = i;
            break;
         }
      }
      if(column_id > 1000)
      {
         //
         // Need a new column, must be adding a new compiler to the table!
         //
         table_data[0].push_back(column_heading);
         for(unsigned i = 1; i < table_data.size(); ++i)
            table_data[i].push_back(std::string());
         column_id = table_data[0].size() - 1;
      }
      //
      // Figure out the row:
      //
      unsigned row_id = 1001;
      for(unsigned i = 1; i < table_data.size(); ++i)
      {
         if(table_data[i][0] == row_name)
         {
            row_id = i;
            break;
         }
      }
      if(row_id > 1000)
      {
         //
         // Need a new row, add it now:
         //
         table_data.push_back(std::vector<std::string>());
         table_data.back().push_back(row_name);
         for(unsigned i = 1; i < table_data[0].size(); ++i)
            table_data.back().push_back(std::string());
         row_id = table_data.size() - 1;
      }
      //
      // Find the best result in this row:
      //
      std::uintmax_t best = (std::numeric_limits<std::uintmax_t>::max)();
      std::vector<std::intmax_t> values;
      for(unsigned i = 1; i < table_data[row_id].size(); ++i)
      {
         if(i == column_id)
         {
            if(val < best)
               best = val;
            values.push_back(val);
         }
         else
         {
            std::cout << "Existing cell value was " << table_data[row_id][i] << std::endl;
            std::uintmax_t cell_val = get_value_from_cell(table_data[row_id][i]);
            std::cout << "Extracted value: " << cell_val << std::endl;
            if(cell_val < best)
               best = cell_val;
            values.push_back(cell_val);
         }
      }
      //
      // Update the row:
      //
      for(unsigned i = 1; i < table_data[row_id].size(); ++i)
      {
         std::string& s = table_data[row_id][i];
         s = "[role ";
         if(values[i - 1] < 0)
         {
            s += "grey -]";
         }
         else
         {
            s += get_colour(values[i - 1], best);
            s += " ";
            s += format_precision(static_cast<double>(values[i - 1]) / best, 2);
            s += "[br](";
            s += boost::lexical_cast<std::string>(values[i - 1]) + "ns)]";
         }
      }
      //
      // Convert back to a string and insert into content:
      std::sort(table_data.begin() + 1, table_data.end(), [](std::vector<std::string> const& a, std::vector<std::string> const& b) { return a[0] < b[0]; } );
      std::string c = save_table(table_data);
      content.replace(table_location.position(1), table_location.length(1), c);
   }
   else
   {
      //
      // Create a new table and try again:
      //
      std::string new_table = "\n[template " + table_id;
      new_table += "[]\n[table:" + table_id;
      new_table += " ";
      new_table += table_name;
      new_table += "\n[[Function][";
      new_table += column_heading;
      new_table += "]]\n";
      new_table += "[[";
      new_table += row_name;
      new_table += "][[role blue 1.00[br](";
      new_table += boost::lexical_cast<std::string>(val);
      new_table += "ns)]]]\n]\n]\n";

      std::string::size_type pos = content.find("[/tables:]");
      if(pos != std::string::npos)
         content.insert(pos + 10, new_table);
      else
         content += "\n\n[/tables:]\n" + new_table;
      //
      // Add a section for this table as well:
      //
      std::string section_id = "section_" + sanitize_string(table_name);
      if(content.find(section_id + "[]") == std::string::npos)
      {
         std::string new_section = "\n[template " + section_id + "[]\n[section:" + section_id + " " + table_name + "]\n[" + table_id + "]\n[endsect]\n]\n";
         pos = content.find("[/sections:]");
         if(pos != std::string::npos)
            content.insert(pos + 12, new_section);
         else
            content += "\n\n[/sections:]\n" + new_section;
         add_to_all_sections(section_id);
      }
      //
      // Add to list of all tables (not in sections):
      //
      add_to_all_sections(table_id, "performance_all_tables");
   }
}

void report_execution_time(double t, std::string table, std::string row, std::string heading)
{
   items_to_add.push_back(std::make_tuple(t, table, row, heading));
   //add_cell(static_cast<std::uintmax_t>(t / 1e-9), table, row, heading);
}

std::string get_compiler_options_name()
{
#if defined(BOOST_MSVC) || defined(__ICL)
   std::string result;
#ifdef _MSC_VER
   result = "cl ";
#else
   result = "icl ";
#endif
#ifdef _M_AMD64
#ifdef __AVX__
   result += "/arch:AVX /Ox";
#else
   result += "/Ox";
#endif
   result += " (x64 build)";
#else
#ifdef _DEBUG
   result +=  "/Od";
#elif defined(__AVX2__)
   result += "/arch:AVX2 /Ox";
#elif defined(__AVX__)
   result += "/arch:AVX /Ox";
#elif _M_IX86_FP == 2
   result += "/arch:sse2 /Ox";
#else
   result += "/arch:ia32 /Ox";
#endif
   result += " (x86 build)";
#endif
   std::cout << "Compiler options are found as: " << result << std::endl;
   return result;
#else
   return "Unknown";
#endif
}

