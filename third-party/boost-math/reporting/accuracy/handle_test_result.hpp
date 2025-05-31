//  (C) Copyright John Maddock 2006-7.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_HANDLE_TEST_RESULT
#define BOOST_MATH_HANDLE_TEST_RESULT

#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>

#include <boost/math/tools/test.hpp>

inline std::string sanitize_string(const std::string& s)
{
   static const boost::regex e("[^a-zA-Z0-9]+");
   return boost::regex_replace(s, e, "_");
}

static std::string content;
boost::filesystem::path path_to_content;

struct content_loader
{
   boost::interprocess::named_mutex mu;
   boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock;
   content_loader() : mu(boost::interprocess::open_or_create, "handle_test_result"), lock(mu)
   {
      boost::filesystem::path p(__FILE__);
      p = p.parent_path();
      p /= "doc";
      p /= "accuracy_tables.qbk";
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
   }
   ~content_loader()
   {
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

void add_to_all_sections(const std::string& id, std::string list_name = "all_sections")
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
         "([^\\[\\]]*(?0)?)*"
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

void add_cell(const std::string& cell_name, const std::string& table_name, const std::string& row_name, const std::string& type_name)
{
   //
   // Load the table, add our data, and re-write:
   //
   std::string table_id = "table_" + sanitize_string(table_name);
   std::string column_heading = BOOST_COMPILER;
   column_heading += "[br]";
   column_heading += BOOST_PLATFORM;
   column_heading += "[br]";
   column_heading += type_name;
   boost::regex table_e("\\[table:" + table_id
      + "\\s[^\\[]+"
         "((\\["
            "([^\\[\\]]*(?2)?)*"
         "\\]\\s*)*\\s*)"
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
      // Update the entry:
      //
      std::string& s = table_data[row_id][column_id];
      if(s.empty())
      {
         std::cout << "Adding " << cell_name << " to empty cell.";
         s = "[" + cell_name + "]";
      }
      else
      {
         if(cell_name.find("_boost_") != std::string::npos)
         {
            std::cout << "Adding " << cell_name << " to start of cell.";
            s.insert(0, "[" + cell_name + "][br][br]");
         }
         else
         {
            std::cout << "Adding " << cell_name << " to end of cell.";
            if((s.find("_boost_") != std::string::npos) && (s.find("[br]") == std::string::npos))
               s += "[br]"; // extra break if we're adding directly after the boost results.
            s += "[br][" + cell_name + "]";
         }
      }
      //
      // Convert back to a string and insert into content:
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
      new_table += " Error rates for ";
      new_table += table_name;
      new_table += "\n[[][";
      new_table += column_heading;
      new_table += "]]\n";
      new_table += "[[";
      new_table += row_name;
      new_table += "][[";
      new_table += cell_name;
      new_table += "]]]\n]\n]\n";

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
      add_to_all_sections(table_id, "all_tables");
   }
}

void set_result(const std::string& cell_name, const std::string& cell_content, const std::string& table_name, const std::string& row_name, const std::string& type_name)
{
   loader.instantiate();
   const boost::regex e("\\[template\\s+" + cell_name + 
      "\\[\\]([^\\n]*)\\]$");

   boost::smatch what;
   if(regex_search(content, what, e))
   {
      content.replace(what.position(1), what.length(1), cell_content);
   }
   else
   {
      // Need to add new content:
      std::string::size_type pos = content.find("[/Cell Content:]");
      std::string t = "\n[template " + cell_name + "[] " + cell_content + "]";
      if(pos != std::string::npos)
         content.insert(pos + 16, t);
      else
      {
         content.insert(0, t);
         content.insert(0, "[/Cell Content:]");
      }
   }
   //
   // Check to verify that our content is actually used somewhere,
   // if not we need to create a place for it:
   //
   if(content.find("[" + cell_name + "]") == std::string::npos)
      add_cell(cell_name, table_name, row_name, type_name);
}

void set_error_content(const std::string& id, const std::string& error_s)
{
   boost::regex content_e("\\[template\\s+" + id + 
      "\\[\\]\\s+"
      "("
         "[^\\]\\[]*"
         "(?:"
            "\\["
            "([^\\[\\]]*(?2)?)*"
            "\\]"
            "[^\\]\\[]*"
         ")*"
         
       ")"
       "\\]");
   boost::smatch what;
   if(regex_search(content, what, content_e))
   {
      // replace existing content:
      content.replace(what.position(1), what.length(1), error_s);
   }
   else
   {
      // add new content:
      std::string::size_type pos = content.find("[/error_content:]");
      if(pos != std::string::npos)
      {
         content.insert(pos + 17, "\n[template " + id + "[]\n" + error_s + "\n]\n");
      }
      else
         content.append("\n[/error_content:]\n[template " + id + "[]\n" + error_s + "\n]\n");
   }
   //
   // Add to all_errors if not already there:
   //
   if(content.find("[" + id + "]") == std::string::npos)
   {
      // Find all_errors template:
      std::string::size_type pos = content.find("[template all_errors[]\n");
      if(pos != std::string::npos)
      {
         content.insert(pos + 23, "[" + id + "]\n");
      }
      else
      {
         content.append("\n[template all_errors[]\n[").append(id).append("]\n]\n");
      }
   }
}

void remove_error_content(const std::string& error_id)
{
   // remove use template first:
   std::string::size_type pos = content.find("[" + error_id + "]");
   if(pos != std::string::npos)
   {
      content.erase(pos, 2 + error_id.size());
   }
   // then the template define itself:
   boost::regex content_e("\\[template\\s+" + error_id +
      "\\[\\]\\s+"
      "("
         "[^\\]\\[]*"
         "(?:"
            "\\["
            "([^\\[\\]]*(?2)?)*"
            "\\]"
            "[^\\]\\[]*"
         ")*"
      ")"
      "\\]");
   boost::smatch what;
   if(regex_search(content, what, content_e))
   {
      content.erase(what.position(), what.length());
   }
}

template <class T, class Seq>
void handle_test_result(const boost::math::tools::test_result<T>& result,
                       const Seq& worst, int row, 
                       const char* type_name, 
                       const char* test_name, 
                       const char* group_name)
{
   T eps = boost::math::tools::epsilon<T>();
   T max_error_found = (result.max)() / eps;
   T mean_error_found = result.rms() / eps;

   std::string cell_name = sanitize_string(BOOST_COMPILER) + "_" + sanitize_string(BOOST_PLATFORM) + "_" + sanitize_string(type_name) 
      + "_" + sanitize_string(test_name) + "_" + sanitize_string(TEST_LIBRARY_NAME) + "_" + sanitize_string(group_name);

   std::stringstream ss;
   ss << std::setprecision(3);
   if(std::string(TEST_LIBRARY_NAME) != "boost")
      ss << "(['" << TEST_LIBRARY_NAME << ":] ";
   else
      ss << "[role blue ";

   if((result.max)() > std::sqrt(eps))
      ss << "[role red ";


   ss << "Max = ";
   if((boost::math::isfinite)(max_error_found))
      ss << max_error_found;
   else
      ss << "+INF";
   ss << "[epsilon] (Mean = ";
   if((boost::math::isfinite)(mean_error_found))
      ss << mean_error_found;
   else
      ss << "+INF";
   ss << "[epsilon])";

   //
   // Now check for error output from gross errors or unexpected exceptions:
   //
   std::stringbuf* pbuf = dynamic_cast<std::stringbuf*>(std::cerr.rdbuf());
   bool have_errors = false;
   std::string error_id = "errors_" + cell_name;
   if(pbuf)
   {
      std::string err_s = pbuf->str();
      if(err_s.size())
      {
         if(err_s.size() > 4096)
         {
            std::string::size_type pos = err_s.find("\n", 4096);
            if(pos != std::string::npos)
            {
               err_s.erase(pos);
               err_s += "\n*** FURTHER CONTENT HAS BEEN TRUNCATED FOR BREVITY ***\n";
            }
         }
         std::string::size_type pos = err_s.find("\n");
         while(pos != std::string::npos)
         {
            err_s.replace(pos, 1, "[br]");
            pos = err_s.find("\n");
         }
         err_s = "[h4 Error Output For " + std::string(test_name) + std::string(" with compiler ") + std::string(BOOST_COMPILER)
            + std::string(" and library ") + std::string(TEST_LIBRARY_NAME) + " and test data "
            + std::string(group_name) + "]\n\n[#" + error_id + "]\n" + err_s + std::string("\n\n\n");
         ss << "  [link " << error_id << " And other failures.]";
         pbuf->str("");
         set_error_content(error_id, err_s);
         have_errors = true;
      }
   }
   if(!have_errors)
      remove_error_content(error_id);


   if(std::string(TEST_LIBRARY_NAME) != "boost")
      ss << ")";
   else
      ss << "]";

   if((result.max)() > std::sqrt(eps))
      ss << "]";

   std::string cell_content = ss.str();

   set_result(cell_name, cell_content, test_name, group_name, type_name);
}

struct error_stream_replacer
{
   std::streambuf* old_buf;
   std::stringstream ss;
   error_stream_replacer()
   {
      old_buf = std::cerr.rdbuf();
      std::cerr.rdbuf(ss.rdbuf());
   }
   ~error_stream_replacer()
   {
      std::cerr.rdbuf(old_buf);
   }
};

#endif // BOOST_MATH_HANDLE_TEST_RESULT

