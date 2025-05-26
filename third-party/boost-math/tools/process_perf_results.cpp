//  Copyright John Maddock 2007.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <map>
#include <fstream>
#include <iostream>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/math/special_functions.hpp>

std::map<std::string, double> results;

std::map<std::string, std::string> extra_text;

void load_file(std::string& s, std::istream& is)
{
   s.erase();
   if(is.bad()) return;
   s.reserve(is.rdbuf()->in_avail());
   char c;
   while(is.get(c))
   {
      if(s.capacity() == s.size())
         s.reserve(s.capacity() * 3);
      s.append(1, c);
   }
}

int main(int argc, const char* argv[])
{
   //
   // Set any additional text that should accompany specific results:
   //
   extra_text["msvc-dist-beta-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   extra_text["msvc-dist-nbinom-R-quantile"] = "[footnote The R library appears to use a linear-search strategy, that can perform very badly in a small number of pathological cases, but may or may not be more efficient in \"typical\" cases]";
   extra_text["gcc-4_3_2-dist-beta-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   extra_text["gcc-4_3_2-dist-nbinom-R-quantile"] = "[footnote The R library appears to use a linear-search strategy, that can perform very badly in a small number of pathological cases, but may or may not be more efficient in \"typical\" cases]";
   extra_text["msvc-dist-hypergeometric-cdf"] = "[footnote This result is somewhat misleading: for small values of the parameters there is  virtually no difference between the two libraries, but for large values the Boost implementation is /much/ slower, albeit with much improved precision.]";
   extra_text["msvc-dist-nt-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   extra_text["msvc-dist-nchisq-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   extra_text["gcc-4_3_2-dist-hypergeometric-cdf"] = "[footnote This result is somewhat misleading: for small values of the parameters there is  virtually no difference between the two libraries, but for large values the Boost implementation is /much/ slower, albeit with much improved precision.]";
   extra_text["gcc-4_3_2-dist-nt-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   extra_text["gcc-4_3_2-dist-nchisq-R-quantile"] = "[footnote There are a small number of our test cases where the R library fails to converge on a result: these tend to dominate the performance result.]";
   
   boost::regex e("^Testing\\s+(\\S+)\\s+(\\S+)");
   std::string f;
   for(int i = 1; i < argc-1; ++i)
   {
      std::ifstream is(argv[i]);
      load_file(f, is);
      boost::sregex_iterator a(f.begin(), f.end(), e), b;
      while(a != b)
      {
         results[(*a).str(1)] = boost::lexical_cast<double>((*a).str(2));
         ++a;
      }
   }
   //
   // Load quickbook file:
   //
   std::ifstream is(argv[argc-1]);
   std::string bak_file = std::string(argv[argc-1]).append(".bak");
   std::ofstream os(bak_file.c_str());
   e.assign(
      "\\[perf\\s+([^\\s.]+)"
      "(?:"
         "\\[[^\\]\\[]*"
            "(?:\\[[^\\]\\[]*\\][^\\]\\[]*)?"
         "\\]"
         "|[^\\]]"
      ")*\\]");
   std::string newfile;
   while(is.good())
   {
      std::getline(is, f);
      os << f << std::endl;
      boost::sregex_iterator i(f.begin(), f.end(), e), j;
      double min = (std::numeric_limits<double>::max)();
      while(i != j)
      {
         std::cout << (*i).str() << std::endl << (*i).str(1) << std::endl;
         std::string item = (*i).str(1);
         if(results.find(item) != results.end())
         {
            double r = results[item];
            if(r < min)
               min = r;
         }
         ++i;
      }
      //
      // Now perform the substitutions:
      //
      std::string newstr;
      std::string tail;
      i = boost::sregex_iterator(f.begin(), f.end(), e);
      while(i != j)
      {
         std::string item = (*i).str(1);
         newstr.append(i->prefix());
         if(results.find(item) != results.end())
         {
            double v = results[item];
            double r = v / min;
            newstr += std::string((*i)[0].first, (*i)[1].second);
            newstr += "..[para ";
            if(r < 1.01)
               newstr += "*";
            newstr += (boost::format("%.2f") % r).str();
            if(r < 1.01)
               newstr += "*";
            if(extra_text.find(item) != extra_text.end())
            {
               newstr += extra_text[item];
            }
            newstr += "][para (";
            newstr += (boost::format("%.3e") % results[item]).str();
            newstr += "s)]]";
         }
         else
         {
            newstr.append(i->str());
            std::cerr << "Item " << item << " not found!!" << std::endl;
         }
         tail = i->suffix();
         ++i;
      }
      if(newstr.size())
         newfile.append(newstr).append(tail);
      else
         newfile.append(f);
      newfile.append("\n");
   }
   is.close();
   std::ofstream ns(argv[argc-1]);
   ns << newfile;
}

