//===-- Regex.cpp - Regular Expression matcher implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a POSIX regular expression matcher.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Regex.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "regex_impl.h"
#include <string>
using namespace llvm;

Regex::Regex(StringRef regex, unsigned Flags) {
  unsigned flags = 0;
  preg = new llvm_regex();
  preg->re_endp = regex.end();
  if (Flags & IgnoreCase) 
    flags |= REG_ICASE;
  if (Flags & Newline)
    flags |= REG_NEWLINE;
  error = llvm_regcomp(preg, regex.data(), flags|REG_EXTENDED|REG_PEND);
}

Regex::~Regex() {
  llvm_regfree(preg);
  delete preg;
}

bool Regex::isValid(std::string &Error) {
  if (!error)
    return true;
  
  size_t len = llvm_regerror(error, preg, NULL, 0);
  
  Error.resize(len);
  llvm_regerror(error, preg, &Error[0], len);
  return false;
}

/// getNumMatches - In a valid regex, return the number of parenthesized
/// matches it contains.
unsigned Regex::getNumMatches() const {
  return preg->re_nsub;
}

bool Regex::match(StringRef String, SmallVectorImpl<StringRef> *Matches){
  unsigned nmatch = Matches ? preg->re_nsub+1 : 0;

  // pmatch needs to have at least one element.
  SmallVector<llvm_regmatch_t, 8> pm;
  pm.resize(nmatch > 0 ? nmatch : 1);
  pm[0].rm_so = 0;
  pm[0].rm_eo = String.size();

  int rc = llvm_regexec(preg, String.data(), nmatch, pm.data(), REG_STARTEND);

  if (rc == REG_NOMATCH)
    return false;
  if (rc != 0) {
    // regexec can fail due to invalid pattern or running out of memory.
    error = rc;
    return false;
  }

  // There was a match.

  if (Matches) { // match position requested
    Matches->clear();
    
    for (unsigned i = 0; i != nmatch; ++i) {
      if (pm[i].rm_so == -1) {
        // this group didn't match
        Matches->push_back(StringRef());
        continue;
      }
      assert(pm[i].rm_eo >= pm[i].rm_so);
      Matches->push_back(StringRef(String.data()+pm[i].rm_so,
                                   pm[i].rm_eo-pm[i].rm_so));
    }
  }

  return true;
}

std::string Regex::sub(StringRef Repl, StringRef String,
                       std::string *Error) {
  SmallVector<StringRef, 8> Matches;

  // Reset error, if given.
  if (Error && !Error->empty()) *Error = "";

  // Return the input if there was no match.
  if (!match(String, &Matches))
    return String;

  // Otherwise splice in the replacement string, starting with the prefix before
  // the match.
  std::string Res(String.begin(), Matches[0].begin());

  // Then the replacement string, honoring possible substitutions.
  while (!Repl.empty()) {
    // Skip to the next escape.
    std::pair<StringRef, StringRef> Split = Repl.split('\\');

    // Add the skipped substring.
    Res += Split.first;

    // Check for terminimation and trailing backslash.
    if (Split.second.empty()) {
      if (Repl.size() != Split.first.size() &&
          Error && Error->empty())
        *Error = "replacement string contained trailing backslash";
      break;
    }

    // Otherwise update the replacement string and interpret escapes.
    Repl = Split.second;

    // FIXME: We should have a StringExtras function for mapping C99 escapes.
    switch (Repl[0]) {
      // Treat all unrecognized characters as self-quoting.
    default:
      Res += Repl[0];
      Repl = Repl.substr(1);
      break;

      // Single character escapes.
    case 't':
      Res += '\t';
      Repl = Repl.substr(1);
      break;
    case 'n':
      Res += '\n';
      Repl = Repl.substr(1);
      break;

      // Decimal escapes are backreferences.
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9': {
      // Extract the backreference number.
      StringRef Ref = Repl.slice(0, Repl.find_first_not_of("0123456789"));
      Repl = Repl.substr(Ref.size());

      unsigned RefValue;
      if (!Ref.getAsInteger(10, RefValue) &&
          RefValue < Matches.size())
        Res += Matches[RefValue];
      else if (Error && Error->empty())
        *Error = "invalid backreference string '" + Ref.str() + "'";
      break;
    }
    }
  }

  // And finally the suffix.
  Res += StringRef(Matches[0].end(), String.end() - Matches[0].end());

  return Res;
}
