//===-- StringExtras.cpp - Implement the StringExtras header --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringExtras.h header
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include <cstring>
using namespace llvm;

/// getToken - This function extracts one token from source, ignoring any
/// leading characters that appear in the Delimiters string, and ending the
/// token at any of the characters that appear in the Delimiters string.  If
/// there are no tokens in the source string, an empty string is returned.
/// The Source source string is updated in place to remove the returned string
/// and any delimiter prefix from it.
std::string llvm::getToken(std::string &Source, const char *Delimiters) {
  size_t NumDelimiters = std::strlen(Delimiters);

  // Figure out where the token starts.
  std::string::size_type Start =
    Source.find_first_not_of(Delimiters, 0, NumDelimiters);
  if (Start == std::string::npos) Start = Source.size();

  // Find the next occurance of the delimiter.
  std::string::size_type End =
    Source.find_first_of(Delimiters, Start, NumDelimiters);
  if (End == std::string::npos) End = Source.size();

  // Create the return token.
  std::string Result = std::string(Source.begin()+Start, Source.begin()+End);

  // Erase the token that we read in.
  Source.erase(Source.begin(), Source.begin()+End);

  return Result;
}

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void llvm::SplitString(const std::string &Source, 
                       std::vector<std::string> &OutFragments,
                       const char *Delimiters) {
  std::string S = Source;
  
  std::string S2 = getToken(S, Delimiters);
  while (!S2.empty()) {
    OutFragments.push_back(S2);
    S2 = getToken(S, Delimiters);
  }
}



/// UnescapeString - Modify the argument string, turning two character sequences
/// @verbatim
/// like '\\' 'n' into '\n'.  This handles: \e \a \b \f \n \r \t \v \' \ and
/// \num (where num is a 1-3 byte octal value).
/// @endverbatim
void llvm::UnescapeString(std::string &Str) {
  for (unsigned i = 0; i != Str.size(); ++i) {
    if (Str[i] == '\\' && i != Str.size()-1) {
      switch (Str[i+1]) {
      default: continue;  // Don't execute the code after the switch.
      case 'a': Str[i] = '\a'; break;
      case 'b': Str[i] = '\b'; break;
      case 'e': Str[i] = 27; break;
      case 'f': Str[i] = '\f'; break;
      case 'n': Str[i] = '\n'; break;
      case 'r': Str[i] = '\r'; break;
      case 't': Str[i] = '\t'; break;
      case 'v': Str[i] = '\v'; break;
      case '\'': Str[i] = '\''; break;
      case '\\': Str[i] = '\\'; break;
      }
      // Nuke the second character.
      Str.erase(Str.begin()+i+1);
    }
  }
}

/// EscapeString - Modify the argument string, turning '\\' and anything that
/// doesn't satisfy std::isprint into an escape sequence.
void llvm::EscapeString(std::string &Str) {
  for (unsigned i = 0; i != Str.size(); ++i) {
    if (Str[i] == '\\') {
      ++i;
      Str.insert(Str.begin()+i, '\\');
    } else if (Str[i] == '\t') {
      Str[i++] = '\\';
      Str.insert(Str.begin()+i, 't');
    } else if (Str[i] == '\n') {
      Str[i++] = '\\';
      Str.insert(Str.begin()+i, 'n');
    } else if (!std::isprint(Str[i])) {
      // Always expand to a 3-digit octal escape.
      unsigned Char = Str[i];
      Str[i++] = '\\';
      Str.insert(Str.begin()+i++, '0'+((Char/64) & 7));
      Str.insert(Str.begin()+i++, '0'+((Char/8)  & 7));
      Str.insert(Str.begin()+i  , '0'+( Char     & 7));
    }
  }
}
