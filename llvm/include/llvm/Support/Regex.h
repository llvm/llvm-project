//===-- Regex.h - Regular Expression matcher implementation -*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a POSIX regular expression matcher.  Both Basic and
// Extended POSIX regular expressions (ERE) are supported.  EREs were extended
// to support backreferences in matches.
// This implementation also supports matching strings with embedded NUL chars.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_REGEX_H
#define LLVM_SUPPORT_REGEX_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include <string>

struct llvm_regex;

namespace llvm {
  class StringRef;
  template<typename T> class SmallVectorImpl;

  class Regex {
  public:
    enum RegexFlags : unsigned {
      NoFlags = 0,
      /// Compile for matching that ignores upper/lower case distinctions.
      IgnoreCase = 1,
      /// Compile for newline-sensitive matching. With this flag '[^' bracket
      /// expressions and '.' never match newline. A ^ anchor matches the
      /// null string after any newline in the string in addition to its normal
      /// function, and the $ anchor matches the null string before any
      /// newline in the string in addition to its normal function.
      Newline = 2,
      /// By default, the POSIX extended regular expression (ERE) syntax is
      /// assumed. Pass this flag to turn on basic regular expressions (BRE)
      /// instead.
      BasicRegex = 4,

      LLVM_MARK_AS_BITMASK_ENUM(BasicRegex)
    };

    LLVM_ABI Regex();
    /// Compiles the given regular expression \p Regex.
    ///
    /// \param Regex - referenced string is no longer needed after this
    /// constructor does finish.  Only its compiled form is kept stored.
    LLVM_ABI Regex(StringRef Regex, RegexFlags Flags = NoFlags);
    LLVM_ABI Regex(StringRef Regex, unsigned Flags);
    Regex(const Regex &) = delete;
    Regex &operator=(Regex regex) {
      std::swap(preg, regex.preg);
      std::swap(error, regex.error);
      return *this;
    }
    LLVM_ABI Regex(Regex &&regex);
    LLVM_ABI ~Regex();

    /// isValid - returns the error encountered during regex compilation, if
    /// any.
    LLVM_ABI bool isValid(std::string &Error) const;
    bool isValid() const { return !error; }

    /// getNumMatches - In a valid regex, return the number of parenthesized
    /// matches it contains.  The number filled in by match will include this
    /// many entries plus one for the whole regex (as element 0).
    LLVM_ABI unsigned getNumMatches() const;

    /// matches - Match the regex against a given \p String.
    ///
    /// \param Matches - If given, on a successful match this will be filled in
    /// with references to the matched group expressions (inside \p String),
    /// the first group is always the entire pattern.
    ///
    /// \param Error - If non-null, any errors in the matching will be recorded
    /// as a non-empty string. If there is no error, it will be an empty string.
    ///
    /// This returns true on a successful match.
    LLVM_ABI bool match(StringRef String,
                        SmallVectorImpl<StringRef> *Matches = nullptr,
                        std::string *Error = nullptr) const;

    /// sub - Return the result of replacing the first match of the regex in
    /// \p String with the \p Repl string. Backreferences like "\0" and "\g<1>"
    /// in the replacement string are replaced with the appropriate match
    /// substring.
    ///
    /// Note that the replacement string has backslash escaping performed on
    /// it. Invalid backreferences are ignored (replaced by empty strings).
    ///
    /// \param Error If non-null, any errors in the substitution (invalid
    /// backreferences, trailing backslashes) will be recorded as a non-empty
    /// string. If there is no error, it will be an empty string.
    LLVM_ABI std::string sub(StringRef Repl, StringRef String,
                             std::string *Error = nullptr) const;

    /// If this function returns true, ^Str$ is an extended regular
    /// expression that matches Str and only Str.
    LLVM_ABI static bool isLiteralERE(StringRef Str);

    /// Turn String into a regex by escaping its special characters.
    LLVM_ABI static std::string escape(StringRef String);

  private:
    struct llvm_regex *preg;
    int error;
  };
}

#endif // LLVM_SUPPORT_REGEX_H
