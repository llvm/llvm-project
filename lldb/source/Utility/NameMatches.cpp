//===-- NameMatches.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Utility/NameMatches.h"
#include "lldb/Utility/RegularExpression.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb_private;

bool lldb_private::NameMatches(llvm::StringRef name, NameMatchType match_type,
                               llvm::StringRef match) {
  if (match_type == eNameMatchIgnore)
    return true;

  if (name == match)
    return true;

  if (name.empty() || match.empty())
    return false;

  switch (match_type) {
  case eNameMatchIgnore: // This case cannot occur: tested before
    return true;
  case eNameMatchEquals:
    return name == match;
  case eNameMatchContains:
    return name.contains(match);
  case eNameMatchStartsWith:
    return name.startswith(match);
  case eNameMatchEndsWith:
    return name.endswith(match);
  case eNameMatchRegularExpression: {
    RegularExpression regex(match);
    return regex.Execute(name);
  } break;
  }
  return false;
}
