//===-- NameMatches.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Utility/NameMatches.h"
#include "lldb/Core/RegularExpression.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb_private;

bool lldb_private::NameMatches(const char *name, NameMatchType match_type,
                               const char *match) {
  if (match_type == eNameMatchIgnore)
    return true;

  if (name == match)
    return true;

  if (name && match) {
    llvm::StringRef name_sref(name);
    llvm::StringRef match_sref(match);
    switch (match_type) {
    case eNameMatchIgnore: // This case cannot occur: tested before
      return true;
    case eNameMatchEquals:
      return name_sref == match_sref;
    case eNameMatchContains:
      return name_sref.find(match_sref) != llvm::StringRef::npos;
    case eNameMatchStartsWith:
      return name_sref.startswith(match_sref);
    case eNameMatchEndsWith:
      return name_sref.endswith(match_sref);
    case eNameMatchRegularExpression: {
      RegularExpression regex(match_sref);
      return regex.Execute(name_sref);
    } break;
    }
  }
  return false;
}
