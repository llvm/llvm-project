//===- TypeName.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TYPENAME_H
#define LLVM_SUPPORT_TYPENAME_H

#include <string_view>

#include "llvm/ADT/StringRef.h"

namespace llvm {

/// We provide a function which tries to compute the (demangled) name of a type
/// statically.
///
/// This routine may fail on some platforms or for particularly unusual types.
/// Do not use it for anything other than logging and debugging aids. It isn't
/// portable or dependendable in any real sense.
///
/// The returned StringRef will point into a static storage duration string.
/// However, it may not be null terminated and may be some strangely aligned
/// inner substring of a larger string.
template <typename DesiredTypeName> inline constexpr StringRef getTypeName() {
#if defined(__clang__) || defined(__GNUC__)
  constexpr std::string_view Name = __PRETTY_FUNCTION__;

  constexpr std::string_view Key = "DesiredTypeName = ";
  constexpr std::string_view TemplateParamsStart = Name.substr(Name.find(Key));
  static_assert(!TemplateParamsStart.empty(),
                "Unable to find the template parameter!");
  constexpr std::string_view SubstitutionKey =
      TemplateParamsStart.substr(Key.size());

  // ends_with() is only available in c++20
  static_assert(!SubstitutionKey.empty() && SubstitutionKey.back() == ']',
                "Name doesn't end in the substitution key!");
  return SubstitutionKey.substr(0, SubstitutionKey.size() - 1);
#elif defined(_MSC_VER)
  constexpr std::string_view Name = __FUNCSIG__;

  constexpr std::string_view Key = "getTypeName<";
  constexpr std::string_view GetTypeNameStart = Name.substr(Name.find(Key));
  static_assert(!GetTypeNameStart.empty(),
                "Unable to find the template parameter!");
  constexpr std::string_view SubstitutionKey =
      GetTypeNameStart.substr(Key.size());

  // starts_with() only available in c++20
  constexpr std::string_view RmPrefixClass =
      SubstitutionKey.find("class ") == 0
          ? SubstitutionKey.substr(sizeof("class ") - 1)
          : SubstitutionKey;
  constexpr std::string_view RmPrefixStruct =
      RmPrefixClass.find("struct ") == 0
          ? RmPrefixClass.substr(sizeof("struct ") - 1)
          : RmPrefixClass;
  constexpr std::string_view RmPrefixUnion =
      RmPrefixStruct.find("union ") == 0
          ? RmPrefixStruct.substr(sizeof("union ") - 1)
          : RmPrefixStruct;
  constexpr std::string_view RmPrefixEnum =
      RmPrefixUnion.find("enum ") == 0
          ? RmPrefixUnion.substr(sizeof("enum ") - 1)
          : RmPrefixUnion;

  constexpr auto AnglePos = RmPrefixEnum.rfind('>');
  static_assert(AnglePos != std::string_view::npos,
                "Unable to find the closing '>'!");
  return RmPrefixEnum.substr(0, AnglePos);
#else
  // No known technique for statically extracting a type name on this compiler.
  // We return a string that is unlikely to look like any type in LLVM.
  return "UNKNOWN_TYPE";
#endif
}

} // namespace llvm

#endif
