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

// Versions of GCC prior to GCC 9 don't declare __PRETTY_FUNCTION__ as constexpr
#if defined(__clang__) || defined(_MSC_VER) ||                                 \
    (defined(__GNUC__) && __GNUC__ >= 9)
#define LLVM_GET_TYPE_NAME_CONSTEXPR constexpr
#define LLVM_GET_TYPE_NAME_STATIC_ASSERT 1
#else
#define LLVM_GET_TYPE_NAME_CONSTEXPR
#define LLVM_GET_TYPE_NAME_STATIC_ASSERT 0
#include <cassert>
#endif

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
template <typename DesiredTypeName>
inline LLVM_GET_TYPE_NAME_CONSTEXPR StringRef getTypeName() {
#if defined(__clang__) || defined(__GNUC__)
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view Name = __PRETTY_FUNCTION__;

  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view Key = "DesiredTypeName = ";
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view TemplateParamsStart =
      Name.substr(Name.find(Key));
#if LLVM_GET_TYPE_NAME_STATIC_ASSERT
  static_assert(!TemplateParamsStart.empty(),
                "Unable to find the template parameter!");
#else
  assert(!TemplateParamsStart.empty() &&
         "Unable to find the template parameter!");
#endif

  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view SubstitutionKey =
      TemplateParamsStart.substr(Key.size());

#if LLVM_GET_TYPE_NAME_STATIC_ASSERT
  // ends_with() is only available in c++20
  static_assert(!SubstitutionKey.empty() && SubstitutionKey.back() == ']',
                "Name doesn't end in the substitution key!");
#else
  assert(!SubstitutionKey.empty() && SubstitutionKey.back() == ']' &&
         "Name doesn't end in the substitution key!");
#endif

  return SubstitutionKey.substr(0, SubstitutionKey.size() - 1);
#elif defined(_MSC_VER)
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view Name = __FUNCSIG__;

  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view Key = "getTypeName<";
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view GetTypeNameStart =
      Name.substr(Name.find(Key));
  // TODO: SWDEV-517818 - Changed from static_assert to assert to ensure
  // compiler compatibility
  assert(!GetTypeNameStart.empty() && "Unable to find the template parameter!");
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view SubstitutionKey =
      GetTypeNameStart.substr(Key.size());

  // starts_with() only available in c++20
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view RmPrefixClass =
      SubstitutionKey.find("class ") == 0
          ? SubstitutionKey.substr(sizeof("class ") - 1)
          : SubstitutionKey;
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view RmPrefixStruct =
      RmPrefixClass.find("struct ") == 0
          ? RmPrefixClass.substr(sizeof("struct ") - 1)
          : RmPrefixClass;
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view RmPrefixUnion =
      RmPrefixStruct.find("union ") == 0
          ? RmPrefixStruct.substr(sizeof("union ") - 1)
          : RmPrefixStruct;
  LLVM_GET_TYPE_NAME_CONSTEXPR std::string_view RmPrefixEnum =
      RmPrefixUnion.find("enum ") == 0
          ? RmPrefixUnion.substr(sizeof("enum ") - 1)
          : RmPrefixUnion;

  LLVM_GET_TYPE_NAME_CONSTEXPR auto AnglePos = RmPrefixEnum.rfind('>');
  // TODO: SWDEV-517818 - Changed from static_assert to assert to ensure
  // compiler compatibility
  assert(AnglePos != std::string_view::npos &&
         "Unable to find the closing '>'!");
  return RmPrefixEnum.substr(0, AnglePos);
#else
  // No known technique for statically extracting a type name on this compiler.
  // We return a string that is unlikely to look like any type in LLVM.
  return "UNKNOWN_TYPE";
#endif
}

} // namespace llvm

// Don't leak out of this header file
#undef LLVM_GET_TYPE_NAME_CONSTEXPR
#undef LLVM_GET_TYPE_NAME_STATIC_ASSERT

#endif
