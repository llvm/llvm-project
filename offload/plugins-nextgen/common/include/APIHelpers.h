//===-- Shared/APIHelpers.h - helpers for external APIs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The header contains helper functions to make interactions with external APIs
// such as CUDA or level zero easier
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_APIHELPERS_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_APIHELPERS_H

#include "DLWrap.h"

#define API_HELPER_STRINGIFY_INNER(x) #x
#define API_HELPER_STRINGIFY(x) API_HELPER_STRINGIFY_INNER(x)

// Macro to mark external symbol as weak, so linker will be okay
// if the symbol is missing. For direct linking only available on Linux, we need
// to check if linker could find the symbol. For symbols loaded using dlsym we
// call name##_loaded function. The name##_loaded function will be nullptr if
// external library was linked directly.
// _Pragma("weak name") retroactively downgrades any prior strong declaration
// (e.g. from a vendor header included before this macro) to weak.
#define API_HELPER_OPTIONAL(return_type, name, ...)                            \
  _Pragma(API_HELPER_STRINGIFY(weak name)) namespace dlwrap {                  \
    bool name##_loaded() __attribute__((weak));                                \
  }                                                                            \
  extern "C" return_type name(__VA_ARGS__) __attribute__((weak));              \
  template <> inline bool api_helper::canCall<name>() {                        \
    if (name == nullptr)                                                       \
      /* Not loaded weak symbol, only possible on Linux */                     \
      return false;                                                            \
    /* If symbol wasn't dlwrapped, i.e name##_loaded == nullptr and is not     \
     * nullptr, it means the symbol was linked directly, so we can call it */  \
    if (dlwrap::name##_loaded == nullptr)                                      \
      return true;                                                             \
    /* Symbol is not nullptr and it was dlwrapped, all symbols on Windows go   \
     * here*/                                                                  \
    return dlwrap::name##_loaded();                                            \
  }

namespace api_helper {

// Default template specialization for extra safety
template <auto Fn> bool canCall() {
  static_assert(sizeof(decltype(Fn) *) == 0,
                "api_helper::canCall() should only be called on symbols "
                "decorated with API_HELPER_OPTIONAL!");
}

} // namespace api_helper

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_APIHELPERS_H
