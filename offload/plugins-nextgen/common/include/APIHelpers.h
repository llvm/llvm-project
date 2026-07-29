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

// Macro to mark external symbol as weak, so linker will be okay
// if the symbol is missing. For direct linking only available on Linux, we need
// to check if linker could find the symbol. For symbols loaded using dlsym we
// call name##_loaded function. The name##_loaded function will be nullptr if
// external library was linked directly.
#ifndef WIN32
#define API_HELPER_OPTIONAL(return_type, name, ...)                            \
  namespace dlwrap {                                                           \
  bool name##_loaded() __attribute__((weak));                                  \
  }                                                                            \
  extern "C" return_type name(__VA_ARGS__) __attribute__((weak));              \
  template <> inline bool api_helper::canCall<name>() {                        \
    if (name == nullptr)                                                       \
      /* Not loaded weak symbol */                                             \
      return false;                                                            \
    /* If symbol wasn't dlwrapped, i.e name##_loaded == nullptr and is not     \
     * nullptr, it means the symbol was linked directly, so we can call it */  \
    if (dlwrap::name##_loaded == nullptr)                                      \
      return true;                                                             \
    /* Symbol is not nullptr and it was dlwrapped */                           \
    return dlwrap::name##_loaded();                                            \
  }
#else
#define API_HELPER_OPTIONAL(return_type, name, ...)                            \
  namespace dlwrap {                                                           \
  bool name##_loaded() __attribute__((weak));                                  \
  }                                                                            \
  extern "C" return_type name(__VA_ARGS__) __attribute__((weak));              \
  template <> inline bool api_helper::canCall<name>() {                        \
    /* Windows has no direct linking, so all symbols go through dlwrap */      \
    return dlwrap::name##_loaded();                                            \
  }
#endif

namespace api_helper {

// Default template specialization for extra safety
template <auto Fn> bool canCall() {
  static_assert(sizeof(decltype(Fn) *) == 0,
                "api_helper::canCall() should only be called on symbols "
                "decorated with API_HELPER_OPTIONAL!");
}

} // namespace api_helper

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_APIHELPERS_H
