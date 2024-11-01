
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_SAFE_C_API_H
#define KMP_SAFE_C_API_H

#include <type_traits>
#include "kmp_platform.h"
#include <string.h>

// Replacement for banned C API

// Not every unsafe call listed here is handled now, but keeping everything
// in one place should be handy for future maintenance.
#if KMP_OS_WINDOWS && KMP_MSVC_COMPAT

#define RSIZE_MAX_STR (4UL << 10) // 4KB

// _malloca was suggested, but it is not a drop-in replacement for _alloca
#define KMP_ALLOCA _alloca

#define KMP_MEMCPY_S memcpy_s
#define KMP_SNPRINTF sprintf_s
#define KMP_SSCANF sscanf_s
#define KMP_STRCPY_S strcpy_s
#define KMP_STRNCPY_S strncpy_s
#define KMP_STRNCAT_S strncat_s

// Use this only when buffer size is unknown
#define KMP_MEMCPY(dst, src, cnt) memcpy_s(dst, cnt, src, cnt)

template <typename T, bool B = std::is_array<T>::value>
struct kmp_get_rmax_t {};
template <typename T> struct kmp_get_rmax_t<T, false> {
  static const size_t value = RSIZE_MAX_STR;
};
template <typename T> struct kmp_get_rmax_t<T, true> {
  static const size_t value = sizeof(T);
};
#define KMP_STRLEN(str) strnlen_s(str, kmp_get_rmax_t<decltype(str)>::value)

// Use this only when buffer size is unknown
#define KMP_STRNCPY(dst, src, cnt) strncpy_s(dst, cnt, src, cnt)

// _TRUNCATE insures buffer size > max string to print.
#define KMP_VSNPRINTF(dst, cnt, fmt, arg)                                      \
  vsnprintf_s(dst, cnt, _TRUNCATE, fmt, arg)

#else // KMP_OS_WINDOWS

// For now, these macros use the existing API.

#define KMP_ALLOCA alloca
#define KMP_MEMCPY_S(dst, bsz, src, cnt) memcpy(dst, src, cnt)
#define KMP_SNPRINTF snprintf
#define KMP_SSCANF sscanf
#define KMP_STRCPY_S(dst, bsz, src) strcpy(dst, src)
#define KMP_STRNCPY_S(dst, bsz, src, cnt) strncpy(dst, src, cnt)
#define KMP_STRNCAT_S(dst, bsz, src, cnt) strncat(dst, src, cnt)
#define KMP_VSNPRINTF vsnprintf
#define KMP_STRNCPY strncpy
#define KMP_STRLEN strlen
#define KMP_MEMCPY memcpy

#endif // KMP_OS_WINDOWS

// Offer truncated version of strncpy
static inline void __kmp_strncpy_truncate(char *buffer, size_t buf_size,
                                          char const *src, size_t src_size) {
  if (src_size >= buf_size) {
    src_size = buf_size - 1;
  }
  KMP_STRNCPY_S(buffer, buf_size, src, src_size);
  buffer[src_size] = '\0';
}

#endif // KMP_SAFE_C_API_H
