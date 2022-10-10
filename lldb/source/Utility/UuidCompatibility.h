//===-- UuidCompatibility.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Include this header if your system does not have a definition of uuid_t

#ifndef utility_UUID_COMPATIBILITY_H
#define utility_UUID_COMPATIBILITY_H

// uuid_t is guaranteed to always be a 16-byte array
typedef unsigned char uuid_t[16];

// Return 1 if uuid is null, that is, all zeroes.
inline __attribute__((always_inline)) int uuid_is_null(uuid_t uuid) {
  for (int i = 0; i < 16; i++)
    if (uuid[i])
      return 0;
  return 1;
}

#endif // utility_UUID_COMPATIBILITY_H
