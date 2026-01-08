//===-- Utility.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "CoreSpec.h"

void add_uint64(std::vector<uint8_t> &buf, uint64_t val) {
  uint8_t *p = reinterpret_cast<uint8_t *>(&val);
  for (int i = 0; i < 8; i++)
    buf.push_back(*p++);
}

void add_uint32(std::vector<uint8_t> &buf, uint32_t val) {
  uint8_t *p = reinterpret_cast<uint8_t *>(&val);
  for (int i = 0; i < 4; i++)
    buf.push_back(*p++);
}
