//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_UTILITY_H
#define YAML2MACHOCOREFILE_UTILITY_H

#include "CoreSpec.h"
#include <vector>

void add_uint64(std::vector<uint8_t> &buf, uint64_t val);
void add_uint32(std::vector<uint8_t> &buf, uint32_t val);

#endif
