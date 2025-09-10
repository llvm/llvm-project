//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Functions to emit LC_THREAD bytes to the corefile's Mach-O load commands,
/// specifying the threads, the register sets ("flavors") within those threads,
/// and all of the registers within those register sets.
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_THREADWRITER_H
#define YAML2MACHOCOREFILE_THREADWRITER_H

#include "CoreSpec.h"

#include <vector>

void add_lc_threads(CoreSpec &spec,
                    std::vector<std::vector<uint8_t>> &load_commands);

#endif
