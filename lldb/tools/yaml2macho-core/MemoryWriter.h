//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Functions to emit the LC_SEGMENT load command, and to provide the bytes
/// that appear later in the corefile.
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_MEMORYWRITER_H
#define YAML2MACHOCOREFILE_MEMORYWRITER_H

#include "CoreSpec.h"

#include <vector>

void create_lc_segment_cmd(const CoreSpec &spec, std::vector<uint8_t> &cmds,
                           const MemoryRegion &memory, off_t data_offset);

void create_memory_bytes(const CoreSpec &spec, const MemoryRegion &memory,
                         std::vector<uint8_t> &buf);

#endif
