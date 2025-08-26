//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_LCNOTEWRITER_H
#define YAML2MACHOCOREFILE_LCNOTEWRITER_H

#include "CoreSpec.h"

#include <stdio.h>
#include <vector>

void create_lc_note_binary_load_cmd(const CoreSpec &spec,
                                    std::vector<uint8_t> &cmds,
                                    std::string uuid, uint64_t slide,
                                    std::vector<uint8_t> &payload_bytes,
                                    off_t data_offset);

#endif
