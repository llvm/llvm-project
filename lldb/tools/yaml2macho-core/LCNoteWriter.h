//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Functions to add an LC_NOTE load command to the corefile's load commands,
/// and supply the payload of that LC_NOTE separately.
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_LCNOTEWRITER_H
#define YAML2MACHOCOREFILE_LCNOTEWRITER_H

#include "CoreSpec.h"

#include <stdio.h>
#include <vector>

void create_lc_note_binary_load_cmd(const CoreSpec &spec,
                                    std::vector<uint8_t> &cmds,
                                    const Binary &binary,
                                    std::vector<uint8_t> &payload_bytes,
                                    off_t data_offset);

void create_lc_note_addressable_bits(const CoreSpec &spec,
                                     std::vector<uint8_t> &cmds,
                                     const AddressableBits &addr_bits,
                                     std::vector<uint8_t> &payload_bytes,
                                     off_t data_offset);

#endif
