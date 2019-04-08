//===-- llvm/Support/CRC.h - Cyclic Redundancy Check-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains basic functions for calculating Cyclic Redundancy Check
// or CRC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CRC_H
#define LLVM_SUPPORT_CRC_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
/// zlib independent CRC32 calculation.
uint32_t crc32(uint32_t CRC, StringRef S);
} // end namespace llvm

#endif
