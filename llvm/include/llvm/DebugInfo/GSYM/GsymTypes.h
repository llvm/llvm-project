//===- GsymTypes.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMTYPES_H
#define LLVM_DEBUGINFO_GSYM_GSYMTYPES_H

#include <stdint.h>

namespace llvm {
namespace gsym {

/// The type of string offset used in the code.
///
/// Note: This may be different from what's serialized into GSYM files, which
/// is version dependent (e.g. V1 uses uint32_t, V2+ uses uint64_t).
typedef uint64_t gsym_strp_t;

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMTYPES_H
