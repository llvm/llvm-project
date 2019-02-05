//===- EnumTables.h - Enum to string conversion tables ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_ENUMTABLES_H
#define LLVM_DEBUGINFO_PDB_RAW_ENUMTABLES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace pdb {
ArrayRef<EnumEntry<uint16_t>> getOMFSegMapDescFlagNames();
}
}

#endif // LLVM_DEBUGINFO_PDB_RAW_ENUMTABLES_H
