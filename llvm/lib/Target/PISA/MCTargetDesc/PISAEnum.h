//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares PISA enumeration and operand option tables.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include <cstdint>

namespace llvm {
namespace PISA {

#define GET_BoolOptionID_DECL
#define GET_EnumOptionClass_DECL
#define GET_LoadCacheControl_DECL
#define GET_StoreCacheControl_DECL
#define GET_AtomicCacheControl_DECL
#include "PISAGenSearchableTables.inc"

struct BoolOptionTableEntry {
  BoolOptionID OptID;
  StringTable::Offset OptStr;
};

struct EnumOptionEntry {
  EnumOptionClass OptClass;
  unsigned Value;
  StringTable::Offset OptStr;
  unsigned CFlags;
};

#define GET_BoolOptionTable_DECL
#define GET_EnumOptionTable_DECL
#include "PISAGenSearchableTables.inc"

} // namespace PISA
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H
