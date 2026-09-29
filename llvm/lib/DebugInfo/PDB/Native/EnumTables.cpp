//===- EnumTables.cpp - Enum to string conversion tables --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/EnumTables.h"
#include "llvm/ADT/Enum.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"

using namespace llvm;

EnumStrings<uint16_t> llvm::pdb::getOMFSegMapDescFlagNames() {
#define PDB_ENUM_CLASS_ENT(enum_class, enum)                                   \
  {                                                                            \
    {#enum}, std::underlying_type_t<enum_class>(enum_class::enum)              \
  }
  constexpr EnumStringDef<uint16_t> Defs[] = {
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, Read),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, Write),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, Execute),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, AddressIs32Bit),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, IsSelector),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, IsAbsoluteAddress),
      PDB_ENUM_CLASS_ENT(OMFSegDescFlags, IsGroup),
  };
  static constexpr auto Names = BUILD_ENUM_STRINGS(Defs);
  return EnumStrings(Names);
}
