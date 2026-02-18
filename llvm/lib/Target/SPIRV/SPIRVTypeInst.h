//===-- SPIRVTypeInst.h - SPIR-V Type Instruction ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIRVTypeInst is used to represent a SPIR-V type instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVTYPEINST_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVTYPEINST_H

#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
class MachineInstr;
class MachineRegisterInfo;

/// @deprecated Use SPIRVTypeInst instead
/// SPIRVType is supposed to represent a MachineInstr that defines a SPIRV Type
/// (e.g. an OpTypeInt intruction). It is misused in several places and we're
/// getting rid of it.
using SPIRVType = const MachineInstr;

class SPIRVTypeInst {
  const MachineInstr *MI;

  // Used by DenseMapInfo to bypass the assertion. The thombstone and empty keys
  // are not null. They are -1 and -2 aligned to the appropiate pointer size.
  struct UncheckedConstructor {};
  SPIRVTypeInst(const MachineInstr *MI, UncheckedConstructor) : MI(MI) {};

public:
  SPIRVTypeInst(const MachineInstr &MI) : SPIRVTypeInst(&MI) {}
  SPIRVTypeInst(const MachineInstr *MI = nullptr);

  // No need to verify the register since it's already verified by the copied
  // object.
  SPIRVTypeInst(const SPIRVTypeInst &Other) = default;
  SPIRVTypeInst &operator=(const SPIRVTypeInst &Other) = default;

  const MachineInstr &operator*() const { return *MI; }
  const MachineInstr *operator->() const { return MI; }
  operator const MachineInstr *() const { return MI; }

  bool operator==(const SPIRVTypeInst &Other) const { return MI == Other.MI; }
  bool operator!=(const SPIRVTypeInst &Other) const { return MI != Other.MI; }

  bool operator==(const MachineInstr *Other) const { return MI == Other; }
  bool operator!=(const MachineInstr *Other) const { return MI != Other; }

  operator bool() const { return MI; }

  friend struct DenseMapInfo<SPIRVTypeInst>;
};

template <> struct DenseMapInfo<SPIRVTypeInst> {
  using MIInfo = DenseMapInfo<MachineInstr *>;
  static SPIRVTypeInst getEmptyKey() {
    return {MIInfo::getEmptyKey(), SPIRVTypeInst::UncheckedConstructor()};
  }
  static SPIRVTypeInst getTombstoneKey() {
    return {MIInfo::getTombstoneKey(), SPIRVTypeInst::UncheckedConstructor()};
  }
  static unsigned getHashValue(SPIRVTypeInst Ty) {
    return MIInfo::getHashValue(Ty.MI);
  }
  static bool isEqual(SPIRVTypeInst Ty1, SPIRVTypeInst Ty2) {
    return Ty1 == Ty2;
  }
};

} // namespace llvm
#endif
