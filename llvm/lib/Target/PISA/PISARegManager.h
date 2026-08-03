//===-- PISARegManager.h - Manage PISA virtual registers ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAREGMANAGER_H
#define LLVM_LIB_TARGET_PISA_PISAREGMANAGER_H

#include "MCTargetDesc/PISARegEncoder.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {
namespace PISA {

class RegManager : public RegEncoder {
public:
  enum Usage { None = 0, NoEmissionDef = (1 << 0) };
  struct RegInfo {
    RegType Type;
    unsigned Idx;
    Usage Flags;
  };

public:
  RegManager(const MachineFunction &MF);
  unsigned getRegIdx(Register Reg) const;
  void setRegIdx(Register Reg, unsigned Idx) { Mapping[Reg].Idx = Idx; }
  unsigned encodeVirtualRegister(RegBank Bank, Register Reg) const;
  auto mapping() const { return make_range(Mapping.begin(), Mapping.end()); }
  bool exists(Register Reg) const { return Mapping.count(Reg) > 0; }

private:
  MapVector<Register, RegInfo> Mapping;
  const MachineFunction &MF;
  const MachineRegisterInfo &MRI;
  void computeMapping();
};

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAREGMANAGER_H
