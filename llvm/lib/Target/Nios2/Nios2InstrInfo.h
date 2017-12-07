//===-- Nios2InstrInfo.h - Nios2 Instruction Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2INSTRINFO_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2INSTRINFO_H

#include "Nios2RegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "Nios2GenInstrInfo.inc"

namespace llvm {

class Nios2Subtarget;

class Nios2InstrInfo : public Nios2GenInstrInfo {
  const Nios2RegisterInfo RI;
  const Nios2Subtarget &Subtarget;
  virtual void anchor();

public:
  explicit Nios2InstrInfo(Nios2Subtarget &ST);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const Nios2RegisterInfo &getRegisterInfo() const { return RI; };

  bool expandPostRAPseudo(MachineInstr &MI) const override;
};
} // namespace llvm

#endif
