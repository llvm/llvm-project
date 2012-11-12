//===-- ARMInstrInfo.h - ARM Instruction Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMINSTRUCTIONINFO_H
#define ARMINSTRUCTIONINFO_H

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"

namespace llvm {
  class ARMSubtarget;

class ARMInstrInfo : public ARMBaseInstrInfo {
  ARMRegisterInfo RI;
public:
  explicit ARMInstrInfo(const ARMSubtarget &STI);

  /// getNoopForMachoTarget - Return the noop instruction to use for a noop.
  void getNoopForMachoTarget(MCInst &NopInst) const;

  // Return the non-pre/post incrementing version of 'Opc'. Return 0
  // if there is not such an opcode.
  unsigned getUnindexedOpcode(unsigned Opc) const;

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const ARMRegisterInfo &getRegisterInfo() const { return RI; }
};

}

#endif
