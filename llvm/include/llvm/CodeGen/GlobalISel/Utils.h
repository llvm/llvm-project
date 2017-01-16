//==-- llvm/CodeGen/GlobalISel/Utils.h ---------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API of helper functions used throughout the
/// GlobalISel pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_UTILS_H
#define LLVM_CODEGEN_GLOBALISEL_UTILS_H

namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineRegisterInfo;
class MCInstrDesc;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetRegisterInfo;

/// Try to constrain Reg so that it is usable by argument OpIdx of the
/// provided MCInstrDesc \p II. If this fails, create a new virtual
/// register in the correct class and insert a COPY before \p InsertPt.
/// The debug location of \p InsertPt is used for the new copy.
///
/// \return The virtual register constrained to the right register class.
unsigned constrainOperandRegClass(const MachineFunction &MF,
                                  const TargetRegisterInfo &TRI,
                                  MachineRegisterInfo &MRI,
                                  const TargetInstrInfo &TII,
                                  const RegisterBankInfo &RBI,
                                  MachineInstr &InsertPt, const MCInstrDesc &II,
                                  unsigned Reg, unsigned OpIdx);

} // End namespace llvm.
#endif
