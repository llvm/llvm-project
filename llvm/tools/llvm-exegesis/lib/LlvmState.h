//===-- LlvmState.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A class to set up and access common LLVM objects.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
#define LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H

#include "Target.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>

namespace exegesis {

class ExegesisTarget;

// An object to initialize LLVM and prepare objects needed to run the
// measurements.
class LLVMState {
public:
  LLVMState();

  LLVMState(const std::string &Triple,
            const std::string &CpuName); // For tests.

  const llvm::TargetMachine &getTargetMachine() const { return *TargetMachine; }
  std::unique_ptr<llvm::LLVMTargetMachine> createTargetMachine() const;

  const ExegesisTarget *getExegesisTarget() const { return TheExegesisTarget; }

  bool canAssemble(const llvm::MCInst &mc_inst) const;

  // For convenience:
  const llvm::MCInstrInfo &getInstrInfo() const {
    return *TargetMachine->getMCInstrInfo();
  }
  const llvm::MCRegisterInfo &getRegInfo() const {
    return *TargetMachine->getMCRegisterInfo();
  }
  const llvm::MCSubtargetInfo &getSubtargetInfo() const {
    return *TargetMachine->getMCSubtargetInfo();
  }

private:
  const ExegesisTarget *TheExegesisTarget = nullptr;
  std::unique_ptr<const llvm::TargetMachine> TargetMachine;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
