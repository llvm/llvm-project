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

#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>

namespace llvm {
namespace exegesis {

class ExegesisTarget;
struct PfmCountersInfo;

// An object to initialize LLVM and prepare objects needed to run the
// measurements.
class LLVMState {
public:
  // Uses the host triple. If CpuName is empty, uses the host CPU.
  LLVMState(const std::string &CpuName);

  LLVMState(const std::string &Triple,
            const std::string &CpuName,
            const std::string &Features = ""); // For tests.

  const llvm::TargetMachine &getTargetMachine() const { return *TargetMachine; }
  std::unique_ptr<llvm::LLVMTargetMachine> createTargetMachine() const;

  const ExegesisTarget &getExegesisTarget() const { return *TheExegesisTarget; }

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

  const RegisterAliasingTrackerCache &getRATC() const { return *RATC; }
  const InstructionsCache &getIC() const { return *IC; }

  const PfmCountersInfo &getPfmCounters() const { return *PfmCounters; }

private:
  const ExegesisTarget *TheExegesisTarget;
  std::unique_ptr<const llvm::TargetMachine> TargetMachine;
  std::unique_ptr<const RegisterAliasingTrackerCache> RATC;
  std::unique_ptr<const InstructionsCache> IC;
  const PfmCountersInfo *PfmCounters;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
