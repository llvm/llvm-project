//===-- LlvmState.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  const TargetMachine &getTargetMachine() const { return *TheTargetMachine; }
  std::unique_ptr<LLVMTargetMachine> createTargetMachine() const;

  const ExegesisTarget &getExegesisTarget() const { return *TheExegesisTarget; }

  bool canAssemble(const MCInst &mc_inst) const;

  // For convenience:
  const MCInstrInfo &getInstrInfo() const {
    return *TheTargetMachine->getMCInstrInfo();
  }
  const MCRegisterInfo &getRegInfo() const {
    return *TheTargetMachine->getMCRegisterInfo();
  }
  const MCSubtargetInfo &getSubtargetInfo() const {
    return *TheTargetMachine->getMCSubtargetInfo();
  }

  const RegisterAliasingTrackerCache &getRATC() const { return *RATC; }
  const InstructionsCache &getIC() const { return *IC; }

  const PfmCountersInfo &getPfmCounters() const { return *PfmCounters; }

private:
  const ExegesisTarget *TheExegesisTarget;
  std::unique_ptr<const TargetMachine> TheTargetMachine;
  std::unique_ptr<const RegisterAliasingTrackerCache> RATC;
  std::unique_ptr<const InstructionsCache> IC;
  const PfmCountersInfo *PfmCounters;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
