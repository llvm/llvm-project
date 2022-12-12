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
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>

static constexpr llvm::StringLiteral kNoRegister("%noreg");

namespace llvm {
namespace exegesis {

class ExegesisTarget;
struct PfmCountersInfo;

// An object to initialize LLVM and prepare objects needed to run the
// measurements.
class LLVMState {
public:
  // Factory function.
  // If `Triple` is empty, uses the host triple.
  // If `CpuName` is empty, uses the host CPU.
  // `Features` is intended for tests.
  static Expected<LLVMState> Create(std::string TripleName, std::string CpuName,
                                    StringRef Features = "");

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

  const StringMap<unsigned> &getOpcodeNameToOpcodeIdxMapping() const {
    assert(OpcodeNameToOpcodeIdxMapping);
    return *OpcodeNameToOpcodeIdxMapping;
  };

  const StringMap<unsigned> &getRegNameToRegNoMapping() const {
    assert(RegNameToRegNoMapping);
    return *RegNameToRegNoMapping;
  }

private:
  std::unique_ptr<const StringMap<unsigned>>
  createOpcodeNameToOpcodeIdxMapping() const;

  std::unique_ptr<const StringMap<unsigned>>
  createRegNameToRegNoMapping() const;

  LLVMState(std::unique_ptr<const TargetMachine> TM, const ExegesisTarget *ET,
            StringRef CpuName);

  const ExegesisTarget *TheExegesisTarget;
  std::unique_ptr<const TargetMachine> TheTargetMachine;
  std::unique_ptr<const RegisterAliasingTrackerCache> RATC;
  std::unique_ptr<const InstructionsCache> IC;
  const PfmCountersInfo *PfmCounters;
  std::unique_ptr<const StringMap<unsigned>> OpcodeNameToOpcodeIdxMapping;
  std::unique_ptr<const StringMap<unsigned>> RegNameToRegNoMapping;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
