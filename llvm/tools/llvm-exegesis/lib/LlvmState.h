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

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>

namespace exegesis {

// An object to initialize LLVM and prepare objects needed to run the
// measurements.
class LLVMState {
public:
  LLVMState();

  LLVMState(const std::string &Triple,
            const std::string &CpuName); // For tests.

  llvm::StringRef getTriple() const { return TheTriple; }
  llvm::StringRef getCpuName() const { return CpuName; }
  llvm::StringRef getFeatures() const { return Features; }

  const llvm::MCInstrInfo &getInstrInfo() const { return *InstrInfo; }

  const llvm::MCRegisterInfo &getRegInfo() const { return *RegInfo; }

  const llvm::MCSubtargetInfo &getSubtargetInfo() const {
    return *SubtargetInfo;
  }

  std::unique_ptr<llvm::LLVMTargetMachine> createTargetMachine() const;

  bool canAssemble(const llvm::MCInst &mc_inst) const;

private:
  std::string TheTriple;
  std::string CpuName;
  std::string Features;
  const llvm::Target *TheTarget = nullptr;
  std::unique_ptr<const llvm::MCSubtargetInfo> SubtargetInfo;
  std::unique_ptr<const llvm::MCInstrInfo> InstrInfo;
  std::unique_ptr<const llvm::MCRegisterInfo> RegInfo;
  std::unique_ptr<const llvm::MCAsmInfo> AsmInfo;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LLVMSTATE_H
