//===--- CodeGenTestBase.h - Utilities for codegen unit tests ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_CODEGEN_CODEGENTESTBASE_H
#define LLVM_UNITTESTS_CODEGEN_CODEGENTESTBASE_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

/// Boilerplate set-up for codegen tests. Sets up all analyses managers for a
/// given target and creates a module from an MIR string.
class CodeGenTestBase : public testing::Test {
public:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MIRParser> MIR;
  std::unique_ptr<Module> Mod;

  LoopAnalysisManager LAM;
  MachineFunctionAnalysisManager MFAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  MachineFunction &getMF(StringRef FuncName) {
    return FAM.getResult<MachineFunctionAnalysis>(*Mod->getFunction(FuncName))
        .getMF();
  }

protected:
  /// Sets up the target machine and analyses managers.
  void setUpImpl(StringRef Triple, StringRef CPU, StringRef FS) {
    llvm::Triple TT(Triple);
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TT, Error);
    if (!T)
      GTEST_SKIP();
    TargetOptions Options;
    TM.reset(T->createTargetMachine(TT, CPU, FS, Options, std::nullopt));
    if (!TM)
      GTEST_SKIP();
    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    PassBuilder PB(TM.get());
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerMachineFunctionAnalyses(MFAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);
    MAM.registerPass([&] { return MachineModuleAnalysis(*MMI); });
  }

  /// Parses \p MIRCode into a module. Returns whether parsing was successful.
  bool parseMIR(StringRef MIRCode) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return false;

    Mod = MIR->parseIRModule();
    Mod->setDataLayout(TM->createDataLayout());
    if (MIR->parseMachineFunctions(*Mod, MAM)) {
      Mod.reset();
      return false;
    }
    return true;
  }
};

} // namespace llvm

#endif // LLVM_UNITTESTS_CODEGEN_CODEGENTESTBASE_H
