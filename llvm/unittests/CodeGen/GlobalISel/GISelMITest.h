//===- GISelMITest.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_UNITTEST_CODEGEN_GLOBALISEL_GISELMI_H
#define LLVM_UNITTEST_CODEGEN_GLOBALISEL_GISELMI_H

#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace MIPatternMatch;

static inline void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
}

// Define a printers to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const LLT Ty);

std::ostream &
operator<<(std::ostream &OS, const MachineFunction &MF);
}

static std::unique_ptr<Module> parseMIR(LLVMContext &Context,
                                        std::unique_ptr<MIRParser> &MIR,
                                        const TargetMachine &TM,
                                        StringRef MIRCode, const char *FuncName,
                                        MachineModuleInfo &MMI) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  MIR = createMIRParser(std::move(MBuffer), Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> M = MIR->parseIRModule();
  if (!M)
    return nullptr;

  M->setDataLayout(TM.createDataLayout());

  if (MIR->parseMachineFunctions(*M, MMI))
    return nullptr;

  return M;
}
static std::pair<std::unique_ptr<Module>, std::unique_ptr<MachineModuleInfo>>
createDummyModule(LLVMContext &Context, const LLVMTargetMachine &TM,
                  StringRef MIRString, const char *FuncName) {
  std::unique_ptr<MIRParser> MIR;
  auto MMI = std::make_unique<MachineModuleInfo>(&TM);
  std::unique_ptr<Module> M =
      parseMIR(Context, MIR, TM, MIRString, FuncName, *MMI);
  return make_pair(std::move(M), std::move(MMI));
}

static MachineFunction *getMFFromMMI(const Module *M,
                                     const MachineModuleInfo *MMI) {
  Function *F = M->getFunction("func");
  auto *MF = MMI->getMachineFunction(*F);
  return MF;
}

static void collectCopies(SmallVectorImpl<Register> &Copies,
                          MachineFunction *MF) {
  for (auto &MBB : *MF)
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == TargetOpcode::COPY)
        Copies.push_back(MI.getOperand(0).getReg());
    }
}

class GISelMITest : public ::testing::Test {
protected:
  GISelMITest() : ::testing::Test() {}

  /// Prepare a target specific LLVMTargetMachine.
  virtual std::unique_ptr<LLVMTargetMachine> createTargetMachine() const = 0;

  /// Get the stub sample MIR test function.
  virtual void getTargetTestModuleString(SmallString<512> &S,
                                         StringRef MIRFunc) const = 0;

  LLVMTargetMachine *
  createTargetMachineAndModule(StringRef ExtraAssembly = "") {
    TheTM = createTargetMachine();
    if (!TheTM)
      return nullptr;

    SmallString<512> MIRString;
    getTargetTestModuleString(MIRString, ExtraAssembly);

    ModuleMMIPair = createDummyModule(Context, *TheTM, MIRString, "func");
    MF = getMFFromMMI(ModuleMMIPair.first.get(), ModuleMMIPair.second.get());
    collectCopies(Copies, MF);
    EntryMBB = &*MF->begin();
    B.setMF(*MF);
    MRI = &MF->getRegInfo();
    B.setInsertPt(*EntryMBB, EntryMBB->end());
    return TheTM.get();
  }

  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TheTM;
  MachineFunction *MF;
  std::pair<std::unique_ptr<Module>, std::unique_ptr<MachineModuleInfo>>
      ModuleMMIPair;
  SmallVector<Register, 4> Copies;
  MachineBasicBlock *EntryMBB;
  MachineIRBuilder B;
  MachineRegisterInfo *MRI;
};

class AArch64GISelMITest : public GISelMITest {
  std::unique_ptr<LLVMTargetMachine> createTargetMachine() const override;
  void getTargetTestModuleString(SmallString<512> &S,
                                 StringRef MIRFunc) const override;
};

class AMDGPUGISelMITest : public GISelMITest {
  std::unique_ptr<LLVMTargetMachine> createTargetMachine() const override;
  void getTargetTestModuleString(SmallString<512> &S,
                                 StringRef MIRFunc) const override;
};

#define DefineLegalizerInfo(Name, SettingUpActionsBlock)                       \
  class Name##Info : public LegalizerInfo {                                    \
  public:                                                                      \
    Name##Info(const TargetSubtargetInfo &ST) {                                \
      using namespace TargetOpcode;                                            \
      const LLT s8 = LLT::scalar(8);                                           \
      (void)s8;                                                                \
      const LLT s16 = LLT::scalar(16);                                         \
      (void)s16;                                                               \
      const LLT s32 = LLT::scalar(32);                                         \
      (void)s32;                                                               \
      const LLT s64 = LLT::scalar(64);                                         \
      (void)s64;                                                               \
      const LLT s128 = LLT::scalar(128);                                       \
      (void)s128;                                                              \
      do                                                                       \
        SettingUpActionsBlock while (0);                                       \
      computeTables();                                                         \
      verify(*ST.getInstrInfo());                                              \
    }                                                                          \
  };

static inline bool CheckMachineFunction(const MachineFunction &MF,
                                        StringRef CheckStr) {
  SmallString<512> Msg;
  raw_svector_ostream OS(Msg);
  MF.print(OS);
  auto OutputBuf = MemoryBuffer::getMemBuffer(Msg, "Output", false);
  auto CheckBuf = MemoryBuffer::getMemBuffer(CheckStr, "");
  SmallString<4096> CheckFileBuffer;
  FileCheckRequest Req;
  FileCheck FC(Req);
  StringRef CheckFileText =
      FC.CanonicalizeFile(*CheckBuf.get(), CheckFileBuffer);
  SourceMgr SM;
  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(CheckFileText, "CheckFile"),
                        SMLoc());
  Regex PrefixRE = FC.buildCheckPrefixRegex();
  if (FC.readCheckFile(SM, CheckFileText, PrefixRE))
    return false;

  auto OutBuffer = OutputBuf->getBuffer();
  SM.AddNewSourceBuffer(std::move(OutputBuf), SMLoc());
  return FC.checkInput(SM, OutBuffer);
}
#endif
