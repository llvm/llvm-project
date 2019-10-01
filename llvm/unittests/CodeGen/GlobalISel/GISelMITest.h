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
#include "llvm/Support/FileCheck.h"
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

/// Create a TargetMachine. As we lack a dedicated always available target for
/// unittests, we go for "AArch64".
static std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  Triple TargetTriple("aarch64--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<LLVMTargetMachine>(
      static_cast<LLVMTargetMachine *>(T->createTargetMachine(
          "AArch64", "", "", Options, None, None, CodeGenOpt::Aggressive)));
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
                  StringRef MIRFunc) {
  SmallString<512> S;
  StringRef MIRString = (Twine(R"MIR(
---
...
name: func
registers:
  - { id: 0, class: _ }
  - { id: 1, class: _ }
  - { id: 2, class: _ }
  - { id: 3, class: _ }
body: |
  bb.1:
    %0(s64) = COPY $x0
    %1(s64) = COPY $x1
    %2(s64) = COPY $x2
)MIR") + Twine(MIRFunc) + Twine("...\n"))
                            .toNullTerminatedStringRef(S);
  std::unique_ptr<MIRParser> MIR;
  auto MMI = make_unique<MachineModuleInfo>(&TM);
  std::unique_ptr<Module> M =
      parseMIR(Context, MIR, TM, MIRString, "func", *MMI);
  return make_pair(std::move(M), std::move(MMI));
}

static MachineFunction *getMFFromMMI(const Module *M,
                                     const MachineModuleInfo *MMI) {
  Function *F = M->getFunction("func");
  auto *MF = MMI->getMachineFunction(*F);
  return MF;
}

static void collectCopies(SmallVectorImpl<unsigned> &Copies,
                          MachineFunction *MF) {
  for (auto &MBB : *MF)
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == TargetOpcode::COPY)
        Copies.push_back(MI.getOperand(0).getReg());
    }
}

class GISelMITest : public ::testing::Test {
protected:
  GISelMITest() : ::testing::Test() {
    TM = createTargetMachine();
    if (!TM)
      return;
    ModuleMMIPair = createDummyModule(Context, *TM, "");
    MF = getMFFromMMI(ModuleMMIPair.first.get(), ModuleMMIPair.second.get());
    collectCopies(Copies, MF);
    EntryMBB = &*MF->begin();
    B.setMF(*MF);
    MRI = &MF->getRegInfo();
    B.setInsertPt(*EntryMBB, EntryMBB->end());
  }
  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  MachineFunction *MF;
  std::pair<std::unique_ptr<Module>, std::unique_ptr<MachineModuleInfo>>
      ModuleMMIPair;
  SmallVector<unsigned, 4> Copies;
  MachineBasicBlock *EntryMBB;
  MachineIRBuilder B;
  MachineRegisterInfo *MRI;
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
  std::vector<FileCheckString> CheckStrings;
  if (FC.ReadCheckFile(SM, CheckFileText, PrefixRE, CheckStrings))
    return false;

  auto OutBuffer = OutputBuf->getBuffer();
  SM.AddNewSourceBuffer(std::move(OutputBuf), SMLoc());
  return FC.CheckInput(SM, OutBuffer, CheckStrings);
}
#endif
