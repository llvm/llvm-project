//===- llvm/unittest/unittests/MC/AMDGPU/Disassembler.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Disassembler.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;

static const char *symbolLookupCallback(void *DisInfo, uint64_t ReferenceValue,
                                        uint64_t *ReferenceType,
                                        uint64_t ReferencePC,
                                        const char **ReferenceName) {
  *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  return nullptr;
}

struct SymbolLookupRecorder {
  unsigned NumCalls = 0;
  uint64_t ReferenceValue = 0;
  uint64_t ReferencePC = 0;
  // What the callback answers.
  const char *Name = nullptr;
  const char *DemangledName = nullptr;
};

static const char *recordingSymbolLookupCallback(void *DisInfo,
                                                 uint64_t ReferenceValue,
                                                 uint64_t *ReferenceType,
                                                 uint64_t ReferencePC,
                                                 const char **ReferenceName) {
  auto *Recorder = static_cast<SymbolLookupRecorder *>(DisInfo);
  ++Recorder->NumCalls;
  Recorder->ReferenceValue = ReferenceValue;
  Recorder->ReferencePC = ReferencePC;
  if (Recorder->DemangledName) {
    *ReferenceType = LLVMDisassembler_ReferenceType_DeMangled_Name;
    *ReferenceName = Recorder->DemangledName;
  } else {
    *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  }
  return Recorder->Name;
}

static constexpr char TripleName[] = "amdgpu10.30--amdpal";
static constexpr char CPUName[] = "";

// Basic smoke test.
TEST(AMDGPUDisassembler, Basic) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  uint8_t Bytes[] = {0x04, 0x00, 0x80, 0xb0};
  uint8_t *BytesP = Bytes;
  const char OutStringSize = 100;
  char OutString[OutStringSize];
  LLVMDisasmContextRef DCR = LLVMCreateDisasmCPU(
      TripleName, CPUName, nullptr, 0, nullptr, symbolLookupCallback);

  // Skip test if AMDGPU not built.
  if (!DCR)
    GTEST_SKIP();

  size_t InstSize;
  unsigned NumBytes = sizeof(Bytes);
  unsigned PC = 0U;

  InstSize = LLVMDisasmInstruction(DCR, BytesP, NumBytes, PC, OutString,
                                   OutStringSize);
  EXPECT_EQ(InstSize, 4U);
  EXPECT_EQ(StringRef(OutString), "\ts_version UC_VERSION_GFX10");

  LLVMDisasmDispose(DCR);
}

// s_branch 0xfff0 at BranchPC jumps to BranchTarget.
static constexpr uint64_t BranchPC = 0x7d5210317ffc;
static constexpr uint64_t BranchTarget = 0x7d5210317fc0;

// Disassembles the branch through the C API. Returns std::nullopt if AMDGPU is
// not built.
static std::optional<std::string>
disassembleBranch(void *DisInfo, LLVMSymbolLookupCallback SymbolLookUp,
                  bool InstrComments) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  LLVMDisasmContextRef DCR = LLVMCreateDisasmCPU(TripleName, CPUName, DisInfo,
                                                 0, nullptr, SymbolLookUp);
  if (!DCR)
    return std::nullopt;
  if (InstrComments) {
    EXPECT_NE(
        LLVMSetDisasmOptions(DCR, LLVMDisassembler_Option_SetInstrComments), 0);
  }

  uint8_t Bytes[] = {0xf0, 0xff, 0x82, 0xbf};
  char OutString[100] = {};
  EXPECT_EQ(LLVMDisasmInstruction(DCR, Bytes, sizeof(Bytes), BranchPC,
                                  OutString, sizeof(OutString)),
            4U);
  LLVMDisasmDispose(DCR);
  return std::string(OutString);
}

TEST(AMDGPUDisassembler, SymbolLookupCallback) {
  SymbolLookupRecorder Recorder;
  std::optional<std::string> Output = disassembleBranch(
      &Recorder, recordingSymbolLookupCallback, /*InstrComments=*/false);
  if (!Output)
    GTEST_SKIP();

  EXPECT_EQ(Recorder.NumCalls, 1U);
  EXPECT_EQ(Recorder.ReferenceValue, BranchTarget);
  EXPECT_EQ(Recorder.ReferencePC, BranchPC);
  EXPECT_EQ(*Output, "\ts_branch 65520 ; 0x7d5210317fc0");
}

TEST(AMDGPUDisassembler, SymbolLookupCallbackInstrComments) {
  SymbolLookupRecorder Recorder;
  std::optional<std::string> Output = disassembleBranch(
      &Recorder, recordingSymbolLookupCallback, /*InstrComments=*/true);
  if (!Output)
    GTEST_SKIP();

  EXPECT_EQ(Recorder.NumCalls, 1U);
  // The comment is padded out to the comment column.
  StringRef Rest(*Output);
  EXPECT_TRUE(Rest.consume_front("\ts_branch 65520"));
  EXPECT_EQ(Rest.ltrim(' '), "; 0x7d5210317fc0");
}

TEST(AMDGPUDisassembler, SymbolLookupCallbackName) {
  SymbolLookupRecorder Recorder;
  Recorder.Name = "target";
  std::optional<std::string> Output = disassembleBranch(
      &Recorder, recordingSymbolLookupCallback, /*InstrComments=*/false);
  if (!Output)
    GTEST_SKIP();

  EXPECT_EQ(*Output, "\ts_branch 65520 ; target");
}

TEST(AMDGPUDisassembler, SymbolLookupCallbackDemangledName) {
  SymbolLookupRecorder Recorder;
  Recorder.Name = "_Z6targetv";
  Recorder.DemangledName = "target()";
  std::optional<std::string> Output = disassembleBranch(
      &Recorder, recordingSymbolLookupCallback, /*InstrComments=*/false);
  if (!Output)
    GTEST_SKIP();

  EXPECT_EQ(*Output, "\ts_branch 65520 ; target()");
}

// llvm-objdump and Mesa pass a SectionSymbolsTy as DisInfo and no callbacks.
TEST(AMDGPUDisassembler, SectionSymbolsWithoutCallbacks) {
  SectionSymbolsTy Symbols;
  Symbols.emplace_back(BranchTarget, "target", ELF::STT_NOTYPE);
  std::optional<std::string> Output =
      disassembleBranch(&Symbols, nullptr, /*InstrComments=*/false);
  if (!Output)
    GTEST_SKIP();

  EXPECT_EQ(*Output, "\ts_branch target");
}

// Check multiple disassemblers in same MCContext.
TEST(AMDGPUDisassembler, MultiDisassembler) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  std::string Error;
  Triple TT(TripleName);
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  // Skip test if AMDGPU not built.
  if (!TheTarget)
    GTEST_SKIP();

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TT, CPUName, ""));
  auto Ctx = std::make_unique<MCContext>(TT, *MAI, *MRI, *STI);

  int AsmPrinterVariant = MAI->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(
      TheTarget->createMCInstPrinter(TT, AsmPrinterVariant, *MAI, *MII, *MRI));

  SmallVector<char, 64> InsnStr, AnnoStr;
  raw_svector_ostream OS(InsnStr);
  raw_svector_ostream Annotations(AnnoStr);
  formatted_raw_ostream FormattedOS(OS);

  char StrBuffer[128];

  uint8_t Bytes[] = {0x04, 0x00, 0x80, 0xb0};
  uint64_t InstSize = 0U;
  MCInst Inst1, Inst2;
  MCDisassembler::DecodeStatus Status;

  // Test disassembler works as expected.
  AnnoStr.clear();
  InsnStr.clear();
  std::unique_ptr<MCDisassembler> DisAsm1(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  Status = DisAsm1->getInstruction(Inst1, InstSize, Bytes, 0, Annotations);
  ASSERT_TRUE(Status == MCDisassembler::Success);
  EXPECT_EQ(InstSize, 4U);

  IP->printInst(&Inst1, 0U, Annotations.str(), *STI, FormattedOS);
  ASSERT_TRUE(InsnStr.size() < (sizeof(StrBuffer) - 1));
  std::memcpy(StrBuffer, InsnStr.data(), InsnStr.size());
  StrBuffer[InsnStr.size()] = '\0';
  EXPECT_EQ(StringRef(StrBuffer), "\ts_version UC_VERSION_GFX10");

  // Test that second disassembler in same context works as expected.
  AnnoStr.clear();
  InsnStr.clear();
  std::unique_ptr<MCDisassembler> DisAsm2(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  Status = DisAsm2->getInstruction(Inst2, InstSize, Bytes, 0, Annotations);
  ASSERT_TRUE(Status == MCDisassembler::Success);
  EXPECT_EQ(InstSize, 4U);

  IP->printInst(&Inst2, 0U, Annotations.str(), *STI, FormattedOS);
  ASSERT_TRUE(InsnStr.size() < (sizeof(StrBuffer) - 1));
  std::memcpy(StrBuffer, InsnStr.data(), InsnStr.size());
  StrBuffer[InsnStr.size()] = '\0';
  EXPECT_EQ(StringRef(StrBuffer), "\ts_version UC_VERSION_GFX10");
}

// Test UC_VERSION symbols can be overriden without crashing.
// There is no valid behaviour if symbols are redefined in this way.
TEST(AMDGPUDisassembler, UCVersionOverride) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  Triple TT(TripleName);

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);

  // Skip test if AMDGPU not built.
  if (!TheTarget)
    GTEST_SKIP();

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TT, CPUName, ""));
  auto Ctx = std::make_unique<MCContext>(TT, *MAI, *MRI, *STI);

  // Define custom UC_VERSION before initializing disassembler.
  const uint8_t UC_VERSION_GFX10_DEFAULT = 0x04;
  const uint8_t UC_VERSION_GFX10_NEW = 0x99;
  auto Sym = Ctx->getOrCreateSymbol("UC_VERSION_GFX10");
  Sym->setVariableValue(MCConstantExpr::create(UC_VERSION_GFX10_NEW, *Ctx));

  int AsmPrinterVariant = MAI->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(
      TheTarget->createMCInstPrinter(TT, AsmPrinterVariant, *MAI, *MII, *MRI));

  testing::internal::CaptureStderr();
  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  std::string Output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(Output.find("<unknown>:0: warning: unsupported redefinition of "
                          "UC_VERSION_GFX10") != std::string::npos);

  SmallVector<char, 64> InsnStr, AnnoStr;
  raw_svector_ostream OS(InsnStr);
  raw_svector_ostream Annotations(AnnoStr);
  formatted_raw_ostream FormattedOS(OS);

  char StrBuffer[128];

  // Decode S_VERSION instruction with original or custom version.
  uint8_t Versions[] = {UC_VERSION_GFX10_DEFAULT, UC_VERSION_GFX10_NEW};
  for (uint8_t Version : Versions) {
    uint8_t Bytes[] = {Version, 0x00, 0x80, 0xb0};
    uint64_t InstSize = 0U;
    MCInst Inst;

    AnnoStr.clear();
    InsnStr.clear();
    MCDisassembler::DecodeStatus Status =
        DisAsm->getInstruction(Inst, InstSize, Bytes, 0, Annotations);
    ASSERT_TRUE(Status == MCDisassembler::Success);
    EXPECT_EQ(InstSize, 4U);

    IP->printInst(&Inst, 0, Annotations.str(), *STI, FormattedOS);
    ASSERT_TRUE(InsnStr.size() < (sizeof(StrBuffer) - 1));
    std::memcpy(StrBuffer, InsnStr.data(), InsnStr.size());
    StrBuffer[InsnStr.size()] = '\0';

    if (Version == UC_VERSION_GFX10_DEFAULT)
      EXPECT_EQ(StringRef(StrBuffer), "\ts_version UC_VERSION_GFX10");
    else
      EXPECT_EQ(StringRef(StrBuffer), "\ts_version 153");
  }
}
