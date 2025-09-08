//===- llvm/unittest/unittests/MC/AMDGPU/Disassembler.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Disassembler.h"
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

using namespace llvm;

static const char *symbolLookupCallback(void *DisInfo, uint64_t ReferenceValue,
                                        uint64_t *ReferenceType,
                                        uint64_t ReferencePC,
                                        const char **ReferenceName) {
  *ReferenceType = LLVMDisassembler_ReferenceType_InOut_None;
  return nullptr;
}

static constexpr char TripleName[] = "amdgcn--amdpal";
static constexpr char CPUName[] = "gfx1030";

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

// Check multiple disassemblers in same MCContext.
TEST(AMDGPUDisassembler, MultiDisassembler) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);

  // Skip test if AMDGPU not built.
  if (!TheTarget)
    GTEST_SKIP();

  Triple TT(TripleName);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TT, CPUName, ""));
  auto Ctx = std::make_unique<MCContext>(TT, MAI.get(), MRI.get(), STI.get());

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

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);

  // Skip test if AMDGPU not built.
  if (!TheTarget)
    GTEST_SKIP();

  Triple TT(TripleName);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TT, CPUName, ""));
  auto Ctx = std::make_unique<MCContext>(TT, MAI.get(), MRI.get(), STI.get());

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
