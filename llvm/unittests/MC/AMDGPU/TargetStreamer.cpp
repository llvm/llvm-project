//===- llvm/unittest/unittests/MC/AMDGPU/TargetStreamer.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

static constexpr char TripleName[] = "amdgcn--amdpal";
static constexpr char CPUName[] = "gfx1201";

// Test that we can generate a data-only AMDGPU ELF directly using the MC layer,
// without asserting due to initializeTargetID() not being called.
TEST(AMDGPUTargetStreamer, ELFStreamerBasic) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUTarget();

  // Create TargetMachine.
  std::string errMsg;
  Triple Trpl(TripleName);
  const Target *T = TargetRegistry::lookupTarget(Trpl, errMsg);
  // Skip test if AMDGPU not built.
  if (!T)
    GTEST_SKIP();
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(Trpl, CPUName, "", {}, {}, {}, {}));

  const MCSubtargetInfo &STI = TM->getMCSubtargetInfo();
  const MCRegisterInfo &RI = TM->getMCRegisterInfo();
  const MCAsmInfo &AI = TM->getMCAsmInfo();

  MCContext Context(Trpl, AI, RI, STI);
  std::unique_ptr<MCObjectFileInfo> ObjectFileInfo(
      T->createMCObjectFileInfo(Context, /*PIC=*/false));
  Context.setObjectFileInfo(ObjectFileInfo.get());

  SmallString<256> OutBuffer;
  raw_svector_ostream Outs(OutBuffer);

  Expected<std::unique_ptr<MCStreamer>> StreamerOr =
      TM->createMCStreamer(Outs, {}, CodeGenFileType::ObjectFile, Context);
  if (Error Err = StreamerOr.takeError())
    report_fatal_error(StringRef(toString(std::move(Err))));
  std::unique_ptr<MCStreamer> Streamer = std::move(*StreamerOr);

  Streamer->initSections(STI);

  MCSymbol *TableSymbol = Context.getOrCreateSymbol("myData");
  Streamer->switchSection(ObjectFileInfo->getReadOnlySection());
  Streamer->emitValueToAlignment(Align(4));
  Streamer->emitSymbolAttribute(TableSymbol, MCSA_Global);
  Streamer->emitSymbolAttribute(TableSymbol, MCSA_ELF_TypeObject);
  Streamer->emitLabel(TableSymbol);

  StringRef DataBlob = "hatstand";
  Streamer->emitBytes(DataBlob);
  Streamer->emitELFSize(TableSymbol,
                        MCConstantExpr::create(DataBlob.size(), Context));
  Streamer->finish();

  EXPECT_NE(OutBuffer.size(), 0U);
}
