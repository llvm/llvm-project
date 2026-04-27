//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Verify that .debug_frame emits distinct CIEs when frame parameters (e.g.
/// the return-address register) differ between functions.  This is a
/// regression test for a bug where only the first CIE was emitted for
/// .debug_frame, causing all FDEs to silently reuse that CIE's
/// return-address register regardless of per-function overrides via
/// emitCFIReturnColumn.
///
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class DwarfDebugFrameCIE : public ::testing::Test {
public:
  static constexpr const char *TripleName = "x86_64-pc-linux";
  Triple TT;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  MCTargetOptions MCOptions;
  const Target *TheTarget = nullptr;

  struct StreamerContext {
    std::unique_ptr<MCObjectFileInfo> MOFI;
    std::unique_ptr<MCContext> Ctx;
    std::unique_ptr<const MCInstrInfo> MII;
    std::unique_ptr<MCStreamer> Streamer;
  };

  DwarfDebugFrameCIE() : TT(TripleName) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "", ""));
  }

  StreamerContext createStreamer(raw_pwrite_stream &OS) {
    StreamerContext Res;
    Res.Ctx = std::make_unique<MCContext>(TT, *MAI, MRI.get(),
                                          /*MSTI=*/nullptr);
    Res.MOFI.reset(TheTarget->createMCObjectFileInfo(*Res.Ctx, /*PIC=*/false));
    Res.Ctx->setObjectFileInfo(Res.MOFI.get());

    Res.MII.reset(TheTarget->createMCInstrInfo());
    MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*Res.MII, *Res.Ctx);
    MCAsmBackend *MAB =
        TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());
    std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);
    Res.Streamer.reset(TheTarget->createMCObjectStreamer(
        TT, *Res.Ctx, std::unique_ptr<MCAsmBackend>(MAB), std::move(OW),
        std::unique_ptr<MCCodeEmitter>(MCE), *STI));
    return Res;
  }

  /// Enable .debug_frame emission (instead of the default .eh_frame).
  void enableDebugFrame(StreamerContext &C) {
    C.Streamer->emitCFISections(/*EH=*/false, /*Debug=*/true, /*SFrame=*/false);
  }

  /// Emit a mock function with the given return-address register in its CIE.
  void emitFunction(StreamerContext &C, StringRef Name, unsigned RAReg) {
    MCStreamer *S = C.Streamer.get();
    MCContext &Ctx = *C.Ctx;

    MCSection *TextSec = C.MOFI->getTextSection();
    TextSec->setHasInstructions(true);
    S->switchSection(TextSec);

    MCSymbol *FuncSym = Ctx.getOrCreateSymbol(Name);
    S->emitLabel(FuncSym);

    S->emitCFIStartProc(/*IsSimple=*/true);
    S->emitCFIReturnColumn(RAReg);
    S->emitNops(4, 1, SMLoc(), *STI);
    S->emitCFIEndProc();
  }
};

/// Test that two functions with different return-address registers produce
/// two distinct CIEs in .debug_frame.
TEST_F(DwarfDebugFrameCIE, DistinctReturnColumnsGetDistinctCIEs) {
  if (!TheTarget)
    GTEST_SKIP();

  SmallString<0> ObjContents;
  raw_svector_ostream VecOS(ObjContents);
  StreamerContext C = createStreamer(VecOS);

  C.Streamer->initSections(*STI);
  enableDebugFrame(C);

  // Function A: return-address register = 16 (x86_64 RA)
  emitFunction(C, "funcA", /*RAReg=*/16);
  // Function B: return-address register = 0 (different)
  emitFunction(C, "funcB", /*RAReg=*/0);

  C.Streamer->finish();

  // Parse the emitted ELF and find .debug_frame.
  std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(
      ObjContents.str(), "", /*RequiresNullTerminator=*/false);
  auto BinOrErr = llvm::object::createBinary(MB->getMemBufferRef());
  ASSERT_TRUE(static_cast<bool>(BinOrErr));
  auto *ELF = dyn_cast<llvm::object::ELFObjectFileBase>(&**BinOrErr);
  ASSERT_NE(ELF, nullptr);

  // Extract .debug_frame section contents.
  StringRef FrameContents;
  bool Found = false;
  for (const object::SectionRef &Section : ELF->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    ASSERT_TRUE(static_cast<bool>(NameOrErr));
    if (*NameOrErr == ".debug_frame") {
      Expected<StringRef> ContentsOrErr = Section.getContents();
      ASSERT_TRUE(static_cast<bool>(ContentsOrErr));
      FrameContents = *ContentsOrErr;
      Found = true;
      break;
    }
  }
  ASSERT_TRUE(Found) << ".debug_frame section not found";
  ASSERT_FALSE(FrameContents.empty());

  // Parse the .debug_frame section using DWARFDebugFrame.
  DWARFDataExtractor Data(FrameContents, /*isLittleEndian=*/true,
                          /*AddressSize=*/8);
  DWARFDebugFrame DebugFrame(Triple::x86_64, /*IsEH=*/false);
  Error Err = DebugFrame.parse(Data);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  // Collect CIEs and their return-address registers.
  SmallVector<uint64_t, 4> CIEReturnRegs;
  unsigned FDECount = 0;
  for (const dwarf::FrameEntry &Entry : DebugFrame) {
    if (const auto *CIEp = dyn_cast<dwarf::CIE>(&Entry))
      CIEReturnRegs.push_back(CIEp->getReturnAddressRegister());
    else if (isa<dwarf::FDE>(Entry))
      ++FDECount;
  }

  // We emitted two functions, so expect two FDEs.
  EXPECT_EQ(FDECount, 2u);
  // The two functions use different return-address registers, so there must
  // be two distinct CIEs.
  ASSERT_EQ(CIEReturnRegs.size(), 2u);
  EXPECT_NE(CIEReturnRegs[0], CIEReturnRegs[1]);
  // Verify both registers are present (order depends on CIEKey sort).
  llvm::sort(CIEReturnRegs);
  EXPECT_EQ(CIEReturnRegs[0], 0u);
  EXPECT_EQ(CIEReturnRegs[1], 16u);
}

/// Test that two functions with the same return-address register share a
/// single CIE (deduplication still works).
TEST_F(DwarfDebugFrameCIE, SameReturnColumnsShareCIE) {
  if (!TheTarget)
    GTEST_SKIP();

  SmallString<0> ObjContents;
  raw_svector_ostream VecOS(ObjContents);
  StreamerContext C = createStreamer(VecOS);

  C.Streamer->initSections(*STI);
  enableDebugFrame(C);

  emitFunction(C, "funcC", /*RAReg=*/16);
  emitFunction(C, "funcD", /*RAReg=*/16);

  C.Streamer->finish();

  std::unique_ptr<MemoryBuffer> MB =
      MemoryBuffer::getMemBuffer(ObjContents.str(), "", false);
  auto BinOrErr = llvm::object::createBinary(MB->getMemBufferRef());
  ASSERT_TRUE(static_cast<bool>(BinOrErr));
  auto *ELF = dyn_cast<llvm::object::ELFObjectFileBase>(&**BinOrErr);
  ASSERT_NE(ELF, nullptr);

  StringRef FrameContents;
  bool Found = false;
  for (const object::SectionRef &Section : ELF->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    ASSERT_TRUE(static_cast<bool>(NameOrErr));
    if (*NameOrErr == ".debug_frame") {
      Expected<StringRef> ContentsOrErr = Section.getContents();
      ASSERT_TRUE(static_cast<bool>(ContentsOrErr));
      FrameContents = *ContentsOrErr;
      Found = true;
      break;
    }
  }
  ASSERT_TRUE(Found) << ".debug_frame section not found";

  DWARFDataExtractor Data(FrameContents, true, 8);
  DWARFDebugFrame DebugFrame(Triple::x86_64, /*IsEH=*/false);
  Error Err = DebugFrame.parse(Data);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  unsigned CIECount = 0;
  unsigned FDECount = 0;
  for (const dwarf::FrameEntry &Entry : DebugFrame) {
    if (isa<dwarf::CIE>(Entry))
      ++CIECount;
    else if (isa<dwarf::FDE>(Entry))
      ++FDECount;
  }

  EXPECT_EQ(FDECount, 2u);
  // Same return-address register -> single shared CIE.
  EXPECT_EQ(CIECount, 1u);
}

} // namespace
