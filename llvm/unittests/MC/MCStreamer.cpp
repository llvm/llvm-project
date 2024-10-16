//===- llvm/unittest/MC/MCStreamer.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MCStreamerTest : public ::testing::Test {
public:
  static constexpr char TripleName[] = "x86_64-pc-linux";
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCSubtargetInfo> STI;
  const Target *TheTarget;

  struct StreamerContext {
    std::unique_ptr<MCContext> Ctx;
    std::unique_ptr<const MCInstrInfo> MII;
    std::unique_ptr<MCInstPrinter> Printer;
    std::unique_ptr<MCStreamer> Streamer;
  };

  MCStreamerTest() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    // If we didn't build the target, do not run the test.
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  }

  /// Create all data structures necessary to operate an assembler.
  StreamerContext createStreamer(raw_pwrite_stream &OS) {
    StreamerContext Res;
    Res.Ctx =
        std::make_unique<MCContext>(Triple(TripleName), MAI.get(), MRI.get(),
                                    /*MSTI=*/nullptr);
    Res.MII.reset(TheTarget->createMCInstrInfo());

    Res.Printer.reset(TheTarget->createMCInstPrinter(
        Triple(TripleName), MAI->getAssemblerDialect(), *MAI, *Res.MII, *MRI));

    MCCodeEmitter *MCE = TheTarget->createMCCodeEmitter(*Res.MII, *Res.Ctx);
    MCAsmBackend *MAB =
        TheTarget->createMCAsmBackend(*STI, *MRI, MCTargetOptions());
    std::unique_ptr<formatted_raw_ostream> Out(new formatted_raw_ostream(OS));
    Res.Streamer.reset(TheTarget->createAsmStreamer(
        *Res.Ctx, std::move(Out), Res.Printer.get(),
        std::unique_ptr<MCCodeEmitter>(MCE),
        std::unique_ptr<MCAsmBackend>(MAB)));
    return Res;
  }
};
} // namespace

TEST_F(MCStreamerTest, AnonymousGroup) {
  if (!MRI)
    GTEST_SKIP();

  SmallString<0> Out;
  raw_svector_ostream S(Out);
  StreamerContext C = createStreamer(S);

  // Test that switching to a group section with no associated signature name
  // doesn't crash.
  C.Streamer->switchSection(
      C.Ctx->getELFSection("foo", ELF::SHT_PROGBITS, ELF::SHF_GROUP));
  C.Streamer->finish();
}
