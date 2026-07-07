//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// LFI-specific MC implementation.
///
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLFI.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

static const char NoteNamespace[] = "LFI";

namespace llvm {

cl::opt<bool> FlagEnableRewriting("lfi-enable-rewriter",
                                  cl::desc("Enable rewriting for LFI."),
                                  cl::init(true), cl::Hidden);

void initializeLFIMCStreamer(MCStreamer &Streamer, MCContext &Ctx,
                             const Triple &TheTriple) {
  assert(TheTriple.isLFI());

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);

  // Create the target-specific MCLFIRewriter.
  assert(TheTarget != nullptr);
  if (FlagEnableRewriting) {
    auto MRI =
        std::unique_ptr<MCRegisterInfo>(TheTarget->createMCRegInfo(TheTriple));
    auto MII = std::unique_ptr<MCInstrInfo>(TheTarget->createMCInstrInfo());
    Streamer.setLFIRewriter(std::unique_ptr<MCLFIRewriter>(
        TheTarget->createMCLFIRewriter(Ctx, std::move(MRI), std::move(MII))));
  }
}

void emitLFINoteSection(MCStreamer &Streamer, MCContext &Ctx) {
  const Triple &TheTriple = Ctx.getTargetTriple();
  assert(TheTriple.isLFI());

  const char *NoteName;
  const char *NoteArch;
  switch (TheTriple.getArch()) {
  case Triple::aarch64:
    NoteName = ".note.LFI.ABI.aarch64";
    NoteArch = "aarch64";
    break;
  default:
    reportFatalUsageError("Unsupported architecture for LFI");
  }

  // Emit an ELF Note section in its own COMDAT group which identifies LFI
  // object files.
  MCSectionELF *Note = Ctx.getELFSection(NoteName, ELF::SHT_NOTE,
                                         ELF::SHF_ALLOC | ELF::SHF_GROUP, 0,
                                         NoteName, /*IsComdat=*/true);

  Streamer.switchSection(Note);
  Streamer.emitIntValue(strlen(NoteNamespace) + 1, 4);
  Streamer.emitIntValue(strlen(NoteArch) + 1, 4);
  Streamer.emitIntValue(ELF::NT_VERSION, 4);
  Streamer.emitBytes(NoteNamespace);
  Streamer.emitIntValue(0, 1); // NUL terminator
  Streamer.emitValueToAlignment(Align(4));
  Streamer.emitBytes(NoteArch);
  Streamer.emitIntValue(0, 1); // NUL terminator
  Streamer.emitValueToAlignment(Align(4));
}

} // namespace llvm
