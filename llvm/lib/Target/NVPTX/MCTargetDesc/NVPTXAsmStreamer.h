//===-- NVPTXMCAsmStreamer.h - NVPTX asm streamer --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTXAsmStreamer class
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXASMSTREAMER_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXASMSTREAMER_H

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmStreamer.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {

class NVPTXAsmStreamer final : public MCAsmBaseStreamer {
  std::unique_ptr<formatted_raw_ostream> OSOwner;
  formatted_raw_ostream &OS;
  const MCAsmInfo *MAI;
  std::unique_ptr<MCInstPrinter> InstPrinter;
  std::unique_ptr<MCAssembler> Assembler;

  SmallString<128> ExplicitCommentToEmit;
  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;
  raw_null_ostream NullStream;

  bool EmittedSectionDirective = false;

  bool IsVerboseAsm = false;
  bool ShowInst = false;
  bool UseDwarfDirectory = false;

public:
  NVPTXAsmStreamer(MCContext &Context,
                   std::unique_ptr<formatted_raw_ostream> os,
                   std::unique_ptr<MCInstPrinter> printer,
                   std::unique_ptr<MCCodeEmitter> emitter,
                   std::unique_ptr<MCAsmBackend> asmbackend)
      : MCAsmBaseStreamer(Context), OSOwner(std::move(os)), OS(*OSOwner),
        MAI(Context.getAsmInfo()), InstPrinter(std::move(printer)),
        Assembler(std::make_unique<MCAssembler>(
            Context, std::move(asmbackend), std::move(emitter),
            (asmbackend) ? asmbackend->createObjectWriter(NullStream)
                         : nullptr)),
        CommentStream(CommentToEmit) {
    // assert(false && "Implement NVPTXAsmStreamer.");
    assert(InstPrinter);
    if (Assembler->getBackendPtr())
      setAllowAutoPadding(Assembler->getBackend().allowAutoPadding());

    Context.setUseNamesOnTempLabels(true);

    auto *TO = Context.getTargetOptions();
    IsVerboseAsm = TO->AsmVerbose;
    if (IsVerboseAsm)
      InstPrinter->setCommentStream(CommentStream);
    ShowInst = TO->ShowMCInst;
    switch (TO->MCUseDwarfDirectory) {
    case MCTargetOptions::DisableDwarfDirectory:
      UseDwarfDirectory = false;
      break;
    case MCTargetOptions::EnableDwarfDirectory:
      UseDwarfDirectory = true;
      break;
    case MCTargetOptions::DefaultDwarfDirectory:
      UseDwarfDirectory =
          Context.getAsmInfo()->enableDwarfFileDirectoryDefault();
      break;
    }
  }

  inline void EmitEOL() {
    // Dump Explicit Comments here.
    emitExplicitComments();
    // If we don't have any comments, just emit a \n.
    if (!IsVerboseAsm) {
      OS << '\n';
      return;
    }
    EmitCommentsAndEOL();
  }

  void EmitCommentsAndEOL();

  void emitExplicitComments() override;

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return false;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}

  void emitRawTextImpl(StringRef String) override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXASMSTREAMER_H
