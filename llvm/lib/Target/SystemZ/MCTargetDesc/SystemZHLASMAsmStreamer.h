//===- SystemZHLASMAsmStreamer.h - HLASM Assembly Text Output ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZHLASMAsmStreamer class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

class SystemZHLASMAsmStreamer final : public MCStreamer {
  std::unique_ptr<formatted_raw_ostream> OSOwner;
  formatted_raw_ostream &OS;
  const MCAsmInfo *MAI;
  std::unique_ptr<MCInstPrinter> InstPrinter;
  std::unique_ptr<MCAssembler> Assembler;
  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;
  raw_null_ostream NullStream;

  bool IsVerboseAsm = false;

public:
  SystemZHLASMAsmStreamer(MCContext &Context,
                          std::unique_ptr<formatted_raw_ostream> os,
                          MCInstPrinter *printer,
                          std::unique_ptr<MCCodeEmitter> emitter,
                          std::unique_ptr<MCAsmBackend> asmbackend)
      : MCStreamer(Context), OSOwner(std::move(os)), OS(*OSOwner),
        MAI(Context.getAsmInfo()), InstPrinter(printer),
        Assembler(std::make_unique<MCAssembler>(
            Context, std::move(asmbackend), std::move(emitter),
            (asmbackend) ? asmbackend->createObjectWriter(NullStream)
                         : nullptr)),
        CommentStream(CommentToEmit) {
    assert(InstPrinter);
    if (Assembler->getBackendPtr())
      setAllowAutoPadding(Assembler->getBackend().allowAutoPadding());

    Context.setUseNamesOnTempLabels(true);
    auto *TO = Context.getTargetOptions();
    if (!TO)
      return;
    IsVerboseAsm = TO->AsmVerbose;
    if (IsVerboseAsm)
      InstPrinter->setCommentStream(CommentStream);
  }

  MCAssembler &getAssembler() { return *Assembler; }

  /// Return true if this streamer supports verbose assembly at all.
  bool isVerboseAsm() const override { return IsVerboseAsm; }

  /// Do we support EmitRawText?
  bool hasRawTextSupport() const override { return true; }

  /// @name MCStreamer Interface
  /// @{

  void changeSection(MCSection *Section, uint32_t Subsection) override;

  void emitLabel(MCSymbol *Symbol, SMLoc Loc) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return false;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}

  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, Align ByteAlignment = Align(1),
                    SMLoc Loc = SMLoc()) override {}

  /// @}
};
