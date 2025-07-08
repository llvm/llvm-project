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

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZHLASMASMSTREAMER_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZHLASMASMSTREAMER_H

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

namespace llvm {

class SystemZHLASMAsmStreamer final : public MCStreamer {
  constexpr static size_t InstLimit = 80;
  constexpr static size_t ContIndicatorColumn = 72;
  constexpr static size_t ContStartColumn = 15;
  constexpr static size_t ContLen = ContIndicatorColumn - ContStartColumn;
  std::unique_ptr<formatted_raw_ostream> FOSOwner;
  formatted_raw_ostream &FOS;
  std::string Str;
  raw_string_ostream OS;
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
                          std::unique_ptr<MCInstPrinter> printer,
                          std::unique_ptr<MCCodeEmitter> emitter,
                          std::unique_ptr<MCAsmBackend> asmbackend)
      : MCStreamer(Context), FOSOwner(std::move(os)), FOS(*FOSOwner), OS(Str),
        MAI(Context.getAsmInfo()), InstPrinter(std::move(printer)),
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

  void EmitEOL();
  void EmitComment();

  /// Add a comment that can be emitted to the generated .s file to make the
  /// output of the compiler more readable. This only affects the MCAsmStreamer
  /// and only when verbose assembly output is enabled.
  void AddComment(const Twine &T, bool EOL = true) override;

  void emitBytes(StringRef Data) override;

  void emitAlignmentDS(uint64_t ByteAlignment, std::optional<int64_t> Value,
                       unsigned ValueSize, unsigned MaxBytesToEmit);
  void emitValueToAlignment(Align Alignment, int64_t Value = 0,
                            unsigned ValueSize = 1,
                            unsigned MaxBytesToEmit = 0) override;

  void emitCodeAlignment(Align Alignment, const MCSubtargetInfo *STI,
                         unsigned MaxBytesToEmit = 0) override;

  /// Return true if this streamer supports verbose assembly at all.
  bool isVerboseAsm() const override { return IsVerboseAsm; }

  /// Do we support EmitRawText?
  bool hasRawTextSupport() const override { return true; }

  /// @name MCStreamer Interface
  /// @{

  void changeSection(MCSection *Section, uint32_t Subsection) override;

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void emitLabel(MCSymbol *Symbol, SMLoc Loc) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return false;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}

  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, Align ByteAlignment = Align(1),
                    SMLoc Loc = SMLoc()) override {}
  void emitRawTextImpl(StringRef String) override;
  void emitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) override;

  void emitHLASMValueImpl(const MCExpr *Value, unsigned Size,
                          bool Parens = false);
  /// @}

  void emitEnd();
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZHLASMASMSTREAMER_H
