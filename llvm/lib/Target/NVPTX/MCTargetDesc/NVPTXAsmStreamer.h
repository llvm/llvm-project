//===-- NVPTXAsmStreamer.h - NVPTX asm streamer ----------------*- C++ -*--===//
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

  bool IsVerboseAsm = false;
  bool ShowInst = false;
  bool UseDwarfDirectory = false;
  bool EmittedSectionDirective = false;

  void PrintQuotedString(StringRef Data, raw_ostream &OS) const;
  void printDwarfFileDirective(unsigned FileNo, StringRef Directory,
                               StringRef Filename,
                               std::optional<MD5::MD5Result> Checksum,
                               std::optional<StringRef> Source,
                               bool UseDwarfDirectory,
                               raw_svector_ostream &OS) const;

  /// Helper to emit common .loc directive flags, isa, and discriminator.
  void emitDwarfLocDirectiveFlags(unsigned Flags, unsigned Isa,
                                  unsigned Discriminator);

  /// Helper to emit the common suffix of .loc directives (flags, comment, EOL,
  /// parent call).
  void emitDwarfLocDirectiveSuffix(unsigned FileNo, unsigned Line,
                                   unsigned Column, unsigned Flags,
                                   unsigned Isa, unsigned Discriminator,
                                   StringRef FileName, StringRef Comment);

public:
  NVPTXAsmStreamer(MCContext &Context,
                   std::unique_ptr<formatted_raw_ostream> os,
                   std::unique_ptr<MCInstPrinter> printer,
                   std::unique_ptr<MCCodeEmitter> emitter,
                   std::unique_ptr<MCAsmBackend> asmbackend);

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

  /// Return true if this streamer supports verbose assembly at all.
  bool isVerboseAsm() const override { return IsVerboseAsm; }

  bool hasRawTextSupport() const override { return true; }

  void AddComment(const Twine &T, bool EOL = true) override;

  /// Return a raw_ostream that comments can be written to.
  /// Unlike AddComment, you are required to terminate comments with \n if you
  /// use this method.
  raw_ostream &getCommentOS() override {
    if (!IsVerboseAsm)
      return nulls(); // Discard comments unless in verbose asm mode.
    return CommentStream;
  }

  void emitRawComment(const Twine &T, bool TabPrefix = true) override;

  void emitExplicitComments() override;

  void addBlankLine() override { EmitEOL(); }

  void switchSection(MCSection *Section, uint32_t Subsection) override;

  void emitBytes(StringRef Data) override;

  void emitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void emitIntValue(uint64_t Value, unsigned Size) override;

  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}

  void emitRelocDirective(const MCExpr &Offset, StringRef Name,
                          const MCExpr *Expr, SMLoc Loc) override;

  void emitRawTextImpl(StringRef String) override;

  Expected<unsigned> tryEmitDwarfFileDirective(
      unsigned FileNo, StringRef Directory, StringRef Filename,
      std::optional<MD5::MD5Result> Checksum = std::nullopt,
      std::optional<StringRef> Source = std::nullopt,
      unsigned CUID = 0) override;

  void emitDwarfFile0Directive(StringRef Directory, StringRef Filename,
                               std::optional<MD5::MD5Result> Checksum,
                               std::optional<StringRef> Source,
                               unsigned CUID = 0) override;

  void emitDwarfLocDirective(unsigned FileNo, unsigned Line, unsigned Column,
                             unsigned Flags, unsigned Isa,
                             unsigned Discriminator, StringRef FileName,
                             StringRef Location = {}) override;

  /// This is same as emitDwarfLocDirective, except also emits inlined function
  /// and inlined callsite information.
  void emitDwarfLocDirectiveWithInlinedAt(unsigned FileNo, unsigned Line,
                                          unsigned Column, unsigned FileIA,
                                          unsigned LineIA, unsigned ColIA,
                                          const MCSymbol *Sym, unsigned Flags,
                                          unsigned Isa, unsigned Discriminator,
                                          StringRef FileName,
                                          StringRef Comment = {}) override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXASMSTREAMER_H
