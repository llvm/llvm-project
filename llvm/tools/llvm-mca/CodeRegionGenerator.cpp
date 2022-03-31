//===----------------------- CodeRegionGenerator.cpp ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines classes responsible for generating llvm-mca
/// CodeRegions from various types of input. llvm-mca only analyzes CodeRegions,
/// so the classes here provide the input-to-CodeRegions translation.
//
//===----------------------------------------------------------------------===//

#include "CodeRegionGenerator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SMLoc.h"
#include <memory>

namespace llvm {
namespace mca {

// This virtual dtor serves as the anchor for the CodeRegionGenerator class.
CodeRegionGenerator::~CodeRegionGenerator() {}

// A comment consumer that parses strings.  The only valid tokens are strings.
class MCACommentConsumer : public AsmCommentConsumer {
public:
  CodeRegions &Regions;

  MCACommentConsumer(CodeRegions &R) : Regions(R) {}
  void HandleComment(SMLoc Loc, StringRef CommentText) override;
};

// This class provides the callbacks that occur when parsing input assembly.
class MCStreamerWrapper final : public MCStreamer {
  CodeRegions &Regions;

public:
  MCStreamerWrapper(MCContext &Context, mca::CodeRegions &R)
      : MCStreamer(Context), Regions(R) {}

  // We only want to intercept the emission of new instructions.
  virtual void emitInstruction(const MCInst &Inst,
                               const MCSubtargetInfo & /* unused */) override {
    Regions.addInstruction(Inst);
  }

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override {}
  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0,
                    SMLoc Loc = SMLoc()) override {}
  void emitGPRel32Value(const MCExpr *Value) override {}
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override {}
  void EmitCOFFSymbolStorageClass(int StorageClass) override {}
  void EmitCOFFSymbolType(int Type) override {}
  void EndCOFFSymbolDef() override {}

  ArrayRef<MCInst> GetInstructionSequence(unsigned Index) const {
    return Regions.getInstructionSequence(Index);
  }
};

void MCACommentConsumer::HandleComment(SMLoc Loc, StringRef CommentText) {
  // Skip empty comments.
  StringRef Comment(CommentText);
  if (Comment.empty())
    return;

  // Skip spaces and tabs.
  unsigned Position = Comment.find_first_not_of(" \t");
  if (Position >= Comment.size())
    // We reached the end of the comment. Bail out.
    return;

  Comment = Comment.drop_front(Position);
  if (Comment.consume_front("LLVM-MCA-END")) {
    // Skip spaces and tabs.
    Position = Comment.find_first_not_of(" \t");
    if (Position < Comment.size())
      Comment = Comment.drop_front(Position);
    Regions.endRegion(Comment, Loc);
    return;
  }

  // Try to parse the LLVM-MCA-BEGIN comment.
  if (!Comment.consume_front("LLVM-MCA-BEGIN"))
    return;

  // Skip spaces and tabs.
  Position = Comment.find_first_not_of(" \t");
  if (Position < Comment.size())
    Comment = Comment.drop_front(Position);
  // Use the rest of the string as a descriptor for this code snippet.
  Regions.beginRegion(Comment, Loc);
}

Expected<const CodeRegions &> AsmCodeRegionGenerator::parseCodeRegions(
    const std::unique_ptr<MCInstPrinter> &IP) {
  MCTargetOptions Opts;
  Opts.PreserveAsmComments = false;
  MCStreamerWrapper Str(Ctx, Regions);

  // Need to initialize an MCTargetStreamer otherwise
  // certain asm directives will cause a segfault.
  // Using nulls() so that anything emitted by the MCTargetStreamer
  // doesn't show up in the llvm-mca output.
  raw_ostream &OSRef = nulls();
  formatted_raw_ostream FOSRef(OSRef);
  TheTarget.createAsmTargetStreamer(Str, FOSRef, IP.get(),
                                    /*IsVerboseAsm=*/true);

  // Create a MCAsmParser and setup the lexer to recognize llvm-mca ASM
  // comments.
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(Regions.getSourceMgr(), Ctx, Str, MAI));
  MCAsmLexer &Lexer = Parser->getLexer();
  MCACommentConsumer CC(Regions);
  Lexer.setCommentConsumer(&CC);
  // Enable support for MASM literal numbers (example: 05h, 101b).
  Lexer.setLexMasmIntegers(true);

  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget.createMCAsmParser(STI, *Parser, MCII, Opts));
  if (!TAP)
    return make_error<StringError>(
        "This target does not support assembly parsing.",
        inconvertibleErrorCode());
  Parser->setTargetParser(*TAP);
  Parser->Run(false);

  // Set the assembler dialect from the input. llvm-mca will use this as the
  // default dialect when printing reports.
  AssemblerDialect = Parser->getAssemblerDialect();
  return Regions;
}

} // namespace mca
} // namespace llvm
