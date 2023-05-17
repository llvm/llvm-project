//===----------------------- CodeRegionGenerator.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares classes responsible for generating llvm-mca
/// CodeRegions from various types of input. llvm-mca only analyzes CodeRegions,
/// so the classes here provide the input-to-CodeRegions translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H
#define LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H

#include "CodeRegion.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace llvm {
namespace mca {

class MCACommentConsumer : public AsmCommentConsumer {
protected:
  bool FoundError;

public:
  MCACommentConsumer() : FoundError(false) {}

  bool hadErr() const { return FoundError; }
};

/// A comment consumer that parses strings.  The only valid tokens are strings.
class AnalysisRegionCommentConsumer : public MCACommentConsumer {
  AnalysisRegions &Regions;

public:
  AnalysisRegionCommentConsumer(AnalysisRegions &R) : Regions(R) {}

  /// Parses a comment. It begins a new region if it is of the form
  /// LLVM-MCA-BEGIN. It ends a region if it is of the form LLVM-MCA-END.
  /// Regions can be optionally named if they are of the form
  /// LLVM-MCA-BEGIN <name> or LLVM-MCA-END <name>. Subregions are
  /// permitted, but a region that begins while another region is active
  /// must be ended before the outer region is ended. If thre is only one
  /// active region, LLVM-MCA-END does not need to provide a name.
  void HandleComment(SMLoc Loc, StringRef CommentText) override;
};

/// A comment consumer that parses strings to create InstrumentRegions.
/// The only valid tokens are strings.
class InstrumentRegionCommentConsumer : public MCACommentConsumer {
  llvm::SourceMgr &SM;

  InstrumentRegions &Regions;

  InstrumentManager &IM;

public:
  InstrumentRegionCommentConsumer(llvm::SourceMgr &SM, InstrumentRegions &R,
                                  InstrumentManager &IM)
      : SM(SM), Regions(R), IM(IM) {}

  /// Parses a comment. It begins a new region if it is of the form
  /// LLVM-MCA-<INSTRUMENTATION_TYPE> <data> where INSTRUMENTATION_TYPE
  /// is a valid InstrumentKind. If there is already an active
  /// region of type INSTRUMENATION_TYPE, then it will end the active
  /// one and begin a new one using the new data.
  void HandleComment(SMLoc Loc, StringRef CommentText) override;
};

/// This abstract class is responsible for parsing the input given to
/// the llvm-mca driver, and converting that into a CodeRegions instance.
class CodeRegionGenerator {
protected:
  CodeRegionGenerator(const CodeRegionGenerator &) = delete;
  CodeRegionGenerator &operator=(const CodeRegionGenerator &) = delete;
  virtual Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) = 0;

public:
  CodeRegionGenerator() {}
  virtual ~CodeRegionGenerator();
};

/// Abastract CodeRegionGenerator with AnalysisRegions member
class AnalysisRegionGenerator : public virtual CodeRegionGenerator {
protected:
  AnalysisRegions Regions;

public:
  AnalysisRegionGenerator(llvm::SourceMgr &SM) : Regions(SM) {}

  virtual Expected<const AnalysisRegions &>
  parseAnalysisRegions(const std::unique_ptr<MCInstPrinter> &IP) = 0;
};

/// Abstract CodeRegionGenerator with InstrumentRegionsRegions member
class InstrumentRegionGenerator : public virtual CodeRegionGenerator {
protected:
  InstrumentRegions Regions;

public:
  InstrumentRegionGenerator(llvm::SourceMgr &SM) : Regions(SM) {}

  virtual Expected<const InstrumentRegions &>
  parseInstrumentRegions(const std::unique_ptr<MCInstPrinter> &IP) = 0;
};

/// This abstract class is responsible for parsing input ASM and
/// generating a CodeRegions instance.
class AsmCodeRegionGenerator : public virtual CodeRegionGenerator {
  const Target &TheTarget;
  MCContext &Ctx;
  const MCAsmInfo &MAI;
  const MCSubtargetInfo &STI;
  const MCInstrInfo &MCII;
  unsigned AssemblerDialect; // This is set during parsing.

public:
  AsmCodeRegionGenerator(const Target &T, MCContext &C, const MCAsmInfo &A,
                         const MCSubtargetInfo &S, const MCInstrInfo &I)
      : TheTarget(T), Ctx(C), MAI(A), STI(S), MCII(I), AssemblerDialect(0) {}

  virtual MCACommentConsumer *getCommentConsumer() = 0;
  virtual CodeRegions &getRegions() = 0;

  unsigned getAssemblerDialect() const { return AssemblerDialect; }
  Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) override;
};

class AsmAnalysisRegionGenerator final : public AnalysisRegionGenerator,
                                         public AsmCodeRegionGenerator {
  AnalysisRegionCommentConsumer CC;

public:
  AsmAnalysisRegionGenerator(const Target &T, llvm::SourceMgr &SM, MCContext &C,
                             const MCAsmInfo &A, const MCSubtargetInfo &S,
                             const MCInstrInfo &I)
      : AnalysisRegionGenerator(SM), AsmCodeRegionGenerator(T, C, A, S, I),
        CC(Regions) {}

  MCACommentConsumer *getCommentConsumer() override { return &CC; };
  CodeRegions &getRegions() override { return Regions; };

  Expected<const AnalysisRegions &>
  parseAnalysisRegions(const std::unique_ptr<MCInstPrinter> &IP) override {
    Expected<const CodeRegions &> RegionsOrErr = parseCodeRegions(IP);
    if (!RegionsOrErr)
      return RegionsOrErr.takeError();
    else
      return static_cast<const AnalysisRegions &>(*RegionsOrErr);
  }

  Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) override {
    return AsmCodeRegionGenerator::parseCodeRegions(IP);
  }
};

class AsmInstrumentRegionGenerator final : public InstrumentRegionGenerator,
                                           public AsmCodeRegionGenerator {
  InstrumentRegionCommentConsumer CC;

public:
  AsmInstrumentRegionGenerator(const Target &T, llvm::SourceMgr &SM,
                               MCContext &C, const MCAsmInfo &A,
                               const MCSubtargetInfo &S, const MCInstrInfo &I,
                               InstrumentManager &IM)
      : InstrumentRegionGenerator(SM), AsmCodeRegionGenerator(T, C, A, S, I),
        CC(SM, Regions, IM) {}

  MCACommentConsumer *getCommentConsumer() override { return &CC; };
  CodeRegions &getRegions() override { return Regions; };

  Expected<const InstrumentRegions &>
  parseInstrumentRegions(const std::unique_ptr<MCInstPrinter> &IP) override {
    Expected<const CodeRegions &> RegionsOrErr = parseCodeRegions(IP);
    if (!RegionsOrErr)
      return RegionsOrErr.takeError();
    else
      return static_cast<const InstrumentRegions &>(*RegionsOrErr);
  }

  Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) override {
    return AsmCodeRegionGenerator::parseCodeRegions(IP);
  }
};

} // namespace mca
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H
