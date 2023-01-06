//===-- SnippetFile.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SnippetFile.h"
#include "Error.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

namespace llvm {
namespace exegesis {
namespace {

// An MCStreamer that reads a BenchmarkCode definition from a file.
class BenchmarkCodeStreamer : public MCStreamer, public AsmCommentConsumer {
public:
  explicit BenchmarkCodeStreamer(
      MCContext *Context, const DenseMap<StringRef, unsigned> &RegNameToRegNo,
      BenchmarkCode *Result)
      : MCStreamer(*Context), RegNameToRegNo(RegNameToRegNo), Result(Result) {}

  // Implementation of the MCStreamer interface. We only care about
  // instructions.
  void emitInstruction(const MCInst &Instruction,
                       const MCSubtargetInfo &STI) override {
    Result->Key.Instructions.push_back(Instruction);
  }

  // Implementation of the AsmCommentConsumer.
  void HandleComment(SMLoc Loc, StringRef CommentText) override {
    CommentText = CommentText.trim();
    if (!CommentText.consume_front("LLVM-EXEGESIS-"))
      return;
    if (CommentText.consume_front("DEFREG")) {
      // LLVM-EXEGESIS-DEFREF <reg> <hex_value>
      RegisterValue RegVal;
      SmallVector<StringRef, 2> Parts;
      CommentText.split(Parts, ' ', /*unlimited splits*/ -1,
                        /*do not keep empty strings*/ false);
      if (Parts.size() != 2) {
        errs() << "invalid comment 'LLVM-EXEGESIS-DEFREG " << CommentText
               << "', expected two parameters <REG> <HEX_VALUE>\n";
        ++InvalidComments;
        return;
      }
      if (!(RegVal.Register = findRegisterByName(Parts[0].trim()))) {
        errs() << "unknown register '" << Parts[0]
               << "' in 'LLVM-EXEGESIS-DEFREG " << CommentText << "'\n";
        ++InvalidComments;
        return;
      }
      const StringRef HexValue = Parts[1].trim();
      RegVal.Value = APInt(
          /* each hex digit is 4 bits */ HexValue.size() * 4, HexValue, 16);
      Result->Key.RegisterInitialValues.push_back(std::move(RegVal));
      return;
    }
    if (CommentText.consume_front("LIVEIN")) {
      // LLVM-EXEGESIS-LIVEIN <reg>
      const auto RegName = CommentText.ltrim();
      if (unsigned Reg = findRegisterByName(RegName))
        Result->LiveIns.push_back(Reg);
      else {
        errs() << "unknown register '" << RegName
               << "' in 'LLVM-EXEGESIS-LIVEIN " << CommentText << "'\n";
        ++InvalidComments;
      }
      return;
    }
  }

  unsigned numInvalidComments() const { return InvalidComments; }

private:
  // We only care about instructions, we don't implement this part of the API.
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return false;
  }
  void emitValueToAlignment(Align Alignment, int64_t Value, unsigned ValueSize,
                            unsigned MaxBytesToEmit) override {}
  void emitZerofill(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    Align ByteAlignment, SMLoc Loc) override {}

  unsigned findRegisterByName(const StringRef RegName) const {
    auto Iter = RegNameToRegNo.find(RegName);
    if (Iter != RegNameToRegNo.end())
      return Iter->second;
    errs() << "'" << RegName
           << "' is not a valid register name for the target\n";
    return 0;
  }

  const DenseMap<StringRef, unsigned> &RegNameToRegNo;
  BenchmarkCode *const Result;
  unsigned InvalidComments = 0;
};

} // namespace

// Reads code snippets from file `Filename`.
Expected<std::vector<BenchmarkCode>> readSnippets(const LLVMState &State,
                                                  StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferPtr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BufferPtr.getError()) {
    return make_error<Failure>("cannot read snippet: " + Filename + ": " +
                               EC.message());
  }
  SourceMgr SM;
  SM.AddNewSourceBuffer(std::move(BufferPtr.get()), SMLoc());

  BenchmarkCode Result;

  const TargetMachine &TM = State.getTargetMachine();
  MCContext Context(TM.getTargetTriple(), TM.getMCAsmInfo(),
                    TM.getMCRegisterInfo(), TM.getMCSubtargetInfo());
  std::unique_ptr<MCObjectFileInfo> ObjectFileInfo(
      TM.getTarget().createMCObjectFileInfo(Context, /*PIC=*/false));
  Context.setObjectFileInfo(ObjectFileInfo.get());
  Context.initInlineSourceManager();
  BenchmarkCodeStreamer Streamer(&Context, State.getRegNameToRegNoMapping(),
                                 &Result);

  std::string Error;
  raw_string_ostream ErrorStream(Error);
  formatted_raw_ostream InstPrinterOStream(ErrorStream);
  const std::unique_ptr<MCInstPrinter> InstPrinter(
      TM.getTarget().createMCInstPrinter(
          TM.getTargetTriple(), TM.getMCAsmInfo()->getAssemblerDialect(),
          *TM.getMCAsmInfo(), *TM.getMCInstrInfo(), *TM.getMCRegisterInfo()));
  // The following call will take care of calling Streamer.setTargetStreamer.
  TM.getTarget().createAsmTargetStreamer(Streamer, InstPrinterOStream,
                                         InstPrinter.get(),
                                         TM.Options.MCOptions.AsmVerbose);
  if (!Streamer.getTargetStreamer())
    return make_error<Failure>("cannot create target asm streamer");

  const std::unique_ptr<MCAsmParser> AsmParser(
      createMCAsmParser(SM, Context, Streamer, *TM.getMCAsmInfo()));
  if (!AsmParser)
    return make_error<Failure>("cannot create asm parser");
  AsmParser->getLexer().setCommentConsumer(&Streamer);

  const std::unique_ptr<MCTargetAsmParser> TargetAsmParser(
      TM.getTarget().createMCAsmParser(*TM.getMCSubtargetInfo(), *AsmParser,
                                       *TM.getMCInstrInfo(),
                                       MCTargetOptions()));

  if (!TargetAsmParser)
    return make_error<Failure>("cannot create target asm parser");
  AsmParser->setTargetParser(*TargetAsmParser);

  if (AsmParser->Run(false))
    return make_error<Failure>("cannot parse asm file");
  if (Streamer.numInvalidComments())
    return make_error<Failure>(Twine("found ")
                                   .concat(Twine(Streamer.numInvalidComments()))
                                   .concat(" invalid LLVM-EXEGESIS comments"));
  return std::vector<BenchmarkCode>{std::move(Result)};
}

} // namespace exegesis
} // namespace llvm
