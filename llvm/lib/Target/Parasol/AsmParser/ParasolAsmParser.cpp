//===-- ParasolAsmParser.cpp - Parse Parasol asm to MCInst instructions----===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "ParasolInstrInfo.h"
#include "ParasolTargetMachine.h"
#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#define DEBUG_TYPE "mcasmparser"

using namespace llvm;

namespace {

class ParasolAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;
  MCInst MCB;
  bool InBrackets;

  MCAsmParser &getParser() const { return Parser; }
  MCAssembler *getAssembler() const {
    MCAssembler *Assembler = nullptr;
    // FIXME: need better way to detect AsmStreamer (upstream removed getKind())
    if (!Parser.getStreamer().hasRawTextSupport()) {
      MCELFStreamer *MES = static_cast<MCELFStreamer *>(&Parser.getStreamer());
      Assembler = &MES->getAssembler();
    }
    return Assembler;
  }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  bool equalIsAsmAssignment() override { return false; }
  bool isLabel(AsmToken &Token) override { llvm_unreachable("Unimplemented"); };

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }
  bool ParseDirectiveFalign(unsigned Size, SMLoc L);

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override {
    llvm_unreachable("Unimplemented");
  }
  bool ParseDirectiveSubsection(SMLoc L);
  bool ParseDirectiveComm(bool IsLocal, SMLoc L);
  bool RegisterMatchesArch(unsigned MatchNum) const;

  bool matchBundleOptions();
  bool handleNoncontigiousRegister(bool Contigious, SMLoc &Loc);
  bool finishBundle(SMLoc IDLoc, MCStreamer &Out);
  void canonicalizeImmediates(MCInst &MCI);
  bool matchOneInstruction(MCInst &MCB, SMLoc IDLoc,
                           OperandVector &InstOperands, uint64_t &ErrorInfo,
                           bool MatchingInlineAsm);
  void eatToEndOfPacket();
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override {
    llvm_unreachable("Unimplemented");
  }

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override {
    llvm_unreachable("Unimplemented");
  }
  bool OutOfRange(SMLoc IDLoc, long long Val, long long Max);
  int processInstruction(MCInst &Inst, OperandVector const &Operands,
                         SMLoc IDLoc);

  unsigned matchRegister(StringRef Name);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "ParasolGenAsmMatcher.inc"

  /// }

public:
  ParasolAsmParser(const MCSubtargetInfo &_STI, MCAsmParser &_Parser,
                   const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, _STI, MII), Parser(_Parser),
        InBrackets(false) {
    MCAsmParserExtension::Initialize(_Parser);
  }

  bool splitIdentifier(OperandVector &Operands);
  bool parseOperand(OperandVector &Operands);
  bool parseInstruction(OperandVector &Operands);
  bool implicitExpressionLocation(OperandVector &Operands);
  bool parseExpressionOrOperand(OperandVector &Operands);
  bool parseExpression(MCExpr const *&Expr);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override {
    llvm_unreachable("Unimplemented");
  }

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name, AsmToken ID,
                        OperandVector &Operands) override {
    llvm_unreachable("Unimplemented");
  }

  bool ParseDirective(AsmToken DirectiveID) override {
    llvm_unreachable("Unimplemented");
  }
};
} // namespace

bool ParasolAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                     SMLoc &EndLoc) {
  llvm_unreachable("unimplemented");
}

void ParasolAsmParser::convertToMapAndConstraints(
    unsigned Kind, const OperandVector &Operands) {
  llvm_unreachable("unimplemented");
}
/// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolAsmParser() {
  RegisterMCAsmParser<ParasolAsmParser> X(getTheParasolTarget());
}
