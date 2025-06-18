//===-- ParasolAsmParser.cpp - Parse Parasol asm to MCInst instructions----===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ParasolInstPrinter.h"
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

/// ARMOperand - Instances of this class represent a parsed ARM machine
/// operand.
class ParasolOperand : public MCParsedAsmOperand {
  enum KindTy {
    k_Token,
    k_Register,
    k_Immediate

  } Kind;

  SMLoc StartLoc, EndLoc, AlignmentLoc;
  SmallVector<unsigned, 8> Registers;

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  union {
    struct TokOp Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
  };

public:
  ParasolOperand(KindTy K) : Kind(K) {}

  bool isImm() const override { return Kind == k_Immediate; }

  bool isToken() const override { return Kind == k_Token; }

  bool isMem() const override { return false; }

  bool isReg() const override { return Kind == k_Register; }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }

  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  /// getLocRange - Get the range between the first and last token of this
  /// operand.
  SMRange getLocRange() const { return SMRange(StartLoc, EndLoc); }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const override {
    assert(Kind == k_Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(isImm() && "Invalid access!");
    return Imm.Val;
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void print(raw_ostream &OS) const override;

  static std::unique_ptr<ParasolOperand> CreateToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<ParasolOperand>(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<ParasolOperand>
  createReg(unsigned RegNo, SMLoc S, SMLoc E, bool IsGPRAsFPR = false) {
    auto Op = std::make_unique<ParasolOperand>(k_Register);
    Op->Reg.RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
};

void ParasolOperand::print(raw_ostream &OS) const {
  auto RegName = [](MCRegister Reg) {
    if (Reg)
      return ParasolInstPrinter::getRegisterName(Reg);
    else
      return "noreg";
  };

  switch (Kind) {
  case k_Token:
    OS << "'" << getToken() << "'";
    break;
  case k_Register:
    OS << "<register " << RegName(getReg()) << ">";
    break;
  case k_Immediate:
    OS << *getImm();
    break;
  }
}

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
  SMLoc getLoc() const { return getParser().getTok().getLoc(); }
  bool equalIsAsmAssignment() override { return false; }
  bool isLabel(AsmToken &Token) override { llvm_unreachable("Unimplemented"); };

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }
  bool ParseDirectiveFalign(unsigned Size, SMLoc L);

  bool parseOperand(OperandVector &, StringRef Mnemonic);
  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;
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
                               bool MatchingInlineAsm) override;

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
  bool implicitExpressionLocation(OperandVector &Operands);
  bool parseExpressionOrOperand(OperandVector &Operands);
  bool parseExpression(MCExpr const *&Expr);

  ParseStatus parseRegister(OperandVector &Operands);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override {
    llvm_unreachable("Unimplemented");
  }
};
} // namespace

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "ParasolGenAsmMatcher.inc"

bool ParasolAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                               OperandVector &Operands,
                                               MCStreamer &Out,
                                               uint64_t &ErrorInfo,
                                               bool MatchingInlineAsm) {
  FeatureBitset MissingFeatures;
  MCInst Inst;

  auto Result = MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                     MatchingInlineAsm);

  switch (Result) {
  default:
    break;
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  }

  llvm_unreachable("Unknown match type detected!");
}

bool ParasolAsmParser::parseOperand(OperandVector &Operands,
                                    StringRef Mnemonic) {
  MCAsmParser &Parser = getParser();
  SMLoc S, E;

  if (parseRegister(Operands).isSuccess()) {
    return false;
  }

  return false;
}

bool ParasolAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                        StringRef Name, SMLoc NameLoc,
                                        OperandVector &Operands) {
  MCAsmParser &Parser = getParser();

  Operands.push_back(ParasolOperand::CreateToken(Name, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Name)) {
      return true;
    }

    while (parseOptionalToken(AsmToken::Comma)) {
      // Parse and remember the operand.
      if (parseOperand(Operands, Name)) {
        return true;
      }
    }
  }

  return false;
}

static MCRegister matchRegisterNameHelper(StringRef Name) {
  MCRegister Reg = MatchRegisterName(Name);
  assert(Reg >= Parasol::X0 && Reg <= Parasol::X63);

  if (!Reg)
    return Parasol::NoRegister;

  return Reg;
}

bool ParasolAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                     SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus ParasolAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                               SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  StringRef Name = getLexer().getTok().getIdentifier();

  Reg = matchRegisterNameHelper(Name);
  if (!Reg)
    return ParseStatus::NoMatch;

  getParser().Lex(); // Eat identifier token.
  return ParseStatus::Success;
}

/// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolAsmParser() {
  RegisterMCAsmParser<ParasolAsmParser> X(getTheParasolTarget());
}

ParseStatus ParasolAsmParser::parseRegister(OperandVector &Operands) {
  SMLoc FirstS = getLoc();
  bool HadParens = false;
  AsmToken LParen;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::Identifier:
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister RegNo = matchRegisterNameHelper(Name);

    SMLoc S = getLoc();
    SMLoc E = SMLoc::getFromPointer(S.getPointer() + Name.size());
    getLexer().Lex();
    Operands.push_back(ParasolOperand::createReg(RegNo, S, E));
  }

  return ParseStatus::Success;
}
