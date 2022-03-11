// LoongArchAsmParser.cpp - Parse LoongArch assembly to MCInst instructions -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/LoongArchInstPrinter.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "TargetInfo/LoongArchTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-asm-parser"

namespace {
struct LoongArchOperand;

class LoongArchAsmParser : public MCTargetAsmParser {
  SMLoc getLoc() const { return getParser().getTok().getLoc(); }

  /// Parse a register as used in CFI directives.
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  OperandMatchResultTy tryParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                        SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override { return true; }

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool generateImmOutOfRangeError(OperandVector &Operands, uint64_t ErrorInfo,
                                  int64_t Lower, int64_t Upper, Twine Msg);

  /// Helper for processing MC instructions that have been successfully matched
  /// by MatchAndEmitInstruction.
  bool processInstruction(MCInst &Inst, SMLoc IDLoc, OperandVector &Operands,
                          MCStreamer &Out);

// Auto-generated instruction matching functions.
#define GET_ASSEMBLER_HEADER
#include "LoongArchGenAsmMatcher.inc"

  OperandMatchResultTy parseRegister(OperandVector &Operands);
  OperandMatchResultTy parseImmediate(OperandVector &Operands);

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

public:
  enum LoongArchMatchResultTy {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "LoongArchGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  LoongArchAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                     const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    Parser.addAliasForDirective(".half", ".2byte");
    Parser.addAliasForDirective(".hword", ".2byte");
    Parser.addAliasForDirective(".word", ".4byte");
    Parser.addAliasForDirective(".dword", ".8byte");

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

/// LoongArchOperand - Instances of this class represent a parsed LoongArch
/// machine instruction.
class LoongArchOperand : public MCParsedAsmOperand {
  enum class KindTy {
    Token,
    Register,
    Immediate,
  } Kind;

  struct RegOp {
    MCRegister RegNum;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  SMLoc StartLoc, EndLoc;
  union {
    StringRef Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
  };

public:
  LoongArchOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  bool isToken() const override { return Kind == KindTy::Token; }
  bool isReg() const override { return Kind == KindTy::Register; }
  bool isImm() const override { return Kind == KindTy::Immediate; }
  bool isMem() const override { return false; }

  static bool evaluateConstantImm(const MCExpr *Expr, int64_t &Imm) {
    if (auto CE = dyn_cast<MCConstantExpr>(Expr)) {
      Imm = CE->getValue();
      return true;
    }

    return false;
  }

  template <unsigned N, int P = 0> bool isUImm() const {
    if (!isImm())
      return false;

    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isUInt<N>(Imm - P);
  }

  template <unsigned N, unsigned S = 0> bool isSImm() const {
    if (!isImm())
      return false;

    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isShiftedInt<N, S>(Imm);
  }

  bool isUImm2() const { return isUImm<2>(); }
  bool isUImm2plus1() const { return isUImm<2, 1>(); }
  bool isUImm3() const { return isUImm<3>(); }
  bool isUImm5() const { return isUImm<5>(); }
  bool isUImm6() const { return isUImm<6>(); }
  bool isUImm12() const { return isUImm<12>(); }
  bool isUImm15() const { return isUImm<15>(); }
  bool isSImm12() const { return isSImm<12>(); }
  bool isSImm14lsl2() const { return isSImm<14, 2>(); }
  bool isSImm16() const { return isSImm<16>(); }
  bool isSImm16lsl2() const { return isSImm<16, 2>(); }
  bool isSImm20() const { return isSImm<20>(); }
  bool isSImm21lsl2() const { return isSImm<21, 2>(); }
  bool isSImm26lsl2() const { return isSImm<26, 2>(); }

  /// Gets location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }
  /// Gets location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  unsigned getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum.id();
  }

  const MCExpr *getImm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val;
  }

  StringRef getToken() const {
    assert(Kind == KindTy::Token && "Invalid type access!");
    return Tok;
  }

  void print(raw_ostream &OS) const override {
    auto RegName = [](unsigned Reg) {
      if (Reg)
        return LoongArchInstPrinter::getRegisterName(Reg);
      else
        return "noreg";
    };

    switch (Kind) {
    case KindTy::Immediate:
      OS << *getImm();
      break;
    case KindTy::Register:
      OS << "<register " << RegName(getReg()) << ">";
      break;
    case KindTy::Token:
      OS << "'" << getToken() << "'";
      break;
    }
  }

  static std::unique_ptr<LoongArchOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<LoongArchOperand>(KindTy::Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<LoongArchOperand> createReg(unsigned RegNo, SMLoc S,
                                                     SMLoc E) {
    auto Op = std::make_unique<LoongArchOperand>(KindTy::Register);
    Op->Reg.RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<LoongArchOperand> createImm(const MCExpr *Val, SMLoc S,
                                                     SMLoc E) {
    auto Op = std::make_unique<LoongArchOperand>(KindTy::Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    if (auto CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code.
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }
  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }
};
} // end anonymous namespace

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "LoongArchGenAsmMatcher.inc"

static bool matchRegisterNameHelper(MCRegister &RegNo, StringRef Name) {
  RegNo = MatchRegisterName(Name);

  if (RegNo == LoongArch::NoRegister)
    RegNo = MatchRegisterAltName(Name);

  return RegNo == LoongArch::NoRegister;
}

bool LoongArchAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                       SMLoc &EndLoc) {
  return Error(getLoc(), "invalid register number");
}

OperandMatchResultTy LoongArchAsmParser::tryParseRegister(unsigned &RegNo,
                                                          SMLoc &StartLoc,
                                                          SMLoc &EndLoc) {
  llvm_unreachable("Unimplemented function.");
}

OperandMatchResultTy
LoongArchAsmParser::parseRegister(OperandVector &Operands) {
  if (getLexer().getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  // Eat the $ prefix.
  getLexer().Lex();
  if (getLexer().getKind() != AsmToken::Identifier)
    return MatchOperand_NoMatch;

  StringRef Name = getLexer().getTok().getIdentifier();
  MCRegister RegNo;
  matchRegisterNameHelper(RegNo, Name);
  if (RegNo == LoongArch::NoRegister)
    return MatchOperand_NoMatch;

  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() + Name.size());
  getLexer().Lex();
  Operands.push_back(LoongArchOperand::createReg(RegNo, S, E));

  return MatchOperand_Success;
}

OperandMatchResultTy
LoongArchAsmParser::parseImmediate(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  if (getParser().parseExpression(Res, E))
    return MatchOperand_ParseFail;

  Operands.push_back(LoongArchOperand::createImm(Res, S, E));
  return MatchOperand_Success;
}

/// Looks at a token type and creates the relevant operand from this
/// information, adding to Operands. Return true upon an error.
bool LoongArchAsmParser::parseOperand(OperandVector &Operands,
                                      StringRef Mnemonic) {
  if (parseRegister(Operands) == MatchOperand_Success ||
      parseImmediate(Operands) == MatchOperand_Success)
    return false;

  // Finally we have exhausted all options and must declare defeat.
  Error(getLoc(), "unknown operand");
  return true;
}

bool LoongArchAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                          StringRef Name, SMLoc NameLoc,
                                          OperandVector &Operands) {
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(LoongArchOperand::createToken(Name, NameLoc));

  // If there are no more operands, then finish.
  if (parseOptionalToken(AsmToken::EndOfStatement))
    return false;

  // Parse first operand.
  if (parseOperand(Operands, Name))
    return true;

  // Parse until end of statement, consuming commas between operands.
  while (parseOptionalToken(AsmToken::Comma))
    if (parseOperand(Operands, Name))
      return true;

  // Parse end of statement and return successfully.
  if (parseOptionalToken(AsmToken::EndOfStatement))
    return false;

  SMLoc Loc = getLexer().getLoc();
  getParser().eatToEndOfStatement();
  return Error(Loc, "unexpected token");
}

bool LoongArchAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                            OperandVector &Operands,
                                            MCStreamer &Out) {
  Inst.setLoc(IDLoc);
  Out.emitInstruction(Inst, getSTI());
  return false;
}

bool LoongArchAsmParser::generateImmOutOfRangeError(
    OperandVector &Operands, uint64_t ErrorInfo, int64_t Lower, int64_t Upper,
    Twine Msg = "immediate must be an integer in the range") {
  SMLoc ErrorLoc = ((LoongArchOperand &)*Operands[ErrorInfo]).getStartLoc();
  return Error(ErrorLoc, Msg + " [" + Twine(Lower) + ", " + Twine(Upper) + "]");
}

bool LoongArchAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                                 OperandVector &Operands,
                                                 MCStreamer &Out,
                                                 uint64_t &ErrorInfo,
                                                 bool MatchingInlineAsm) {
  MCInst Inst;
  FeatureBitset MissingFeatures;

  auto Result = MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                     MatchingInlineAsm);
  switch (Result) {
  default:
    break;
  case Match_Success:
    return processInstruction(Inst, IDLoc, Operands, Out);
  case Match_MissingFeature: {
    assert(MissingFeatures.any() && "Unknown missing features!");
    bool FirstFeature = true;
    std::string Msg = "instruction requires the following:";
    for (unsigned i = 0, e = MissingFeatures.size(); i != e; ++i) {
      if (MissingFeatures[i]) {
        Msg += FirstFeature ? " " : ", ";
        Msg += getSubtargetFeatureName(i);
        FirstFeature = false;
      }
    }
    return Error(IDLoc, Msg);
  }
  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion = LoongArchMnemonicSpellCheck(
        ((LoongArchOperand &)*Operands[0]).getToken(), FBS, 0);
    return Error(IDLoc, "unrecognized instruction mnemonic" + Suggestion);
  }
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((LoongArchOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  // Handle the case when the error message is of specific type
  // other than the generic Match_InvalidOperand, and the
  // corresponding operand is missing.
  if (Result > FIRST_TARGET_MATCH_RESULT_TY) {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL && ErrorInfo >= Operands.size())
      return Error(ErrorLoc, "too few operands for instruction");
  }

  switch (Result) {
  default:
    break;
  case Match_InvalidUImm2:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 2) - 1);
  case Match_InvalidUImm2plus1:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/1,
                                      /*Upper=*/(1 << 2));
  case Match_InvalidUImm3:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 3) - 1);
  case Match_InvalidUImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 5) - 1);
  case Match_InvalidUImm6:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 6) - 1);
  case Match_InvalidUImm12:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 12) - 1);
  case Match_InvalidUImm15:
    return generateImmOutOfRangeError(Operands, ErrorInfo, /*Lower=*/0,
                                      /*Upper=*/(1 << 15) - 1);
  case Match_InvalidSImm12:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 11),
                                      (1 << 11) - 1);
  case Match_InvalidSImm14lsl2:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, /*Lower=*/-(1 << 15), /*Upper=*/(1 << 15) - 4,
        "immediate must be a multiple of 4 in the range");
  case Match_InvalidSImm16:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 15),
                                      (1 << 15) - 1);
  case Match_InvalidSImm16lsl2:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, /*Lower=*/-(1 << 17), /*Upper=*/(1 << 17) - 4,
        "immediate must be a multiple of 4 in the range");
  case Match_InvalidSImm20:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 19),
                                      (1 << 19) - 1);
  case Match_InvalidSImm21lsl2:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, /*Lower=*/-(1 << 22), /*Upper=*/(1 << 22) - 4,
        "immediate must be a multiple of 4 in the range");
  case Match_InvalidSImm26lsl2:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, /*Lower=*/-(1 << 27), /*Upper=*/(1 << 27) - 4,
        "immediate must be a multiple of 4 in the range");
  }
  llvm_unreachable("Unknown match type detected!");
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLoongArchAsmParser() {
  RegisterMCAsmParser<LoongArchAsmParser> X(getTheLoongArch32Target());
  RegisterMCAsmParser<LoongArchAsmParser> Y(getTheLoongArch64Target());
}
