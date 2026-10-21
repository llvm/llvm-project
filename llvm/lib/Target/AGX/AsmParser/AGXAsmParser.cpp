//===-- AGXAsmParser.cpp - Parse AGX assembly to MCInst instructions ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Accepts the syntax printed by AGXInstPrinter, which follows the applegpu
// disassembler.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace {

/// A parsed register: the first register of a tuple and the tuple length.
struct ParsedReg {
  AGX::RegKind Kind = AGX::RegKind::Invalid;
  unsigned HalfIndex = 0;
  unsigned Count = 1;
};

class AGXOperand : public MCParsedAsmOperand {
public:
  enum KindTy { Token, Register, Immediate, Expression } Kind;
  /// What an immediate was parsed as.
  enum ImmKindTy {
    IK_Int,
    IK_Float,
    IK_Shift,
    IK_BitfieldMask,
    IK_TruthTable,
    IK_ICond,
    IK_FCond,
    IK_ConvertMode,
    IK_RoundMode,
    IK_MemFormat,
    IK_MemMask,
    IK_Signedness,
    IK_AtomicOp,
    IK_SpecialReg,
    IK_NestCounter,
  };

  SMLoc StartLoc, EndLoc;
  StringRef Tok;
  MCRegister Reg;
  AGX::RegKind RegKind = AGX::RegKind::Invalid;
  unsigned Mods = 0;
  int64_t Imm = 0;
  double FPImm = 0;
  ImmKindTy ImmKind = IK_Int;
  const MCExpr *Expr = nullptr;

  explicit AGXOperand(KindTy K) : Kind(K) {}

  static std::unique_ptr<AGXOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<AGXOperand>(Token);
    Op->Tok = Str;
    Op->StartLoc = Op->EndLoc = S;
    return Op;
  }
  static std::unique_ptr<AGXOperand> createReg(MCRegister R, AGX::RegKind K,
                                               unsigned Mods, SMLoc S,
                                               SMLoc E) {
    auto Op = std::make_unique<AGXOperand>(Register);
    Op->Reg = R;
    Op->RegKind = K;
    Op->Mods = Mods;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
  static std::unique_ptr<AGXOperand> createImm(int64_t V, ImmKindTy IK,
                                               SMLoc S, SMLoc E) {
    auto Op = std::make_unique<AGXOperand>(Immediate);
    Op->Imm = V;
    Op->ImmKind = IK;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
  static std::unique_ptr<AGXOperand> createFPImm(double V, SMLoc S, SMLoc E) {
    auto Op = createImm(0, IK_Float, S, E);
    Op->FPImm = V;
    return Op;
  }
  static std::unique_ptr<AGXOperand> createExpr(const MCExpr *Ex, SMLoc S,
                                                SMLoc E) {
    auto Op = std::make_unique<AGXOperand>(Expression);
    Op->Expr = Ex;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  bool isToken() const override { return Kind == Token; }
  bool isImm() const override {
    return Kind == Immediate && ImmKind == IK_Int;
  }
  bool isAGXImm() const { return isImm(); }
  bool isReg() const override { return Kind == Register; }
  bool isMem() const override { return false; }
  MCRegister getReg() const override { return Reg; }
  StringRef getToken() const { return Tok; }
  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }
  void print(raw_ostream &OS, const MCAsmInfo &) const override {
    switch (Kind) {
    case Token:
      OS << "'" << Tok << "'";
      break;
    case Register:
      OS << "<register " << Reg.id() << ">";
      break;
    case Immediate:
      OS << Imm;
      break;
    case Expression:
      OS << "<expr>";
      break;
    }
  }

  bool isRegKind(std::initializer_list<AGX::RegKind> Kinds) const {
    return Kind == Register && is_contained(Kinds, RegKind);
  }
  bool isIntImm8() const {
    return Kind == Immediate && ImmKind == IK_Int && isUInt<8>(Imm);
  }
  bool hasOnlyMods(unsigned Allowed) const { return (Mods & ~Allowed) == 0; }

  static constexpr unsigned HintMods =
      AGX::SRC_MOD_CACHE | AGX::SRC_MOD_DISCARD;
  static constexpr unsigned FPMods = AGX::SRC_MOD_ABS | AGX::SRC_MOD_NEG;

  bool isSrcImpl(std::initializer_list<AGX::RegKind> Kinds,
                 unsigned AllowedMods) const {
    if (Kind == Immediate)
      return isIntImm8();
    return isRegKind(Kinds) && hasOnlyMods(HintMods | AllowedMods);
  }

  bool isSrc() const {
    return isSrcImpl({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::UGPR16, AGX::RegKind::UGPR32},
                     0);
  }
  bool isSrc64() const {
    return isSrcImpl({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::GPR64, AGX::RegKind::UGPR16,
                      AGX::RegKind::UGPR32},
                     0);
  }
  bool isSrc16() const {
    return isSrcImpl(
        {AGX::RegKind::GPR16, AGX::RegKind::UGPR16, AGX::RegKind::UGPR32}, 0);
  }
  bool isSrcSX() const {
    return isSrcImpl({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::UGPR16, AGX::RegKind::UGPR32},
                     AGX::SRC_MOD_SX);
  }
  bool isSrcSX64() const {
    return isSrcImpl({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::GPR64, AGX::RegKind::UGPR16,
                      AGX::RegKind::UGPR32},
                     AGX::SRC_MOD_SX);
  }
  bool isFPImm() const {
    if (Kind != Immediate)
      return false;
    if (ImmKind == IK_Float)
      return AGX::encodeFPImm8(FPImm).has_value();
    return isIntImm8();
  }
  bool isSrcF() const {
    if (Kind == Immediate)
      return isFPImm();
    return isRegKind({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::UGPR16, AGX::RegKind::UGPR32}) &&
           hasOnlyMods(HintMods | FPMods);
  }
  bool isSrcF16() const {
    if (Kind == Immediate)
      return isFPImm();
    return isRegKind({AGX::RegKind::GPR16, AGX::RegKind::UGPR16,
                      AGX::RegKind::UGPR32}) &&
           hasOnlyMods(HintMods | FPMods);
  }
  bool isCmpselSrc() const { return isSrc(); }

  bool isDstImpl(std::initializer_list<AGX::RegKind> Kinds) const {
    return isRegKind(Kinds) && hasOnlyMods(AGX::SRC_MOD_CACHE);
  }
  bool isDst() const {
    return isDstImpl({AGX::RegKind::GPR16, AGX::RegKind::GPR32});
  }
  bool isDst64() const {
    return isDstImpl(
        {AGX::RegKind::GPR16, AGX::RegKind::GPR32, AGX::RegKind::GPR64});
  }
  bool isDst16() const { return isDstImpl({AGX::RegKind::GPR16}); }
  bool isDst32() const { return isDstImpl({AGX::RegKind::GPR32}); }

  bool isImmKind(ImmKindTy K) const {
    return Kind == Immediate && ImmKind == K;
  }
  bool isShift() const { return isImmKind(IK_Shift); }
  bool isBitfieldMask() const { return isImmKind(IK_BitfieldMask); }
  bool isTruthTable() const { return isImmKind(IK_TruthTable); }
  bool isICond() const { return isImmKind(IK_ICond); }
  bool isFCond() const { return isImmKind(IK_FCond); }
  bool isICond3() const { return isImmKind(IK_ICond) && Imm < 8; }
  bool isFCond3() const { return isImmKind(IK_FCond) && Imm < 8; }
  bool isConvertMode() const { return isImmKind(IK_ConvertMode); }
  bool isRoundMode() const { return isImmKind(IK_RoundMode); }
  bool isMemFormat() const { return isImmKind(IK_MemFormat); }
  bool isMemMask() const { return isImmKind(IK_MemMask); }
  bool isSignedness() const { return isImmKind(IK_Signedness); }
  bool isAtomicOp() const { return isImmKind(IK_AtomicOp); }
  bool isSpecialReg() const { return isImmKind(IK_SpecialReg); }
  bool isNestCounter() const { return isImmKind(IK_NestCounter); }

  bool isMemReg() const {
    return isRegKind({AGX::RegKind::GPR16, AGX::RegKind::GPR32,
                      AGX::RegKind::GPR64, AGX::RegKind::GPR96,
                      AGX::RegKind::GPR128}) &&
           !Mods;
  }
  bool isAtomicDst() const {
    return (Kind == Register && !Reg) || isRegKind({AGX::RegKind::GPR32});
  }
  bool isLocalAtomicDst() const {
    return (Kind == Register && !Reg) ||
           isRegKind({AGX::RegKind::GPR16, AGX::RegKind::GPR32});
  }
  bool isStoreFlag() const {
    return Kind == Immediate && ImmKind == IK_Int && Imm >= 0 && Imm <= 3;
  }
  bool isAtomicSrc() const {
    return isRegKind({AGX::RegKind::GPR32, AGX::RegKind::GPR64});
  }
  bool isMemBase() const {
    return isRegKind({AGX::RegKind::GPR64, AGX::RegKind::UGPR64});
  }
  bool isMemIndex() const {
    return (Kind == Immediate && ImmKind == IK_Int &&
            (isInt<16>(Imm) || isUInt<16>(Imm))) ||
           isRegKind({AGX::RegKind::GPR32});
  }
  bool isLocalBase() const {
    return (Kind == Immediate && ImmKind == IK_Int && Imm == 0) ||
           isRegKind({AGX::RegKind::GPR16, AGX::RegKind::UGPR16});
  }
  bool isLocalIndex() const {
    return (Kind == Immediate && ImmKind == IK_Int &&
            (isInt<16>(Imm) || isUInt<16>(Imm))) ||
           isRegKind({AGX::RegKind::GPR16});
  }
  bool isBrTarget() const {
    return Kind == Expression || (Kind == Immediate && ImmKind == IK_Int);
  }

  void addSrcOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "invalid number of operands");
    if (Kind == Immediate) {
      Inst.addOperand(MCOperand::createImm(0));
      int64_t V = ImmKind == IK_Float ? *AGX::encodeFPImm8(FPImm) : Imm;
      Inst.addOperand(MCOperand::createImm(V));
      return;
    }
    Inst.addOperand(MCOperand::createImm(Mods));
    Inst.addOperand(MCOperand::createReg(Reg));
  }
  void addDstOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "invalid number of operands");
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createImm((Mods & AGX::SRC_MOD_CACHE) ? 1 : 0));
  }
  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "invalid number of operands");
    Inst.addOperand(MCOperand::createImm(Imm));
  }
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "invalid number of operands");
    Inst.addOperand(MCOperand::createReg(Reg));
  }
  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "invalid number of operands");
    if (Kind == Register)
      Inst.addOperand(MCOperand::createReg(Reg));
    else
      Inst.addOperand(MCOperand::createImm(Imm));
  }
  void addBrTargetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "invalid number of operands");
    if (Kind == Expression)
      Inst.addOperand(MCOperand::createExpr(Expr));
    else
      Inst.addOperand(MCOperand::createImm(Imm));
  }
};

class AGXAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

#define GET_ASSEMBLER_HEADER
#include "AGXGenAsmMatcher.inc"

public:
  enum AGXMatchResultTy {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "AGXGenAsmMatcher.inc"
  };

private:

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;
  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  /// Parse a register name, with an optional tuple ("r0_r1") and flag
  /// suffixes (".cache", ".neg", ...).
  bool parseRegName(StringRef Name, ParsedReg &PR, unsigned &Mods);
  ParseStatus parseRegOperand(OperandVector &Operands, bool AllowFlags);
  ParseStatus parseIntOperand(OperandVector &Operands,
                              AGXOperand::ImmKindTy IK);

  ParseStatus parseSrc(OperandVector &Operands);
  ParseStatus parseDst(OperandVector &Operands);
  ParseStatus parseImm(OperandVector &Operands);
  ParseStatus parseShift(OperandVector &Operands);
  ParseStatus parseBitfieldMask(OperandVector &Operands);
  ParseStatus parseTruthTable(OperandVector &Operands);
  ParseStatus parseNamedEnum(OperandVector &Operands, AGXOperand::ImmKindTy IK,
                             std::optional<unsigned> (*Lookup)(StringRef));
  ParseStatus parseICond(OperandVector &Operands);
  ParseStatus parseFCond(OperandVector &Operands);
  ParseStatus parseICond3(OperandVector &Operands) {
    return parseICond(Operands);
  }
  ParseStatus parseFCond3(OperandVector &Operands) {
    return parseFCond(Operands);
  }
  ParseStatus parseConvertMode(OperandVector &Operands);
  ParseStatus parseRoundMode(OperandVector &Operands);
  ParseStatus parseMemFormat(OperandVector &Operands);
  ParseStatus parseMemMask(OperandVector &Operands);
  ParseStatus parseSignedness(OperandVector &Operands);
  ParseStatus parseAtomicOp(OperandVector &Operands);
  ParseStatus parseSpecialReg(OperandVector &Operands);
  ParseStatus parseNestCounter(OperandVector &Operands);
  ParseStatus parseMemReg(OperandVector &Operands);
  ParseStatus parseMemIndex(OperandVector &Operands);
  ParseStatus parseBrTarget(OperandVector &Operands);

  std::unique_ptr<AGXOperand> defaultZeroOperand() const {
    return AGXOperand::createImm(0, AGXOperand::IK_Int, SMLoc(), SMLoc());
  }

  const MCRegisterInfo &MRI() { return *getContext().getRegisterInfo(); }

public:
  AGXAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
               const MCInstrInfo &MII)
      : MCTargetAsmParser(STI, MII), Parser(Parser) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

} // namespace

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "AGXGenAsmMatcher.inc"

/// Parse "<prefix><number><suffix>" where prefix is r or u and suffix is l,
/// h or empty.
static bool parseOneReg(StringRef S, char &File, unsigned &Number,
                        char &Half) {
  if (S.size() < 2 || (S[0] != 'r' && S[0] != 'u'))
    return false;
  File = S[0];
  S = S.drop_front();
  Half = 0;
  if (S.ends_with("l") || S.ends_with("h")) {
    Half = S.back();
    S = S.drop_back();
  }
  if (S.empty() || S.getAsInteger(10, Number))
    return false;
  return true;
}

bool AGXAsmParser::parseRegName(StringRef Name, ParsedReg &PR,
                                unsigned &Mods) {
  SmallVector<StringRef, 4> Parts;
  Name.split(Parts, '.');
  Mods = 0;
  for (StringRef Flag : drop_begin(Parts)) {
    unsigned F = StringSwitch<unsigned>(Flag)
                     .Case("cache", AGX::SRC_MOD_CACHE)
                     .Case("discard", AGX::SRC_MOD_DISCARD)
                     .Case("abs", AGX::SRC_MOD_ABS)
                     .Case("neg", AGX::SRC_MOD_NEG)
                     .Case("sx", AGX::SRC_MOD_SX)
                     .Default(0);
    if (!F)
      return false;
    Mods |= F;
  }

  SmallVector<StringRef, 4> Elts;
  Parts[0].split(Elts, '_');
  char File = 0, Half = 0;
  unsigned Number = 0;
  if (!parseOneReg(Elts[0], File, Number, Half))
    return false;
  unsigned Limit = File == 'r' ? 128 : 256;
  if (Number >= Limit)
    return false;
  unsigned First = Number * 2 + (Half == 'h' ? 1 : 0);
  bool Is16 = Half != 0;

  // Tuples must name consecutive registers of the same size.
  for (unsigned I = 1; I < Elts.size(); ++I) {
    char F2 = 0, H2 = 0;
    unsigned N2 = 0;
    if (!parseOneReg(Elts[I], F2, N2, H2) || F2 != File || (H2 != 0) != Is16)
      return false;
    unsigned Idx = N2 * 2 + (H2 == 'h' ? 1 : 0);
    if (Idx != First + I * (Is16 ? 1 : 2))
      return false;
  }

  PR.HalfIndex = First;
  PR.Count = Elts.size();
  if (File == 'u') {
    if (Is16 && PR.Count == 1)
      PR.Kind = AGX::RegKind::UGPR16;
    else if (!Is16 && PR.Count == 1)
      PR.Kind = AGX::RegKind::UGPR32;
    else if (!Is16 && PR.Count == 2)
      PR.Kind = AGX::RegKind::UGPR64;
    else
      return false;
    return true;
  }
  if (Is16) {
    PR.Kind = AGX::RegKind::GPR16;
    return PR.Count <= 4;
  }
  switch (PR.Count) {
  case 1:
    PR.Kind = AGX::RegKind::GPR32;
    return true;
  case 2:
    PR.Kind = AGX::RegKind::GPR64;
    return true;
  case 3:
    PR.Kind = AGX::RegKind::GPR96;
    return true;
  case 4:
    PR.Kind = AGX::RegKind::GPR128;
    return true;
  default:
    return false;
  }
}

bool AGXAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                 SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register");
  return false;
}

ParseStatus AGXAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                           SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;
  ParsedReg PR;
  unsigned Mods;
  if (!parseRegName(Tok.getIdentifier(), PR, Mods) || Mods ||
      (PR.Kind == AGX::RegKind::GPR16 && PR.Count != 1))
    return ParseStatus::NoMatch;
  Reg = AGX::getRegister(PR.Kind, PR.HalfIndex, MRI());
  if (!Reg)
    return ParseStatus::NoMatch;
  Parser.Lex();
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseRegOperand(OperandVector &Operands,
                                          bool AllowFlags) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;
  ParsedReg PR;
  unsigned Mods;
  if (!parseRegName(Tok.getIdentifier(), PR, Mods))
    return ParseStatus::NoMatch;
  if (Mods && !AllowFlags)
    return Error(Tok.getLoc(), "unexpected register flags");
  // Tuples of 16-bit registers only appear as memory operands, which are
  // parsed by parseMemReg.
  if (PR.Kind == AGX::RegKind::GPR16 && PR.Count != 1)
    return Error(Tok.getLoc(), "unexpected register tuple");
  MCRegister Reg = AGX::getRegister(PR.Kind, PR.HalfIndex, MRI());
  if (!Reg)
    return Error(Tok.getLoc(), "invalid register");
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  Parser.Lex();
  Operands.push_back(AGXOperand::createReg(Reg, PR.Kind, Mods, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseIntOperand(OperandVector &Operands,
                                          AGXOperand::ImmKindTy IK) {
  SMLoc S = getParser().getTok().getLoc();
  if (getParser().getTok().isNot(AsmToken::Integer) &&
      getParser().getTok().isNot(AsmToken::Minus))
    return ParseStatus::NoMatch;
  const MCExpr *Ex;
  SMLoc E;
  if (getParser().parseExpression(Ex, E))
    return ParseStatus::Failure;
  int64_t V;
  if (!Ex->evaluateAsAbsolute(V))
    return Error(S, "expected an absolute expression");
  Operands.push_back(AGXOperand::createImm(V, IK, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseSrc(OperandVector &Operands) {
  ParseStatus Res = parseRegOperand(Operands, /*AllowFlags=*/true);
  if (!Res.isNoMatch())
    return Res;

  // Integer or floating point immediate.
  SMLoc S = getParser().getTok().getLoc();
  bool Neg = false;
  if (getParser().getTok().is(AsmToken::Minus) &&
      getLexer().peekTok().is(AsmToken::Real)) {
    Neg = true;
    Parser.Lex();
  }
  if (getParser().getTok().is(AsmToken::Real)) {
    double V;
    if (getParser().getTok().getString().getAsDouble(V))
      return Error(S, "invalid floating point immediate");
    SMLoc E = getParser().getTok().getEndLoc();
    Parser.Lex();
    Operands.push_back(AGXOperand::createFPImm(Neg ? -V : V, S, E));
    return ParseStatus::Success;
  }
  return parseIntOperand(Operands, AGXOperand::IK_Int);
}

ParseStatus AGXAsmParser::parseDst(OperandVector &Operands) {
  return parseRegOperand(Operands, /*AllowFlags=*/true);
}

ParseStatus AGXAsmParser::parseImm(OperandVector &Operands) {
  return parseIntOperand(Operands, AGXOperand::IK_Int);
}

ParseStatus AGXAsmParser::parseShift(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier) || Tok.getIdentifier() != "lsl")
    return ParseStatus::NoMatch;
  SMLoc S = Tok.getLoc();
  Parser.Lex();
  int64_t V;
  if (getParser().parseAbsoluteExpression(V))
    return ParseStatus::Failure;
  if (V < 0 || V > 7)
    return Error(S, "shift out of range");
  Operands.push_back(
      AGXOperand::createImm(V, AGXOperand::IK_Shift, S, getLexer().getLoc()));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseBitfieldMask(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier) || Tok.getIdentifier() != "mask")
    return ParseStatus::NoMatch;
  SMLoc S = Tok.getLoc();
  Parser.Lex();
  int64_t V;
  if (getParser().parseAbsoluteExpression(V))
    return ParseStatus::Failure;
  uint64_t Plus1 = uint64_t(V) + 1;
  if (V <= 0 || !isPowerOf2_64(Plus1) || Log2_64(Plus1) > 32)
    return Error(S, "mask must be 2^n - 1 with 0 < n <= 32");
  unsigned M = Log2_64(Plus1);
  Operands.push_back(AGXOperand::createImm(M == 32 ? 0 : M,
                                           AGXOperand::IK_BitfieldMask, S,
                                           getLexer().getLoc()));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseTruthTable(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Integer))
    return ParseStatus::NoMatch;
  StringRef Str = Tok.getString();
  if (Str.size() != 4 || Str.find_first_not_of("01") != StringRef::npos)
    return Error(Tok.getLoc(), "truth table must be four binary digits");
  unsigned TT = 0;
  for (unsigned I = 0; I < 4; ++I)
    TT |= unsigned(Str[I] - '0') << I;
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  Parser.Lex();
  Operands.push_back(
      AGXOperand::createImm(TT, AGXOperand::IK_TruthTable, S, E));
  return ParseStatus::Success;
}

ParseStatus
AGXAsmParser::parseNamedEnum(OperandVector &Operands, AGXOperand::ImmKindTy IK,
                             std::optional<unsigned> (*Lookup)(StringRef)) {
  const AsmToken &Tok = getParser().getTok();
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  if (Tok.is(AsmToken::Integer)) {
    int64_t V = Tok.getIntVal();
    Parser.Lex();
    Operands.push_back(AGXOperand::createImm(V, IK, S, E));
    return ParseStatus::Success;
  }
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;
  std::optional<unsigned> V = Lookup(Tok.getIdentifier());
  if (!V)
    return ParseStatus::NoMatch;
  Parser.Lex();
  Operands.push_back(AGXOperand::createImm(*V, IK, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseICond(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_ICond, AGX::parseICond);
}

ParseStatus AGXAsmParser::parseFCond(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_FCond, AGX::parseFCond);
}

ParseStatus AGXAsmParser::parseConvertMode(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_ConvertMode,
                        AGX::parseConvertMode);
}

ParseStatus AGXAsmParser::parseRoundMode(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_RoundMode,
                        AGX::parseRoundMode);
}

ParseStatus AGXAsmParser::parseMemFormat(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_MemFormat,
                        AGX::parseMemFormat);
}

ParseStatus AGXAsmParser::parseMemMask(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_MemMask, AGX::parseMemMask);
}

static std::optional<unsigned> parseSignednessName(StringRef Name) {
  if (Name == "signed")
    return 0;
  if (Name == "unsigned")
    return 1;
  return std::nullopt;
}

ParseStatus AGXAsmParser::parseSignedness(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_Signedness,
                        parseSignednessName);
}

ParseStatus AGXAsmParser::parseAtomicOp(OperandVector &Operands) {
  return parseNamedEnum(Operands, AGXOperand::IK_AtomicOp, AGX::parseAtomicOp);
}

ParseStatus AGXAsmParser::parseSpecialReg(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier) || !Tok.getIdentifier().starts_with("sr"))
    return ParseStatus::NoMatch;
  unsigned SR;
  if (Tok.getIdentifier().drop_front(2).getAsInteger(10, SR) || SR > 255)
    return Error(Tok.getLoc(), "invalid special register");
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  Parser.Lex();
  // An optional "(name)" annotation, as printed.
  if (getParser().getTok().is(AsmToken::LParen)) {
    while (getParser().getTok().isNot(AsmToken::RParen) &&
           getParser().getTok().isNot(AsmToken::EndOfStatement))
      Parser.Lex();
    if (getParser().getTok().is(AsmToken::RParen))
      Parser.Lex();
  }
  Operands.push_back(
      AGXOperand::createImm(SR, AGXOperand::IK_SpecialReg, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseNestCounter(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;
  StringRef Id = Tok.getIdentifier();
  if (Id != "r0l" && Id != "r0l.cache")
    return ParseStatus::NoMatch;
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  Parser.Lex();
  Operands.push_back(AGXOperand::createImm(Id == "r0l.cache" ? 1 : 0,
                                           AGXOperand::IK_NestCounter, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseMemReg(OperandVector &Operands) {
  const AsmToken &Tok = getParser().getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;
  SMLoc S = Tok.getLoc(), E = Tok.getEndLoc();
  if (Tok.getIdentifier() == "_") {
    Parser.Lex();
    Operands.push_back(AGXOperand::createReg(MCRegister(), AGX::RegKind::GPR32,
                                             0, S, E));
    return ParseStatus::Success;
  }
  ParsedReg PR;
  unsigned Mods;
  if (!parseRegName(Tok.getIdentifier(), PR, Mods) || Mods)
    return ParseStatus::NoMatch;
  // Register tuples are kept whole; memory instructions encode their first
  // register. 16-bit tuples are represented by their first half register.
  AGX::RegKind Kind = PR.Kind;
  MCRegister Reg = AGX::getRegister(Kind, PR.HalfIndex, MRI());
  if (!Reg)
    return Error(S, "invalid register");
  Parser.Lex();
  Operands.push_back(AGXOperand::createReg(Reg, Kind, 0, S, E));
  return ParseStatus::Success;
}

ParseStatus AGXAsmParser::parseMemIndex(OperandVector &Operands) {
  ParseStatus Res = parseRegOperand(Operands, /*AllowFlags=*/false);
  if (!Res.isNoMatch())
    return Res;
  return parseIntOperand(Operands, AGXOperand::IK_Int);
}

ParseStatus AGXAsmParser::parseBrTarget(OperandVector &Operands) {
  SMLoc S = getParser().getTok().getLoc();
  // pc+N / pc-N: a byte offset relative to the branch, as printed.
  const AsmToken &Tok = getParser().getTok();
  if (Tok.is(AsmToken::Identifier) && Tok.getIdentifier() == "pc" &&
      (getLexer().peekTok().is(AsmToken::Plus) ||
       getLexer().peekTok().is(AsmToken::Minus))) {
    Parser.Lex();
    bool Neg = getLexer().is(AsmToken::Minus);
    Parser.Lex();
    int64_t V;
    if (getParser().parseAbsoluteExpression(V))
      return ParseStatus::Failure;
    Operands.push_back(AGXOperand::createImm(Neg ? -V : V, AGXOperand::IK_Int,
                                             S, getLexer().getLoc()));
    return ParseStatus::Success;
  }
  const MCExpr *Ex;
  SMLoc E;
  if (getParser().parseExpression(Ex, E))
    return ParseStatus::Failure;
  int64_t V;
  if (Ex->evaluateAsAbsolute(V))
    Operands.push_back(AGXOperand::createImm(V, AGXOperand::IK_Int, S, E));
  else
    Operands.push_back(AGXOperand::createExpr(Ex, S, E));
  return ParseStatus::Success;
}

bool AGXAsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  Operands.push_back(AGXOperand::createToken(Name, NameLoc));
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Parser.Lex();
    return false;
  }

  while (true) {
    ParseStatus Res = MatchOperandParserImpl(Operands, Name);
    if (Res.isFailure())
      return true;
    if (Res.isNoMatch()) {
      // Fall back to a register or an expression.
      Res = parseRegOperand(Operands, /*AllowFlags=*/true);
      if (Res.isFailure())
        return true;
      if (Res.isNoMatch() &&
          parseIntOperand(Operands, AGXOperand::IK_Int).isNoMatch())
        return Error(getLexer().getLoc(), "unknown operand");
    }
    if (getLexer().is(AsmToken::EndOfStatement))
      break;
    if (getLexer().isNot(AsmToken::Comma))
      return Error(getLexer().getLoc(), "unexpected token in operand list");
    Parser.Lex();
  }
  Parser.Lex();
  return false;
}

bool AGXAsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                           OperandVector &Operands,
                                           MCStreamer &Out,
                                           uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  MCInst Inst;
  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    Opcode = Inst.getOpcode();
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction not supported by the target encoding");
  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion = AGXMnemonicSpellCheck(
        static_cast<AGXOperand &>(*Operands[0]).getToken(), FBS, 0);
    return Error(IDLoc, "invalid instruction" + Suggestion);
  }
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");
      ErrorLoc = Operands[ErrorInfo]->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  default:
    return Error(IDLoc, "invalid instruction");
  }
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXAsmParser() {
  RegisterMCAsmParser<AGXAsmParser> X(getTheAGXTarget());
}
