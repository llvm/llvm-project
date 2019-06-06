//===-- DPUAsmParser.cpp - Parse DPU assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/DPUAsmCondition.h"
#include "MCTargetDesc/DPUMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#define GET_REGINFO_ENUM
#include "DPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

using namespace llvm;

#define DEBUG_TYPE "dpu-asm-parser"

namespace llvm {

class MCInstrInfo;

} // end namespace llvm

namespace {

static unsigned MatchRegisterName(StringRef Name);

struct DPUOperand;

class DPUAsmParser : public MCTargetAsmParser {
#define GET_ASSEMBLER_HEADER
#include "DPUGenAsmMatcher.inc"

  const MCSubtargetInfo &SubtargetInfo;

  bool mnemonicIsValid(StringRef Mnemonic, unsigned VariantID);

  bool parseOperand(OperandVector &, StringRef Mnemonic);

public:
  enum DPUMatchResultTy {
    Match_RequiresDifferentSrcAndDst = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "DPUGenAsmMatcher.inc"
  };

  DPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &parser,
               const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), SubtargetInfo(STI) {
    parser.addAliasForDirective(".half", ".2byte");
    parser.addAliasForDirective(".hword", ".2byte");
    parser.addAliasForDirective(".word", ".4byte");
    parser.addAliasForDirective(".dword", ".8byte");
    setAvailableFeatures(ComputeAvailableFeatures(SubtargetInfo.getFeatureBits()));
    MCAsmParserExtension::Initialize(parser);
  }

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  OperandMatchResultTy parseAnyRegister(OperandVector &Operands);
  OperandMatchResultTy parseAnyImmediate(OperandVector &Operands);
  OperandMatchResultTy parseAnyEndianness(OperandVector &Operands);

  OperandMatchResultTy parseAnyConditionAsAcquire_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::AcquireCC);
  }
  OperandMatchResultTy parseAnyConditionAsAdd_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::AddCC);
  }
  OperandMatchResultTy parseAnyConditionAsAdd_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Add_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsBoot_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::BootCC);
  }
  OperandMatchResultTy
  parseAnyConditionAsConst_cc_ge0(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::ConstCC_ge0);
  }
  OperandMatchResultTy
  parseAnyConditionAsConst_cc_geu(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::ConstCC_geu);
  }
  OperandMatchResultTy
  parseAnyConditionAsConst_cc_zero(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::ConstCC_zero);
  }
  OperandMatchResultTy parseAnyConditionAsCount_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::CountCC);
  }
  OperandMatchResultTy parseAnyConditionAsCount_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Count_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsDiv_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::DivCC);
  }
  OperandMatchResultTy parseAnyConditionAsDiv_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Div_nzCC);
  }
  OperandMatchResultTy
  parseAnyConditionAsExt_sub_set_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Ext_sub_setCC);
  }
  OperandMatchResultTy parseAnyConditionAsFalse_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::FalseCC);
  }
  OperandMatchResultTy
  parseAnyConditionAsImm_shift_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Imm_shiftCC);
  }
  OperandMatchResultTy
  parseAnyConditionAsImm_shift_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Imm_shift_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsLog_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::LogCC);
  }
  OperandMatchResultTy parseAnyConditionAsLog_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Log_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsLog_set_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Log_setCC);
  }
  OperandMatchResultTy parseAnyConditionAsMul_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::MulCC);
  }
  OperandMatchResultTy parseAnyConditionAsMul_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Mul_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsNo_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::NoCC);
  }
  OperandMatchResultTy parseAnyConditionAsRelease_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::ReleaseCC);
  }
  OperandMatchResultTy parseAnyConditionAsShift_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::ShiftCC);
  }
  OperandMatchResultTy parseAnyConditionAsShift_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Shift_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsSub_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::SubCC);
  }
  OperandMatchResultTy parseAnyConditionAsSub_nz_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Sub_nzCC);
  }
  OperandMatchResultTy parseAnyConditionAsSub_set_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::Sub_setCC);
  }
  OperandMatchResultTy parseAnyConditionAsTrue_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands, DPUAsmCondition::ConditionClass::TrueCC);
  }
  OperandMatchResultTy
  parseAnyConditionAsTrue_false_cc(OperandVector &Operands) {
    return parseAnyCondition(Operands,
                             DPUAsmCondition::ConditionClass::True_falseCC);
  }

private:
  OperandMatchResultTy
  parseAnyCondition(OperandVector &Operands,
                    DPUAsmCondition::ConditionClass CondClass);
};

/// DPUOperand - Instances of this class represent a parsed DPU machine
/// instruction.
struct DPUOperand : public MCParsedAsmOperand {
private:
  enum KindTy {
    Immediate,  /// An immediate (possibly involving symbol references)
    Register,   /// A register index in one or more RegKind.
    Endianness, /// An endianness value.
    Condition,  /// A condition value.
    Token,      /// A simple token
  } Kind;

  struct TokTy {
    const char *Data;
    unsigned Length;
  };

  struct RegTy {
    unsigned RegNum;
  };

  struct ImmTy {
    const MCExpr *Val;
  };

  struct EndianTy {
    bool isLittleEndian;
  };

  struct CondTy {
    DPUAsmCondition::Condition Cond;
    DPUAsmCondition::ConditionClass CondClass;
  };

  union {
    struct TokTy Tok;
    struct RegTy Reg;
    struct ImmTy Imm;
    struct EndianTy Endian;
    struct CondTy Cond;
  };

  SMLoc StartLoc, EndLoc;

  /// For diagnostics, and checking the assembler temporary
  DPUAsmParser &AsmParser;

public:
  explicit DPUOperand(KindTy K, DPUAsmParser &Parser)
      : MCParsedAsmOperand(), Kind(K), AsmParser(Parser) {}

  static std::unique_ptr<DPUOperand> CreateToken(StringRef Str, SMLoc S,
                                                 DPUAsmParser &Parser) {
    auto Op = llvm::make_unique<DPUOperand>(Token, Parser);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<DPUOperand> CreateImm(const MCExpr *Val, SMLoc S,
                                               SMLoc E, DPUAsmParser &Parser) {
    auto Op = llvm::make_unique<DPUOperand>(Immediate, Parser);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<DPUOperand> CreateReg(unsigned int RegNum, SMLoc S,
                                               SMLoc E, DPUAsmParser &Parser) {
    auto Op = llvm::make_unique<DPUOperand>(Register, Parser);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<DPUOperand>
  CreateEndian(bool isLittleEndian, SMLoc S, SMLoc E, DPUAsmParser &Parser) {
    auto Op = llvm::make_unique<DPUOperand>(Endianness, Parser);
    Op->Endian.isLittleEndian = isLittleEndian;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<DPUOperand>
  CreateCondition(DPUAsmCondition::Condition Cond,
                  DPUAsmCondition::ConditionClass CondClass, SMLoc S, SMLoc E,
                  DPUAsmParser &Parser) {
    auto Op = llvm::make_unique<DPUOperand>(Condition, Parser);
    Op->Cond.Cond = Cond;
    Op->Cond.CondClass = CondClass;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  unsigned getReg() const override {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  bool getEndian() const {
    assert(Kind == Endianness && "Invalid access!");
    return Endian.isLittleEndian;
  }

  DPUAsmCondition::Condition getCond() const {
    assert(Kind == Condition && "Invalid access!");
    return Cond.Cond;
  }

  DPUAsmCondition::ConditionClass getCondClass() const {
    assert(Kind == Condition && "Invalid access!");
    return Cond.CondClass;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createExpr(getImm()));
  }

  void addConditionOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(getCond()));
  }

  void addEndianOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(getEndian() ? 0 : 1));
  }

  bool isToken() const override { return Kind == Token; }
  bool isImm() const override { return Kind == Immediate; }
  bool isReg() const override { return Kind == Register; }
  bool isMem() const override { return false; }

  bool isEndian() const { return Kind == Endianness; }
  bool isCondition() const { return Kind == Condition; }

  bool isRegOfClass(unsigned RegClassID) const {
    return isReg() && isRegOfClass(getReg(), RegClassID);
  }

  static bool isRegOfClass(unsigned RegNum, unsigned RegClassID) {
    return DPUMCRegisterClasses[RegClassID].contains(RegNum);
  }

  bool isZERO_REG_AsmReg() const { return isRegOfClass(DPU::ZERO_REGRegClassID); }
  bool isGP_REG_AsmReg() const { return isRegOfClass(DPU::GP_REGRegClassID); }
  bool isGP64_REG_AsmReg() const {
    return isRegOfClass(DPU::GP64_REGRegClassID);
  }
  bool isOP_REG_AsmReg() const { return isRegOfClass(DPU::OP_REGRegClassID); }
  bool isSAFE_OP_REG_AsmReg() const {
    return isRegOfClass(DPU::SAFE_OP_REGRegClassID);
  }
  bool isSAFE_REG_AsmReg() const {
    return isRegOfClass(DPU::SAFE_REGRegClassID);
  }
  bool isCST_REG_AsmReg() const { return isRegOfClass(DPU::CST_REGRegClassID); }
  bool isNZ_OP_REG_AsmReg() const {
    return isRegOfClass(DPU::NZ_OP_REGRegClassID);
  }

  bool isConditionOfClass(DPUAsmCondition::ConditionClass CondClassID) const {
    return isCondition() &&
           DPUAsmCondition::isInConditionClass(getCond(), CondClassID);
  }

  bool isAcquire_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::AcquireCC);
  }
  bool isAdd_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::AddCC);
  }
  bool isAdd_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Add_nzCC);
  }
  bool isBoot_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::BootCC);
  }
  bool isConst_cc_ge0() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::AcquireCC);
  }
  bool isConst_cc_geu() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::AcquireCC);
  }
  bool isConst_cc_zero() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::AcquireCC);
  }
  bool isCount_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::CountCC);
  }
  bool isCount_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Count_nzCC);
  }
  bool isDiv_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::DivCC);
  }
  bool isDiv_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Div_nzCC);
  }
  bool isExt_sub_set_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Ext_sub_setCC);
  }
  bool isFalse_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::FalseCC);
  }
  bool isImm_shift_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Imm_shiftCC);
  }
  bool isImm_shift_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Imm_shift_nzCC);
  }
  bool isLog_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::LogCC);
  }
  bool isLog_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Log_nzCC);
  }
  bool isLog_set_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Log_setCC);
  }
  bool isMul_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::MulCC);
  }
  bool isMul_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Mul_nzCC);
  }
  bool isNo_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::NoCC);
  }
  bool isRelease_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::ReleaseCC);
  }
  bool isShift_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::ShiftCC);
  }
  bool isShift_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Shift_nzCC);
  }
  bool isSub_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::SubCC);
  }
  bool isSub_nz_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Sub_nzCC);
  }
  bool isSub_set_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::Sub_setCC);
  }
  bool isTrue_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::TrueCC);
  }
  bool isTrue_false_cc() const {
    return isConditionOfClass(DPUAsmCondition::ConditionClass::True_falseCC);
  }

  bool isConstantImm() const {
    int64_t Res;
    return getImm()->evaluateAsAbsolute(Res);
  }

  int64_t getConstantImm() const {
    const MCExpr *Val = getImm();
    int64_t Value = 0;
    (void)Val->evaluateAsAbsolute(Value);
    return Value;
  }

  template <unsigned Bits> bool isSImm() const {
    return isImm() && (!isConstantImm() || isInt<Bits>(getConstantImm()));
  }
  template <unsigned Bits> bool isUImm() const {
    return isImm() && (!isConstantImm() || isUInt<Bits>(getConstantImm()));
  }
  template <unsigned Bits> bool isSUImm() const {
    return isImm() && (!isConstantImm() || isUInt<Bits>(getConstantImm()) ||
                       isInt<Bits>(getConstantImm()));
  }
  template <unsigned Bits> bool isPCImm() const {
    return isImm() && (!isConstantImm() || isUInt<Bits>(getConstantImm()) ||
                       isInt<Bits>(getConstantImm()));
  }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case Immediate:
      OS << "Imm<";
      OS << *Imm.Val;
      OS << ">";
      break;
    case Register:
      OS << "Reg<" << Reg.RegNum << ">";
      break;
    case Endianness:
      OS << "Endian<" << Endian.isLittleEndian << ">";
      break;
    case Condition:
      OS << "Cond<" << Cond.Cond << ">";
      break;
    case Token:
      OS << getToken();
      break;
    }
  }
}; // struct DPUOperand

bool DPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                 SMLoc &EndLoc) {
  LLVM_DEBUG(dbgs() << "ParseRegister\n");
  // todo
  llvm_unreachable("ParseRegister");
}

bool DPUAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "ParseInstruction\n");

  // Check if we have valid mnemonic
  if (!mnemonicIsValid(Name, 0)) {
    return Error(NameLoc, "unknown instruction");
  }
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(DPUOperand::CreateToken(Name, NameLoc, *this));

  auto parseOp = [&]() -> bool { return parseOperand(Operands, Name); };

  if (parseMany(parseOp, true)) {
    return true;
  }

  Parser.clearPendingErrors();
  return false;
}

bool DPUAsmParser::ParseDirective(AsmToken /*DirectiveID*/) {
  LLVM_DEBUG(dbgs() << "ParseDirective\n");
  return true;
}

bool DPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                           OperandVector &Operands,
                                           MCStreamer &Out, uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  MCInst Inst;
  SMLoc ErrorLoc;
  LLVM_DEBUG(dbgs() << "MatchAndEmitInstruction\n");

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  case Match_Success:
    Out.EmitInstruction(Inst, SubtargetInfo);
    Opcode = Inst.getOpcode();
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "Instruction use requires option to be enabled");
  case Match_MnemonicFail:
    return Error(IDLoc, "Unrecognized instruction mnemonic");
  case Match_InvalidOperand: {
    ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "Too few operands for instruction");

      ErrorLoc = ((DPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "Invalid operand for instruction");
  }
  default:
    break;
  }

  llvm_unreachable("Unknown match type detected!");
}

bool DPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "parseOperand\n");

  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  if (ResTy == MatchOperand_NoMatch) {
    Error(Parser.getTok().getLoc(), "unknown operand");
  }

  return ResTy != MatchOperand_Success;
}

OperandMatchResultTy DPUAsmParser::parseAnyRegister(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "parseAnyRegister\n");

  auto Token = Parser.getTok();

  SMLoc S = Token.getLoc();

  if (Token.is(AsmToken::Percent)) {
    Parser.Lex(); // % (before the register identifier)
    Token = Parser.getTok();
  }

  if (Token.isNot(AsmToken::Identifier)) {
    return MatchOperand_NoMatch;
  }

  unsigned RegNum = MatchRegisterName(Token.getIdentifier());

  if (RegNum == 0) {
    return MatchOperand_NoMatch;
  }

  Parser.Lex(); // identifier

  Operands.push_back(DPUOperand::CreateReg(RegNum, S, Token.getLoc(), *this));

  return MatchOperand_Success;
}

OperandMatchResultTy DPUAsmParser::parseAnyImmediate(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "parseAnyImmediate\n");

  SMLoc Start = Parser.getTok().getLoc();
  SMLoc End = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  const MCExpr *ExprVal;

  if (parseAnyRegister(Operands) == MatchOperand_Success) {
    return MatchOperand_Success;
  }

  if (Parser.parseExpression(ExprVal)) {
    return MatchOperand_NoMatch;
  }

  Operands.push_back(DPUOperand::CreateImm(ExprVal, Start, End, *this));

  return MatchOperand_Success;
}

OperandMatchResultTy DPUAsmParser::parseAnyEndianness(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "parseAnyEndianness\n");

  auto Token = Parser.getTok();

  SMLoc S = Token.getLoc();

  if (Token.isNot(AsmToken::Exclaim)) {
    return MatchOperand_NoMatch;
  }

  Parser.Lex(); // ! (before the endianness)
  auto IdToken = Parser.getTok();

  if (IdToken.isNot(AsmToken::Identifier)) {
    Error(S, "unknown endianness");
    return MatchOperand_ParseFail;
  }

  StringRef Id = IdToken.getIdentifier();
  bool isLittleEndian;
  if (Id.compare("little") == 0) {
    isLittleEndian = true;
  } else if (Id.compare("big") == 0) {
    isLittleEndian = false;
  } else {
    Error(S, "unknown endianness");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // identifier

  Operands.push_back(
      DPUOperand::CreateEndian(isLittleEndian, S, Token.getLoc(), *this));

  return MatchOperand_Success;
}

OperandMatchResultTy
DPUAsmParser::parseAnyCondition(OperandVector &Operands,
                                DPUAsmCondition::ConditionClass CondClass) {
  MCAsmParser &Parser = getParser();
  LLVM_DEBUG(dbgs() << "parseAnyCondition\n");

  auto Token = Parser.getTok();

  SMLoc S = Token.getLoc();
  std::string CondStr;

  if (Token.is(AsmToken::Identifier)) {
    if (Token.getString().startswith("?")) {
      CondStr = Token.getIdentifier().str().erase(0, 1);
    } else {
      CondStr = Token.getIdentifier().str();
    }
  } else if (Token.is(AsmToken::Exclaim)) {
    Parser.Lex(); // ! (before the condition identifier)
    auto IdToken = Parser.getTok();

    if (IdToken.isNot(AsmToken::Identifier)) {
      Error(S, "unknown condition");
      return MatchOperand_ParseFail;
    }

    StringRef Id = IdToken.getIdentifier();

    CondStr = Id.str();
  } else {
    return MatchOperand_NoMatch;
  }

  DPUAsmCondition::Condition Condition;

  if (DPUAsmCondition::fromString(CondStr, Condition)) {
    Error(S, "unknown condition");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // identifier

  Operands.push_back(DPUOperand::CreateCondition(Condition, CondClass, S,
                                                 Token.getLoc(), *this));

  return MatchOperand_Success;
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "DPUGenAsmMatcher.inc"

bool DPUAsmParser::mnemonicIsValid(StringRef Mnemonic, unsigned VariantID) {
  // Find the appropriate table for this asm variant.
  const MatchEntry *Start, *End;
  switch (VariantID) {
  default:
    llvm_unreachable("invalid variant!");
  case 0:
    Start = std::begin(MatchTable0);
    End = std::end(MatchTable0);
    break;
  }
  // Search the table.
  auto MnemonicRange = std::equal_range(Start, End, Mnemonic, LessOpcode());
  return MnemonicRange.first != MnemonicRange.second;
}

} // end anonymous namespace

extern "C" void LLVMInitializeDPUAsmParser() {
  RegisterMCAsmParser<DPUAsmParser> X(TheDPUTarget);
}
