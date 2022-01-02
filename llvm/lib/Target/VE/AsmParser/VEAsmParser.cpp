//===-- VEAsmParser.cpp - Parse VE assembly to MCInst instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/VEMCExpr.h"
#include "MCTargetDesc/VEMCTargetDesc.h"
#include "TargetInfo/VETargetInfo.h"
#include "VE.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "ve-asmparser"

namespace {

class VEOperand;

class VEAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "VEGenAsmMatcher.inc"

  /// }

  // public interface of the MCTargetAsmParser.
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  int parseRegisterName(unsigned (*matchFn)(StringRef));
  OperandMatchResultTy tryParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                        SMLoc &EndLoc) override;
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  bool ParseDirective(AsmToken DirectiveID) override;

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  // Custom parse functions for VE specific operands.
  OperandMatchResultTy parseMEMOperand(OperandVector &Operands);
  OperandMatchResultTy parseMEMAsOperand(OperandVector &Operands);
  OperandMatchResultTy parseCCOpOperand(OperandVector &Operands);
  OperandMatchResultTy parseRDOpOperand(OperandVector &Operands);
  OperandMatchResultTy parseMImmOperand(OperandVector &Operands);
  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Name);
  OperandMatchResultTy parseVEAsmOperand(std::unique_ptr<VEOperand> &Operand);

  // Helper function to parse expression with a symbol.
  const MCExpr *extractModifierFromExpr(const MCExpr *E,
                                        VEMCExpr::VariantKind &Variant);
  const MCExpr *fixupVariantKind(const MCExpr *E);
  bool parseExpression(const MCExpr *&EVal);

  // Split the mnemonic stripping conditional code and quantifiers
  StringRef splitMnemonic(StringRef Name, SMLoc NameLoc,
                          OperandVector *Operands);

  bool parseLiteralValues(unsigned Size, SMLoc L);

public:
  VEAsmParser(const MCSubtargetInfo &sti, MCAsmParser &parser,
              const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, sti, MII), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
  }
};

} // end anonymous namespace

static const MCPhysReg I32Regs[64] = {
    VE::SW0,  VE::SW1,  VE::SW2,  VE::SW3,  VE::SW4,  VE::SW5,  VE::SW6,
    VE::SW7,  VE::SW8,  VE::SW9,  VE::SW10, VE::SW11, VE::SW12, VE::SW13,
    VE::SW14, VE::SW15, VE::SW16, VE::SW17, VE::SW18, VE::SW19, VE::SW20,
    VE::SW21, VE::SW22, VE::SW23, VE::SW24, VE::SW25, VE::SW26, VE::SW27,
    VE::SW28, VE::SW29, VE::SW30, VE::SW31, VE::SW32, VE::SW33, VE::SW34,
    VE::SW35, VE::SW36, VE::SW37, VE::SW38, VE::SW39, VE::SW40, VE::SW41,
    VE::SW42, VE::SW43, VE::SW44, VE::SW45, VE::SW46, VE::SW47, VE::SW48,
    VE::SW49, VE::SW50, VE::SW51, VE::SW52, VE::SW53, VE::SW54, VE::SW55,
    VE::SW56, VE::SW57, VE::SW58, VE::SW59, VE::SW60, VE::SW61, VE::SW62,
    VE::SW63};

static const MCPhysReg F32Regs[64] = {
    VE::SF0,  VE::SF1,  VE::SF2,  VE::SF3,  VE::SF4,  VE::SF5,  VE::SF6,
    VE::SF7,  VE::SF8,  VE::SF9,  VE::SF10, VE::SF11, VE::SF12, VE::SF13,
    VE::SF14, VE::SF15, VE::SF16, VE::SF17, VE::SF18, VE::SF19, VE::SF20,
    VE::SF21, VE::SF22, VE::SF23, VE::SF24, VE::SF25, VE::SF26, VE::SF27,
    VE::SF28, VE::SF29, VE::SF30, VE::SF31, VE::SF32, VE::SF33, VE::SF34,
    VE::SF35, VE::SF36, VE::SF37, VE::SF38, VE::SF39, VE::SF40, VE::SF41,
    VE::SF42, VE::SF43, VE::SF44, VE::SF45, VE::SF46, VE::SF47, VE::SF48,
    VE::SF49, VE::SF50, VE::SF51, VE::SF52, VE::SF53, VE::SF54, VE::SF55,
    VE::SF56, VE::SF57, VE::SF58, VE::SF59, VE::SF60, VE::SF61, VE::SF62,
    VE::SF63};

static const MCPhysReg F128Regs[32] = {
    VE::Q0,  VE::Q1,  VE::Q2,  VE::Q3,  VE::Q4,  VE::Q5,  VE::Q6,  VE::Q7,
    VE::Q8,  VE::Q9,  VE::Q10, VE::Q11, VE::Q12, VE::Q13, VE::Q14, VE::Q15,
    VE::Q16, VE::Q17, VE::Q18, VE::Q19, VE::Q20, VE::Q21, VE::Q22, VE::Q23,
    VE::Q24, VE::Q25, VE::Q26, VE::Q27, VE::Q28, VE::Q29, VE::Q30, VE::Q31};

static const MCPhysReg VM512Regs[8] = {VE::VMP0, VE::VMP1, VE::VMP2, VE::VMP3,
                                       VE::VMP4, VE::VMP5, VE::VMP6, VE::VMP7};

static const MCPhysReg MISCRegs[31] = {
    VE::USRCC,      VE::PSW,        VE::SAR,        VE::NoRegister,
    VE::NoRegister, VE::NoRegister, VE::NoRegister, VE::PMMR,
    VE::PMCR0,      VE::PMCR1,      VE::PMCR2,      VE::PMCR3,
    VE::NoRegister, VE::NoRegister, VE::NoRegister, VE::NoRegister,
    VE::PMC0,       VE::PMC1,       VE::PMC2,       VE::PMC3,
    VE::PMC4,       VE::PMC5,       VE::PMC6,       VE::PMC7,
    VE::PMC8,       VE::PMC9,       VE::PMC10,      VE::PMC11,
    VE::PMC12,      VE::PMC13,      VE::PMC14};

namespace {

/// VEOperand - Instances of this class represent a parsed VE machine
/// instruction.
class VEOperand : public MCParsedAsmOperand {
private:
  enum KindTy {
    k_Token,
    k_Register,
    k_Immediate,
    // SX-Aurora ASX form is disp(index, base).
    k_MemoryRegRegImm,  // base=reg, index=reg, disp=imm
    k_MemoryRegImmImm,  // base=reg, index=imm, disp=imm
    k_MemoryZeroRegImm, // base=0, index=reg, disp=imm
    k_MemoryZeroImmImm, // base=0, index=imm, disp=imm
    // SX-Aurora AS form is disp(base).
    k_MemoryRegImm,  // base=reg, disp=imm
    k_MemoryZeroImm, // base=0, disp=imm
    // Other special cases for Aurora VE
    k_CCOp,   // condition code
    k_RDOp,   // rounding mode
    k_MImmOp, // Special immediate value of sequential bit stream of 0 or 1.
  } Kind;

  SMLoc StartLoc, EndLoc;

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned Base;
    unsigned IndexReg;
    const MCExpr *Index;
    const MCExpr *Offset;
  };

  struct CCOp {
    unsigned CCVal;
  };

  struct RDOp {
    unsigned RDVal;
  };

  struct MImmOp {
    const MCExpr *Val;
    bool M0Flag;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
    struct CCOp CC;
    struct RDOp RD;
    struct MImmOp MImm;
  };

public:
  VEOperand(KindTy K) : Kind(K) {}

  bool isToken() const override { return Kind == k_Token; }
  bool isReg() const override { return Kind == k_Register; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isMem() const override {
    return isMEMrri() || isMEMrii() || isMEMzri() || isMEMzii() || isMEMri() ||
           isMEMzi();
  }
  bool isMEMrri() const { return Kind == k_MemoryRegRegImm; }
  bool isMEMrii() const { return Kind == k_MemoryRegImmImm; }
  bool isMEMzri() const { return Kind == k_MemoryZeroRegImm; }
  bool isMEMzii() const { return Kind == k_MemoryZeroImmImm; }
  bool isMEMri() const { return Kind == k_MemoryRegImm; }
  bool isMEMzi() const { return Kind == k_MemoryZeroImm; }
  bool isCCOp() const { return Kind == k_CCOp; }
  bool isRDOp() const { return Kind == k_RDOp; }
  bool isZero() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return Value == 0;
    }
    return false;
  }
  bool isUImm0to2() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return Value >= 0 && Value < 3;
    }
    return false;
  }
  bool isUImm1() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<1>(Value);
    }
    return false;
  }
  bool isUImm2() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<2>(Value);
    }
    return false;
  }
  bool isUImm3() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<3>(Value);
    }
    return false;
  }
  bool isUImm4() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<4>(Value);
    }
    return false;
  }
  bool isUImm6() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<6>(Value);
    }
    return false;
  }
  bool isUImm7() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<7>(Value);
    }
    return false;
  }
  bool isSImm7() {
    if (!isImm())
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isInt<7>(Value);
    }
    return false;
  }
  bool isMImm() const {
    if (Kind != k_MImmOp)
      return false;

    // Constant case
    if (const auto *ConstExpr = dyn_cast<MCConstantExpr>(MImm.Val)) {
      int64_t Value = ConstExpr->getValue();
      return isUInt<6>(Value);
    }
    return false;
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const override {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_MemoryRegRegImm || Kind == k_MemoryRegImmImm ||
            Kind == k_MemoryRegImm) &&
           "Invalid access!");
    return Mem.Base;
  }

  unsigned getMemIndexReg() const {
    assert((Kind == k_MemoryRegRegImm || Kind == k_MemoryZeroRegImm) &&
           "Invalid access!");
    return Mem.IndexReg;
  }

  const MCExpr *getMemIndex() const {
    assert((Kind == k_MemoryRegImmImm || Kind == k_MemoryZeroImmImm) &&
           "Invalid access!");
    return Mem.Index;
  }

  const MCExpr *getMemOffset() const {
    assert((Kind == k_MemoryRegRegImm || Kind == k_MemoryRegImmImm ||
            Kind == k_MemoryZeroImmImm || Kind == k_MemoryZeroRegImm ||
            Kind == k_MemoryRegImm || Kind == k_MemoryZeroImm) &&
           "Invalid access!");
    return Mem.Offset;
  }

  void setMemOffset(const MCExpr *off) {
    assert((Kind == k_MemoryRegRegImm || Kind == k_MemoryRegImmImm ||
            Kind == k_MemoryZeroImmImm || Kind == k_MemoryZeroRegImm ||
            Kind == k_MemoryRegImm || Kind == k_MemoryZeroImm) &&
           "Invalid access!");
    Mem.Offset = off;
  }

  unsigned getCCVal() const {
    assert((Kind == k_CCOp) && "Invalid access!");
    return CC.CCVal;
  }

  unsigned getRDVal() const {
    assert((Kind == k_RDOp) && "Invalid access!");
    return RD.RDVal;
  }

  const MCExpr *getMImmVal() const {
    assert((Kind == k_MImmOp) && "Invalid access!");
    return MImm.Val;
  }
  bool getM0Flag() const {
    assert((Kind == k_MImmOp) && "Invalid access!");
    return MImm.M0Flag;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case k_Token:
      OS << "Token: " << getToken() << "\n";
      break;
    case k_Register:
      OS << "Reg: #" << getReg() << "\n";
      break;
    case k_Immediate:
      OS << "Imm: " << getImm() << "\n";
      break;
    case k_MemoryRegRegImm:
      assert(getMemOffset() != nullptr);
      OS << "Mem: #" << getMemBase() << "+#" << getMemIndexReg() << "+"
         << *getMemOffset() << "\n";
      break;
    case k_MemoryRegImmImm:
      assert(getMemIndex() != nullptr && getMemOffset() != nullptr);
      OS << "Mem: #" << getMemBase() << "+" << *getMemIndex() << "+"
         << *getMemOffset() << "\n";
      break;
    case k_MemoryZeroRegImm:
      assert(getMemOffset() != nullptr);
      OS << "Mem: 0+#" << getMemIndexReg() << "+" << *getMemOffset() << "\n";
      break;
    case k_MemoryZeroImmImm:
      assert(getMemIndex() != nullptr && getMemOffset() != nullptr);
      OS << "Mem: 0+" << *getMemIndex() << "+" << *getMemOffset() << "\n";
      break;
    case k_MemoryRegImm:
      assert(getMemOffset() != nullptr);
      OS << "Mem: #" << getMemBase() << "+" << *getMemOffset() << "\n";
      break;
    case k_MemoryZeroImm:
      assert(getMemOffset() != nullptr);
      OS << "Mem: 0+" << *getMemOffset() << "\n";
      break;
    case k_CCOp:
      OS << "CCOp: " << getCCVal() << "\n";
      break;
    case k_RDOp:
      OS << "RDOp: " << getRDVal() << "\n";
      break;
    case k_MImmOp:
      OS << "MImm: (" << getMImmVal() << (getM0Flag() ? ")0" : ")1") << "\n";
      break;
    }
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addZeroOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm0to2Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm1Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm2Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm3Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm4Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm6Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addUImm7Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addSImm7Operands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediate when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addMEMrriOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));
    Inst.addOperand(MCOperand::createReg(getMemIndexReg()));
    addExpr(Inst, getMemOffset());
  }

  void addMEMriiOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));
    addExpr(Inst, getMemIndex());
    addExpr(Inst, getMemOffset());
  }

  void addMEMzriOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createImm(0));
    Inst.addOperand(MCOperand::createReg(getMemIndexReg()));
    addExpr(Inst, getMemOffset());
  }

  void addMEMziiOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createImm(0));
    addExpr(Inst, getMemIndex());
    addExpr(Inst, getMemOffset());
  }

  void addMEMriOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));
    addExpr(Inst, getMemOffset());
  }

  void addMEMziOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createImm(0));
    addExpr(Inst, getMemOffset());
  }

  void addCCOpOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createImm(getCCVal()));
  }

  void addRDOpOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createImm(getRDVal()));
  }

  void addMImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const auto *ConstExpr = dyn_cast<MCConstantExpr>(getMImmVal());
    assert(ConstExpr && "Null operands!");
    int64_t Value = ConstExpr->getValue();
    if (getM0Flag())
      Value += 64;
    Inst.addOperand(MCOperand::createImm(Value));
  }

  static std::unique_ptr<VEOperand> CreateToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<VEOperand>(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<VEOperand> CreateReg(unsigned RegNum, SMLoc S,
                                              SMLoc E) {
    auto Op = std::make_unique<VEOperand>(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<VEOperand> CreateImm(const MCExpr *Val, SMLoc S,
                                              SMLoc E) {
    auto Op = std::make_unique<VEOperand>(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<VEOperand> CreateCCOp(unsigned CCVal, SMLoc S,
                                               SMLoc E) {
    auto Op = std::make_unique<VEOperand>(k_CCOp);
    Op->CC.CCVal = CCVal;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<VEOperand> CreateRDOp(unsigned RDVal, SMLoc S,
                                               SMLoc E) {
    auto Op = std::make_unique<VEOperand>(k_RDOp);
    Op->RD.RDVal = RDVal;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<VEOperand> CreateMImm(const MCExpr *Val, bool Flag,
                                               SMLoc S, SMLoc E) {
    auto Op = std::make_unique<VEOperand>(k_MImmOp);
    Op->MImm.Val = Val;
    Op->MImm.M0Flag = Flag;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static bool MorphToI32Reg(VEOperand &Op) {
    unsigned Reg = Op.getReg();
    unsigned regIdx = Reg - VE::SX0;
    if (regIdx > 63)
      return false;
    Op.Reg.RegNum = I32Regs[regIdx];
    return true;
  }

  static bool MorphToF32Reg(VEOperand &Op) {
    unsigned Reg = Op.getReg();
    unsigned regIdx = Reg - VE::SX0;
    if (regIdx > 63)
      return false;
    Op.Reg.RegNum = F32Regs[regIdx];
    return true;
  }

  static bool MorphToF128Reg(VEOperand &Op) {
    unsigned Reg = Op.getReg();
    unsigned regIdx = Reg - VE::SX0;
    if (regIdx % 2 || regIdx > 63)
      return false;
    Op.Reg.RegNum = F128Regs[regIdx / 2];
    return true;
  }

  static bool MorphToVM512Reg(VEOperand &Op) {
    unsigned Reg = Op.getReg();
    unsigned regIdx = Reg - VE::VM0;
    if (regIdx % 2 || regIdx > 15)
      return false;
    Op.Reg.RegNum = VM512Regs[regIdx / 2];
    return true;
  }

  static bool MorphToMISCReg(VEOperand &Op) {
    const auto *ConstExpr = dyn_cast<MCConstantExpr>(Op.getImm());
    if (!ConstExpr)
      return false;
    unsigned regIdx = ConstExpr->getValue();
    if (regIdx > 31 || MISCRegs[regIdx] == VE::NoRegister)
      return false;
    Op.Kind = k_Register;
    Op.Reg.RegNum = MISCRegs[regIdx];
    return true;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMri(unsigned Base, std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryRegImm;
    Op->Mem.Base = Base;
    Op->Mem.IndexReg = 0;
    Op->Mem.Index = nullptr;
    Op->Mem.Offset = Imm;
    return Op;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMzi(std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryZeroImm;
    Op->Mem.Base = 0;
    Op->Mem.IndexReg = 0;
    Op->Mem.Index = nullptr;
    Op->Mem.Offset = Imm;
    return Op;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMrri(unsigned Base, unsigned Index, std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryRegRegImm;
    Op->Mem.Base = Base;
    Op->Mem.IndexReg = Index;
    Op->Mem.Index = nullptr;
    Op->Mem.Offset = Imm;
    return Op;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMrii(unsigned Base, const MCExpr *Index,
                std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryRegImmImm;
    Op->Mem.Base = Base;
    Op->Mem.IndexReg = 0;
    Op->Mem.Index = Index;
    Op->Mem.Offset = Imm;
    return Op;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMzri(unsigned Index, std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryZeroRegImm;
    Op->Mem.Base = 0;
    Op->Mem.IndexReg = Index;
    Op->Mem.Index = nullptr;
    Op->Mem.Offset = Imm;
    return Op;
  }

  static std::unique_ptr<VEOperand>
  MorphToMEMzii(const MCExpr *Index, std::unique_ptr<VEOperand> Op) {
    const MCExpr *Imm = Op->getImm();
    Op->Kind = k_MemoryZeroImmImm;
    Op->Mem.Base = 0;
    Op->Mem.IndexReg = 0;
    Op->Mem.Index = Index;
    Op->Mem.Offset = Imm;
    return Op;
  }
};

} // end anonymous namespace

bool VEAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                          OperandVector &Operands,
                                          MCStreamer &Out, uint64_t &ErrorInfo,
                                          bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);
  switch (MatchResult) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;

  case Match_MissingFeature:
    return Error(IDLoc,
                 "instruction requires a CPU feature not currently enabled");

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((VEOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction mnemonic");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool VEAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                SMLoc &EndLoc) {
  if (tryParseRegister(RegNo, StartLoc, EndLoc) != MatchOperand_Success)
    return Error(StartLoc, "invalid register name");
  return false;
}

/// Parses a register name using a given matching function.
/// Checks for lowercase or uppercase if necessary.
int VEAsmParser::parseRegisterName(unsigned (*matchFn)(StringRef)) {
  StringRef Name = Parser.getTok().getString();

  int RegNum = matchFn(Name);

  // GCC supports case insensitive register names. All of the VE registers
  // are all lower case.
  if (RegNum == VE::NoRegister) {
    RegNum = matchFn(Name.lower());
  }

  return RegNum;
}

/// Maps from the set of all register names to a register number.
/// \note Generated by TableGen.
static unsigned MatchRegisterName(StringRef Name);

/// Maps from the set of all alternative registernames to a register number.
/// \note Generated by TableGen.
static unsigned MatchRegisterAltName(StringRef Name);

OperandMatchResultTy
VEAsmParser::tryParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  if (getLexer().getKind() != AsmToken::Percent)
    return MatchOperand_NoMatch;
  Parser.Lex();

  RegNo = parseRegisterName(&MatchRegisterName);
  if (RegNo == VE::NoRegister)
    RegNo = parseRegisterName(&MatchRegisterAltName);

  if (RegNo != VE::NoRegister) {
    Parser.Lex();
    return MatchOperand_Success;
  }

  getLexer().UnLex(Tok);
  return MatchOperand_NoMatch;
}

static StringRef parseCC(StringRef Name, unsigned Prefix, unsigned Suffix,
                         bool IntegerCC, bool OmitCC, SMLoc NameLoc,
                         OperandVector *Operands) {
  // Parse instructions with a conditional code. For example, 'bne' is
  // converted into two operands 'b' and 'ne'.
  StringRef Cond = Name.slice(Prefix, Suffix);
  VECC::CondCode CondCode =
      IntegerCC ? stringToVEICondCode(Cond) : stringToVEFCondCode(Cond);

  // If OmitCC is enabled, CC_AT and CC_AF is treated as a part of mnemonic.
  if (CondCode != VECC::UNKNOWN &&
      (!OmitCC || (CondCode != VECC::CC_AT && CondCode != VECC::CC_AF))) {
    StringRef SuffixStr = Name.substr(Suffix);
    // Push "b".
    Name = Name.slice(0, Prefix);
    Operands->push_back(VEOperand::CreateToken(Name, NameLoc));
    // Push $cond part.
    SMLoc CondLoc = SMLoc::getFromPointer(NameLoc.getPointer() + Prefix);
    SMLoc SuffixLoc = SMLoc::getFromPointer(NameLoc.getPointer() + Suffix);
    Operands->push_back(VEOperand::CreateCCOp(CondCode, CondLoc, SuffixLoc));
    // push suffix like ".l.t"
    if (!SuffixStr.empty())
      Operands->push_back(VEOperand::CreateToken(SuffixStr, SuffixLoc));
  } else {
    Operands->push_back(VEOperand::CreateToken(Name, NameLoc));
  }
  return Name;
}

static StringRef parseRD(StringRef Name, unsigned Prefix, SMLoc NameLoc,
                         OperandVector *Operands) {
  // Parse instructions with a conditional code. For example, 'cvt.w.d.sx.rz'
  // is converted into two operands 'cvt.w.d.sx' and '.rz'.
  StringRef RD = Name.substr(Prefix);
  VERD::RoundingMode RoundingMode = stringToVERD(RD);

  if (RoundingMode != VERD::UNKNOWN) {
    Name = Name.slice(0, Prefix);
    // push 1st like `cvt.w.d.sx`
    Operands->push_back(VEOperand::CreateToken(Name, NameLoc));
    SMLoc SuffixLoc =
        SMLoc::getFromPointer(NameLoc.getPointer() + (RD.data() - Name.data()));
    SMLoc SuffixEnd =
        SMLoc::getFromPointer(NameLoc.getPointer() + (RD.end() - Name.data()));
    // push $round if it has rounding mode
    Operands->push_back(
        VEOperand::CreateRDOp(RoundingMode, SuffixLoc, SuffixEnd));
  } else {
    Operands->push_back(VEOperand::CreateToken(Name, NameLoc));
  }
  return Name;
}

// Split the mnemonic into ASM operand, conditional code and instruction
// qualifier (half-word, byte).
StringRef VEAsmParser::splitMnemonic(StringRef Name, SMLoc NameLoc,
                                     OperandVector *Operands) {
  // Create the leading tokens for the mnemonic
  StringRef Mnemonic = Name;

  if (Name[0] == 'b') {
    // Match b?? or br??.
    size_t Start = 1;
    size_t Next = Name.find('.');
    // Adjust position of CondCode.
    if (Name.size() > 1 && Name[1] == 'r')
      Start = 2;
    // Check suffix.
    bool ICC = true;
    if (Next + 1 < Name.size() &&
        (Name[Next + 1] == 'd' || Name[Next + 1] == 's'))
      ICC = false;
    Mnemonic = parseCC(Name, Start, Next, ICC, true, NameLoc, Operands);
  } else if (Name.startswith("cmov.l.") || Name.startswith("cmov.w.") ||
             Name.startswith("cmov.d.") || Name.startswith("cmov.s.")) {
    bool ICC = Name[5] == 'l' || Name[5] == 'w';
    Mnemonic = parseCC(Name, 7, Name.size(), ICC, false, NameLoc, Operands);
  } else if (Name.startswith("cvt.w.d.sx") || Name.startswith("cvt.w.d.zx") ||
             Name.startswith("cvt.w.s.sx") || Name.startswith("cvt.w.s.zx")) {
    Mnemonic = parseRD(Name, 10, NameLoc, Operands);
  } else if (Name.startswith("cvt.l.d")) {
    Mnemonic = parseRD(Name, 7, NameLoc, Operands);
  } else if (Name.startswith("vcvt.w.d.sx") || Name.startswith("vcvt.w.d.zx") ||
             Name.startswith("vcvt.w.s.sx") || Name.startswith("vcvt.w.s.zx")) {
    Mnemonic = parseRD(Name, 11, NameLoc, Operands);
  } else if (Name.startswith("vcvt.l.d")) {
    Mnemonic = parseRD(Name, 8, NameLoc, Operands);
  } else if (Name.startswith("pvcvt.w.s.lo") ||
             Name.startswith("pvcvt.w.s.up")) {
    Mnemonic = parseRD(Name, 12, NameLoc, Operands);
  } else if (Name.startswith("pvcvt.w.s")) {
    Mnemonic = parseRD(Name, 9, NameLoc, Operands);
  } else if (Name.startswith("vfmk.l.") || Name.startswith("vfmk.w.") ||
             Name.startswith("vfmk.d.") || Name.startswith("vfmk.s.")) {
    bool ICC = Name[5] == 'l' || Name[5] == 'w' ? true : false;
    Mnemonic = parseCC(Name, 7, Name.size(), ICC, true, NameLoc, Operands);
  } else if (Name.startswith("pvfmk.w.lo.") || Name.startswith("pvfmk.w.up.") ||
             Name.startswith("pvfmk.s.lo.") || Name.startswith("pvfmk.s.up.")) {
    bool ICC = Name[6] == 'l' || Name[6] == 'w' ? true : false;
    Mnemonic = parseCC(Name, 11, Name.size(), ICC, true, NameLoc, Operands);
  } else {
    Operands->push_back(VEOperand::CreateToken(Mnemonic, NameLoc));
  }

  return Mnemonic;
}

static void applyMnemonicAliases(StringRef &Mnemonic,
                                 const FeatureBitset &Features,
                                 unsigned VariantID);

bool VEAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                   SMLoc NameLoc, OperandVector &Operands) {
  // If the target architecture uses MnemonicAlias, call it here to parse
  // operands correctly.
  applyMnemonicAliases(Name, getAvailableFeatures(), 0);

  // Split name to first token and the rest, e.g. "bgt.l.t" to "b", "gt", and
  // ".l.t".  We treat "b" as a mnemonic, "gt" as first operand, and ".l.t"
  // as second operand.
  StringRef Mnemonic = splitMnemonic(Name, NameLoc, &Operands);

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Mnemonic) != MatchOperand_Success) {
      SMLoc Loc = getLexer().getLoc();
      return Error(Loc, "unexpected token");
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      // Parse and remember the operand.
      if (parseOperand(Operands, Mnemonic) != MatchOperand_Success) {
        SMLoc Loc = getLexer().getLoc();
        return Error(Loc, "unexpected token");
      }
    }
  }
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    return Error(Loc, "unexpected token");
  }
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool VEAsmParser::ParseDirective(AsmToken DirectiveID) {
  std::string IDVal = DirectiveID.getIdentifier().lower();

  // Defines VE specific directives.  Reference is "Vector Engine Assembly
  // Language Reference Manual":
  // https://www.hpc.nec/documents/sdk/pdfs/VectorEngine-as-manual-v1.3.pdf

  // The .word is 4 bytes long on VE.
  if (IDVal == ".word")
    return parseLiteralValues(4, DirectiveID.getLoc());

  // The .long is 8 bytes long on VE.
  if (IDVal == ".long")
    return parseLiteralValues(8, DirectiveID.getLoc());

  // The .llong is 8 bytes long on VE.
  if (IDVal == ".llong")
    return parseLiteralValues(8, DirectiveID.getLoc());

  // Let the MC layer to handle other directives.
  return true;
}

/// parseLiteralValues
///  ::= .word expression [, expression]*
///  ::= .long expression [, expression]*
///  ::= .llong expression [, expression]*
bool VEAsmParser::parseLiteralValues(unsigned Size, SMLoc L) {
  auto parseOne = [&]() -> bool {
    const MCExpr *Value;
    if (getParser().parseExpression(Value))
      return true;
    getParser().getStreamer().emitValue(Value, Size, L);
    return false;
  };
  return (parseMany(parseOne));
}

/// Extract \code @lo32/@hi32/etc \endcode modifier from expression.
/// Recursively scan the expression and check for VK_VE_HI32/LO32/etc
/// symbol variants.  If all symbols with modifier use the same
/// variant, return the corresponding VEMCExpr::VariantKind,
/// and a modified expression using the default symbol variant.
/// Otherwise, return NULL.
const MCExpr *
VEAsmParser::extractModifierFromExpr(const MCExpr *E,
                                     VEMCExpr::VariantKind &Variant) {
  MCContext &Context = getParser().getContext();
  Variant = VEMCExpr::VK_VE_None;

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return nullptr;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    switch (SRE->getKind()) {
    case MCSymbolRefExpr::VK_None:
      // Use VK_VE_REFLONG to a symbol without modifiers.
      Variant = VEMCExpr::VK_VE_REFLONG;
      break;
    case MCSymbolRefExpr::VK_VE_HI32:
      Variant = VEMCExpr::VK_VE_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_LO32:
      Variant = VEMCExpr::VK_VE_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_PC_HI32:
      Variant = VEMCExpr::VK_VE_PC_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_PC_LO32:
      Variant = VEMCExpr::VK_VE_PC_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_GOT_HI32:
      Variant = VEMCExpr::VK_VE_GOT_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_GOT_LO32:
      Variant = VEMCExpr::VK_VE_GOT_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_GOTOFF_HI32:
      Variant = VEMCExpr::VK_VE_GOTOFF_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_GOTOFF_LO32:
      Variant = VEMCExpr::VK_VE_GOTOFF_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_PLT_HI32:
      Variant = VEMCExpr::VK_VE_PLT_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_PLT_LO32:
      Variant = VEMCExpr::VK_VE_PLT_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_TLS_GD_HI32:
      Variant = VEMCExpr::VK_VE_TLS_GD_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_TLS_GD_LO32:
      Variant = VEMCExpr::VK_VE_TLS_GD_LO32;
      break;
    case MCSymbolRefExpr::VK_VE_TPOFF_HI32:
      Variant = VEMCExpr::VK_VE_TPOFF_HI32;
      break;
    case MCSymbolRefExpr::VK_VE_TPOFF_LO32:
      Variant = VEMCExpr::VK_VE_TPOFF_LO32;
      break;
    default:
      return nullptr;
    }

    return MCSymbolRefExpr::create(&SRE->getSymbol(), Context);
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = extractModifierFromExpr(UE->getSubExpr(), Variant);
    if (!Sub)
      return nullptr;
    return MCUnaryExpr::create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    VEMCExpr::VariantKind LHSVariant, RHSVariant;
    const MCExpr *LHS = extractModifierFromExpr(BE->getLHS(), LHSVariant);
    const MCExpr *RHS = extractModifierFromExpr(BE->getRHS(), RHSVariant);

    if (!LHS && !RHS)
      return nullptr;

    if (!LHS)
      LHS = BE->getLHS();
    if (!RHS)
      RHS = BE->getRHS();

    if (LHSVariant == VEMCExpr::VK_VE_None)
      Variant = RHSVariant;
    else if (RHSVariant == VEMCExpr::VK_VE_None)
      Variant = LHSVariant;
    else if (LHSVariant == RHSVariant)
      Variant = LHSVariant;
    else
      return nullptr;

    return MCBinaryExpr::create(BE->getOpcode(), LHS, RHS, Context);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

const MCExpr *VEAsmParser::fixupVariantKind(const MCExpr *E) {
  MCContext &Context = getParser().getContext();

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
  case MCExpr::SymbolRef:
    return E;

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = fixupVariantKind(UE->getSubExpr());
    if (Sub == UE->getSubExpr())
      return E;
    return MCUnaryExpr::create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = fixupVariantKind(BE->getLHS());
    const MCExpr *RHS = fixupVariantKind(BE->getRHS());
    if (LHS == BE->getLHS() && RHS == BE->getRHS())
      return E;
    return MCBinaryExpr::create(BE->getOpcode(), LHS, RHS, Context);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// ParseExpression.  This differs from the default "parseExpression" in that
/// it handles modifiers.
bool VEAsmParser::parseExpression(const MCExpr *&EVal) {
  // Handle \code symbol @lo32/@hi32/etc \endcode.
  if (getParser().parseExpression(EVal))
    return true;

  // Convert MCSymbolRefExpr with VK_* to MCExpr with VK_*.
  EVal = fixupVariantKind(EVal);
  VEMCExpr::VariantKind Variant;
  const MCExpr *E = extractModifierFromExpr(EVal, Variant);
  if (E)
    EVal = VEMCExpr::create(Variant, E, getParser().getContext());

  return false;
}

OperandMatchResultTy VEAsmParser::parseMEMOperand(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << "parseMEMOperand\n");
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  SMLoc E = Tok.getEndLoc();
  // Parse ASX format
  //   disp
  //   disp(, base)
  //   disp(index)
  //   disp(index, base)
  //   (, base)
  //   (index)
  //   (index, base)

  std::unique_ptr<VEOperand> Offset;
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;

  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Dot:
  case AsmToken::Identifier: {
    const MCExpr *EVal;
    if (!parseExpression(EVal))
      Offset = VEOperand::CreateImm(EVal, S, E);
    else
      return MatchOperand_NoMatch;
    break;
  }

  case AsmToken::LParen:
    // empty disp (= 0)
    Offset =
        VEOperand::CreateImm(MCConstantExpr::create(0, getContext()), S, E);
    break;
  }

  switch (getLexer().getKind()) {
  default:
    return MatchOperand_ParseFail;

  case AsmToken::EndOfStatement:
    Operands.push_back(VEOperand::MorphToMEMzii(
        MCConstantExpr::create(0, getContext()), std::move(Offset)));
    return MatchOperand_Success;

  case AsmToken::LParen:
    Parser.Lex(); // Eat the (
    break;
  }

  const MCExpr *IndexValue = nullptr;
  unsigned IndexReg = 0;

  switch (getLexer().getKind()) {
  default:
    if (ParseRegister(IndexReg, S, E))
      return MatchOperand_ParseFail;
    break;

  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Dot:
    if (getParser().parseExpression(IndexValue, E))
      return MatchOperand_ParseFail;
    break;

  case AsmToken::Comma:
    // empty index
    IndexValue = MCConstantExpr::create(0, getContext());
    break;
  }

  switch (getLexer().getKind()) {
  default:
    return MatchOperand_ParseFail;

  case AsmToken::RParen:
    Parser.Lex(); // Eat the )
    Operands.push_back(
        IndexValue ? VEOperand::MorphToMEMzii(IndexValue, std::move(Offset))
                   : VEOperand::MorphToMEMzri(IndexReg, std::move(Offset)));
    return MatchOperand_Success;

  case AsmToken::Comma:
    Parser.Lex(); // Eat the ,
    break;
  }

  unsigned BaseReg = 0;
  if (ParseRegister(BaseReg, S, E))
    return MatchOperand_ParseFail;

  if (!Parser.getTok().is(AsmToken::RParen))
    return MatchOperand_ParseFail;

  Parser.Lex(); // Eat the )
  Operands.push_back(
      IndexValue
          ? VEOperand::MorphToMEMrii(BaseReg, IndexValue, std::move(Offset))
          : VEOperand::MorphToMEMrri(BaseReg, IndexReg, std::move(Offset)));

  return MatchOperand_Success;
}

OperandMatchResultTy VEAsmParser::parseMEMAsOperand(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << "parseMEMAsOperand\n");
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();
  SMLoc E = Tok.getEndLoc();
  // Parse AS format
  //   disp
  //   disp(, base)
  //   disp(base)
  //   disp()
  //   (, base)
  //   (base)
  //   base

  unsigned BaseReg = VE::NoRegister;
  std::unique_ptr<VEOperand> Offset;
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;

  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Dot:
  case AsmToken::Identifier: {
    const MCExpr *EVal;
    if (!parseExpression(EVal))
      Offset = VEOperand::CreateImm(EVal, S, E);
    else
      return MatchOperand_NoMatch;
    break;
  }

  case AsmToken::Percent:
    if (ParseRegister(BaseReg, S, E))
      return MatchOperand_NoMatch;
    Offset =
        VEOperand::CreateImm(MCConstantExpr::create(0, getContext()), S, E);
    break;

  case AsmToken::LParen:
    // empty disp (= 0)
    Offset =
        VEOperand::CreateImm(MCConstantExpr::create(0, getContext()), S, E);
    break;
  }

  switch (getLexer().getKind()) {
  default:
    return MatchOperand_ParseFail;

  case AsmToken::EndOfStatement:
  case AsmToken::Comma:
    Operands.push_back(BaseReg != VE::NoRegister
                           ? VEOperand::MorphToMEMri(BaseReg, std::move(Offset))
                           : VEOperand::MorphToMEMzi(std::move(Offset)));
    return MatchOperand_Success;

  case AsmToken::LParen:
    if (BaseReg != VE::NoRegister)
      return MatchOperand_ParseFail;
    Parser.Lex(); // Eat the (
    break;
  }

  switch (getLexer().getKind()) {
  default:
    if (ParseRegister(BaseReg, S, E))
      return MatchOperand_ParseFail;
    break;

  case AsmToken::Comma:
    Parser.Lex(); // Eat the ,
    if (ParseRegister(BaseReg, S, E))
      return MatchOperand_ParseFail;
    break;

  case AsmToken::RParen:
    break;
  }

  if (!Parser.getTok().is(AsmToken::RParen))
    return MatchOperand_ParseFail;

  Parser.Lex(); // Eat the )
  Operands.push_back(BaseReg != VE::NoRegister
                         ? VEOperand::MorphToMEMri(BaseReg, std::move(Offset))
                         : VEOperand::MorphToMEMzi(std::move(Offset)));

  return MatchOperand_Success;
}

OperandMatchResultTy VEAsmParser::parseMImmOperand(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << "parseMImmOperand\n");

  // Parsing "(" + number + ")0/1"
  const AsmToken Tok1 = Parser.getTok();
  if (!Tok1.is(AsmToken::LParen))
    return MatchOperand_NoMatch;

  Parser.Lex(); // Eat the '('.

  const AsmToken Tok2 = Parser.getTok();
  SMLoc E;
  const MCExpr *EVal;
  if (!Tok2.is(AsmToken::Integer) || getParser().parseExpression(EVal, E)) {
    getLexer().UnLex(Tok1);
    return MatchOperand_NoMatch;
  }

  const AsmToken Tok3 = Parser.getTok();
  if (!Tok3.is(AsmToken::RParen)) {
    getLexer().UnLex(Tok2);
    getLexer().UnLex(Tok1);
    return MatchOperand_NoMatch;
  }
  Parser.Lex(); // Eat the ')'.

  const AsmToken &Tok4 = Parser.getTok();
  StringRef Suffix = Tok4.getString();
  if (Suffix != "1" && Suffix != "0") {
    getLexer().UnLex(Tok3);
    getLexer().UnLex(Tok2);
    getLexer().UnLex(Tok1);
    return MatchOperand_NoMatch;
  }
  Parser.Lex(); // Eat the value.
  SMLoc EndLoc = SMLoc::getFromPointer(Suffix.end());
  Operands.push_back(
      VEOperand::CreateMImm(EVal, Suffix == "0", Tok1.getLoc(), EndLoc));
  return MatchOperand_Success;
}

OperandMatchResultTy VEAsmParser::parseOperand(OperandVector &Operands,
                                               StringRef Mnemonic) {
  LLVM_DEBUG(dbgs() << "parseOperand\n");
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail)
    return ResTy;

  switch (getLexer().getKind()) {
  case AsmToken::LParen: {
    // Parsing "(" + %vreg + ", " + %vreg + ")"
    const AsmToken Tok1 = Parser.getTok();
    Parser.Lex(); // Eat the '('.

    unsigned RegNo1;
    SMLoc S1, E1;
    if (tryParseRegister(RegNo1, S1, E1) != MatchOperand_Success) {
      getLexer().UnLex(Tok1);
      return MatchOperand_NoMatch;
    }

    if (!Parser.getTok().is(AsmToken::Comma))
      return MatchOperand_ParseFail;
    Parser.Lex(); // Eat the ','.

    unsigned RegNo2;
    SMLoc S2, E2;
    if (tryParseRegister(RegNo2, S2, E2) != MatchOperand_Success)
      return MatchOperand_ParseFail;

    if (!Parser.getTok().is(AsmToken::RParen))
      return MatchOperand_ParseFail;

    Operands.push_back(VEOperand::CreateToken(Tok1.getString(), Tok1.getLoc()));
    Operands.push_back(VEOperand::CreateReg(RegNo1, S1, E1));
    Operands.push_back(VEOperand::CreateReg(RegNo2, S2, E2));
    Operands.push_back(VEOperand::CreateToken(Parser.getTok().getString(),
                                              Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the ')'.
    break;
  }
  default: {
    std::unique_ptr<VEOperand> Op;
    ResTy = parseVEAsmOperand(Op);
    if (ResTy != MatchOperand_Success || !Op)
      return MatchOperand_ParseFail;

    // Push the parsed operand into the list of operands
    Operands.push_back(std::move(Op));

    if (!Parser.getTok().is(AsmToken::LParen))
      break;

    // Parsing %vec-reg + "(" + %sclar-reg/number + ")"
    std::unique_ptr<VEOperand> Op1 = VEOperand::CreateToken(
        Parser.getTok().getString(), Parser.getTok().getLoc());
    Parser.Lex(); // Eat the '('.

    std::unique_ptr<VEOperand> Op2;
    ResTy = parseVEAsmOperand(Op2);
    if (ResTy != MatchOperand_Success || !Op2)
      return MatchOperand_ParseFail;

    if (!Parser.getTok().is(AsmToken::RParen))
      return MatchOperand_ParseFail;

    Operands.push_back(std::move(Op1));
    Operands.push_back(std::move(Op2));
    Operands.push_back(VEOperand::CreateToken(Parser.getTok().getString(),
                                              Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the ')'.
    break;
  }
  }

  return MatchOperand_Success;
}

OperandMatchResultTy
VEAsmParser::parseVEAsmOperand(std::unique_ptr<VEOperand> &Op) {
  LLVM_DEBUG(dbgs() << "parseVEAsmOperand\n");
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  const MCExpr *EVal;

  Op = nullptr;
  switch (getLexer().getKind()) {
  default:
    break;

  case AsmToken::Percent:
    unsigned RegNo;
    if (tryParseRegister(RegNo, S, E) == MatchOperand_Success)
      Op = VEOperand::CreateReg(RegNo, S, E);
    break;

  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Dot:
  case AsmToken::Identifier:
    if (!parseExpression(EVal))
      Op = VEOperand::CreateImm(EVal, S, E);
    break;
  }
  return (Op) ? MatchOperand_Success : MatchOperand_ParseFail;
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeVEAsmParser() {
  RegisterMCAsmParser<VEAsmParser> A(getTheVETarget());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "VEGenAsmMatcher.inc"

unsigned VEAsmParser::validateTargetOperandClass(MCParsedAsmOperand &GOp,
                                                 unsigned Kind) {
  VEOperand &Op = (VEOperand &)GOp;

  // VE uses identical register name for all registers like both
  // F32 and I32 uses "%s23".  Need to convert the name of them
  // for validation.
  switch (Kind) {
  default:
    break;
  case MCK_F32:
    if (Op.isReg() && VEOperand::MorphToF32Reg(Op))
      return MCTargetAsmParser::Match_Success;
    break;
  case MCK_I32:
    if (Op.isReg() && VEOperand::MorphToI32Reg(Op))
      return MCTargetAsmParser::Match_Success;
    break;
  case MCK_F128:
    if (Op.isReg() && VEOperand::MorphToF128Reg(Op))
      return MCTargetAsmParser::Match_Success;
    break;
  case MCK_VM512:
    if (Op.isReg() && VEOperand::MorphToVM512Reg(Op))
      return MCTargetAsmParser::Match_Success;
    break;
  case MCK_MISC:
    if (Op.isImm() && VEOperand::MorphToMISCReg(Op))
      return MCTargetAsmParser::Match_Success;
    break;
  }
  return Match_InvalidOperand;
}
