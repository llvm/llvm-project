//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsRegisterInfo.h"
#include "MipsTargetStreamer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/APInt.h"

using namespace llvm;

namespace llvm {
class MCInstrInfo;
}

namespace {
class MipsAssemblerOptions {
public:
  MipsAssemblerOptions() : aTReg(1), reorder(true), macro(true) {}

  unsigned getATRegNum() { return aTReg; }
  bool setATReg(unsigned Reg);

  bool isReorder() { return reorder; }
  void setReorder() { reorder = true; }
  void setNoreorder() { reorder = false; }

  bool isMacro() { return macro; }
  void setMacro() { macro = true; }
  void setNomacro() { macro = false; }

private:
  unsigned aTReg;
  bool reorder;
  bool macro;
};
}

namespace {
class MipsAsmParser : public MCTargetAsmParser {

  MipsTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = Parser.getStreamer().getTargetStreamer();
    return static_cast<MipsTargetStreamer &>(TS);
  }

  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  MipsAssemblerOptions Options;
  bool hasConsumedDollar;

#define GET_ASSEMBLER_HEADER
#include "MipsGenAsmMatcher.inc"

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  bool ParseDirective(AsmToken DirectiveID);

  MipsAsmParser::OperandMatchResultTy
  parseRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands, int RegKind);

  MipsAsmParser::OperandMatchResultTy
  parseMSARegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands, int RegKind);

  MipsAsmParser::OperandMatchResultTy
  parseMSACtrlRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                   int RegKind);

  MipsAsmParser::OperandMatchResultTy
  parseMemOperand(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  bool parsePtrReg(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                   int RegKind);

  MipsAsmParser::OperandMatchResultTy
  parsePtrReg(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseGPR32(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseGPR64(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseHWRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseCCRRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseAFGR64Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseFGR64Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseFGR32Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseFGRH32Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseFCCRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseACC64DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseLO32DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseHI32DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseCOP2(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseMSA128BRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseMSA128HRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseMSA128WRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseMSA128DRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseMSA128CtrlRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseInvNum(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  MipsAsmParser::OperandMatchResultTy
  parseLSAImm(SmallVectorImpl<MCParsedAsmOperand *> &Operands);

  bool searchSymbolAlias(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                         unsigned RegKind);

  bool ParseOperand(SmallVectorImpl<MCParsedAsmOperand *> &,
                    StringRef Mnemonic);

  int tryParseRegister(bool is64BitReg);

  bool tryParseRegisterOperand(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                               bool is64BitReg);

  bool needsExpansion(MCInst &Inst);

  void expandInstruction(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions);
  void expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);
  void expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);
  void expandLoadAddressReg(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);
  void expandMemInst(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions, bool isLoad,
                     bool isImmOpnd);
  bool reportParseError(StringRef ErrorMsg);

  bool parseMemOffset(const MCExpr *&Res, bool isParenExpr);
  bool parseRelocOperand(const MCExpr *&Res);

  const MCExpr *evaluateRelocExpr(const MCExpr *Expr, StringRef RelocStr);

  bool isEvaluated(const MCExpr *Expr);
  bool parseDirectiveSet();
  bool parseDirectiveMipsHackStocg();
  bool parseDirectiveMipsHackELFFlags();

  bool parseSetAtDirective();
  bool parseSetNoAtDirective();
  bool parseSetMacroDirective();
  bool parseSetNoMacroDirective();
  bool parseSetReorderDirective();
  bool parseSetNoReorderDirective();

  bool parseSetAssignment();

  bool parseDirectiveWord(unsigned Size, SMLoc L);
  bool parseDirectiveGpWord();

  MCSymbolRefExpr::VariantKind getVariantKind(StringRef Symbol);

  bool isMips64() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64) != 0;
  }

  bool isFP64() const {
    return (STI.getFeatureBits() & Mips::FeatureFP64Bit) != 0;
  }

  bool isN64() const { return STI.getFeatureBits() & Mips::FeatureN64; }

  int matchRegisterName(StringRef Symbol, bool is64BitReg);

  int matchCPURegisterName(StringRef Symbol);

  int matchRegisterByNumber(unsigned RegNum, unsigned RegClass);

  int matchFPURegisterName(StringRef Name);

  int matchFCCRegisterName(StringRef Name);

  int matchACRegisterName(StringRef Name);

  int matchMSA128RegisterName(StringRef Name);

  int matchMSA128CtrlRegisterName(StringRef Name);

  int regKindToRegClass(int RegKind);

  unsigned getReg(int RC, int RegNo);

  int getATReg();

  bool processInstruction(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  // Helper function that checks if the value of a vector index is within the
  // boundaries of accepted values for each RegisterKind
  // Example: INSERT.B $w0[n], $1 => 16 > n >= 0
  bool validateMSAIndex(int Val, int RegKind);

public:
  MipsAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser,
                const MCInstrInfo &MII)
      : MCTargetAsmParser(), STI(sti), Parser(parser),
        hasConsumedDollar(false) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }
};
}

namespace {

/// MipsOperand - Instances of this class represent a parsed Mips machine
/// instruction.
class MipsOperand : public MCParsedAsmOperand {

public:
  enum RegisterKind {
    Kind_None,
    Kind_GPR32,
    Kind_GPR64,
    Kind_HWRegs,
    Kind_FGR32Regs,
    Kind_FGRH32Regs,
    Kind_FGR64Regs,
    Kind_AFGR64Regs,
    Kind_CCRRegs,
    Kind_FCCRegs,
    Kind_ACC64DSP,
    Kind_LO32DSP,
    Kind_HI32DSP,
    Kind_COP2,
    Kind_MSA128BRegs,
    Kind_MSA128HRegs,
    Kind_MSA128WRegs,
    Kind_MSA128DRegs,
    Kind_MSA128CtrlRegs
  };

private:
  enum KindTy {
    k_CondCode,
    k_CoprocNum,
    k_Immediate,
    k_Memory,
    k_PostIndexRegister,
    k_Register,
    k_PtrReg,
    k_Token,
    k_LSAImm
  } Kind;

  MipsOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
    RegisterKind Kind;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned Base;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

  SMLoc StartLoc, EndLoc;

public:
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addPtrRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getPtrReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediate when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  bool isReg() const { return Kind == k_Register; }
  bool isImm() const { return Kind == k_Immediate; }
  bool isToken() const { return Kind == k_Token; }
  bool isMem() const { return Kind == k_Memory; }
  bool isPtrReg() const { return Kind == k_PtrReg; }
  bool isInvNum() const { return Kind == k_Immediate; }
  bool isLSAImm() const { return Kind == k_LSAImm; }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  unsigned getPtrReg() const {
    assert((Kind == k_PtrReg) && "Invalid access!");
    return Reg.RegNum;
  }

  void setRegKind(RegisterKind RegKind) {
    assert((Kind == k_Register || Kind == k_PtrReg) && "Invalid access!");
    Reg.Kind = RegKind;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate || Kind == k_LSAImm) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Base;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Off;
  }

  static MipsOperand *CreateToken(StringRef Str, SMLoc S) {
    MipsOperand *Op = new MipsOperand(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static MipsOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreatePtrReg(unsigned RegNum, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_PtrReg);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateLSAImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_LSAImm);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MipsOperand *CreateMem(unsigned Base, const MCExpr *Off,
                                SMLoc S, SMLoc E) {
    MipsOperand *Op = new MipsOperand(k_Memory);
    Op->Mem.Base = Base;
    Op->Mem.Off = Off;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  bool isGPR32Asm() const {
    return Kind == k_Register && Reg.Kind == Kind_GPR32;
  }
  void addRegAsmOperands(MCInst &Inst, unsigned N) const {
    Inst.addOperand(MCOperand::CreateReg(Reg.RegNum));
  }

  bool isGPR64Asm() const {
    return Kind == k_Register && Reg.Kind == Kind_GPR64;
  }

  bool isHWRegsAsm() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.Kind == Kind_HWRegs;
  }

  bool isCCRAsm() const {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.Kind == Kind_CCRRegs;
  }

  bool isAFGR64Asm() const {
    return Kind == k_Register && Reg.Kind == Kind_AFGR64Regs;
  }

  bool isFGR64Asm() const {
    return Kind == k_Register && Reg.Kind == Kind_FGR64Regs;
  }

  bool isFGR32Asm() const {
    return (Kind == k_Register) && Reg.Kind == Kind_FGR32Regs;
  }

  bool isFGRH32Asm() const {
    return (Kind == k_Register) && Reg.Kind == Kind_FGRH32Regs;
  }

  bool isFCCRegsAsm() const {
    return (Kind == k_Register) && Reg.Kind == Kind_FCCRegs;
  }

  bool isACC64DSPAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_ACC64DSP;
  }

  bool isLO32DSPAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_LO32DSP;
  }

  bool isHI32DSPAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_HI32DSP;
  }

  bool isCOP2Asm() const { return Kind == k_Register && Reg.Kind == Kind_COP2; }

  bool isMSA128BAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_MSA128BRegs;
  }

  bool isMSA128HAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_MSA128HRegs;
  }

  bool isMSA128WAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_MSA128WRegs;
  }

  bool isMSA128DAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_MSA128DRegs;
  }

  bool isMSA128CRAsm() const {
    return Kind == k_Register && Reg.Kind == Kind_MSA128CtrlRegs;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  virtual void print(raw_ostream &OS) const {
    llvm_unreachable("unimplemented!");
  }
}; // class MipsOperand
} // namespace

namespace llvm {
extern const MCInstrDesc MipsInsts[];
}
static const MCInstrDesc &getInstDesc(unsigned Opcode) {
  return MipsInsts[Opcode];
}

bool MipsAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                       SmallVectorImpl<MCInst> &Instructions) {
  const MCInstrDesc &MCID = getInstDesc(Inst.getOpcode());
  Inst.setLoc(IDLoc);
  if (MCID.hasDelaySlot() && Options.isReorder()) {
    // If this instruction has a delay slot and .set reorder is active,
    // emit a NOP after it.
    Instructions.push_back(Inst);
    MCInst NopInst;
    NopInst.setOpcode(Mips::SLL);
    NopInst.addOperand(MCOperand::CreateReg(Mips::ZERO));
    NopInst.addOperand(MCOperand::CreateReg(Mips::ZERO));
    NopInst.addOperand(MCOperand::CreateImm(0));
    Instructions.push_back(NopInst);
    return false;
  }

  if (MCID.mayLoad() || MCID.mayStore()) {
    // Check the offset of memory operand, if it is a symbol
    // reference or immediate we may have to expand instructions.
    for (unsigned i = 0; i < MCID.getNumOperands(); i++) {
      const MCOperandInfo &OpInfo = MCID.OpInfo[i];
      if ((OpInfo.OperandType == MCOI::OPERAND_MEMORY) ||
          (OpInfo.OperandType == MCOI::OPERAND_UNKNOWN)) {
        MCOperand &Op = Inst.getOperand(i);
        if (Op.isImm()) {
          int MemOffset = Op.getImm();
          if (MemOffset < -32768 || MemOffset > 32767) {
            // Offset can't exceed 16bit value.
            expandMemInst(Inst, IDLoc, Instructions, MCID.mayLoad(), true);
            return false;
          }
        } else if (Op.isExpr()) {
          const MCExpr *Expr = Op.getExpr();
          if (Expr->getKind() == MCExpr::SymbolRef) {
            const MCSymbolRefExpr *SR =
                static_cast<const MCSymbolRefExpr *>(Expr);
            if (SR->getKind() == MCSymbolRefExpr::VK_None) {
              // Expand symbol.
              expandMemInst(Inst, IDLoc, Instructions, MCID.mayLoad(), false);
              return false;
            }
          } else if (!isEvaluated(Expr)) {
            expandMemInst(Inst, IDLoc, Instructions, MCID.mayLoad(), false);
            return false;
          }
        }
      }
    } // for
  }   // if load/store

  if (needsExpansion(Inst))
    expandInstruction(Inst, IDLoc, Instructions);
  else
    Instructions.push_back(Inst);

  return false;
}

bool MipsAsmParser::needsExpansion(MCInst &Inst) {

  switch (Inst.getOpcode()) {
  case Mips::LoadImm32Reg:
  case Mips::LoadAddr32Imm:
  case Mips::LoadAddr32Reg:
    return true;
  default:
    return false;
  }
}

void MipsAsmParser::expandInstruction(MCInst &Inst, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions) {
  switch (Inst.getOpcode()) {
  case Mips::LoadImm32Reg:
    return expandLoadImm(Inst, IDLoc, Instructions);
  case Mips::LoadAddr32Imm:
    return expandLoadAddressImm(Inst, IDLoc, Instructions);
  case Mips::LoadAddr32Reg:
    return expandLoadAddressReg(Inst, IDLoc, Instructions);
  }
}

void MipsAsmParser::expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");

  int ImmValue = ImmOp.getImm();
  tmpInst.setLoc(IDLoc);
  if (0 <= ImmValue && ImmValue <= 65535) {
    // For 0 <= j <= 65535.
    // li d,j => ori d,$zero,j
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else if (ImmValue < 0 && ImmValue >= -32768) {
    // For -32768 <= j < 0.
    // li d,j => addiu d,$zero,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    // For any other value of j that is representable as a 32-bit integer.
    // li d,j => lui d,hi16(j)
    //           ori d,d,lo16(j)
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    tmpInst.setLoc(IDLoc);
    Instructions.push_back(tmpInst);
  }
}

void
MipsAsmParser::expandLoadAddressReg(MCInst &Inst, SMLoc IDLoc,
                                    SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(2);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &SrcRegOp = Inst.getOperand(1);
  assert(SrcRegOp.isReg() && "expected register operand kind");
  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");
  int ImmValue = ImmOp.getImm();
  if (-32768 <= ImmValue && ImmValue <= 65535) {
    // For -32768 <= j <= 65535.
    // la d,j(s) => addiu d,s,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(SrcRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    // For any other value of j that is representable as a 32-bit integer.
    // la d,j(s) => lui d,hi16(j)
    //              ori d,d,lo16(j)
    //              addu d,d,s
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ADDu);
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(DstRegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(SrcRegOp.getReg()));
    Instructions.push_back(tmpInst);
  }
}

void
MipsAsmParser::expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                                    SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");
  int ImmValue = ImmOp.getImm();
  if (-32768 <= ImmValue && ImmValue <= 65535) {
    // For -32768 <= j <= 65535.
    // la d,j => addiu d,$zero,j
    tmpInst.setOpcode(Mips::ADDiu);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(Mips::ZERO));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    // For any other value of j that is representable as a 32-bit integer.
    // la d,j => lui d,hi16(j)
    //           ori d,d,lo16(j)
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(Mips::ORi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm(ImmValue & 0xffff));
    Instructions.push_back(tmpInst);
  }
}

void MipsAsmParser::expandMemInst(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions,
                                  bool isLoad, bool isImmOpnd) {
  const MCSymbolRefExpr *SR;
  MCInst TempInst;
  unsigned ImmOffset, HiOffset, LoOffset;
  const MCExpr *ExprOffset;
  unsigned TmpRegNum;
  unsigned AtRegNum = getReg(
      (isMips64()) ? Mips::GPR64RegClassID : Mips::GPR32RegClassID, getATReg());
  // 1st operand is either the source or destination register.
  assert(Inst.getOperand(0).isReg() && "expected register operand kind");
  unsigned RegOpNum = Inst.getOperand(0).getReg();
  // 2nd operand is the base register.
  assert(Inst.getOperand(1).isReg() && "expected register operand kind");
  unsigned BaseRegNum = Inst.getOperand(1).getReg();
  // 3rd operand is either an immediate or expression.
  if (isImmOpnd) {
    assert(Inst.getOperand(2).isImm() && "expected immediate operand kind");
    ImmOffset = Inst.getOperand(2).getImm();
    LoOffset = ImmOffset & 0x0000ffff;
    HiOffset = (ImmOffset & 0xffff0000) >> 16;
    // If msb of LoOffset is 1(negative number) we must increment HiOffset.
    if (LoOffset & 0x8000)
      HiOffset++;
  } else
    ExprOffset = Inst.getOperand(2).getExpr();
  // All instructions will have the same location.
  TempInst.setLoc(IDLoc);
  // 1st instruction in expansion is LUi. For load instruction we can use
  // the dst register as a temporary if base and dst are different,
  // but for stores we must use $at.
  TmpRegNum = (isLoad && (BaseRegNum != RegOpNum)) ? RegOpNum : AtRegNum;
  TempInst.setOpcode(Mips::LUi);
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  if (isImmOpnd)
    TempInst.addOperand(MCOperand::CreateImm(HiOffset));
  else {
    if (ExprOffset->getKind() == MCExpr::SymbolRef) {
      SR = static_cast<const MCSymbolRefExpr *>(ExprOffset);
      const MCSymbolRefExpr *HiExpr = MCSymbolRefExpr::Create(
          SR->getSymbol().getName(), MCSymbolRefExpr::VK_Mips_ABS_HI,
          getContext());
      TempInst.addOperand(MCOperand::CreateExpr(HiExpr));
    } else {
      const MCExpr *HiExpr = evaluateRelocExpr(ExprOffset, "hi");
      TempInst.addOperand(MCOperand::CreateExpr(HiExpr));
    }
  }
  // Add the instruction to the list.
  Instructions.push_back(TempInst);
  // Prepare TempInst for next instruction.
  TempInst.clear();
  // Add temp register to base.
  TempInst.setOpcode(Mips::ADDu);
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  TempInst.addOperand(MCOperand::CreateReg(BaseRegNum));
  Instructions.push_back(TempInst);
  TempInst.clear();
  // And finaly, create original instruction with low part
  // of offset and new base.
  TempInst.setOpcode(Inst.getOpcode());
  TempInst.addOperand(MCOperand::CreateReg(RegOpNum));
  TempInst.addOperand(MCOperand::CreateReg(TmpRegNum));
  if (isImmOpnd)
    TempInst.addOperand(MCOperand::CreateImm(LoOffset));
  else {
    if (ExprOffset->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *LoExpr = MCSymbolRefExpr::Create(
          SR->getSymbol().getName(), MCSymbolRefExpr::VK_Mips_ABS_LO,
          getContext());
      TempInst.addOperand(MCOperand::CreateExpr(LoExpr));
    } else {
      const MCExpr *LoExpr = evaluateRelocExpr(ExprOffset, "lo");
      TempInst.addOperand(MCOperand::CreateExpr(LoExpr));
    }
  }
  Instructions.push_back(TempInst);
  TempInst.clear();
}

bool MipsAsmParser::MatchAndEmitInstruction(
    SMLoc IDLoc, unsigned &Opcode,
    SmallVectorImpl<MCParsedAsmOperand *> &Operands, MCStreamer &Out,
    unsigned &ErrorInfo, bool MatchingInlineAsm) {
  MCInst Inst;
  SmallVector<MCInst, 8> Instructions;
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);

  switch (MatchResult) {
  default:
    break;
  case Match_Success: {
    if (processInstruction(Inst, IDLoc, Instructions))
      return true;
    for (unsigned i = 0; i < Instructions.size(); i++)
      Out.EmitInstruction(Instructions[i]);
    return false;
  }
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((MipsOperand *)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }
  return true;
}

int MipsAsmParser::matchCPURegisterName(StringRef Name) {
  int CC;

  if (Name == "at")
    return getATReg();

  CC = StringSwitch<unsigned>(Name)
           .Case("zero", 0)
           .Case("a0", 4)
           .Case("a1", 5)
           .Case("a2", 6)
           .Case("a3", 7)
           .Case("v0", 2)
           .Case("v1", 3)
           .Case("s0", 16)
           .Case("s1", 17)
           .Case("s2", 18)
           .Case("s3", 19)
           .Case("s4", 20)
           .Case("s5", 21)
           .Case("s6", 22)
           .Case("s7", 23)
           .Case("k0", 26)
           .Case("k1", 27)
           .Case("sp", 29)
           .Case("fp", 30)
           .Case("gp", 28)
           .Case("ra", 31)
           .Case("t0", 8)
           .Case("t1", 9)
           .Case("t2", 10)
           .Case("t3", 11)
           .Case("t4", 12)
           .Case("t5", 13)
           .Case("t6", 14)
           .Case("t7", 15)
           .Case("t8", 24)
           .Case("t9", 25)
           .Default(-1);

  // Although SGI documentation just cuts out t0-t3 for n32/n64,
  // GNU pushes the values of t0-t3 to override the o32/o64 values for t4-t7
  // We are supporting both cases, so for t0-t3 we'll just push them to t4-t7.
  if (isMips64() && 8 <= CC && CC <= 11)
    CC += 4;

  if (CC == -1 && isMips64())
    CC = StringSwitch<unsigned>(Name)
             .Case("a4", 8)
             .Case("a5", 9)
             .Case("a6", 10)
             .Case("a7", 11)
             .Case("kt0", 26)
             .Case("kt1", 27)
             .Case("s8", 30)
             .Default(-1);

  return CC;
}

int MipsAsmParser::matchFPURegisterName(StringRef Name) {

  if (Name[0] == 'f') {
    StringRef NumString = Name.substr(1);
    unsigned IntVal;
    if (NumString.getAsInteger(10, IntVal))
      return -1;     // This is not an integer.
    if (IntVal > 31) // Maximum index for fpu register.
      return -1;
    return IntVal;
  }
  return -1;
}

int MipsAsmParser::matchFCCRegisterName(StringRef Name) {

  if (Name.startswith("fcc")) {
    StringRef NumString = Name.substr(3);
    unsigned IntVal;
    if (NumString.getAsInteger(10, IntVal))
      return -1;    // This is not an integer.
    if (IntVal > 7) // There are only 8 fcc registers.
      return -1;
    return IntVal;
  }
  return -1;
}

int MipsAsmParser::matchACRegisterName(StringRef Name) {

  if (Name.startswith("ac")) {
    StringRef NumString = Name.substr(2);
    unsigned IntVal;
    if (NumString.getAsInteger(10, IntVal))
      return -1;    // This is not an integer.
    if (IntVal > 3) // There are only 3 acc registers.
      return -1;
    return IntVal;
  }
  return -1;
}

int MipsAsmParser::matchMSA128RegisterName(StringRef Name) {
  unsigned IntVal;

  if (Name.front() != 'w' || Name.drop_front(1).getAsInteger(10, IntVal))
    return -1;

  if (IntVal > 31)
    return -1;

  return IntVal;
}

int MipsAsmParser::matchMSA128CtrlRegisterName(StringRef Name) {
  int CC;

  CC = StringSwitch<unsigned>(Name)
           .Case("msair", 0)
           .Case("msacsr", 1)
           .Case("msaaccess", 2)
           .Case("msasave", 3)
           .Case("msamodify", 4)
           .Case("msarequest", 5)
           .Case("msamap", 6)
           .Case("msaunmap", 7)
           .Default(-1);

  return CC;
}

int MipsAsmParser::matchRegisterName(StringRef Name, bool is64BitReg) {

  int CC;
  CC = matchCPURegisterName(Name);
  if (CC != -1)
    return matchRegisterByNumber(CC, is64BitReg ? Mips::GPR64RegClassID
                                                : Mips::GPR32RegClassID);
  CC = matchFPURegisterName(Name);
  // TODO: decide about fpu register class
  if (CC != -1)
    return matchRegisterByNumber(CC, isFP64() ? Mips::FGR64RegClassID
                                              : Mips::FGR32RegClassID);
  return matchMSA128RegisterName(Name);
}

int MipsAsmParser::regKindToRegClass(int RegKind) {

  switch (RegKind) {
  case MipsOperand::Kind_GPR32:
    return Mips::GPR32RegClassID;
  case MipsOperand::Kind_GPR64:
    return Mips::GPR64RegClassID;
  case MipsOperand::Kind_HWRegs:
    return Mips::HWRegsRegClassID;
  case MipsOperand::Kind_FGR32Regs:
    return Mips::FGR32RegClassID;
  case MipsOperand::Kind_FGRH32Regs:
    return Mips::FGRH32RegClassID;
  case MipsOperand::Kind_FGR64Regs:
    return Mips::FGR64RegClassID;
  case MipsOperand::Kind_AFGR64Regs:
    return Mips::AFGR64RegClassID;
  case MipsOperand::Kind_CCRRegs:
    return Mips::CCRRegClassID;
  case MipsOperand::Kind_ACC64DSP:
    return Mips::ACC64DSPRegClassID;
  case MipsOperand::Kind_FCCRegs:
    return Mips::FCCRegClassID;
  case MipsOperand::Kind_MSA128BRegs:
    return Mips::MSA128BRegClassID;
  case MipsOperand::Kind_MSA128HRegs:
    return Mips::MSA128HRegClassID;
  case MipsOperand::Kind_MSA128WRegs:
    return Mips::MSA128WRegClassID;
  case MipsOperand::Kind_MSA128DRegs:
    return Mips::MSA128DRegClassID;
  case MipsOperand::Kind_MSA128CtrlRegs:
    return Mips::MSACtrlRegClassID;
  default:
    return -1;
  }
}

bool MipsAssemblerOptions::setATReg(unsigned Reg) {
  if (Reg > 31)
    return false;

  aTReg = Reg;
  return true;
}

int MipsAsmParser::getATReg() { return Options.getATRegNum(); }

unsigned MipsAsmParser::getReg(int RC, int RegNo) {
  return *(getContext().getRegisterInfo()->getRegClass(RC).begin() + RegNo);
}

int MipsAsmParser::matchRegisterByNumber(unsigned RegNum, unsigned RegClass) {
  if (RegNum >
      getContext().getRegisterInfo()->getRegClass(RegClass).getNumRegs())
    return -1;

  return getReg(RegClass, RegNum);
}

int MipsAsmParser::tryParseRegister(bool is64BitReg) {
  const AsmToken &Tok = Parser.getTok();
  int RegNum = -1;

  if (Tok.is(AsmToken::Identifier)) {
    std::string lowerCase = Tok.getString().lower();
    RegNum = matchRegisterName(lowerCase, is64BitReg);
  } else if (Tok.is(AsmToken::Integer))
    RegNum = matchRegisterByNumber(static_cast<unsigned>(Tok.getIntVal()),
                                   is64BitReg ? Mips::GPR64RegClassID
                                              : Mips::GPR32RegClassID);
  return RegNum;
}

bool MipsAsmParser::tryParseRegisterOperand(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands, bool is64BitReg) {

  SMLoc S = Parser.getTok().getLoc();
  int RegNo = -1;

  RegNo = tryParseRegister(is64BitReg);
  if (RegNo == -1)
    return true;

  Operands.push_back(
      MipsOperand::CreateReg(RegNo, S, Parser.getTok().getLoc()));
  Parser.Lex(); // Eat register token.
  return false;
}

bool
MipsAsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                            StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);
  if (ResTy == MatchOperand_Success)
    return false;
  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_ParseFail)
    return true;

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Dollar: {
    // Parse the register.
    SMLoc S = Parser.getTok().getLoc();
    Parser.Lex(); // Eat dollar token.
    // Parse the register operand.
    if (!tryParseRegisterOperand(Operands, isMips64())) {
      if (getLexer().is(AsmToken::LParen)) {
        // Check if it is indexed addressing operand.
        Operands.push_back(MipsOperand::CreateToken("(", S));
        Parser.Lex(); // Eat the parenthesis.
        if (getLexer().isNot(AsmToken::Dollar))
          return true;

        Parser.Lex(); // Eat the dollar
        if (tryParseRegisterOperand(Operands, isMips64()))
          return true;

        if (!getLexer().is(AsmToken::RParen))
          return true;

        S = Parser.getTok().getLoc();
        Operands.push_back(MipsOperand::CreateToken(")", S));
        Parser.Lex();
      }
      return false;
    }
    // Maybe it is a symbol reference.
    StringRef Identifier;
    if (Parser.parseIdentifier(Identifier))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    MCSymbol *Sym = getContext().GetOrCreateSymbol("$" + Identifier);
    // Otherwise create a symbol reference.
    const MCExpr *Res =
        MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());

    Operands.push_back(MipsOperand::CreateImm(Res, S, E));
    return false;
  }
  case AsmToken::Identifier:
    // For instruction aliases like "bc1f $Label" dedicated parser will
    // eat the '$' sign before failing. So in order to look for appropriate
    // label we must check first if we have already consumed '$'.
    if (hasConsumedDollar) {
      hasConsumedDollar = false;
      SMLoc S = Parser.getTok().getLoc();
      StringRef Identifier;
      if (Parser.parseIdentifier(Identifier))
        return true;
      SMLoc E =
          SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      MCSymbol *Sym = getContext().GetOrCreateSymbol("$" + Identifier);
      // Create a symbol reference.
      const MCExpr *Res =
          MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());

      Operands.push_back(MipsOperand::CreateImm(Res, S, E));
      return false;
    }
    // Look for the existing symbol, we should check if
    // we need to assigne the propper RegisterKind.
    if (searchSymbolAlias(Operands, MipsOperand::Kind_None))
      return false;
  // Else drop to expression parsing.
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::String: {
    // Quoted label names.
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc();
    if (getParser().parseExpression(IdVal))
      return true;
    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
    return false;
  }
  case AsmToken::Percent: {
    // It is a symbol reference or constant expression.
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc(); // Start location of the operand.
    if (parseRelocOperand(IdVal))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
    return false;
  } // case AsmToken::Percent
  } // switch(getLexer().getKind())
  return true;
}

const MCExpr *MipsAsmParser::evaluateRelocExpr(const MCExpr *Expr,
                                               StringRef RelocStr) {
  const MCExpr *Res;
  // Check the type of the expression.
  if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Expr)) {
    // It's a constant, evaluate lo or hi value.
    if (RelocStr == "lo") {
      short Val = MCE->getValue();
      Res = MCConstantExpr::Create(Val, getContext());
    } else if (RelocStr == "hi") {
      int Val = MCE->getValue();
      int LoSign = Val & 0x8000;
      Val = (Val & 0xffff0000) >> 16;
      // Lower part is treated as a signed int, so if it is negative
      // we must add 1 to the hi part to compensate.
      if (LoSign)
        Val++;
      Res = MCConstantExpr::Create(Val, getContext());
    } else {
      llvm_unreachable("Invalid RelocStr value");
    }
    return Res;
  }

  if (const MCSymbolRefExpr *MSRE = dyn_cast<MCSymbolRefExpr>(Expr)) {
    // It's a symbol, create a symbolic expression from the symbol.
    StringRef Symbol = MSRE->getSymbol().getName();
    MCSymbolRefExpr::VariantKind VK = getVariantKind(RelocStr);
    Res = MCSymbolRefExpr::Create(Symbol, VK, getContext());
    return Res;
  }

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    const MCExpr *LExp = evaluateRelocExpr(BE->getLHS(), RelocStr);
    const MCExpr *RExp = evaluateRelocExpr(BE->getRHS(), RelocStr);
    Res = MCBinaryExpr::Create(BE->getOpcode(), LExp, RExp, getContext());
    return Res;
  }

  if (const MCUnaryExpr *UN = dyn_cast<MCUnaryExpr>(Expr)) {
    const MCExpr *UnExp = evaluateRelocExpr(UN->getSubExpr(), RelocStr);
    Res = MCUnaryExpr::Create(UN->getOpcode(), UnExp, getContext());
    return Res;
  }
  // Just return the original expression.
  return Expr;
}

bool MipsAsmParser::isEvaluated(const MCExpr *Expr) {

  switch (Expr->getKind()) {
  case MCExpr::Constant:
    return true;
  case MCExpr::SymbolRef:
    return (cast<MCSymbolRefExpr>(Expr)->getKind() != MCSymbolRefExpr::VK_None);
  case MCExpr::Binary:
    if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
      if (!isEvaluated(BE->getLHS()))
        return false;
      return isEvaluated(BE->getRHS());
    }
  case MCExpr::Unary:
    return isEvaluated(cast<MCUnaryExpr>(Expr)->getSubExpr());
  default:
    return false;
  }
  return false;
}

bool MipsAsmParser::parseRelocOperand(const MCExpr *&Res) {
  Parser.Lex();                          // Eat the % token.
  const AsmToken &Tok = Parser.getTok(); // Get next token, operation.
  if (Tok.isNot(AsmToken::Identifier))
    return true;

  std::string Str = Tok.getIdentifier().str();

  Parser.Lex(); // Eat the identifier.
  // Now make an expression from the rest of the operand.
  const MCExpr *IdVal;
  SMLoc EndLoc;

  if (getLexer().getKind() == AsmToken::LParen) {
    while (1) {
      Parser.Lex(); // Eat the '(' token.
      if (getLexer().getKind() == AsmToken::Percent) {
        Parser.Lex(); // Eat the % token.
        const AsmToken &nextTok = Parser.getTok();
        if (nextTok.isNot(AsmToken::Identifier))
          return true;
        Str += "(%";
        Str += nextTok.getIdentifier();
        Parser.Lex(); // Eat the identifier.
        if (getLexer().getKind() != AsmToken::LParen)
          return true;
      } else
        break;
    }
    if (getParser().parseParenExpression(IdVal, EndLoc))
      return true;

    while (getLexer().getKind() == AsmToken::RParen)
      Parser.Lex(); // Eat the ')' token.

  } else
    return true; // Parenthesis must follow the relocation operand.

  Res = evaluateRelocExpr(IdVal, Str);
  return false;
}

bool MipsAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                  SMLoc &EndLoc) {
  StartLoc = Parser.getTok().getLoc();
  RegNo = tryParseRegister(isMips64());
  EndLoc = Parser.getTok().getLoc();
  return (RegNo == (unsigned)-1);
}

bool MipsAsmParser::parseMemOffset(const MCExpr *&Res, bool isParenExpr) {
  SMLoc S;
  bool Result = true;

  while (getLexer().getKind() == AsmToken::LParen)
    Parser.Lex();

  switch (getLexer().getKind()) {
  default:
    return true;
  case AsmToken::Identifier:
  case AsmToken::LParen:
  case AsmToken::Integer:
  case AsmToken::Minus:
  case AsmToken::Plus:
    if (isParenExpr)
      Result = getParser().parseParenExpression(Res, S);
    else
      Result = (getParser().parseExpression(Res));
    while (getLexer().getKind() == AsmToken::RParen)
      Parser.Lex();
    break;
  case AsmToken::Percent:
    Result = parseRelocOperand(Res);
  }
  return Result;
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMemOperand(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {

  const MCExpr *IdVal = 0;
  SMLoc S;
  bool isParenExpr = false;
  MipsAsmParser::OperandMatchResultTy Res = MatchOperand_NoMatch;
  // First operand is the offset.
  S = Parser.getTok().getLoc();

  if (getLexer().getKind() == AsmToken::LParen) {
    Parser.Lex();
    isParenExpr = true;
  }

  if (getLexer().getKind() != AsmToken::Dollar) {
    if (parseMemOffset(IdVal, isParenExpr))
      return MatchOperand_ParseFail;

    const AsmToken &Tok = Parser.getTok(); // Get the next token.
    if (Tok.isNot(AsmToken::LParen)) {
      MipsOperand *Mnemonic = static_cast<MipsOperand *>(Operands[0]);
      if (Mnemonic->getToken() == "la") {
        SMLoc E =
            SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
        Operands.push_back(MipsOperand::CreateImm(IdVal, S, E));
        return MatchOperand_Success;
      }
      if (Tok.is(AsmToken::EndOfStatement)) {
        SMLoc E =
            SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

        // Zero register assumed, add a memory operand with ZERO as its base.
        Operands.push_back(MipsOperand::CreateMem(
            isMips64() ? Mips::ZERO_64 : Mips::ZERO, IdVal, S, E));
        return MatchOperand_Success;
      }
      Error(Parser.getTok().getLoc(), "'(' expected");
      return MatchOperand_ParseFail;
    }

    Parser.Lex(); // Eat the '(' token.
  }

  Res = parseRegs(Operands, isMips64() ? (int)MipsOperand::Kind_GPR64
                                       : (int)MipsOperand::Kind_GPR32);
  if (Res != MatchOperand_Success)
    return Res;

  if (Parser.getTok().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "')' expected");
    return MatchOperand_ParseFail;
  }

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  Parser.Lex(); // Eat the ')' token.

  if (IdVal == 0)
    IdVal = MCConstantExpr::Create(0, getContext());

  // Replace the register operand with the memory operand.
  MipsOperand *op = static_cast<MipsOperand *>(Operands.back());
  int RegNo = op->getReg();
  // Remove the register from the operands.
  Operands.pop_back();
  // Add the memory operand.
  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(IdVal)) {
    int64_t Imm;
    if (IdVal->EvaluateAsAbsolute(Imm))
      IdVal = MCConstantExpr::Create(Imm, getContext());
    else if (BE->getLHS()->getKind() != MCExpr::SymbolRef)
      IdVal = MCBinaryExpr::Create(BE->getOpcode(), BE->getRHS(), BE->getLHS(),
                                   getContext());
  }

  Operands.push_back(MipsOperand::CreateMem(RegNo, IdVal, S, E));
  delete op;
  return MatchOperand_Success;
}

bool MipsAsmParser::parsePtrReg(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                                int RegKind) {
  // If the first token is not '$' we have an error.
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return false;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex();
  AsmToken::TokenKind TkKind = getLexer().getKind();
  int Reg;

  if (TkKind == AsmToken::Integer) {
    Reg = matchRegisterByNumber(Parser.getTok().getIntVal(),
                                regKindToRegClass(RegKind));
    if (Reg == -1)
      return false;
  } else if (TkKind == AsmToken::Identifier) {
    if ((Reg = matchCPURegisterName(Parser.getTok().getString().lower())) == -1)
      return false;
    Reg = getReg(regKindToRegClass(RegKind), Reg);
  } else {
    return false;
  }

  MipsOperand *Op = MipsOperand::CreatePtrReg(Reg, S, Parser.getTok().getLoc());
  Op->setRegKind((MipsOperand::RegisterKind)RegKind);
  Operands.push_back(Op);
  Parser.Lex();
  return true;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parsePtrReg(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  MipsOperand::RegisterKind RegKind =
      isN64() ? MipsOperand::Kind_GPR64 : MipsOperand::Kind_GPR32;

  // Parse index register.
  if (!parsePtrReg(Operands, RegKind))
    return MatchOperand_NoMatch;

  // Parse '('.
  if (Parser.getTok().isNot(AsmToken::LParen))
    return MatchOperand_NoMatch;

  Operands.push_back(MipsOperand::CreateToken("(", getLexer().getLoc()));
  Parser.Lex();

  // Parse base register.
  if (!parsePtrReg(Operands, RegKind))
    return MatchOperand_NoMatch;

  // Parse ')'.
  if (Parser.getTok().isNot(AsmToken::RParen))
    return MatchOperand_NoMatch;

  Operands.push_back(MipsOperand::CreateToken(")", getLexer().getLoc()));
  Parser.Lex();

  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                         int RegKind) {
  MipsOperand::RegisterKind Kind = (MipsOperand::RegisterKind)RegKind;
  if (getLexer().getKind() == AsmToken::Identifier && !hasConsumedDollar) {
    if (searchSymbolAlias(Operands, Kind))
      return MatchOperand_Success;
    return MatchOperand_NoMatch;
  }
  SMLoc S = Parser.getTok().getLoc();
  // If the first token is not '$', we have an error.
  if (Parser.getTok().isNot(AsmToken::Dollar) && !hasConsumedDollar)
    return MatchOperand_NoMatch;
  if (!hasConsumedDollar) {
    Parser.Lex(); // Eat the '$'
    hasConsumedDollar = true;
  }
  if (getLexer().getKind() == AsmToken::Identifier) {
    int RegNum = -1;
    std::string RegName = Parser.getTok().getString().lower();
    // Match register by name
    switch (RegKind) {
    case MipsOperand::Kind_GPR32:
    case MipsOperand::Kind_GPR64:
      RegNum = matchCPURegisterName(RegName);
      break;
    case MipsOperand::Kind_AFGR64Regs:
    case MipsOperand::Kind_FGR64Regs:
    case MipsOperand::Kind_FGR32Regs:
    case MipsOperand::Kind_FGRH32Regs:
      RegNum = matchFPURegisterName(RegName);
      if (RegKind == MipsOperand::Kind_AFGR64Regs)
        RegNum /= 2;
      else if (RegKind == MipsOperand::Kind_FGRH32Regs && !isFP64())
        if (RegNum != -1 && RegNum % 2 != 0)
          Warning(S, "Float register should be even.");
      break;
    case MipsOperand::Kind_FCCRegs:
      RegNum = matchFCCRegisterName(RegName);
      break;
    case MipsOperand::Kind_ACC64DSP:
      RegNum = matchACRegisterName(RegName);
      break;
    default:
      break; // No match, value is set to -1.
    }
    // No match found, return _NoMatch to give a chance to other round.
    if (RegNum < 0)
      return MatchOperand_NoMatch;

    int RegVal = getReg(regKindToRegClass(Kind), RegNum);
    if (RegVal == -1)
      return MatchOperand_NoMatch;

    MipsOperand *Op =
        MipsOperand::CreateReg(RegVal, S, Parser.getTok().getLoc());
    Op->setRegKind(Kind);
    Operands.push_back(Op);
    hasConsumedDollar = false;
    Parser.Lex(); // Eat the register name.
    return MatchOperand_Success;
  } else if (getLexer().getKind() == AsmToken::Integer) {
    unsigned RegNum = Parser.getTok().getIntVal();
    if (Kind == MipsOperand::Kind_HWRegs) {
      if (RegNum != 29)
        return MatchOperand_NoMatch;
      // Only hwreg 29 is supported, found at index 0.
      RegNum = 0;
    }
    int Reg = matchRegisterByNumber(RegNum, regKindToRegClass(Kind));
    if (Reg == -1)
      return MatchOperand_NoMatch;
    MipsOperand *Op = MipsOperand::CreateReg(Reg, S, Parser.getTok().getLoc());
    Op->setRegKind(Kind);
    Operands.push_back(Op);
    hasConsumedDollar = false;
    Parser.Lex(); // Eat the register number.
    if ((RegKind == MipsOperand::Kind_GPR32) &&
        (getLexer().is(AsmToken::LParen))) {
      // Check if it is indexed addressing operand.
      Operands.push_back(MipsOperand::CreateToken("(", getLexer().getLoc()));
      Parser.Lex(); // Eat the parenthesis.
      if (parseRegs(Operands, RegKind) != MatchOperand_Success)
        return MatchOperand_NoMatch;
      if (getLexer().isNot(AsmToken::RParen))
        return MatchOperand_NoMatch;
      Operands.push_back(MipsOperand::CreateToken(")", getLexer().getLoc()));
      Parser.Lex();
    }
    return MatchOperand_Success;
  }
  return MatchOperand_NoMatch;
}

bool MipsAsmParser::validateMSAIndex(int Val, int RegKind) {
  MipsOperand::RegisterKind Kind = (MipsOperand::RegisterKind)RegKind;

  if (Val < 0)
    return false;

  switch (Kind) {
  default:
    return false;
  case MipsOperand::Kind_MSA128BRegs:
    return Val < 16;
  case MipsOperand::Kind_MSA128HRegs:
    return Val < 8;
  case MipsOperand::Kind_MSA128WRegs:
    return Val < 4;
  case MipsOperand::Kind_MSA128DRegs:
    return Val < 2;
  }
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseMSARegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                            int RegKind) {
  MipsOperand::RegisterKind Kind = (MipsOperand::RegisterKind)RegKind;
  SMLoc S = Parser.getTok().getLoc();
  std::string RegName;

  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  switch (RegKind) {
  default:
    return MatchOperand_ParseFail;
  case MipsOperand::Kind_MSA128BRegs:
  case MipsOperand::Kind_MSA128HRegs:
  case MipsOperand::Kind_MSA128WRegs:
  case MipsOperand::Kind_MSA128DRegs:
    break;
  }

  Parser.Lex(); // Eat the '$'.
  if (getLexer().getKind() == AsmToken::Identifier)
    RegName = Parser.getTok().getString().lower();
  else
    return MatchOperand_ParseFail;

  int RegNum = matchMSA128RegisterName(RegName);

  if (RegNum < 0 || RegNum > 31)
    return MatchOperand_ParseFail;

  int RegVal = getReg(regKindToRegClass(Kind), RegNum);
  if (RegVal == -1)
    return MatchOperand_ParseFail;

  MipsOperand *Op = MipsOperand::CreateReg(RegVal, S, Parser.getTok().getLoc());
  Op->setRegKind(Kind);
  Operands.push_back(Op);

  Parser.Lex(); // Eat the register identifier.

  // MSA registers may be suffixed with an index in the form of:
  // 1) Immediate expression.
  // 2) General Purpose Register.
  // Examples:
  //   1) copy_s.b $29,$w0[0]
  //   2) sld.b $w0,$w1[$1]

  if (Parser.getTok().isNot(AsmToken::LBrac))
    return MatchOperand_Success;

  MipsOperand *Mnemonic = static_cast<MipsOperand *>(Operands[0]);

  Operands.push_back(MipsOperand::CreateToken("[", Parser.getTok().getLoc()));
  Parser.Lex(); // Parse the '[' token.

  if (Parser.getTok().is(AsmToken::Dollar)) {
    // This must be a GPR.
    MipsOperand *RegOp;
    SMLoc VIdx = Parser.getTok().getLoc();
    Parser.Lex(); // Parse the '$' token.

    // GPR have aliases and we must account for that. Example: $30 == $fp
    if (getLexer().getKind() == AsmToken::Integer) {
      unsigned RegNum = Parser.getTok().getIntVal();
      int Reg = matchRegisterByNumber(
          RegNum, regKindToRegClass(MipsOperand::Kind_GPR32));
      if (Reg == -1) {
        Error(VIdx, "invalid general purpose register");
        return MatchOperand_ParseFail;
      }

      RegOp = MipsOperand::CreateReg(Reg, VIdx, Parser.getTok().getLoc());
    } else if (getLexer().getKind() == AsmToken::Identifier) {
      int RegNum = -1;
      std::string RegName = Parser.getTok().getString().lower();

      RegNum = matchCPURegisterName(RegName);
      if (RegNum == -1) {
        Error(VIdx, "general purpose register expected");
        return MatchOperand_ParseFail;
      }
      RegNum = getReg(regKindToRegClass(MipsOperand::Kind_GPR32), RegNum);
      RegOp = MipsOperand::CreateReg(RegNum, VIdx, Parser.getTok().getLoc());
    } else
      return MatchOperand_ParseFail;

    RegOp->setRegKind(MipsOperand::Kind_GPR32);
    Operands.push_back(RegOp);
    Parser.Lex(); // Eat the register identifier.

    if (Parser.getTok().isNot(AsmToken::RBrac))
      return MatchOperand_ParseFail;

    Operands.push_back(MipsOperand::CreateToken("]", Parser.getTok().getLoc()));
    Parser.Lex(); // Parse the ']' token.

    return MatchOperand_Success;
  }

  // The index must be a constant expression then.
  SMLoc VIdx = Parser.getTok().getLoc();
  const MCExpr *ImmVal;

  if (getParser().parseExpression(ImmVal))
    return MatchOperand_ParseFail;

  const MCConstantExpr *expr = dyn_cast<MCConstantExpr>(ImmVal);
  if (!expr || !validateMSAIndex((int)expr->getValue(), Kind)) {
    Error(VIdx, "invalid immediate value");
    return MatchOperand_ParseFail;
  }

  SMLoc E = Parser.getTok().getEndLoc();

  if (Parser.getTok().isNot(AsmToken::RBrac))
    return MatchOperand_ParseFail;

  bool insve =
      Mnemonic->getToken() == "insve.b" || Mnemonic->getToken() == "insve.h" ||
      Mnemonic->getToken() == "insve.w" || Mnemonic->getToken() == "insve.d";

  // The second vector index of insve instructions is always 0.
  if (insve && Operands.size() > 6) {
    if (expr->getValue() != 0) {
      Error(VIdx, "immediate value must be 0");
      return MatchOperand_ParseFail;
    }
    Operands.push_back(MipsOperand::CreateToken("0", VIdx));
  } else
    Operands.push_back(MipsOperand::CreateImm(expr, VIdx, E));

  Operands.push_back(MipsOperand::CreateToken("]", Parser.getTok().getLoc()));

  Parser.Lex(); // Parse the ']' token.

  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseMSACtrlRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands,
                                int RegKind) {
  MipsOperand::RegisterKind Kind = (MipsOperand::RegisterKind)RegKind;

  if (Kind != MipsOperand::Kind_MSA128CtrlRegs)
    return MatchOperand_NoMatch;

  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_ParseFail;

  SMLoc S = Parser.getTok().getLoc();

  Parser.Lex(); // Eat the '$' symbol.

  int RegNum = -1;
  if (getLexer().getKind() == AsmToken::Identifier)
    RegNum = matchMSA128CtrlRegisterName(Parser.getTok().getString().lower());
  else if (getLexer().getKind() == AsmToken::Integer)
    RegNum = Parser.getTok().getIntVal();
  else
    return MatchOperand_ParseFail;

  if (RegNum < 0 || RegNum > 7)
    return MatchOperand_ParseFail;

  int RegVal = getReg(regKindToRegClass(Kind), RegNum);
  if (RegVal == -1)
    return MatchOperand_ParseFail;

  MipsOperand *RegOp =
      MipsOperand::CreateReg(RegVal, S, Parser.getTok().getLoc());
  RegOp->setRegKind(MipsOperand::Kind_MSA128CtrlRegs);
  Operands.push_back(RegOp);
  Parser.Lex(); // Eat the register identifier.

  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseGPR64(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {

  if (!isMips64())
    return MatchOperand_NoMatch;
  return parseRegs(Operands, (int)MipsOperand::Kind_GPR64);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseGPR32(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_GPR32);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseAFGR64Regs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {

  if (isFP64())
    return MatchOperand_NoMatch;
  return parseRegs(Operands, (int)MipsOperand::Kind_AFGR64Regs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseFGR64Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  if (!isFP64())
    return MatchOperand_NoMatch;
  return parseRegs(Operands, (int)MipsOperand::Kind_FGR64Regs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseFGR32Regs(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_FGR32Regs);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseFGRH32Regs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_FGRH32Regs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseFCCRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_FCCRegs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseACC64DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_ACC64DSP);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseLO32DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  // If the first token is not '$' we have an error.
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat the '$'

  const AsmToken &Tok = Parser.getTok(); // Get next token.

  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  if (!Tok.getIdentifier().startswith("ac"))
    return MatchOperand_NoMatch;

  StringRef NumString = Tok.getIdentifier().substr(2);

  unsigned IntVal;
  if (NumString.getAsInteger(10, IntVal))
    return MatchOperand_NoMatch;

  unsigned Reg = matchRegisterByNumber(IntVal, Mips::LO32DSPRegClassID);

  MipsOperand *Op = MipsOperand::CreateReg(Reg, S, Parser.getTok().getLoc());
  Op->setRegKind(MipsOperand::Kind_LO32DSP);
  Operands.push_back(Op);

  Parser.Lex(); // Eat the register number.
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseHI32DSP(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  // If the first token is not '$' we have an error.
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat the '$'

  const AsmToken &Tok = Parser.getTok(); // Get next token.

  if (Tok.isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  if (!Tok.getIdentifier().startswith("ac"))
    return MatchOperand_NoMatch;

  StringRef NumString = Tok.getIdentifier().substr(2);

  unsigned IntVal;
  if (NumString.getAsInteger(10, IntVal))
    return MatchOperand_NoMatch;

  unsigned Reg = matchRegisterByNumber(IntVal, Mips::HI32DSPRegClassID);

  MipsOperand *Op = MipsOperand::CreateReg(Reg, S, Parser.getTok().getLoc());
  Op->setRegKind(MipsOperand::Kind_HI32DSP);
  Operands.push_back(Op);

  Parser.Lex(); // Eat the register number.
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseCOP2(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  // If the first token is not '$' we have an error.
  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat the '$'

  const AsmToken &Tok = Parser.getTok(); // Get next token.

  if (Tok.isNot(AsmToken::Integer))
    return MatchOperand_NoMatch;

  unsigned IntVal = Tok.getIntVal();

  unsigned Reg = matchRegisterByNumber(IntVal, Mips::COP2RegClassID);

  MipsOperand *Op = MipsOperand::CreateReg(Reg, S, Parser.getTok().getLoc());
  Op->setRegKind(MipsOperand::Kind_COP2);
  Operands.push_back(Op);

  Parser.Lex(); // Eat the register number.
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMSA128BRegs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseMSARegs(Operands, (int)MipsOperand::Kind_MSA128BRegs);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMSA128HRegs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseMSARegs(Operands, (int)MipsOperand::Kind_MSA128HRegs);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMSA128WRegs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseMSARegs(Operands, (int)MipsOperand::Kind_MSA128WRegs);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMSA128DRegs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseMSARegs(Operands, (int)MipsOperand::Kind_MSA128DRegs);
}

MipsAsmParser::OperandMatchResultTy MipsAsmParser::parseMSA128CtrlRegs(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseMSACtrlRegs(Operands, (int)MipsOperand::Kind_MSA128CtrlRegs);
}

bool MipsAsmParser::searchSymbolAlias(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands, unsigned RegKind) {

  MCSymbol *Sym = getContext().LookupSymbol(Parser.getTok().getIdentifier());
  if (Sym) {
    SMLoc S = Parser.getTok().getLoc();
    const MCExpr *Expr;
    if (Sym->isVariable())
      Expr = Sym->getVariableValue();
    else
      return false;
    if (Expr->getKind() == MCExpr::SymbolRef) {
      MipsOperand::RegisterKind Kind = (MipsOperand::RegisterKind)RegKind;
      const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);
      const StringRef DefSymbol = Ref->getSymbol().getName();
      if (DefSymbol.startswith("$")) {
        int RegNum = -1;
        APInt IntVal(32, -1);
        if (!DefSymbol.substr(1).getAsInteger(10, IntVal))
          RegNum = matchRegisterByNumber(IntVal.getZExtValue(),
                                         isMips64() ? Mips::GPR64RegClassID
                                                    : Mips::GPR32RegClassID);
        else {
          // Lookup for the register with the corresponding name.
          switch (Kind) {
          case MipsOperand::Kind_AFGR64Regs:
          case MipsOperand::Kind_FGR64Regs:
            RegNum = matchFPURegisterName(DefSymbol.substr(1));
            break;
          case MipsOperand::Kind_FGR32Regs:
            RegNum = matchFPURegisterName(DefSymbol.substr(1));
            break;
          case MipsOperand::Kind_GPR64:
          case MipsOperand::Kind_GPR32:
          default:
            RegNum = matchCPURegisterName(DefSymbol.substr(1));
            break;
          }
          if (RegNum > -1)
            RegNum = getReg(regKindToRegClass(Kind), RegNum);
        }
        if (RegNum > -1) {
          Parser.Lex();
          MipsOperand *op =
              MipsOperand::CreateReg(RegNum, S, Parser.getTok().getLoc());
          op->setRegKind(Kind);
          Operands.push_back(op);
          return true;
        }
      }
    } else if (Expr->getKind() == MCExpr::Constant) {
      Parser.Lex();
      const MCConstantExpr *Const = static_cast<const MCConstantExpr *>(Expr);
      MipsOperand *op =
          MipsOperand::CreateImm(Const, S, Parser.getTok().getLoc());
      Operands.push_back(op);
      return true;
    }
  }
  return false;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseHWRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_HWRegs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseCCRRegs(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  return parseRegs(Operands, (int)MipsOperand::Kind_CCRRegs);
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseInvNum(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  const MCExpr *IdVal;
  // If the first token is '$' we may have register operand.
  if (Parser.getTok().is(AsmToken::Dollar))
    return MatchOperand_NoMatch;
  SMLoc S = Parser.getTok().getLoc();
  if (getParser().parseExpression(IdVal))
    return MatchOperand_ParseFail;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(IdVal);
  assert(MCE && "Unexpected MCExpr type.");
  int64_t Val = MCE->getValue();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(MipsOperand::CreateImm(
      MCConstantExpr::Create(0 - Val, getContext()), S, E));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseLSAImm(SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::LParen:
  case AsmToken::Plus:
  case AsmToken::Minus:
  case AsmToken::Integer:
    break;
  }

  const MCExpr *Expr;
  SMLoc S = Parser.getTok().getLoc();

  if (getParser().parseExpression(Expr))
    return MatchOperand_ParseFail;

  int64_t Val;
  if (!Expr->EvaluateAsAbsolute(Val)) {
    Error(S, "expected immediate value");
    return MatchOperand_ParseFail;
  }

  // The LSA instruction allows a 2-bit unsigned immediate. For this reason
  // and because the CPU always adds one to the immediate field, the allowed
  // range becomes 1..4. We'll only check the range here and will deal
  // with the addition/subtraction when actually decoding/encoding
  // the instruction.
  if (Val < 1 || Val > 4) {
    Error(S, "immediate not in range (1..4)");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(MipsOperand::CreateLSAImm(Expr, S,
                                               Parser.getTok().getLoc()));
  return MatchOperand_Success;
}

MCSymbolRefExpr::VariantKind MipsAsmParser::getVariantKind(StringRef Symbol) {

  MCSymbolRefExpr::VariantKind VK =
      StringSwitch<MCSymbolRefExpr::VariantKind>(Symbol)
          .Case("hi", MCSymbolRefExpr::VK_Mips_ABS_HI)
          .Case("lo", MCSymbolRefExpr::VK_Mips_ABS_LO)
          .Case("gp_rel", MCSymbolRefExpr::VK_Mips_GPREL)
          .Case("call16", MCSymbolRefExpr::VK_Mips_GOT_CALL)
          .Case("got", MCSymbolRefExpr::VK_Mips_GOT)
          .Case("tlsgd", MCSymbolRefExpr::VK_Mips_TLSGD)
          .Case("tlsldm", MCSymbolRefExpr::VK_Mips_TLSLDM)
          .Case("dtprel_hi", MCSymbolRefExpr::VK_Mips_DTPREL_HI)
          .Case("dtprel_lo", MCSymbolRefExpr::VK_Mips_DTPREL_LO)
          .Case("gottprel", MCSymbolRefExpr::VK_Mips_GOTTPREL)
          .Case("tprel_hi", MCSymbolRefExpr::VK_Mips_TPREL_HI)
          .Case("tprel_lo", MCSymbolRefExpr::VK_Mips_TPREL_LO)
          .Case("got_disp", MCSymbolRefExpr::VK_Mips_GOT_DISP)
          .Case("got_page", MCSymbolRefExpr::VK_Mips_GOT_PAGE)
          .Case("got_ofst", MCSymbolRefExpr::VK_Mips_GOT_OFST)
          .Case("hi(%neg(%gp_rel", MCSymbolRefExpr::VK_Mips_GPOFF_HI)
          .Case("lo(%neg(%gp_rel", MCSymbolRefExpr::VK_Mips_GPOFF_LO)
          .Default(MCSymbolRefExpr::VK_None);

  return VK;
}

bool MipsAsmParser::ParseInstruction(
    ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  // Check if we have valid mnemonic
  if (!mnemonicIsValid(Name, 0)) {
    Parser.eatToEndOfStatement();
    return Error(NameLoc, "Unknown instruction");
  }
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(MipsOperand::CreateToken(Name, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      // Parse and remember the operand.
      if (ParseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
    }
  }
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::reportParseError(StringRef ErrorMsg) {
  SMLoc Loc = getLexer().getLoc();
  Parser.eatToEndOfStatement();
  return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::parseSetNoAtDirective() {
  // Line should look like: ".set noat".
  // set at reg to 0.
  Options.setATReg(0);
  // eat noat
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetAtDirective() {
  // Line can be .set at - defaults to $1
  // or .set at=$reg
  int AtRegNo;
  getParser().Lex();
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Options.setATReg(1);
    Parser.Lex(); // Consume the EndOfStatement.
    return false;
  } else if (getLexer().is(AsmToken::Equal)) {
    getParser().Lex(); // Eat the '='.
    if (getLexer().isNot(AsmToken::Dollar)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    Parser.Lex(); // Eat the '$'.
    const AsmToken &Reg = Parser.getTok();
    if (Reg.is(AsmToken::Identifier)) {
      AtRegNo = matchCPURegisterName(Reg.getIdentifier());
    } else if (Reg.is(AsmToken::Integer)) {
      AtRegNo = Reg.getIntVal();
    } else {
      reportParseError("unexpected token in statement");
      return false;
    }

    if (AtRegNo < 1 || AtRegNo > 31) {
      reportParseError("unexpected token in statement");
      return false;
    }

    if (!Options.setATReg(AtRegNo)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    getParser().Lex(); // Eat the register.

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token in statement");
      return false;
    }
    Parser.Lex(); // Consume the EndOfStatement.
    return false;
  } else {
    reportParseError("unexpected token in statement");
    return false;
  }
}

bool MipsAsmParser::parseSetReorderDirective() {
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setReorder();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoReorderDirective() {
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setNoreorder();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetMacroDirective() {
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setMacro();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoMacroDirective() {
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  if (Options.isReorder()) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  Options.setNomacro();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetAssignment() {
  StringRef Name;
  const MCExpr *Value;

  if (Parser.parseIdentifier(Name))
    reportParseError("expected identifier after .set");

  if (getLexer().isNot(AsmToken::Comma))
    return reportParseError("unexpected token in .set directive");
  Lex(); // Eat comma

  if (getLexer().is(AsmToken::Dollar)) {
    MCSymbol *Symbol;
    SMLoc DollarLoc = getLexer().getLoc();
    // Consume the dollar sign, and check for a following identifier.
    Parser.Lex();
    // We have a '$' followed by something, make sure they are adjacent.
    if (DollarLoc.getPointer() + 1 != getTok().getLoc().getPointer())
      return true;
    StringRef Res =
        StringRef(DollarLoc.getPointer(),
                  getTok().getEndLoc().getPointer() - DollarLoc.getPointer());
    Symbol = getContext().GetOrCreateSymbol(Res);
    Parser.Lex();
    Value =
        MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, getContext());
  } else if (Parser.parseExpression(Value))
    return reportParseError("expected valid expression after comma");

  // Check if the Name already exists as a symbol.
  MCSymbol *Sym = getContext().LookupSymbol(Name);
  if (Sym)
    return reportParseError("symbol already defined");
  Sym = getContext().GetOrCreateSymbol(Name);
  Sym->setVariableValue(Value);

  return false;
}

bool MipsAsmParser::parseDirectiveSet() {

  // Get the next token.
  const AsmToken &Tok = Parser.getTok();

  if (Tok.getString() == "noat") {
    return parseSetNoAtDirective();
  } else if (Tok.getString() == "at") {
    return parseSetAtDirective();
  } else if (Tok.getString() == "reorder") {
    return parseSetReorderDirective();
  } else if (Tok.getString() == "noreorder") {
    return parseSetNoReorderDirective();
  } else if (Tok.getString() == "macro") {
    return parseSetMacroDirective();
  } else if (Tok.getString() == "nomacro") {
    return parseSetNoMacroDirective();
  } else if (Tok.getString() == "nomips16") {
    // Ignore this directive for now.
    Parser.eatToEndOfStatement();
    return false;
  } else if (Tok.getString() == "nomicromips") {
    // Ignore this directive for now.
    Parser.eatToEndOfStatement();
    return false;
  } else {
    // It is just an identifier, look for an assignment.
    parseSetAssignment();
    return false;
  }

  return true;
}

bool MipsAsmParser::parseDirectiveMipsHackStocg() {
  MCAsmParser &Parser = getParser();
  StringRef Name;
  if (Parser.parseIdentifier(Name))
    reportParseError("expected identifier");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);
  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token");
  Lex();

  int64_t Flags = 0;
  if (Parser.parseAbsoluteExpression(Flags))
    return TokError("unexpected token");

  getTargetStreamer().emitMipsHackSTOCG(Sym, Flags);
  return false;
}

bool MipsAsmParser::parseDirectiveMipsHackELFFlags() {
  int64_t Flags = 0;
  if (Parser.parseAbsoluteExpression(Flags))
    return TokError("unexpected token");

  getTargetStreamer().emitMipsHackELFFlags(Flags);
  return false;
}

/// parseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool MipsAsmParser::parseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// parseDirectiveGpWord
///  ::= .gpword local_sym
bool MipsAsmParser::parseDirectiveGpWord() {
  const MCExpr *Value;
  // EmitGPRel32Value requires an expression, so we are using base class
  // method to evaluate the expression.
  if (getParser().parseExpression(Value))
    return true;
  getParser().getStreamer().EmitGPRel32Value(Value);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(), "unexpected token in directive");
  Parser.Lex(); // Eat EndOfStatement token.
  return false;
}

bool MipsAsmParser::ParseDirective(AsmToken DirectiveID) {

  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".ent") {
    // Ignore this directive for now.
    Parser.Lex();
    return false;
  }

  if (IDVal == ".end") {
    // Ignore this directive for now.
    Parser.Lex();
    return false;
  }

  if (IDVal == ".frame") {
    // Ignore this directive for now.
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".set") {
    return parseDirectiveSet();
  }

  if (IDVal == ".fmask") {
    // Ignore this directive for now.
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".mask") {
    // Ignore this directive for now.
    Parser.eatToEndOfStatement();
    return false;
  }

  if (IDVal == ".gpword") {
    // Ignore this directive for now.
    parseDirectiveGpWord();
    return false;
  }

  if (IDVal == ".word") {
    parseDirectiveWord(4, DirectiveID.getLoc());
    return false;
  }

  if (IDVal == ".mips_hack_stocg")
    return parseDirectiveMipsHackStocg();

  if (IDVal == ".mips_hack_elf_flags")
    return parseDirectiveMipsHackELFFlags();

  return true;
}

extern "C" void LLVMInitializeMipsAsmParser() {
  RegisterMCAsmParser<MipsAsmParser> X(TheMipsTarget);
  RegisterMCAsmParser<MipsAsmParser> Y(TheMipselTarget);
  RegisterMCAsmParser<MipsAsmParser> A(TheMips64Target);
  RegisterMCAsmParser<MipsAsmParser> B(TheMips64elTarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MipsGenAsmMatcher.inc"
