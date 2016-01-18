//===-- SystemZAsmParser.cpp - Parse SystemZ assembly instructions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// Return true if Expr is in the range [MinValue, MaxValue].
static bool inRange(const MCExpr *Expr, int64_t MinValue, int64_t MaxValue) {
  if (auto *CE = dyn_cast<MCConstantExpr>(Expr)) {
    int64_t Value = CE->getValue();
    return Value >= MinValue && Value <= MaxValue;
  }
  return false;
}

namespace {
enum RegisterKind {
  GR32Reg,
  GRH32Reg,
  GR64Reg,
  GR128Reg,
  ADDR32Reg,
  ADDR64Reg,
  FP32Reg,
  FP64Reg,
  FP128Reg,
  VR32Reg,
  VR64Reg,
  VR128Reg
};

enum MemoryKind {
  BDMem,
  BDXMem,
  BDLMem,
  BDVMem
};

class SystemZOperand : public MCParsedAsmOperand {
public:
private:
  enum OperandKind {
    KindInvalid,
    KindToken,
    KindReg,
    KindAccessReg,
    KindImm,
    KindImmTLS,
    KindMem
  };

  OperandKind Kind;
  SMLoc StartLoc, EndLoc;

  // A string of length Length, starting at Data.
  struct TokenOp {
    const char *Data;
    unsigned Length;
  };

  // LLVM register Num, which has kind Kind.  In some ways it might be
  // easier for this class to have a register bank (general, floating-point
  // or access) and a raw register number (0-15).  This would postpone the
  // interpretation of the operand to the add*() methods and avoid the need
  // for context-dependent parsing.  However, we do things the current way
  // because of the virtual getReg() method, which needs to distinguish
  // between (say) %r0 used as a single register and %r0 used as a pair.
  // Context-dependent parsing can also give us slightly better error
  // messages when invalid pairs like %r1 are used.
  struct RegOp {
    RegisterKind Kind;
    unsigned Num;
  };

  // Base + Disp + Index, where Base and Index are LLVM registers or 0.
  // MemKind says what type of memory this is and RegKind says what type
  // the base register has (ADDR32Reg or ADDR64Reg).  Length is the operand
  // length for D(L,B)-style operands, otherwise it is null.
  struct MemOp {
    unsigned Base : 12;
    unsigned Index : 12;
    unsigned MemKind : 4;
    unsigned RegKind : 4;
    const MCExpr *Disp;
    const MCExpr *Length;
  };

  // Imm is an immediate operand, and Sym is an optional TLS symbol
  // for use with a __tls_get_offset marker relocation.
  struct ImmTLSOp {
    const MCExpr *Imm;
    const MCExpr *Sym;
  };

  union {
    TokenOp Token;
    RegOp Reg;
    unsigned AccessReg;
    const MCExpr *Imm;
    ImmTLSOp ImmTLS;
    MemOp Mem;
  };

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

public:
  SystemZOperand(OperandKind kind, SMLoc startLoc, SMLoc endLoc)
      : Kind(kind), StartLoc(startLoc), EndLoc(endLoc) {}

  // Create particular kinds of operand.
  static std::unique_ptr<SystemZOperand> createInvalid(SMLoc StartLoc,
                                                       SMLoc EndLoc) {
    return make_unique<SystemZOperand>(KindInvalid, StartLoc, EndLoc);
  }
  static std::unique_ptr<SystemZOperand> createToken(StringRef Str, SMLoc Loc) {
    auto Op = make_unique<SystemZOperand>(KindToken, Loc, Loc);
    Op->Token.Data = Str.data();
    Op->Token.Length = Str.size();
    return Op;
  }
  static std::unique_ptr<SystemZOperand>
  createReg(RegisterKind Kind, unsigned Num, SMLoc StartLoc, SMLoc EndLoc) {
    auto Op = make_unique<SystemZOperand>(KindReg, StartLoc, EndLoc);
    Op->Reg.Kind = Kind;
    Op->Reg.Num = Num;
    return Op;
  }
  static std::unique_ptr<SystemZOperand>
  createAccessReg(unsigned Num, SMLoc StartLoc, SMLoc EndLoc) {
    auto Op = make_unique<SystemZOperand>(KindAccessReg, StartLoc, EndLoc);
    Op->AccessReg = Num;
    return Op;
  }
  static std::unique_ptr<SystemZOperand>
  createImm(const MCExpr *Expr, SMLoc StartLoc, SMLoc EndLoc) {
    auto Op = make_unique<SystemZOperand>(KindImm, StartLoc, EndLoc);
    Op->Imm = Expr;
    return Op;
  }
  static std::unique_ptr<SystemZOperand>
  createMem(MemoryKind MemKind, RegisterKind RegKind, unsigned Base,
            const MCExpr *Disp, unsigned Index, const MCExpr *Length,
            SMLoc StartLoc, SMLoc EndLoc) {
    auto Op = make_unique<SystemZOperand>(KindMem, StartLoc, EndLoc);
    Op->Mem.MemKind = MemKind;
    Op->Mem.RegKind = RegKind;
    Op->Mem.Base = Base;
    Op->Mem.Index = Index;
    Op->Mem.Disp = Disp;
    Op->Mem.Length = Length;
    return Op;
  }
  static std::unique_ptr<SystemZOperand>
  createImmTLS(const MCExpr *Imm, const MCExpr *Sym,
               SMLoc StartLoc, SMLoc EndLoc) {
    auto Op = make_unique<SystemZOperand>(KindImmTLS, StartLoc, EndLoc);
    Op->ImmTLS.Imm = Imm;
    Op->ImmTLS.Sym = Sym;
    return Op;
  }

  // Token operands
  bool isToken() const override {
    return Kind == KindToken;
  }
  StringRef getToken() const {
    assert(Kind == KindToken && "Not a token");
    return StringRef(Token.Data, Token.Length);
  }

  // Register operands.
  bool isReg() const override {
    return Kind == KindReg;
  }
  bool isReg(RegisterKind RegKind) const {
    return Kind == KindReg && Reg.Kind == RegKind;
  }
  unsigned getReg() const override {
    assert(Kind == KindReg && "Not a register");
    return Reg.Num;
  }

  // Access register operands.  Access registers aren't exposed to LLVM
  // as registers.
  bool isAccessReg() const {
    return Kind == KindAccessReg;
  }

  // Immediate operands.
  bool isImm() const override {
    return Kind == KindImm;
  }
  bool isImm(int64_t MinValue, int64_t MaxValue) const {
    return Kind == KindImm && inRange(Imm, MinValue, MaxValue);
  }
  const MCExpr *getImm() const {
    assert(Kind == KindImm && "Not an immediate");
    return Imm;
  }

  // Immediate operands with optional TLS symbol.
  bool isImmTLS() const {
    return Kind == KindImmTLS;
  }

  // Memory operands.
  bool isMem() const override {
    return Kind == KindMem;
  }
  bool isMem(MemoryKind MemKind) const {
    return (Kind == KindMem &&
            (Mem.MemKind == MemKind ||
             // A BDMem can be treated as a BDXMem in which the index
             // register field is 0.
             (Mem.MemKind == BDMem && MemKind == BDXMem)));
  }
  bool isMem(MemoryKind MemKind, RegisterKind RegKind) const {
    return isMem(MemKind) && Mem.RegKind == RegKind;
  }
  bool isMemDisp12(MemoryKind MemKind, RegisterKind RegKind) const {
    return isMem(MemKind, RegKind) && inRange(Mem.Disp, 0, 0xfff);
  }
  bool isMemDisp20(MemoryKind MemKind, RegisterKind RegKind) const {
    return isMem(MemKind, RegKind) && inRange(Mem.Disp, -524288, 524287);
  }
  bool isMemDisp12Len8(RegisterKind RegKind) const {
    return isMemDisp12(BDLMem, RegKind) && inRange(Mem.Length, 1, 0x100);
  }
  void addBDVAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands");
    assert(isMem(BDVMem) && "Invalid operand type");
    Inst.addOperand(MCOperand::createReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
    Inst.addOperand(MCOperand::createReg(Mem.Index));
  }

  // Override MCParsedAsmOperand.
  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }
  void print(raw_ostream &OS) const override;

  // Used by the TableGen code to add particular types of operand
  // to an instruction.
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }
  void addAccessRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    assert(Kind == KindAccessReg && "Invalid operand type");
    Inst.addOperand(MCOperand::createImm(AccessReg));
  }
  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    addExpr(Inst, getImm());
  }
  void addBDAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands");
    assert(isMem(BDMem) && "Invalid operand type");
    Inst.addOperand(MCOperand::createReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
  }
  void addBDXAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands");
    assert(isMem(BDXMem) && "Invalid operand type");
    Inst.addOperand(MCOperand::createReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
    Inst.addOperand(MCOperand::createReg(Mem.Index));
  }
  void addBDLAddrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 3 && "Invalid number of operands");
    assert(isMem(BDLMem) && "Invalid operand type");
    Inst.addOperand(MCOperand::createReg(Mem.Base));
    addExpr(Inst, Mem.Disp);
    addExpr(Inst, Mem.Length);
  }
  void addImmTLSOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands");
    assert(Kind == KindImmTLS && "Invalid operand type");
    addExpr(Inst, ImmTLS.Imm);
    if (ImmTLS.Sym)
      addExpr(Inst, ImmTLS.Sym);
  }

  // Used by the TableGen code to check for particular operand types.
  bool isGR32() const { return isReg(GR32Reg); }
  bool isGRH32() const { return isReg(GRH32Reg); }
  bool isGRX32() const { return false; }
  bool isGR64() const { return isReg(GR64Reg); }
  bool isGR128() const { return isReg(GR128Reg); }
  bool isADDR32() const { return isReg(ADDR32Reg); }
  bool isADDR64() const { return isReg(ADDR64Reg); }
  bool isADDR128() const { return false; }
  bool isFP32() const { return isReg(FP32Reg); }
  bool isFP64() const { return isReg(FP64Reg); }
  bool isFP128() const { return isReg(FP128Reg); }
  bool isVR32() const { return isReg(VR32Reg); }
  bool isVR64() const { return isReg(VR64Reg); }
  bool isVF128() const { return false; }
  bool isVR128() const { return isReg(VR128Reg); }
  bool isBDAddr32Disp12() const { return isMemDisp12(BDMem, ADDR32Reg); }
  bool isBDAddr32Disp20() const { return isMemDisp20(BDMem, ADDR32Reg); }
  bool isBDAddr64Disp12() const { return isMemDisp12(BDMem, ADDR64Reg); }
  bool isBDAddr64Disp20() const { return isMemDisp20(BDMem, ADDR64Reg); }
  bool isBDXAddr64Disp12() const { return isMemDisp12(BDXMem, ADDR64Reg); }
  bool isBDXAddr64Disp20() const { return isMemDisp20(BDXMem, ADDR64Reg); }
  bool isBDLAddr64Disp12Len8() const { return isMemDisp12Len8(ADDR64Reg); }
  bool isBDVAddr64Disp12() const { return isMemDisp12(BDVMem, ADDR64Reg); }
  bool isU1Imm() const { return isImm(0, 1); }
  bool isU2Imm() const { return isImm(0, 3); }
  bool isU3Imm() const { return isImm(0, 7); }
  bool isU4Imm() const { return isImm(0, 15); }
  bool isU6Imm() const { return isImm(0, 63); }
  bool isU8Imm() const { return isImm(0, 255); }
  bool isS8Imm() const { return isImm(-128, 127); }
  bool isU12Imm() const { return isImm(0, 4095); }
  bool isU16Imm() const { return isImm(0, 65535); }
  bool isS16Imm() const { return isImm(-32768, 32767); }
  bool isU32Imm() const { return isImm(0, (1LL << 32) - 1); }
  bool isS32Imm() const { return isImm(-(1LL << 31), (1LL << 31) - 1); }
};

class SystemZAsmParser : public MCTargetAsmParser {
#define GET_ASSEMBLER_HEADER
#include "SystemZGenAsmMatcher.inc"

private:
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  enum RegisterGroup {
    RegGR,
    RegFP,
    RegV,
    RegAccess
  };
  struct Register {
    RegisterGroup Group;
    unsigned Num;
    SMLoc StartLoc, EndLoc;
  };

  bool parseRegister(Register &Reg);

  bool parseRegister(Register &Reg, RegisterGroup Group, const unsigned *Regs,
                     bool IsAddress = false);

  OperandMatchResultTy parseRegister(OperandVector &Operands,
                                     RegisterGroup Group, const unsigned *Regs,
                                     RegisterKind Kind);

  bool parseAddress(unsigned &Base, const MCExpr *&Disp,
                    unsigned &Index, bool &IsVector, const MCExpr *&Length,
                    const unsigned *Regs, RegisterKind RegKind);

  OperandMatchResultTy parseAddress(OperandVector &Operands,
                                    MemoryKind MemKind, const unsigned *Regs,
                                    RegisterKind RegKind);

  OperandMatchResultTy parsePCRel(OperandVector &Operands, int64_t MinVal,
                                  int64_t MaxVal, bool AllowTLS);

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

public:
  SystemZAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser,
                   const MCInstrInfo &MII,
                   const MCTargetOptions &Options)
      : MCTargetAsmParser(Options), STI(sti), Parser(parser) {
    MCAsmParserExtension::Initialize(Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  // Override MCTargetAsmParser.
  bool ParseDirective(AsmToken DirectiveID) override;
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  // Used by the TableGen code to parse particular operand types.
  OperandMatchResultTy parseGR32(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GR32Regs, GR32Reg);
  }
  OperandMatchResultTy parseGRH32(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GRH32Regs, GRH32Reg);
  }
  OperandMatchResultTy parseGRX32(OperandVector &Operands) {
    llvm_unreachable("GRX32 should only be used for pseudo instructions");
  }
  OperandMatchResultTy parseGR64(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GR64Regs, GR64Reg);
  }
  OperandMatchResultTy parseGR128(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GR128Regs, GR128Reg);
  }
  OperandMatchResultTy parseADDR32(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GR32Regs, ADDR32Reg);
  }
  OperandMatchResultTy parseADDR64(OperandVector &Operands) {
    return parseRegister(Operands, RegGR, SystemZMC::GR64Regs, ADDR64Reg);
  }
  OperandMatchResultTy parseADDR128(OperandVector &Operands) {
    llvm_unreachable("Shouldn't be used as an operand");
  }
  OperandMatchResultTy parseFP32(OperandVector &Operands) {
    return parseRegister(Operands, RegFP, SystemZMC::FP32Regs, FP32Reg);
  }
  OperandMatchResultTy parseFP64(OperandVector &Operands) {
    return parseRegister(Operands, RegFP, SystemZMC::FP64Regs, FP64Reg);
  }
  OperandMatchResultTy parseFP128(OperandVector &Operands) {
    return parseRegister(Operands, RegFP, SystemZMC::FP128Regs, FP128Reg);
  }
  OperandMatchResultTy parseVR32(OperandVector &Operands) {
    return parseRegister(Operands, RegV, SystemZMC::VR32Regs, VR32Reg);
  }
  OperandMatchResultTy parseVR64(OperandVector &Operands) {
    return parseRegister(Operands, RegV, SystemZMC::VR64Regs, VR64Reg);
  }
  OperandMatchResultTy parseVF128(OperandVector &Operands) {
    llvm_unreachable("Shouldn't be used as an operand");
  }
  OperandMatchResultTy parseVR128(OperandVector &Operands) {
    return parseRegister(Operands, RegV, SystemZMC::VR128Regs, VR128Reg);
  }
  OperandMatchResultTy parseBDAddr32(OperandVector &Operands) {
    return parseAddress(Operands, BDMem, SystemZMC::GR32Regs, ADDR32Reg);
  }
  OperandMatchResultTy parseBDAddr64(OperandVector &Operands) {
    return parseAddress(Operands, BDMem, SystemZMC::GR64Regs, ADDR64Reg);
  }
  OperandMatchResultTy parseBDXAddr64(OperandVector &Operands) {
    return parseAddress(Operands, BDXMem, SystemZMC::GR64Regs, ADDR64Reg);
  }
  OperandMatchResultTy parseBDLAddr64(OperandVector &Operands) {
    return parseAddress(Operands, BDLMem, SystemZMC::GR64Regs, ADDR64Reg);
  }
  OperandMatchResultTy parseBDVAddr64(OperandVector &Operands) {
    return parseAddress(Operands, BDVMem, SystemZMC::GR64Regs, ADDR64Reg);
  }
  OperandMatchResultTy parseAccessReg(OperandVector &Operands);
  OperandMatchResultTy parsePCRel16(OperandVector &Operands) {
    return parsePCRel(Operands, -(1LL << 16), (1LL << 16) - 1, false);
  }
  OperandMatchResultTy parsePCRel32(OperandVector &Operands) {
    return parsePCRel(Operands, -(1LL << 32), (1LL << 32) - 1, false);
  }
  OperandMatchResultTy parsePCRelTLS16(OperandVector &Operands) {
    return parsePCRel(Operands, -(1LL << 16), (1LL << 16) - 1, true);
  }
  OperandMatchResultTy parsePCRelTLS32(OperandVector &Operands) {
    return parsePCRel(Operands, -(1LL << 32), (1LL << 32) - 1, true);
  }
};
} // end anonymous namespace

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#include "SystemZGenAsmMatcher.inc"

void SystemZOperand::print(raw_ostream &OS) const {
  llvm_unreachable("Not implemented");
}

// Parse one register of the form %<prefix><number>.
bool SystemZAsmParser::parseRegister(Register &Reg) {
  Reg.StartLoc = Parser.getTok().getLoc();

  // Eat the % prefix.
  if (Parser.getTok().isNot(AsmToken::Percent))
    return Error(Parser.getTok().getLoc(), "register expected");
  Parser.Lex();

  // Expect a register name.
  if (Parser.getTok().isNot(AsmToken::Identifier))
    return Error(Reg.StartLoc, "invalid register");

  // Check that there's a prefix.
  StringRef Name = Parser.getTok().getString();
  if (Name.size() < 2)
    return Error(Reg.StartLoc, "invalid register");
  char Prefix = Name[0];

  // Treat the rest of the register name as a register number.
  if (Name.substr(1).getAsInteger(10, Reg.Num))
    return Error(Reg.StartLoc, "invalid register");

  // Look for valid combinations of prefix and number.
  if (Prefix == 'r' && Reg.Num < 16)
    Reg.Group = RegGR;
  else if (Prefix == 'f' && Reg.Num < 16)
    Reg.Group = RegFP;
  else if (Prefix == 'v' && Reg.Num < 32)
    Reg.Group = RegV;
  else if (Prefix == 'a' && Reg.Num < 16)
    Reg.Group = RegAccess;
  else
    return Error(Reg.StartLoc, "invalid register");

  Reg.EndLoc = Parser.getTok().getLoc();
  Parser.Lex();
  return false;
}

// Parse a register of group Group.  If Regs is nonnull, use it to map
// the raw register number to LLVM numbering, with zero entries
// indicating an invalid register.  IsAddress says whether the
// register appears in an address context. Allow FP Group if expecting
// RegV Group, since the f-prefix yields the FP group even while used
// with vector instructions.
bool SystemZAsmParser::parseRegister(Register &Reg, RegisterGroup Group,
                                     const unsigned *Regs, bool IsAddress) {
  if (parseRegister(Reg))
    return true;
  if (Reg.Group != Group && !(Reg.Group == RegFP && Group == RegV))
    return Error(Reg.StartLoc, "invalid operand for instruction");
  if (Regs && Regs[Reg.Num] == 0)
    return Error(Reg.StartLoc, "invalid register pair");
  if (Reg.Num == 0 && IsAddress)
    return Error(Reg.StartLoc, "%r0 used in an address");
  if (Regs)
    Reg.Num = Regs[Reg.Num];
  return false;
}

// Parse a register and add it to Operands.  The other arguments are as above.
SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseRegister(OperandVector &Operands, RegisterGroup Group,
                                const unsigned *Regs, RegisterKind Kind) {
  if (Parser.getTok().isNot(AsmToken::Percent))
    return MatchOperand_NoMatch;

  Register Reg;
  bool IsAddress = (Kind == ADDR32Reg || Kind == ADDR64Reg);
  if (parseRegister(Reg, Group, Regs, IsAddress))
    return MatchOperand_ParseFail;

  Operands.push_back(SystemZOperand::createReg(Kind, Reg.Num,
                                               Reg.StartLoc, Reg.EndLoc));
  return MatchOperand_Success;
}

// Parse a memory operand into Base, Disp, Index and Length.
// Regs maps asm register numbers to LLVM register numbers and RegKind
// says what kind of address register we're using (ADDR32Reg or ADDR64Reg).
bool SystemZAsmParser::parseAddress(unsigned &Base, const MCExpr *&Disp,
                                    unsigned &Index, bool &IsVector,
                                    const MCExpr *&Length, const unsigned *Regs,
                                    RegisterKind RegKind) {
  // Parse the displacement, which must always be present.
  if (getParser().parseExpression(Disp))
    return true;

  // Parse the optional base and index.
  Index = 0;
  Base = 0;
  IsVector = false;
  Length = nullptr;
  if (getLexer().is(AsmToken::LParen)) {
    Parser.Lex();

    if (getLexer().is(AsmToken::Percent)) {
      // Parse the first register and decide whether it's a base or an index.
      Register Reg;
      if (parseRegister(Reg))
        return true;
      if (Reg.Group == RegV) {
        // A vector index register.  The base register is optional.
        IsVector = true;
        Index = SystemZMC::VR128Regs[Reg.Num];
      } else if (Reg.Group == RegGR) {
        if (Reg.Num == 0)
          return Error(Reg.StartLoc, "%r0 used in an address");
        // If the are two registers, the first one is the index and the
        // second is the base.
        if (getLexer().is(AsmToken::Comma))
          Index = Regs[Reg.Num];
        else
          Base = Regs[Reg.Num];
      } else
        return Error(Reg.StartLoc, "invalid address register");
    } else {
      // Parse the length.
      if (getParser().parseExpression(Length))
        return true;
    }

    // Check whether there's a second register.  It's the base if so.
    if (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();
      Register Reg;
      if (parseRegister(Reg, RegGR, Regs, RegKind))
        return true;
      Base = Reg.Num;
    }

    // Consume the closing bracket.
    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "unexpected token in address");
    Parser.Lex();
  }
  return false;
}

// Parse a memory operand and add it to Operands.  The other arguments
// are as above.
SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseAddress(OperandVector &Operands, MemoryKind MemKind,
                               const unsigned *Regs, RegisterKind RegKind) {
  SMLoc StartLoc = Parser.getTok().getLoc();
  unsigned Base, Index;
  bool IsVector;
  const MCExpr *Disp;
  const MCExpr *Length;
  if (parseAddress(Base, Disp, Index, IsVector, Length, Regs, RegKind))
    return MatchOperand_ParseFail;

  if (IsVector && MemKind != BDVMem) {
    Error(StartLoc, "invalid use of vector addressing");
    return MatchOperand_ParseFail;
  }

  if (!IsVector && MemKind == BDVMem) {
    Error(StartLoc, "vector index required in address");
    return MatchOperand_ParseFail;
  }

  if (Index && MemKind != BDXMem && MemKind != BDVMem) {
    Error(StartLoc, "invalid use of indexed addressing");
    return MatchOperand_ParseFail;
  }

  if (Length && MemKind != BDLMem) {
    Error(StartLoc, "invalid use of length addressing");
    return MatchOperand_ParseFail;
  }

  if (!Length && MemKind == BDLMem) {
    Error(StartLoc, "missing length in address");
    return MatchOperand_ParseFail;
  }

  SMLoc EndLoc =
    SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(SystemZOperand::createMem(MemKind, RegKind, Base, Disp,
                                               Index, Length, StartLoc,
                                               EndLoc));
  return MatchOperand_Success;
}

bool SystemZAsmParser::ParseDirective(AsmToken DirectiveID) {
  return true;
}

bool SystemZAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                     SMLoc &EndLoc) {
  Register Reg;
  if (parseRegister(Reg))
    return true;
  if (Reg.Group == RegGR)
    RegNo = SystemZMC::GR64Regs[Reg.Num];
  else if (Reg.Group == RegFP)
    RegNo = SystemZMC::FP64Regs[Reg.Num];
  else if (Reg.Group == RegV)
    RegNo = SystemZMC::VR128Regs[Reg.Num];
  else
    // FIXME: Access registers aren't modelled as LLVM registers yet.
    return Error(Reg.StartLoc, "invalid operand for instruction");
  StartLoc = Reg.StartLoc;
  EndLoc = Reg.EndLoc;
  return false;
}

bool SystemZAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                        StringRef Name, SMLoc NameLoc,
                                        OperandVector &Operands) {
  Operands.push_back(SystemZOperand::createToken(Name, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Name)) {
      Parser.eatToEndOfStatement();
      return true;
    }

    // Read any subsequent operands.
    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();
      if (parseOperand(Operands, Name)) {
        Parser.eatToEndOfStatement();
        return true;
      }
    }
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  // Consume the EndOfStatement.
  Parser.Lex();
  return false;
}

bool SystemZAsmParser::parseOperand(OperandVector &Operands,
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

  // Check for a register.  All real register operands should have used
  // a context-dependent parse routine, which gives the required register
  // class.  The code is here to mop up other cases, like those where
  // the instruction isn't recognized.
  if (Parser.getTok().is(AsmToken::Percent)) {
    Register Reg;
    if (parseRegister(Reg))
      return true;
    Operands.push_back(SystemZOperand::createInvalid(Reg.StartLoc, Reg.EndLoc));
    return false;
  }

  // The only other type of operand is an immediate or address.  As above,
  // real address operands should have used a context-dependent parse routine,
  // so we treat any plain expression as an immediate.
  SMLoc StartLoc = Parser.getTok().getLoc();
  unsigned Base, Index;
  bool IsVector;
  const MCExpr *Expr, *Length;
  if (parseAddress(Base, Expr, Index, IsVector, Length, SystemZMC::GR64Regs,
                   ADDR64Reg))
    return true;

  SMLoc EndLoc =
    SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  if (Base || Index || Length)
    Operands.push_back(SystemZOperand::createInvalid(StartLoc, EndLoc));
  else
    Operands.push_back(SystemZOperand::createImm(Expr, StartLoc, EndLoc));
  return false;
}

bool SystemZAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                               OperandVector &Operands,
                                               MCStreamer &Out,
                                               uint64_t &ErrorInfo,
                                               bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult;

  MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                     MatchingInlineAsm);
  switch (MatchResult) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst, STI);
    return false;

  case Match_MissingFeature: {
    assert(ErrorInfo && "Unknown missing feature!");
    // Special case the error message for the very common case where only
    // a single subtarget feature is missing
    std::string Msg = "instruction requires:";
    uint64_t Mask = 1;
    for (unsigned I = 0; I < sizeof(ErrorInfo) * 8 - 1; ++I) {
      if (ErrorInfo & Mask) {
        Msg += " ";
        Msg += getSubtargetFeatureName(ErrorInfo & Mask);
      }
      Mask <<= 1;
    }
    return Error(IDLoc, Msg);
  }

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((SystemZOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }

  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }

  llvm_unreachable("Unexpected match type");
}

SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parseAccessReg(OperandVector &Operands) {
  if (Parser.getTok().isNot(AsmToken::Percent))
    return MatchOperand_NoMatch;

  Register Reg;
  if (parseRegister(Reg, RegAccess, nullptr))
    return MatchOperand_ParseFail;

  Operands.push_back(SystemZOperand::createAccessReg(Reg.Num,
                                                     Reg.StartLoc,
                                                     Reg.EndLoc));
  return MatchOperand_Success;
}

SystemZAsmParser::OperandMatchResultTy
SystemZAsmParser::parsePCRel(OperandVector &Operands, int64_t MinVal,
                             int64_t MaxVal, bool AllowTLS) {
  MCContext &Ctx = getContext();
  MCStreamer &Out = getStreamer();
  const MCExpr *Expr;
  SMLoc StartLoc = Parser.getTok().getLoc();
  if (getParser().parseExpression(Expr))
    return MatchOperand_NoMatch;

  // For consistency with the GNU assembler, treat immediates as offsets
  // from ".".
  if (auto *CE = dyn_cast<MCConstantExpr>(Expr)) {
    int64_t Value = CE->getValue();
    if ((Value & 1) || Value < MinVal || Value > MaxVal) {
      Error(StartLoc, "offset out of range");
      return MatchOperand_ParseFail;
    }
    MCSymbol *Sym = Ctx.createTempSymbol();
    Out.EmitLabel(Sym);
    const MCExpr *Base = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None,
                                                 Ctx);
    Expr = Value == 0 ? Base : MCBinaryExpr::createAdd(Base, Expr, Ctx);
  }

  // Optionally match :tls_gdcall: or :tls_ldcall: followed by a TLS symbol.
  const MCExpr *Sym = nullptr;
  if (AllowTLS && getLexer().is(AsmToken::Colon)) {
    Parser.Lex();

    if (Parser.getTok().isNot(AsmToken::Identifier)) {
      Error(Parser.getTok().getLoc(), "unexpected token");
      return MatchOperand_ParseFail;
    }

    MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
    StringRef Name = Parser.getTok().getString();
    if (Name == "tls_gdcall")
      Kind = MCSymbolRefExpr::VK_TLSGD;
    else if (Name == "tls_ldcall")
      Kind = MCSymbolRefExpr::VK_TLSLDM;
    else {
      Error(Parser.getTok().getLoc(), "unknown TLS tag");
      return MatchOperand_ParseFail;
    }
    Parser.Lex();

    if (Parser.getTok().isNot(AsmToken::Colon)) {
      Error(Parser.getTok().getLoc(), "unexpected token");
      return MatchOperand_ParseFail;
    }
    Parser.Lex();

    if (Parser.getTok().isNot(AsmToken::Identifier)) {
      Error(Parser.getTok().getLoc(), "unexpected token");
      return MatchOperand_ParseFail;
    }

    StringRef Identifier = Parser.getTok().getString();
    Sym = MCSymbolRefExpr::create(Ctx.getOrCreateSymbol(Identifier),
                                  Kind, Ctx);
    Parser.Lex();
  }

  SMLoc EndLoc =
    SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  if (AllowTLS)
    Operands.push_back(SystemZOperand::createImmTLS(Expr, Sym,
                                                    StartLoc, EndLoc));
  else
    Operands.push_back(SystemZOperand::createImm(Expr, StartLoc, EndLoc));

  return MatchOperand_Success;
}

// Force static initialization.
extern "C" void LLVMInitializeSystemZAsmParser() {
  RegisterMCAsmParser<SystemZAsmParser> X(TheSystemZTarget);
}
