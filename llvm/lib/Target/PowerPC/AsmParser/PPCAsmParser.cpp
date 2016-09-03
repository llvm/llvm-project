//===-- PPCAsmParser.cpp - Parse PowerPC asm to MCInst instructions -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCExpr.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "PPCTargetStreamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static const MCPhysReg RRegs[32] = {
  PPC::R0,  PPC::R1,  PPC::R2,  PPC::R3,
  PPC::R4,  PPC::R5,  PPC::R6,  PPC::R7,
  PPC::R8,  PPC::R9,  PPC::R10, PPC::R11,
  PPC::R12, PPC::R13, PPC::R14, PPC::R15,
  PPC::R16, PPC::R17, PPC::R18, PPC::R19,
  PPC::R20, PPC::R21, PPC::R22, PPC::R23,
  PPC::R24, PPC::R25, PPC::R26, PPC::R27,
  PPC::R28, PPC::R29, PPC::R30, PPC::R31
};
static const MCPhysReg RRegsNoR0[32] = {
  PPC::ZERO,
            PPC::R1,  PPC::R2,  PPC::R3,
  PPC::R4,  PPC::R5,  PPC::R6,  PPC::R7,
  PPC::R8,  PPC::R9,  PPC::R10, PPC::R11,
  PPC::R12, PPC::R13, PPC::R14, PPC::R15,
  PPC::R16, PPC::R17, PPC::R18, PPC::R19,
  PPC::R20, PPC::R21, PPC::R22, PPC::R23,
  PPC::R24, PPC::R25, PPC::R26, PPC::R27,
  PPC::R28, PPC::R29, PPC::R30, PPC::R31
};
static const MCPhysReg XRegs[32] = {
  PPC::X0,  PPC::X1,  PPC::X2,  PPC::X3,
  PPC::X4,  PPC::X5,  PPC::X6,  PPC::X7,
  PPC::X8,  PPC::X9,  PPC::X10, PPC::X11,
  PPC::X12, PPC::X13, PPC::X14, PPC::X15,
  PPC::X16, PPC::X17, PPC::X18, PPC::X19,
  PPC::X20, PPC::X21, PPC::X22, PPC::X23,
  PPC::X24, PPC::X25, PPC::X26, PPC::X27,
  PPC::X28, PPC::X29, PPC::X30, PPC::X31
};
static const MCPhysReg XRegsNoX0[32] = {
  PPC::ZERO8,
            PPC::X1,  PPC::X2,  PPC::X3,
  PPC::X4,  PPC::X5,  PPC::X6,  PPC::X7,
  PPC::X8,  PPC::X9,  PPC::X10, PPC::X11,
  PPC::X12, PPC::X13, PPC::X14, PPC::X15,
  PPC::X16, PPC::X17, PPC::X18, PPC::X19,
  PPC::X20, PPC::X21, PPC::X22, PPC::X23,
  PPC::X24, PPC::X25, PPC::X26, PPC::X27,
  PPC::X28, PPC::X29, PPC::X30, PPC::X31
};
static const MCPhysReg FRegs[32] = {
  PPC::F0,  PPC::F1,  PPC::F2,  PPC::F3,
  PPC::F4,  PPC::F5,  PPC::F6,  PPC::F7,
  PPC::F8,  PPC::F9,  PPC::F10, PPC::F11,
  PPC::F12, PPC::F13, PPC::F14, PPC::F15,
  PPC::F16, PPC::F17, PPC::F18, PPC::F19,
  PPC::F20, PPC::F21, PPC::F22, PPC::F23,
  PPC::F24, PPC::F25, PPC::F26, PPC::F27,
  PPC::F28, PPC::F29, PPC::F30, PPC::F31
};
static const MCPhysReg VRegs[32] = {
  PPC::V0,  PPC::V1,  PPC::V2,  PPC::V3,
  PPC::V4,  PPC::V5,  PPC::V6,  PPC::V7,
  PPC::V8,  PPC::V9,  PPC::V10, PPC::V11,
  PPC::V12, PPC::V13, PPC::V14, PPC::V15,
  PPC::V16, PPC::V17, PPC::V18, PPC::V19,
  PPC::V20, PPC::V21, PPC::V22, PPC::V23,
  PPC::V24, PPC::V25, PPC::V26, PPC::V27,
  PPC::V28, PPC::V29, PPC::V30, PPC::V31
};
static const MCPhysReg VSRegs[64] = {
  PPC::VSL0,  PPC::VSL1,  PPC::VSL2,  PPC::VSL3,
  PPC::VSL4,  PPC::VSL5,  PPC::VSL6,  PPC::VSL7,
  PPC::VSL8,  PPC::VSL9,  PPC::VSL10, PPC::VSL11,
  PPC::VSL12, PPC::VSL13, PPC::VSL14, PPC::VSL15,
  PPC::VSL16, PPC::VSL17, PPC::VSL18, PPC::VSL19,
  PPC::VSL20, PPC::VSL21, PPC::VSL22, PPC::VSL23,
  PPC::VSL24, PPC::VSL25, PPC::VSL26, PPC::VSL27,
  PPC::VSL28, PPC::VSL29, PPC::VSL30, PPC::VSL31,

  PPC::VSH0,  PPC::VSH1,  PPC::VSH2,  PPC::VSH3,
  PPC::VSH4,  PPC::VSH5,  PPC::VSH6,  PPC::VSH7,
  PPC::VSH8,  PPC::VSH9,  PPC::VSH10, PPC::VSH11,
  PPC::VSH12, PPC::VSH13, PPC::VSH14, PPC::VSH15,
  PPC::VSH16, PPC::VSH17, PPC::VSH18, PPC::VSH19,
  PPC::VSH20, PPC::VSH21, PPC::VSH22, PPC::VSH23,
  PPC::VSH24, PPC::VSH25, PPC::VSH26, PPC::VSH27,
  PPC::VSH28, PPC::VSH29, PPC::VSH30, PPC::VSH31
};
static const MCPhysReg VSFRegs[64] = {
  PPC::F0,  PPC::F1,  PPC::F2,  PPC::F3,
  PPC::F4,  PPC::F5,  PPC::F6,  PPC::F7,
  PPC::F8,  PPC::F9,  PPC::F10, PPC::F11,
  PPC::F12, PPC::F13, PPC::F14, PPC::F15,
  PPC::F16, PPC::F17, PPC::F18, PPC::F19,
  PPC::F20, PPC::F21, PPC::F22, PPC::F23,
  PPC::F24, PPC::F25, PPC::F26, PPC::F27,
  PPC::F28, PPC::F29, PPC::F30, PPC::F31,

  PPC::VF0,  PPC::VF1,  PPC::VF2,  PPC::VF3,
  PPC::VF4,  PPC::VF5,  PPC::VF6,  PPC::VF7,
  PPC::VF8,  PPC::VF9,  PPC::VF10, PPC::VF11,
  PPC::VF12, PPC::VF13, PPC::VF14, PPC::VF15,
  PPC::VF16, PPC::VF17, PPC::VF18, PPC::VF19,
  PPC::VF20, PPC::VF21, PPC::VF22, PPC::VF23,
  PPC::VF24, PPC::VF25, PPC::VF26, PPC::VF27,
  PPC::VF28, PPC::VF29, PPC::VF30, PPC::VF31
};
static const MCPhysReg VSSRegs[64] = {
  PPC::F0,  PPC::F1,  PPC::F2,  PPC::F3,
  PPC::F4,  PPC::F5,  PPC::F6,  PPC::F7,
  PPC::F8,  PPC::F9,  PPC::F10, PPC::F11,
  PPC::F12, PPC::F13, PPC::F14, PPC::F15,
  PPC::F16, PPC::F17, PPC::F18, PPC::F19,
  PPC::F20, PPC::F21, PPC::F22, PPC::F23,
  PPC::F24, PPC::F25, PPC::F26, PPC::F27,
  PPC::F28, PPC::F29, PPC::F30, PPC::F31,

  PPC::VF0,  PPC::VF1,  PPC::VF2,  PPC::VF3,
  PPC::VF4,  PPC::VF5,  PPC::VF6,  PPC::VF7,
  PPC::VF8,  PPC::VF9,  PPC::VF10, PPC::VF11,
  PPC::VF12, PPC::VF13, PPC::VF14, PPC::VF15,
  PPC::VF16, PPC::VF17, PPC::VF18, PPC::VF19,
  PPC::VF20, PPC::VF21, PPC::VF22, PPC::VF23,
  PPC::VF24, PPC::VF25, PPC::VF26, PPC::VF27,
  PPC::VF28, PPC::VF29, PPC::VF30, PPC::VF31
};
static unsigned QFRegs[32] = {
  PPC::QF0,  PPC::QF1,  PPC::QF2,  PPC::QF3,
  PPC::QF4,  PPC::QF5,  PPC::QF6,  PPC::QF7,
  PPC::QF8,  PPC::QF9,  PPC::QF10, PPC::QF11,
  PPC::QF12, PPC::QF13, PPC::QF14, PPC::QF15,
  PPC::QF16, PPC::QF17, PPC::QF18, PPC::QF19,
  PPC::QF20, PPC::QF21, PPC::QF22, PPC::QF23,
  PPC::QF24, PPC::QF25, PPC::QF26, PPC::QF27,
  PPC::QF28, PPC::QF29, PPC::QF30, PPC::QF31
};
static const MCPhysReg CRBITRegs[32] = {
  PPC::CR0LT, PPC::CR0GT, PPC::CR0EQ, PPC::CR0UN,
  PPC::CR1LT, PPC::CR1GT, PPC::CR1EQ, PPC::CR1UN,
  PPC::CR2LT, PPC::CR2GT, PPC::CR2EQ, PPC::CR2UN,
  PPC::CR3LT, PPC::CR3GT, PPC::CR3EQ, PPC::CR3UN,
  PPC::CR4LT, PPC::CR4GT, PPC::CR4EQ, PPC::CR4UN,
  PPC::CR5LT, PPC::CR5GT, PPC::CR5EQ, PPC::CR5UN,
  PPC::CR6LT, PPC::CR6GT, PPC::CR6EQ, PPC::CR6UN,
  PPC::CR7LT, PPC::CR7GT, PPC::CR7EQ, PPC::CR7UN
};
static const MCPhysReg CRRegs[8] = {
  PPC::CR0, PPC::CR1, PPC::CR2, PPC::CR3,
  PPC::CR4, PPC::CR5, PPC::CR6, PPC::CR7
};

// Evaluate an expression containing condition register
// or condition register field symbols.  Returns positive
// value on success, or -1 on error.
static int64_t
EvaluateCRExpr(const MCExpr *E) {
  switch (E->getKind()) {
  case MCExpr::Target:
    return -1;

  case MCExpr::Constant: {
    int64_t Res = cast<MCConstantExpr>(E)->getValue();
    return Res < 0 ? -1 : Res;
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);
    StringRef Name = SRE->getSymbol().getName();

    if (Name == "lt") return 0;
    if (Name == "gt") return 1;
    if (Name == "eq") return 2;
    if (Name == "so") return 3;
    if (Name == "un") return 3;

    if (Name == "cr0") return 0;
    if (Name == "cr1") return 1;
    if (Name == "cr2") return 2;
    if (Name == "cr3") return 3;
    if (Name == "cr4") return 4;
    if (Name == "cr5") return 5;
    if (Name == "cr6") return 6;
    if (Name == "cr7") return 7;

    return -1;
  }

  case MCExpr::Unary:
    return -1;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    int64_t LHSVal = EvaluateCRExpr(BE->getLHS());
    int64_t RHSVal = EvaluateCRExpr(BE->getRHS());
    int64_t Res;

    if (LHSVal < 0 || RHSVal < 0)
      return -1;

    switch (BE->getOpcode()) {
    default: return -1;
    case MCBinaryExpr::Add: Res = LHSVal + RHSVal; break;
    case MCBinaryExpr::Mul: Res = LHSVal * RHSVal; break;
    }

    return Res < 0 ? -1 : Res;
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

namespace {

struct PPCOperand;

class PPCAsmParser : public MCTargetAsmParser {
  const MCInstrInfo &MII;
  bool IsPPC64;
  bool IsDarwin;

  void Warning(SMLoc L, const Twine &Msg) { getParser().Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return getParser().Error(L, Msg); }

  bool isPPC64() const { return IsPPC64; }
  bool isDarwin() const { return IsDarwin; }

  bool MatchRegisterName(const AsmToken &Tok,
                         unsigned &RegNo, int64_t &IntVal);

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  const MCExpr *ExtractModifierFromExpr(const MCExpr *E,
                                        PPCMCExpr::VariantKind &Variant);
  const MCExpr *FixupVariantKind(const MCExpr *E);
  bool ParseExpression(const MCExpr *&EVal);
  bool ParseDarwinExpression(const MCExpr *&EVal);

  bool ParseOperand(OperandVector &Operands);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveTC(unsigned Size, SMLoc L);
  bool ParseDirectiveMachine(SMLoc L);
  bool ParseDarwinDirectiveMachine(SMLoc L);
  bool ParseDirectiveAbiVersion(SMLoc L);
  bool ParseDirectiveLocalEntry(SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  void ProcessInstruction(MCInst &Inst, const OperandVector &Ops);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "PPCGenAsmMatcher.inc"

  /// }


public:
  PPCAsmParser(const MCSubtargetInfo &STI, MCAsmParser &,
               const MCInstrInfo &MII, const MCTargetOptions &Options)
    : MCTargetAsmParser(Options, STI), MII(MII) {
    // Check for 64-bit vs. 32-bit pointer mode.
    const Triple &TheTriple = STI.getTargetTriple();
    IsPPC64 = (TheTriple.getArch() == Triple::ppc64 ||
               TheTriple.getArch() == Triple::ppc64le);
    IsDarwin = TheTriple.isMacOSX();
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  const MCExpr *applyModifierToExpr(const MCExpr *E,
                                    MCSymbolRefExpr::VariantKind,
                                    MCContext &Ctx) override;
};

/// PPCOperand - Instances of this class represent a parsed PowerPC machine
/// instruction.
struct PPCOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    ContextImmediate,
    Expression,
    TLSRegister
  } Kind;

  SMLoc StartLoc, EndLoc;
  bool IsPPC64;

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    int64_t Val;
  };

  struct ExprOp {
    const MCExpr *Val;
    int64_t CRVal;     // Cached result of EvaluateCRExpr(Val)
  };

  struct TLSRegOp {
    const MCSymbolRefExpr *Sym;
  };

  union {
    struct TokOp Tok;
    struct ImmOp Imm;
    struct ExprOp Expr;
    struct TLSRegOp TLSReg;
  };

  PPCOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}
public:
  PPCOperand(const PPCOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    IsPPC64 = o.IsPPC64;
    switch (Kind) {
    case Token:
      Tok = o.Tok;
      break;
    case Immediate:
    case ContextImmediate:
      Imm = o.Imm;
      break;
    case Expression:
      Expr = o.Expr;
      break;
    case TLSRegister:
      TLSReg = o.TLSReg;
      break;
    }
  }

  // Disable use of sized deallocation due to overallocation of PPCOperand
  // objects in CreateTokenWithStringCopy.
  void operator delete(void *p) { ::operator delete(p); }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }

  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  /// isPPC64 - True if this operand is for an instruction in 64-bit mode.
  bool isPPC64() const { return IsPPC64; }

  int64_t getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }
  int64_t getImmS16Context() const {
    assert((Kind == Immediate || Kind == ContextImmediate) &&
           "Invalid access!");
    if (Kind == Immediate)
      return Imm.Val;
    return static_cast<int16_t>(Imm.Val);
  }
  int64_t getImmU16Context() const {
    assert((Kind == Immediate || Kind == ContextImmediate) &&
           "Invalid access!");
    return Imm.Val;
  }

  const MCExpr *getExpr() const {
    assert(Kind == Expression && "Invalid access!");
    return Expr.Val;
  }

  int64_t getExprCRVal() const {
    assert(Kind == Expression && "Invalid access!");
    return Expr.CRVal;
  }

  const MCExpr *getTLSReg() const {
    assert(Kind == TLSRegister && "Invalid access!");
    return TLSReg.Sym;
  }

  unsigned getReg() const override {
    assert(isRegNumber() && "Invalid access!");
    return (unsigned) Imm.Val;
  }

  unsigned getVSReg() const {
    assert(isVSRegNumber() && "Invalid access!");
    return (unsigned) Imm.Val;
  }

  unsigned getCCReg() const {
    assert(isCCRegNumber() && "Invalid access!");
    return (unsigned) (Kind == Immediate ? Imm.Val : Expr.CRVal);
  }

  unsigned getCRBit() const {
    assert(isCRBitNumber() && "Invalid access!");
    return (unsigned) (Kind == Immediate ? Imm.Val : Expr.CRVal);
  }

  unsigned getCRBitMask() const {
    assert(isCRBitMask() && "Invalid access!");
    return 7 - countTrailingZeros<uint64_t>(Imm.Val);
  }

  bool isToken() const override { return Kind == Token; }
  bool isImm() const override {
    return Kind == Immediate || Kind == Expression;
  }
  bool isU1Imm() const { return Kind == Immediate && isUInt<1>(getImm()); }
  bool isU2Imm() const { return Kind == Immediate && isUInt<2>(getImm()); }
  bool isU3Imm() const { return Kind == Immediate && isUInt<3>(getImm()); }
  bool isU4Imm() const { return Kind == Immediate && isUInt<4>(getImm()); }
  bool isU5Imm() const { return Kind == Immediate && isUInt<5>(getImm()); }
  bool isS5Imm() const { return Kind == Immediate && isInt<5>(getImm()); }
  bool isU6Imm() const { return Kind == Immediate && isUInt<6>(getImm()); }
  bool isU6ImmX2() const { return Kind == Immediate &&
                                  isUInt<6>(getImm()) &&
                                  (getImm() & 1) == 0; }
  bool isU7Imm() const { return Kind == Immediate && isUInt<7>(getImm()); }
  bool isU7ImmX4() const { return Kind == Immediate &&
                                  isUInt<7>(getImm()) &&
                                  (getImm() & 3) == 0; }
  bool isU8Imm() const { return Kind == Immediate && isUInt<8>(getImm()); }
  bool isU8ImmX8() const { return Kind == Immediate &&
                                  isUInt<8>(getImm()) &&
                                  (getImm() & 7) == 0; }

  bool isU10Imm() const { return Kind == Immediate && isUInt<10>(getImm()); }
  bool isU12Imm() const { return Kind == Immediate && isUInt<12>(getImm()); }
  bool isU16Imm() const {
    switch (Kind) {
      case Expression:
        return true;
      case Immediate:
      case ContextImmediate:
        return isUInt<16>(getImmU16Context());
      default:
        return false;
    }
  }
  bool isS16Imm() const {
    switch (Kind) {
      case Expression:
        return true;
      case Immediate:
      case ContextImmediate:
        return isInt<16>(getImmS16Context());
      default:
        return false;
    }
  }
  bool isS16ImmX4() const { return Kind == Expression ||
                                   (Kind == Immediate && isInt<16>(getImm()) &&
                                    (getImm() & 3) == 0); }
  bool isS16ImmX16() const { return Kind == Expression ||
                                    (Kind == Immediate && isInt<16>(getImm()) &&
                                     (getImm() & 15) == 0); }
  bool isS17Imm() const {
    switch (Kind) {
      case Expression:
        return true;
      case Immediate:
      case ContextImmediate:
        return isInt<17>(getImmS16Context());
      default:
        return false;
    }
  }
  bool isTLSReg() const { return Kind == TLSRegister; }
  bool isDirectBr() const {
    if (Kind == Expression)
      return true;
    if (Kind != Immediate)
      return false;
    // Operand must be 64-bit aligned, signed 27-bit immediate.
    if ((getImm() & 3) != 0)
      return false;
    if (isInt<26>(getImm()))
      return true;
    if (!IsPPC64) {
      // In 32-bit mode, large 32-bit quantities wrap around.
      if (isUInt<32>(getImm()) && isInt<26>(static_cast<int32_t>(getImm())))
        return true;
    }
    return false;
  }
  bool isCondBr() const { return Kind == Expression ||
                                 (Kind == Immediate && isInt<16>(getImm()) &&
                                  (getImm() & 3) == 0); }
  bool isRegNumber() const { return Kind == Immediate && isUInt<5>(getImm()); }
  bool isVSRegNumber() const {
    return Kind == Immediate && isUInt<6>(getImm());
  }
  bool isCCRegNumber() const { return (Kind == Expression
                                       && isUInt<3>(getExprCRVal())) ||
                                      (Kind == Immediate
                                       && isUInt<3>(getImm())); }
  bool isCRBitNumber() const { return (Kind == Expression
                                       && isUInt<5>(getExprCRVal())) ||
                                      (Kind == Immediate
                                       && isUInt<5>(getImm())); }
  bool isCRBitMask() const { return Kind == Immediate && isUInt<8>(getImm()) &&
                                    isPowerOf2_32(getImm()); }
  bool isATBitsAsHint() const { return false; }
  bool isMem() const override { return false; }
  bool isReg() const override { return false; }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("addRegOperands");
  }

  void addRegGPRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(RRegs[getReg()]));
  }

  void addRegGPRCNoR0Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(RRegsNoR0[getReg()]));
  }

  void addRegG8RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(XRegs[getReg()]));
  }

  void addRegG8RCNoX0Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(XRegsNoX0[getReg()]));
  }

  void addRegGxRCOperands(MCInst &Inst, unsigned N) const {
    if (isPPC64())
      addRegG8RCOperands(Inst, N);
    else
      addRegGPRCOperands(Inst, N);
  }

  void addRegGxRCNoR0Operands(MCInst &Inst, unsigned N) const {
    if (isPPC64())
      addRegG8RCNoX0Operands(Inst, N);
    else
      addRegGPRCNoR0Operands(Inst, N);
  }

  void addRegF4RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(FRegs[getReg()]));
  }

  void addRegF8RCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(FRegs[getReg()]));
  }

  void addRegVRRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(VRegs[getReg()]));
  }

  void addRegVSRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(VSRegs[getVSReg()]));
  }

  void addRegVSFRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(VSFRegs[getVSReg()]));
  }

  void addRegVSSRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(VSSRegs[getVSReg()]));
  }

  void addRegQFRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(QFRegs[getReg()]));
  }

  void addRegQSRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(QFRegs[getReg()]));
  }

  void addRegQBRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(QFRegs[getReg()]));
  }

  void addRegCRBITRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(CRBITRegs[getCRBit()]));
  }

  void addRegCRRCOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(CRRegs[getCCReg()]));
  }

  void addCRBitMaskOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(CRRegs[getCRBitMask()]));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == Immediate)
      Inst.addOperand(MCOperand::createImm(getImm()));
    else
      Inst.addOperand(MCOperand::createExpr(getExpr()));
  }

  void addS16ImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    switch (Kind) {
      case Immediate:
        Inst.addOperand(MCOperand::createImm(getImm()));
        break;
      case ContextImmediate:
        Inst.addOperand(MCOperand::createImm(getImmS16Context()));
        break;
      default:
        Inst.addOperand(MCOperand::createExpr(getExpr()));
        break;
    }
  }

  void addU16ImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    switch (Kind) {
      case Immediate:
        Inst.addOperand(MCOperand::createImm(getImm()));
        break;
      case ContextImmediate:
        Inst.addOperand(MCOperand::createImm(getImmU16Context()));
        break;
      default:
        Inst.addOperand(MCOperand::createExpr(getExpr()));
        break;
    }
  }

  void addBranchTargetOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == Immediate)
      Inst.addOperand(MCOperand::createImm(getImm() / 4));
    else
      Inst.addOperand(MCOperand::createExpr(getExpr()));
  }

  void addTLSRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createExpr(getTLSReg()));
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  void print(raw_ostream &OS) const override;

  static std::unique_ptr<PPCOperand> CreateToken(StringRef Str, SMLoc S,
                                                 bool IsPPC64) {
    auto Op = make_unique<PPCOperand>(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand>
  CreateTokenWithStringCopy(StringRef Str, SMLoc S, bool IsPPC64) {
    // Allocate extra memory for the string and copy it.
    // FIXME: This is incorrect, Operands are owned by unique_ptr with a default
    // deleter which will destroy them by simply using "delete", not correctly
    // calling operator delete on this extra memory after calling the dtor
    // explicitly.
    void *Mem = ::operator new(sizeof(PPCOperand) + Str.size());
    std::unique_ptr<PPCOperand> Op(new (Mem) PPCOperand(Token));
    Op->Tok.Data = reinterpret_cast<const char *>(Op.get() + 1);
    Op->Tok.Length = Str.size();
    std::memcpy(const_cast<char *>(Op->Tok.Data), Str.data(), Str.size());
    Op->StartLoc = S;
    Op->EndLoc = S;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand> CreateImm(int64_t Val, SMLoc S, SMLoc E,
                                               bool IsPPC64) {
    auto Op = make_unique<PPCOperand>(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand> CreateExpr(const MCExpr *Val, SMLoc S,
                                                SMLoc E, bool IsPPC64) {
    auto Op = make_unique<PPCOperand>(Expression);
    Op->Expr.Val = Val;
    Op->Expr.CRVal = EvaluateCRExpr(Val);
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand>
  CreateTLSReg(const MCSymbolRefExpr *Sym, SMLoc S, SMLoc E, bool IsPPC64) {
    auto Op = make_unique<PPCOperand>(TLSRegister);
    Op->TLSReg.Sym = Sym;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand>
  CreateContextImm(int64_t Val, SMLoc S, SMLoc E, bool IsPPC64) {
    auto Op = make_unique<PPCOperand>(ContextImmediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    Op->IsPPC64 = IsPPC64;
    return Op;
  }

  static std::unique_ptr<PPCOperand>
  CreateFromMCExpr(const MCExpr *Val, SMLoc S, SMLoc E, bool IsPPC64) {
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Val))
      return CreateImm(CE->getValue(), S, E, IsPPC64);

    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(Val))
      if (SRE->getKind() == MCSymbolRefExpr::VK_PPC_TLS)
        return CreateTLSReg(SRE, S, E, IsPPC64);

    if (const PPCMCExpr *TE = dyn_cast<PPCMCExpr>(Val)) {
      int64_t Res;
      if (TE->evaluateAsConstant(Res))
        return CreateContextImm(Res, S, E, IsPPC64);
    }

    return CreateExpr(Val, S, E, IsPPC64);
  }
};

} // end anonymous namespace.

void PPCOperand::print(raw_ostream &OS) const {
  switch (Kind) {
  case Token:
    OS << "'" << getToken() << "'";
    break;
  case Immediate:
  case ContextImmediate:
    OS << getImm();
    break;
  case Expression:
    OS << *getExpr();
    break;
  case TLSRegister:
    OS << *getTLSReg();
    break;
  }
}

static void
addNegOperand(MCInst &Inst, MCOperand &Op, MCContext &Ctx) {
  if (Op.isImm()) {
    Inst.addOperand(MCOperand::createImm(-Op.getImm()));
    return;
  }
  const MCExpr *Expr = Op.getExpr();
  if (const MCUnaryExpr *UnExpr = dyn_cast<MCUnaryExpr>(Expr)) {
    if (UnExpr->getOpcode() == MCUnaryExpr::Minus) {
      Inst.addOperand(MCOperand::createExpr(UnExpr->getSubExpr()));
      return;
    }
  } else if (const MCBinaryExpr *BinExpr = dyn_cast<MCBinaryExpr>(Expr)) {
    if (BinExpr->getOpcode() == MCBinaryExpr::Sub) {
      const MCExpr *NE = MCBinaryExpr::createSub(BinExpr->getRHS(),
                                                 BinExpr->getLHS(), Ctx);
      Inst.addOperand(MCOperand::createExpr(NE));
      return;
    }
  }
  Inst.addOperand(MCOperand::createExpr(MCUnaryExpr::createMinus(Expr, Ctx)));
}

void PPCAsmParser::ProcessInstruction(MCInst &Inst,
                                      const OperandVector &Operands) {
  int Opcode = Inst.getOpcode();
  switch (Opcode) {
  case PPC::DCBTx:
  case PPC::DCBTT:
  case PPC::DCBTSTx:
  case PPC::DCBTSTT: {
    MCInst TmpInst;
    TmpInst.setOpcode((Opcode == PPC::DCBTx || Opcode == PPC::DCBTT) ?
                      PPC::DCBT : PPC::DCBTST);
    TmpInst.addOperand(MCOperand::createImm(
      (Opcode == PPC::DCBTx || Opcode == PPC::DCBTSTx) ? 0 : 16));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::DCBTCT:
  case PPC::DCBTDS: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::DCBT);
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::DCBTSTCT:
  case PPC::DCBTSTDS: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::DCBTST);
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::DCBFx:
  case PPC::DCBFL:
  case PPC::DCBFLP: {
    int L = 0;
    if (Opcode == PPC::DCBFL)
      L = 1;
    else if (Opcode == PPC::DCBFLP)
      L = 3;

    MCInst TmpInst;
    TmpInst.setOpcode(PPC::DCBF);
    TmpInst.addOperand(MCOperand::createImm(L));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::LAx: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::LA);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(Inst.getOperand(1));
    Inst = TmpInst;
    break;
  }
  case PPC::SUBI: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::ADDI);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    addNegOperand(TmpInst, Inst.getOperand(2), getContext());
    Inst = TmpInst;
    break;
  }
  case PPC::SUBIS: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::ADDIS);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    addNegOperand(TmpInst, Inst.getOperand(2), getContext());
    Inst = TmpInst;
    break;
  }
  case PPC::SUBIC: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::ADDIC);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    addNegOperand(TmpInst, Inst.getOperand(2), getContext());
    Inst = TmpInst;
    break;
  }
  case PPC::SUBICo: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::ADDICo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    addNegOperand(TmpInst, Inst.getOperand(2), getContext());
    Inst = TmpInst;
    break;
  }
  case PPC::EXTLWI:
  case PPC::EXTLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(B));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(N - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTRWI:
  case PPC::EXTRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(B + N));
    TmpInst.addOperand(MCOperand::createImm(32 - N));
    TmpInst.addOperand(MCOperand::createImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::INSLWI:
  case PPC::INSLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSLWI? PPC::RLWIMI : PPC::RLWIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(32 - B));
    TmpInst.addOperand(MCOperand::createImm(B));
    TmpInst.addOperand(MCOperand::createImm((B + N) - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::INSRWI:
  case PPC::INSRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSRWI? PPC::RLWIMI : PPC::RLWIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(32 - (B + N)));
    TmpInst.addOperand(MCOperand::createImm(B));
    TmpInst.addOperand(MCOperand::createImm((B + N) - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::ROTRWI:
  case PPC::ROTRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::ROTRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(32 - N));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::SLWI:
  case PPC::SLWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(N));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRWI:
  case PPC::SRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(32 - N));
    TmpInst.addOperand(MCOperand::createImm(N));
    TmpInst.addOperand(MCOperand::createImm(31));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRRWI:
  case PPC::CLRRWIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRRWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRLSLWI:
  case PPC::CLRLSLWIo: {
    MCInst TmpInst;
    int64_t B = Inst.getOperand(2).getImm();
    int64_t N = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRLSLWI? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(N));
    TmpInst.addOperand(MCOperand::createImm(B - N));
    TmpInst.addOperand(MCOperand::createImm(31 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTLDI:
  case PPC::EXTLDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTLDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(B));
    TmpInst.addOperand(MCOperand::createImm(N - 1));
    Inst = TmpInst;
    break;
  }
  case PPC::EXTRDI:
  case PPC::EXTRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::EXTRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(B + N));
    TmpInst.addOperand(MCOperand::createImm(64 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::INSRDI:
  case PPC::INSRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    int64_t B = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::INSRDI? PPC::RLDIMI : PPC::RLDIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(64 - (B + N)));
    TmpInst.addOperand(MCOperand::createImm(B));
    Inst = TmpInst;
    break;
  }
  case PPC::ROTRDI:
  case PPC::ROTRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::ROTRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(64 - N));
    TmpInst.addOperand(MCOperand::createImm(0));
    Inst = TmpInst;
    break;
  }
  case PPC::SLDI:
  case PPC::SLDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SLDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(N));
    TmpInst.addOperand(MCOperand::createImm(63 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::SRDI:
  case PPC::SRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::SRDI? PPC::RLDICL : PPC::RLDICLo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(64 - N));
    TmpInst.addOperand(MCOperand::createImm(N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRRDI:
  case PPC::CLRRDIo: {
    MCInst TmpInst;
    int64_t N = Inst.getOperand(2).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRRDI? PPC::RLDICR : PPC::RLDICRo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(0));
    TmpInst.addOperand(MCOperand::createImm(63 - N));
    Inst = TmpInst;
    break;
  }
  case PPC::CLRLSLDI:
  case PPC::CLRLSLDIo: {
    MCInst TmpInst;
    int64_t B = Inst.getOperand(2).getImm();
    int64_t N = Inst.getOperand(3).getImm();
    TmpInst.setOpcode(Opcode == PPC::CLRLSLDI? PPC::RLDIC : PPC::RLDICo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(N));
    TmpInst.addOperand(MCOperand::createImm(B - N));
    Inst = TmpInst;
    break;
  }
  case PPC::RLWINMbm:
  case PPC::RLWINMobm: {
    unsigned MB, ME;
    int64_t BM = Inst.getOperand(3).getImm();
    if (!isRunOfOnes(BM, MB, ME))
      break;

    MCInst TmpInst;
    TmpInst.setOpcode(Opcode == PPC::RLWINMbm ? PPC::RLWINM : PPC::RLWINMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(MCOperand::createImm(MB));
    TmpInst.addOperand(MCOperand::createImm(ME));
    Inst = TmpInst;
    break;
  }
  case PPC::RLWIMIbm:
  case PPC::RLWIMIobm: {
    unsigned MB, ME;
    int64_t BM = Inst.getOperand(3).getImm();
    if (!isRunOfOnes(BM, MB, ME))
      break;

    MCInst TmpInst;
    TmpInst.setOpcode(Opcode == PPC::RLWIMIbm ? PPC::RLWIMI : PPC::RLWIMIo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(0)); // The tied operand.
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(MCOperand::createImm(MB));
    TmpInst.addOperand(MCOperand::createImm(ME));
    Inst = TmpInst;
    break;
  }
  case PPC::RLWNMbm:
  case PPC::RLWNMobm: {
    unsigned MB, ME;
    int64_t BM = Inst.getOperand(3).getImm();
    if (!isRunOfOnes(BM, MB, ME))
      break;

    MCInst TmpInst;
    TmpInst.setOpcode(Opcode == PPC::RLWNMbm ? PPC::RLWNM : PPC::RLWNMo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(Inst.getOperand(2));
    TmpInst.addOperand(MCOperand::createImm(MB));
    TmpInst.addOperand(MCOperand::createImm(ME));
    Inst = TmpInst;
    break;
  }
  case PPC::MFTB: {
    if (getSTI().getFeatureBits()[PPC::FeatureMFTB]) {
      assert(Inst.getNumOperands() == 2 && "Expecting two operands");
      Inst.setOpcode(PPC::MFSPR);
    }
    break;
  }
  case PPC::CP_COPYx:
  case PPC::CP_COPY_FIRST: {
    MCInst TmpInst;
    TmpInst.setOpcode(PPC::CP_COPY);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(Opcode == PPC::CP_COPYx ? 0 : 1));

    Inst = TmpInst;
    break;
  }
  case PPC::CP_PASTEx :
  case PPC::CP_PASTE_LAST: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opcode == PPC::CP_PASTEx ?
                      PPC::CP_PASTE : PPC::CP_PASTEo);
    TmpInst.addOperand(Inst.getOperand(0));
    TmpInst.addOperand(Inst.getOperand(1));
    TmpInst.addOperand(MCOperand::createImm(Opcode == PPC::CP_PASTEx ? 0 : 1));

    Inst = TmpInst;
    break;
  }
  }
}

bool PPCAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                           OperandVector &Operands,
                                           MCStreamer &Out, uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  MCInst Inst;

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  case Match_Success:
    // Post-process instructions (typically extended mnemonics)
    ProcessInstruction(Inst, Operands);
    Inst.setLoc(IDLoc);
    Out.EmitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction use requires an option to be enabled");
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((PPCOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  llvm_unreachable("Implement any new match types added!");
}

bool PPCAsmParser::
MatchRegisterName(const AsmToken &Tok, unsigned &RegNo, int64_t &IntVal) {
  if (Tok.is(AsmToken::Identifier)) {
    StringRef Name = Tok.getString();

    if (Name.equals_lower("lr")) {
      RegNo = isPPC64()? PPC::LR8 : PPC::LR;
      IntVal = 8;
      return false;
    } else if (Name.equals_lower("ctr")) {
      RegNo = isPPC64()? PPC::CTR8 : PPC::CTR;
      IntVal = 9;
      return false;
    } else if (Name.equals_lower("vrsave")) {
      RegNo = PPC::VRSAVE;
      IntVal = 256;
      return false;
    } else if (Name.startswith_lower("r") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = isPPC64()? XRegs[IntVal] : RRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("f") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = FRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("vs") &&
               !Name.substr(2).getAsInteger(10, IntVal) && IntVal < 64) {
      RegNo = VSRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("v") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = VRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("q") &&
               !Name.substr(1).getAsInteger(10, IntVal) && IntVal < 32) {
      RegNo = QFRegs[IntVal];
      return false;
    } else if (Name.startswith_lower("cr") &&
               !Name.substr(2).getAsInteger(10, IntVal) && IntVal < 8) {
      RegNo = CRRegs[IntVal];
      return false;
    }
  }

  return true;
}

bool PPCAsmParser::
ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  int64_t IntVal;

  if (!MatchRegisterName(Tok, RegNo, IntVal)) {
    Parser.Lex(); // Eat identifier token.
    return false;
  }

  return Error(StartLoc, "invalid register name");
}

/// Extract \code @l/@ha \endcode modifier from expression.  Recursively scan
/// the expression and check for VK_PPC_LO/HI/HA
/// symbol variants.  If all symbols with modifier use the same
/// variant, return the corresponding PPCMCExpr::VariantKind,
/// and a modified expression using the default symbol variant.
/// Otherwise, return NULL.
const MCExpr *PPCAsmParser::
ExtractModifierFromExpr(const MCExpr *E,
                        PPCMCExpr::VariantKind &Variant) {
  MCContext &Context = getParser().getContext();
  Variant = PPCMCExpr::VK_PPC_None;

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return nullptr;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    switch (SRE->getKind()) {
    case MCSymbolRefExpr::VK_PPC_LO:
      Variant = PPCMCExpr::VK_PPC_LO;
      break;
    case MCSymbolRefExpr::VK_PPC_HI:
      Variant = PPCMCExpr::VK_PPC_HI;
      break;
    case MCSymbolRefExpr::VK_PPC_HA:
      Variant = PPCMCExpr::VK_PPC_HA;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHER:
      Variant = PPCMCExpr::VK_PPC_HIGHER;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHERA:
      Variant = PPCMCExpr::VK_PPC_HIGHERA;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHEST:
      Variant = PPCMCExpr::VK_PPC_HIGHEST;
      break;
    case MCSymbolRefExpr::VK_PPC_HIGHESTA:
      Variant = PPCMCExpr::VK_PPC_HIGHESTA;
      break;
    default:
      return nullptr;
    }

    return MCSymbolRefExpr::create(&SRE->getSymbol(), Context);
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = ExtractModifierFromExpr(UE->getSubExpr(), Variant);
    if (!Sub)
      return nullptr;
    return MCUnaryExpr::create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    PPCMCExpr::VariantKind LHSVariant, RHSVariant;
    const MCExpr *LHS = ExtractModifierFromExpr(BE->getLHS(), LHSVariant);
    const MCExpr *RHS = ExtractModifierFromExpr(BE->getRHS(), RHSVariant);

    if (!LHS && !RHS)
      return nullptr;

    if (!LHS) LHS = BE->getLHS();
    if (!RHS) RHS = BE->getRHS();

    if (LHSVariant == PPCMCExpr::VK_PPC_None)
      Variant = RHSVariant;
    else if (RHSVariant == PPCMCExpr::VK_PPC_None)
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

/// Find all VK_TLSGD/VK_TLSLD symbol references in expression and replace
/// them by VK_PPC_TLSGD/VK_PPC_TLSLD.  This is necessary to avoid having
/// _GLOBAL_OFFSET_TABLE_ created via ELFObjectWriter::RelocNeedsGOT.
/// FIXME: This is a hack.
const MCExpr *PPCAsmParser::
FixupVariantKind(const MCExpr *E) {
  MCContext &Context = getParser().getContext();

  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return E;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);
    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;

    switch (SRE->getKind()) {
    case MCSymbolRefExpr::VK_TLSGD:
      Variant = MCSymbolRefExpr::VK_PPC_TLSGD;
      break;
    case MCSymbolRefExpr::VK_TLSLD:
      Variant = MCSymbolRefExpr::VK_PPC_TLSLD;
      break;
    default:
      return E;
    }
    return MCSymbolRefExpr::create(&SRE->getSymbol(), Variant, Context);
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = FixupVariantKind(UE->getSubExpr());
    if (Sub == UE->getSubExpr())
      return E;
    return MCUnaryExpr::create(UE->getOpcode(), Sub, Context);
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = FixupVariantKind(BE->getLHS());
    const MCExpr *RHS = FixupVariantKind(BE->getRHS());
    if (LHS == BE->getLHS() && RHS == BE->getRHS())
      return E;
    return MCBinaryExpr::create(BE->getOpcode(), LHS, RHS, Context);
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// ParseExpression.  This differs from the default "parseExpression" in that
/// it handles modifiers.
bool PPCAsmParser::
ParseExpression(const MCExpr *&EVal) {

  if (isDarwin())
    return ParseDarwinExpression(EVal);

  // (ELF Platforms)
  // Handle \code @l/@ha \endcode
  if (getParser().parseExpression(EVal))
    return true;

  EVal = FixupVariantKind(EVal);

  PPCMCExpr::VariantKind Variant;
  const MCExpr *E = ExtractModifierFromExpr(EVal, Variant);
  if (E)
    EVal = PPCMCExpr::create(Variant, E, false, getParser().getContext());

  return false;
}

/// ParseDarwinExpression.  (MachO Platforms)
/// This differs from the default "parseExpression" in that it handles detection
/// of the \code hi16(), ha16() and lo16() \endcode modifiers.  At present,
/// parseExpression() doesn't recognise the modifiers when in the Darwin/MachO
/// syntax form so it is done here.  TODO: Determine if there is merit in
/// arranging for this to be done at a higher level.
bool PPCAsmParser::
ParseDarwinExpression(const MCExpr *&EVal) {
  MCAsmParser &Parser = getParser();
  PPCMCExpr::VariantKind Variant = PPCMCExpr::VK_PPC_None;
  switch (getLexer().getKind()) {
  default:
    break;
  case AsmToken::Identifier:
    // Compiler-generated Darwin identifiers begin with L,l,_ or "; thus
    // something starting with any other char should be part of the
    // asm syntax.  If handwritten asm includes an identifier like lo16,
    // then all bets are off - but no-one would do that, right?
    StringRef poss = Parser.getTok().getString();
    if (poss.equals_lower("lo16")) {
      Variant = PPCMCExpr::VK_PPC_LO;
    } else if (poss.equals_lower("hi16")) {
      Variant = PPCMCExpr::VK_PPC_HI;
    } else if (poss.equals_lower("ha16")) {
      Variant = PPCMCExpr::VK_PPC_HA;
    }
    if (Variant != PPCMCExpr::VK_PPC_None) {
      Parser.Lex(); // Eat the xx16
      if (getLexer().isNot(AsmToken::LParen))
        return Error(Parser.getTok().getLoc(), "expected '('");
      Parser.Lex(); // Eat the '('
    }
    break;
  }

  if (getParser().parseExpression(EVal))
    return true;

  if (Variant != PPCMCExpr::VK_PPC_None) {
    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "expected ')'");
    Parser.Lex(); // Eat the ')'
    EVal = PPCMCExpr::create(Variant, EVal, false, getParser().getContext());
  }
  return false;
}

/// ParseOperand
/// This handles registers in the form 'NN', '%rNN' for ELF platforms and
/// rNN for MachO.
bool PPCAsmParser::ParseOperand(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  const MCExpr *EVal;

  // Attempt to parse the next token as an immediate
  switch (getLexer().getKind()) {
  // Special handling for register names.  These are interpreted
  // as immediates corresponding to the register number.
  case AsmToken::Percent:
    Parser.Lex(); // Eat the '%'.
    unsigned RegNo;
    int64_t IntVal;
    if (!MatchRegisterName(Parser.getTok(), RegNo, IntVal)) {
      Parser.Lex(); // Eat the identifier token.
      Operands.push_back(PPCOperand::CreateImm(IntVal, S, E, isPPC64()));
      return false;
    }
    return Error(S, "invalid register name");

  case AsmToken::Identifier:
    // Note that non-register-name identifiers from the compiler will begin
    // with '_', 'L'/'l' or '"'.  Of course, handwritten asm could include
    // identifiers like r31foo - so we fall through in the event that parsing
    // a register name fails.
    if (isDarwin()) {
      unsigned RegNo;
      int64_t IntVal;
      if (!MatchRegisterName(Parser.getTok(), RegNo, IntVal)) {
        Parser.Lex(); // Eat the identifier token.
        Operands.push_back(PPCOperand::CreateImm(IntVal, S, E, isPPC64()));
        return false;
      }
    }
    // Fall-through to process non-register-name identifiers as expression.
    LLVM_FALLTHROUGH;
  // All other expressions
  case AsmToken::LParen:
  case AsmToken::Plus:
  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Dot:
  case AsmToken::Dollar:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
    if (!ParseExpression(EVal))
      break;
    /* fall through */
  default:
    return Error(S, "unknown operand");
  }

  // Push the parsed operand into the list of operands
  Operands.push_back(PPCOperand::CreateFromMCExpr(EVal, S, E, isPPC64()));

  // Check whether this is a TLS call expression
  bool TLSCall = false;
  if (const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(EVal))
    TLSCall = Ref->getSymbol().getName() == "__tls_get_addr";

  if (TLSCall && getLexer().is(AsmToken::LParen)) {
    const MCExpr *TLSSym;

    Parser.Lex(); // Eat the '('.
    S = Parser.getTok().getLoc();
    if (ParseExpression(TLSSym))
      return Error(S, "invalid TLS call expression");
    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "missing ')'");
    E = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the ')'.

    Operands.push_back(PPCOperand::CreateFromMCExpr(TLSSym, S, E, isPPC64()));
  }

  // Otherwise, check for D-form memory operands
  if (!TLSCall && getLexer().is(AsmToken::LParen)) {
    Parser.Lex(); // Eat the '('.
    S = Parser.getTok().getLoc();

    int64_t IntVal;
    switch (getLexer().getKind()) {
    case AsmToken::Percent:
      Parser.Lex(); // Eat the '%'.
      unsigned RegNo;
      if (MatchRegisterName(Parser.getTok(), RegNo, IntVal))
        return Error(S, "invalid register name");
      Parser.Lex(); // Eat the identifier token.
      break;

    case AsmToken::Integer:
      if (!isDarwin()) {
        if (getParser().parseAbsoluteExpression(IntVal) ||
          IntVal < 0 || IntVal > 31)
        return Error(S, "invalid register number");
      } else {
        return Error(S, "unexpected integer value");
      }
      break;

   case AsmToken::Identifier:
    if (isDarwin()) {
      unsigned RegNo;
      if (!MatchRegisterName(Parser.getTok(), RegNo, IntVal)) {
        Parser.Lex(); // Eat the identifier token.
        break;
      }
    }
    LLVM_FALLTHROUGH;

    default:
      return Error(S, "invalid memory operand");
    }

    if (getLexer().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "missing ')'");
    E = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the ')'.

    Operands.push_back(PPCOperand::CreateImm(IntVal, S, E, isPPC64()));
  }

  return false;
}

/// Parse an instruction mnemonic followed by its operands.
bool PPCAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  // The first operand is the token for the instruction name.
  // If the next character is a '+' or '-', we need to add it to the
  // instruction name, to match what TableGen is doing.
  std::string NewOpcode;
  if (getLexer().is(AsmToken::Plus)) {
    getLexer().Lex();
    NewOpcode = Name;
    NewOpcode += '+';
    Name = NewOpcode;
  }
  if (getLexer().is(AsmToken::Minus)) {
    getLexer().Lex();
    NewOpcode = Name;
    NewOpcode += '-';
    Name = NewOpcode;
  }
  // If the instruction ends in a '.', we need to create a separate
  // token for it, to match what TableGen is doing.
  size_t Dot = Name.find('.');
  StringRef Mnemonic = Name.slice(0, Dot);
  if (!NewOpcode.empty()) // Underlying memory for Name is volatile.
    Operands.push_back(
        PPCOperand::CreateTokenWithStringCopy(Mnemonic, NameLoc, isPPC64()));
  else
    Operands.push_back(PPCOperand::CreateToken(Mnemonic, NameLoc, isPPC64()));
  if (Dot != StringRef::npos) {
    SMLoc DotLoc = SMLoc::getFromPointer(NameLoc.getPointer() + Dot);
    StringRef DotStr = Name.slice(Dot, StringRef::npos);
    if (!NewOpcode.empty()) // Underlying memory for Name is volatile.
      Operands.push_back(
          PPCOperand::CreateTokenWithStringCopy(DotStr, DotLoc, isPPC64()));
    else
      Operands.push_back(PPCOperand::CreateToken(DotStr, DotLoc, isPPC64()));
  }

  // If there are no more operands then finish
  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Parse the first operand
  if (ParseOperand(Operands))
    return true;

  while (getLexer().isNot(AsmToken::EndOfStatement) &&
         getLexer().is(AsmToken::Comma)) {
    // Consume the comma token
    Lex();

    // Parse the next operand
    if (ParseOperand(Operands))
      return true;
  }

  // We'll now deal with an unfortunate special case: the syntax for the dcbt
  // and dcbtst instructions differs for server vs. embedded cores.
  //  The syntax for dcbt is:
  //    dcbt ra, rb, th [server]
  //    dcbt th, ra, rb [embedded]
  //  where th can be omitted when it is 0. dcbtst is the same. We take the
  //  server form to be the default, so swap the operands if we're parsing for
  //  an embedded core (they'll be swapped again upon printing).
  if (getSTI().getFeatureBits()[PPC::FeatureBookE] &&
      Operands.size() == 4 &&
      (Name == "dcbt" || Name == "dcbtst")) {
    std::swap(Operands[1], Operands[3]);
    std::swap(Operands[2], Operands[1]);
  }

  return false;
}

/// ParseDirective parses the PPC specific directives
bool PPCAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (!isDarwin()) {
    if (IDVal == ".word")
      return ParseDirectiveWord(2, DirectiveID.getLoc());
    if (IDVal == ".llong")
      return ParseDirectiveWord(8, DirectiveID.getLoc());
    if (IDVal == ".tc")
      return ParseDirectiveTC(isPPC64()? 8 : 4, DirectiveID.getLoc());
    if (IDVal == ".machine")
      return ParseDirectiveMachine(DirectiveID.getLoc());
    if (IDVal == ".abiversion")
      return ParseDirectiveAbiVersion(DirectiveID.getLoc());
    if (IDVal == ".localentry")
      return ParseDirectiveLocalEntry(DirectiveID.getLoc());
  } else {
    if (IDVal == ".machine")
      return ParseDarwinDirectiveMachine(DirectiveID.getLoc());
  }
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool PPCAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = getLexer().getLoc();
      if (getParser().parseExpression(Value))
        return false;

      if (const auto *MCE = dyn_cast<MCConstantExpr>(Value)) {
        assert(Size <= 8 && "Invalid size");
        uint64_t IntValue = MCE->getValue();
        if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
          return Error(ExprLoc, "literal value out of range for directive");
        getStreamer().EmitIntValue(IntValue, Size);
      } else {
        getStreamer().EmitValue(Value, Size, ExprLoc);
      }

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// ParseDirectiveTC
///  ::= .tc [ symbol (, expression)* ]
bool PPCAsmParser::ParseDirectiveTC(unsigned Size, SMLoc L) {
  MCAsmParser &Parser = getParser();
  // Skip TC symbol, which is only used with XCOFF.
  while (getLexer().isNot(AsmToken::EndOfStatement)
         && getLexer().isNot(AsmToken::Comma))
    Parser.Lex();
  if (getLexer().isNot(AsmToken::Comma)) {
    Error(L, "unexpected token in directive");
    return false;
  }
  Parser.Lex();

  // Align to word size.
  getParser().getStreamer().EmitValueToAlignment(Size);

  // Emit expressions.
  return ParseDirectiveWord(Size, L);
}

/// ParseDirectiveMachine (ELF platforms)
///  ::= .machine [ cpu | "push" | "pop" ]
bool PPCAsmParser::ParseDirectiveMachine(SMLoc L) {
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::Identifier) &&
      getLexer().isNot(AsmToken::String)) {
    Error(L, "unexpected token in directive");
    return false;
  }

  StringRef CPU = Parser.getTok().getIdentifier();
  Parser.Lex();

  // FIXME: Right now, the parser always allows any available
  // instruction, so the .machine directive is not useful.
  // Implement ".machine any" (by doing nothing) for the benefit
  // of existing assembler code.  Likewise, we can then implement
  // ".machine push" and ".machine pop" as no-op.
  if (CPU != "any" && CPU != "push" && CPU != "pop") {
    Error(L, "unrecognized machine type");
    return false;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Error(L, "unexpected token in directive");
    return false;
  }
  PPCTargetStreamer &TStreamer =
      *static_cast<PPCTargetStreamer *>(
           getParser().getStreamer().getTargetStreamer());
  TStreamer.emitMachine(CPU);

  return false;
}

/// ParseDarwinDirectiveMachine (Mach-o platforms)
///  ::= .machine cpu-identifier
bool PPCAsmParser::ParseDarwinDirectiveMachine(SMLoc L) {
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::Identifier) &&
      getLexer().isNot(AsmToken::String)) {
    Error(L, "unexpected token in directive");
    return false;
  }

  StringRef CPU = Parser.getTok().getIdentifier();
  Parser.Lex();

  // FIXME: this is only the 'default' set of cpu variants.
  // However we don't act on this information at present, this is simply
  // allowing parsing to proceed with minimal sanity checking.
  if (CPU != "ppc7400" && CPU != "ppc" && CPU != "ppc64") {
    Error(L, "unrecognized cpu type");
    return false;
  }

  if (isPPC64() && (CPU == "ppc7400" || CPU == "ppc")) {
    Error(L, "wrong cpu type specified for 64bit");
    return false;
  }
  if (!isPPC64() && CPU == "ppc64") {
    Error(L, "wrong cpu type specified for 32bit");
    return false;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Error(L, "unexpected token in directive");
    return false;
  }

  return false;
}

/// ParseDirectiveAbiVersion
///  ::= .abiversion constant-expression
bool PPCAsmParser::ParseDirectiveAbiVersion(SMLoc L) {
  int64_t AbiVersion;
  if (getParser().parseAbsoluteExpression(AbiVersion)){
    Error(L, "expected constant expression");
    return false;
  }
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Error(L, "unexpected token in directive");
    return false;
  }

  PPCTargetStreamer &TStreamer =
      *static_cast<PPCTargetStreamer *>(
           getParser().getStreamer().getTargetStreamer());
  TStreamer.emitAbiVersion(AbiVersion);

  return false;
}

/// ParseDirectiveLocalEntry
///  ::= .localentry symbol, expression
bool PPCAsmParser::ParseDirectiveLocalEntry(SMLoc L) {
  StringRef Name;
  if (getParser().parseIdentifier(Name)) {
    Error(L, "expected identifier in directive");
    return false;
  }
  MCSymbolELF *Sym = cast<MCSymbolELF>(getContext().getOrCreateSymbol(Name));

  if (getLexer().isNot(AsmToken::Comma)) {
    Error(L, "unexpected token in directive");
    return false;
  }
  Lex();

  const MCExpr *Expr;
  if (getParser().parseExpression(Expr)) {
    Error(L, "expected expression");
    return false;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    Error(L, "unexpected token in directive");
    return false;
  }

  PPCTargetStreamer &TStreamer =
      *static_cast<PPCTargetStreamer *>(
           getParser().getStreamer().getTargetStreamer());
  TStreamer.emitLocalEntry(Sym, Expr);

  return false;
}



/// Force static initialization.
extern "C" void LLVMInitializePowerPCAsmParser() {
  RegisterMCAsmParser<PPCAsmParser> A(ThePPC32Target);
  RegisterMCAsmParser<PPCAsmParser> B(ThePPC64Target);
  RegisterMCAsmParser<PPCAsmParser> C(ThePPC64LETarget);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "PPCGenAsmMatcher.inc"

// Define this matcher function after the auto-generated include so we
// have the match class enum definitions.
unsigned PPCAsmParser::validateTargetOperandClass(MCParsedAsmOperand &AsmOp,
                                                  unsigned Kind) {
  // If the kind is a token for a literal immediate, check if our asm
  // operand matches. This is for InstAliases which have a fixed-value
  // immediate in the syntax.
  int64_t ImmVal;
  switch (Kind) {
    case MCK_0: ImmVal = 0; break;
    case MCK_1: ImmVal = 1; break;
    case MCK_2: ImmVal = 2; break;
    case MCK_3: ImmVal = 3; break;
    case MCK_4: ImmVal = 4; break;
    case MCK_5: ImmVal = 5; break;
    case MCK_6: ImmVal = 6; break;
    case MCK_7: ImmVal = 7; break;
    default: return Match_InvalidOperand;
  }

  PPCOperand &Op = static_cast<PPCOperand &>(AsmOp);
  if (Op.isImm() && Op.getImm() == ImmVal)
    return Match_Success;

  return Match_InvalidOperand;
}

const MCExpr *
PPCAsmParser::applyModifierToExpr(const MCExpr *E,
                                  MCSymbolRefExpr::VariantKind Variant,
                                  MCContext &Ctx) {
  switch (Variant) {
  case MCSymbolRefExpr::VK_PPC_LO:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_LO, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HI:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HI, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HA:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HA, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHER:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HIGHER, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHERA:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HIGHERA, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHEST:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HIGHEST, E, false, Ctx);
  case MCSymbolRefExpr::VK_PPC_HIGHESTA:
    return PPCMCExpr::create(PPCMCExpr::VK_PPC_HIGHESTA, E, false, Ctx);
  default:
    return nullptr;
  }
}
