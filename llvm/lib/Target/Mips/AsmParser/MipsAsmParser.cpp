//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsMCExpr.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsRegisterInfo.h"
#include "MipsTargetStreamer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "mips-asm-parser"

namespace llvm {
class MCInstrInfo;
}

namespace {
class MipsAssemblerOptions {
public:
  MipsAssemblerOptions(uint64_t Features_) : 
    ATReg(1), Reorder(true), Macro(true), Features(Features_) {}

  MipsAssemblerOptions(const MipsAssemblerOptions *Opts) {
    ATReg = Opts->getATRegNum();
    Reorder = Opts->isReorder();
    Macro = Opts->isMacro();
    Features = Opts->getFeatures();
  }

  unsigned getATRegNum() const { return ATReg; }
  bool setATReg(unsigned Reg);

  bool isReorder() const { return Reorder; }
  void setReorder() { Reorder = true; }
  void setNoReorder() { Reorder = false; }

  bool isMacro() const { return Macro; }
  void setMacro() { Macro = true; }
  void setNoMacro() { Macro = false; }

  uint64_t getFeatures() const { return Features; }
  void setFeatures(uint64_t Features_) { Features = Features_; }

private:
  unsigned ATReg;
  bool Reorder;
  bool Macro;
  uint64_t Features;
};
}

namespace {
class MipsAsmParser : public MCTargetAsmParser {
  MipsTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *Parser.getStreamer().getTargetStreamer();
    return static_cast<MipsTargetStreamer &>(TS);
  }

  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  SmallVector<std::unique_ptr<MipsAssemblerOptions>, 2> AssemblerOptions;

#define GET_ASSEMBLER_HEADER
#include "MipsGenAsmMatcher.inc"

  unsigned checkTargetMatchPredicate(MCInst &Inst) override;

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               unsigned &ErrorInfo,
                               bool MatchingInlineAsm) override;

  /// Parse a register as used in CFI directives
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  bool ParseParenSuffix(StringRef Name, OperandVector &Operands);

  bool ParseBracketSuffix(StringRef Name, OperandVector &Operands);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  MipsAsmParser::OperandMatchResultTy parseMemOperand(OperandVector &Operands);

  MipsAsmParser::OperandMatchResultTy
  MatchAnyRegisterNameWithoutDollar(OperandVector &Operands,
                                    StringRef Identifier, SMLoc S);

  MipsAsmParser::OperandMatchResultTy
  MatchAnyRegisterWithoutDollar(OperandVector &Operands, SMLoc S);

  MipsAsmParser::OperandMatchResultTy ParseAnyRegister(OperandVector &Operands);

  MipsAsmParser::OperandMatchResultTy ParseImm(OperandVector &Operands);

  MipsAsmParser::OperandMatchResultTy ParseJumpTarget(OperandVector &Operands);

  MipsAsmParser::OperandMatchResultTy parseInvNum(OperandVector &Operands);

  MipsAsmParser::OperandMatchResultTy ParseLSAImm(OperandVector &Operands);

  bool searchSymbolAlias(OperandVector &Operands);

  bool ParseOperand(OperandVector &, StringRef Mnemonic);

  bool needsExpansion(MCInst &Inst);

  // Expands assembly pseudo instructions.
  // Returns false on success, true otherwise.
  bool expandInstruction(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions);

  bool expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);

  bool expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);

  bool expandLoadAddressReg(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);

  void expandMemInst(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions, bool isLoad,
                     bool isImmOpnd);
  bool reportParseError(Twine ErrorMsg);
  bool reportParseError(SMLoc Loc, Twine ErrorMsg);

  bool parseMemOffset(const MCExpr *&Res, bool isParenExpr);
  bool parseRelocOperand(const MCExpr *&Res);

  const MCExpr *evaluateRelocExpr(const MCExpr *Expr, StringRef RelocStr);

  bool isEvaluated(const MCExpr *Expr);
  bool parseSetFeature(uint64_t Feature);
  bool parseDirectiveCPLoad(SMLoc Loc);
  bool parseDirectiveCPSetup();
  bool parseDirectiveNaN();
  bool parseDirectiveSet();
  bool parseDirectiveOption();

  bool parseSetAtDirective();
  bool parseSetNoAtDirective();
  bool parseSetMacroDirective();
  bool parseSetNoMacroDirective();
  bool parseSetReorderDirective();
  bool parseSetNoReorderDirective();
  bool parseSetNoMips16Directive();
  bool parseSetFpDirective();
  bool parseSetPopDirective();
  bool parseSetPushDirective();

  bool parseSetAssignment();

  bool parseDataDirective(unsigned Size, SMLoc L);
  bool parseDirectiveGpWord();
  bool parseDirectiveGpDWord();
  bool parseDirectiveModule();
  bool parseDirectiveModuleFP();
  bool parseFpABIValue(MipsABIFlagsSection::FpABIKind &FpABI,
                       StringRef Directive);

  MCSymbolRefExpr::VariantKind getVariantKind(StringRef Symbol);

  bool eatComma(StringRef ErrorStr);

  int matchCPURegisterName(StringRef Symbol);

  int matchRegisterByNumber(unsigned RegNum, unsigned RegClass);

  int matchFPURegisterName(StringRef Name);

  int matchFCCRegisterName(StringRef Name);

  int matchACRegisterName(StringRef Name);

  int matchMSA128RegisterName(StringRef Name);

  int matchMSA128CtrlRegisterName(StringRef Name);

  unsigned getReg(int RC, int RegNo);

  unsigned getGPR(int RegNo);

  int getATReg(SMLoc Loc);

  bool processInstruction(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  // Helper function that checks if the value of a vector index is within the
  // boundaries of accepted values for each RegisterKind
  // Example: INSERT.B $w0[n], $1 => 16 > n >= 0
  bool validateMSAIndex(int Val, int RegKind);

  void setFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (!(STI.getFeatureBits() & Feature)) {
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
    AssemblerOptions.back()->setFeatures(getAvailableFeatures());
  }

  void clearFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (STI.getFeatureBits() & Feature) {
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
    AssemblerOptions.back()->setFeatures(getAvailableFeatures());
  }

public:
  enum MipsMatchResultTy {
    Match_RequiresDifferentSrcAndDst = FIRST_TARGET_MATCH_RESULT_TY
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "MipsGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES

  };

  MipsAsmParser(MCSubtargetInfo &sti, MCAsmParser &parser,
                const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(), STI(sti), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
    
    // Remember the initial assembler options. The user can not modify these.
    MipsAssemblerOptions *TmpAOPtr = 
      new MipsAssemblerOptions(getAvailableFeatures());
    AssemblerOptions.push_back(std::unique_ptr<MipsAssemblerOptions>(TmpAOPtr));
    
    // Create an assembler options environment for the user to modify.
    TmpAOPtr = new MipsAssemblerOptions(getAvailableFeatures());
    AssemblerOptions.push_back(std::unique_ptr<MipsAssemblerOptions>(TmpAOPtr));
    TmpAOPtr = nullptr;

    getTargetStreamer().updateABIInfo(*this);

    // Assert exactly one ABI was chosen.
    assert((((STI.getFeatureBits() & Mips::FeatureO32) != 0) +
            ((STI.getFeatureBits() & Mips::FeatureEABI) != 0) +
            ((STI.getFeatureBits() & Mips::FeatureN32) != 0) +
            ((STI.getFeatureBits() & Mips::FeatureN64) != 0)) == 1);

    if (!isABI_O32() && !useOddSPReg() != 0)
      report_fatal_error("-mno-odd-spreg requires the O32 ABI");
  }

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  /// True if all of $fcc0 - $fcc7 exist for the current ISA.
  bool hasEightFccRegisters() const { return hasMips4() || hasMips32(); }

  bool isGP64bit() const { return STI.getFeatureBits() & Mips::FeatureGP64Bit; }
  bool isFP64bit() const { return STI.getFeatureBits() & Mips::FeatureFP64Bit; }
  bool isABI_N32() const { return STI.getFeatureBits() & Mips::FeatureN32; }
  bool isABI_N64() const { return STI.getFeatureBits() & Mips::FeatureN64; }
  bool isABI_O32() const { return STI.getFeatureBits() & Mips::FeatureO32; }
  bool isABI_FPXX() const { return STI.getFeatureBits() & Mips::FeatureFPXX; }

  bool useOddSPReg() const {
    return !(STI.getFeatureBits() & Mips::FeatureNoOddSPReg);
  }

  bool inMicroMipsMode() const {
    return STI.getFeatureBits() & Mips::FeatureMicroMips;
  }
  bool hasMips1() const { return STI.getFeatureBits() & Mips::FeatureMips1; }
  bool hasMips2() const { return STI.getFeatureBits() & Mips::FeatureMips2; }
  bool hasMips3() const { return STI.getFeatureBits() & Mips::FeatureMips3; }
  bool hasMips4() const { return STI.getFeatureBits() & Mips::FeatureMips4; }
  bool hasMips5() const { return STI.getFeatureBits() & Mips::FeatureMips5; }
  bool hasMips32() const {
    return (STI.getFeatureBits() & Mips::FeatureMips32);
  }
  bool hasMips64() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64);
  }
  bool hasMips32r2() const {
    return (STI.getFeatureBits() & Mips::FeatureMips32r2);
  }
  bool hasMips64r2() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64r2);
  }
  bool hasMips32r6() const {
    return (STI.getFeatureBits() & Mips::FeatureMips32r6);
  }
  bool hasMips64r6() const {
    return (STI.getFeatureBits() & Mips::FeatureMips64r6);
  }
  bool hasDSP() const { return (STI.getFeatureBits() & Mips::FeatureDSP); }
  bool hasDSPR2() const { return (STI.getFeatureBits() & Mips::FeatureDSPR2); }
  bool hasMSA() const { return (STI.getFeatureBits() & Mips::FeatureMSA); }

  bool inMips16Mode() const {
    return STI.getFeatureBits() & Mips::FeatureMips16;
  }
  // TODO: see how can we get this info.
  bool abiUsesSoftFloat() const { return false; }

  /// Warn if RegNo is the current assembler temporary.
  void WarnIfAssemblerTemporary(int RegNo, SMLoc Loc);
};
}

namespace {

/// MipsOperand - Instances of this class represent a parsed Mips machine
/// instruction.
class MipsOperand : public MCParsedAsmOperand {
public:
  /// Broad categories of register classes
  /// The exact class is finalized by the render method.
  enum RegKind {
    RegKind_GPR = 1,      /// GPR32 and GPR64 (depending on isGP64bit())
    RegKind_FGR = 2,      /// FGR32, FGR64, AFGR64 (depending on context and
                          /// isFP64bit())
    RegKind_FCC = 4,      /// FCC
    RegKind_MSA128 = 8,   /// MSA128[BHWD] (makes no difference which)
    RegKind_MSACtrl = 16, /// MSA control registers
    RegKind_COP2 = 32,    /// COP2
    RegKind_ACC = 64,     /// HI32DSP, LO32DSP, and ACC64DSP (depending on
                          /// context).
    RegKind_CCR = 128,    /// CCR
    RegKind_HWRegs = 256, /// HWRegs
    RegKind_COP3 = 512,   /// COP3

    /// Potentially any (e.g. $1)
    RegKind_Numeric = RegKind_GPR | RegKind_FGR | RegKind_FCC | RegKind_MSA128 |
                      RegKind_MSACtrl | RegKind_COP2 | RegKind_ACC |
                      RegKind_CCR | RegKind_HWRegs | RegKind_COP3
  };

private:
  enum KindTy {
    k_Immediate,     /// An immediate (possibly involving symbol references)
    k_Memory,        /// Base + Offset Memory Address
    k_PhysRegister,  /// A physical register from the Mips namespace
    k_RegisterIndex, /// A register index in one or more RegKind.
    k_Token          /// A simple token
  } Kind;

public:
  MipsOperand(KindTy K, MipsAsmParser &Parser)
      : MCParsedAsmOperand(), Kind(K), AsmParser(Parser) {}

private:
  /// For diagnostics, and checking the assembler temporary
  MipsAsmParser &AsmParser;

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct PhysRegOp {
    unsigned Num; /// Register Number
  };

  struct RegIdxOp {
    unsigned Index; /// Index into the register class
    RegKind Kind;   /// Bitfield of the kinds it could possibly be
    const MCRegisterInfo *RegInfo;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    MipsOperand *Base;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct PhysRegOp PhysReg;
    struct RegIdxOp RegIdx;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

  SMLoc StartLoc, EndLoc;

  /// Internal constructor for register kinds
  static std::unique_ptr<MipsOperand> CreateReg(unsigned Index, RegKind RegKind,
                                                const MCRegisterInfo *RegInfo,
                                                SMLoc S, SMLoc E,
                                                MipsAsmParser &Parser) {
    auto Op = make_unique<MipsOperand>(k_RegisterIndex, Parser);
    Op->RegIdx.Index = Index;
    Op->RegIdx.RegInfo = RegInfo;
    Op->RegIdx.Kind = RegKind;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

public:
  /// Coerce the register to GPR32 and return the real register for the current
  /// target.
  unsigned getGPR32Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_GPR) && "Invalid access!");
    AsmParser.WarnIfAssemblerTemporary(RegIdx.Index, StartLoc);
    unsigned ClassID = Mips::GPR32RegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to GPR64 and return the real register for the current
  /// target.
  unsigned getGPR64Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_GPR) && "Invalid access!");
    unsigned ClassID = Mips::GPR64RegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

private:
  /// Coerce the register to AFGR64 and return the real register for the current
  /// target.
  unsigned getAFGR64Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_FGR) && "Invalid access!");
    if (RegIdx.Index % 2 != 0)
      AsmParser.Warning(StartLoc, "Float register should be even.");
    return RegIdx.RegInfo->getRegClass(Mips::AFGR64RegClassID)
        .getRegister(RegIdx.Index / 2);
  }

  /// Coerce the register to FGR64 and return the real register for the current
  /// target.
  unsigned getFGR64Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_FGR) && "Invalid access!");
    return RegIdx.RegInfo->getRegClass(Mips::FGR64RegClassID)
        .getRegister(RegIdx.Index);
  }

  /// Coerce the register to FGR32 and return the real register for the current
  /// target.
  unsigned getFGR32Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_FGR) && "Invalid access!");
    return RegIdx.RegInfo->getRegClass(Mips::FGR32RegClassID)
        .getRegister(RegIdx.Index);
  }

  /// Coerce the register to FGRH32 and return the real register for the current
  /// target.
  unsigned getFGRH32Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_FGR) && "Invalid access!");
    return RegIdx.RegInfo->getRegClass(Mips::FGRH32RegClassID)
        .getRegister(RegIdx.Index);
  }

  /// Coerce the register to FCC and return the real register for the current
  /// target.
  unsigned getFCCReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_FCC) && "Invalid access!");
    return RegIdx.RegInfo->getRegClass(Mips::FCCRegClassID)
        .getRegister(RegIdx.Index);
  }

  /// Coerce the register to MSA128 and return the real register for the current
  /// target.
  unsigned getMSA128Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_MSA128) && "Invalid access!");
    // It doesn't matter which of the MSA128[BHWD] classes we use. They are all
    // identical
    unsigned ClassID = Mips::MSA128BRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to MSACtrl and return the real register for the
  /// current target.
  unsigned getMSACtrlReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_MSACtrl) && "Invalid access!");
    unsigned ClassID = Mips::MSACtrlRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to COP2 and return the real register for the
  /// current target.
  unsigned getCOP2Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_COP2) && "Invalid access!");
    unsigned ClassID = Mips::COP2RegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to COP3 and return the real register for the
  /// current target.
  unsigned getCOP3Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_COP3) && "Invalid access!");
    unsigned ClassID = Mips::COP3RegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to ACC64DSP and return the real register for the
  /// current target.
  unsigned getACC64DSPReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_ACC) && "Invalid access!");
    unsigned ClassID = Mips::ACC64DSPRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to HI32DSP and return the real register for the
  /// current target.
  unsigned getHI32DSPReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_ACC) && "Invalid access!");
    unsigned ClassID = Mips::HI32DSPRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to LO32DSP and return the real register for the
  /// current target.
  unsigned getLO32DSPReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_ACC) && "Invalid access!");
    unsigned ClassID = Mips::LO32DSPRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to CCR and return the real register for the
  /// current target.
  unsigned getCCRReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_CCR) && "Invalid access!");
    unsigned ClassID = Mips::CCRRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to HWRegs and return the real register for the
  /// current target.
  unsigned getHWRegsReg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_HWRegs) && "Invalid access!");
    unsigned ClassID = Mips::HWRegsRegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

public:
  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediate when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("Use a custom parser instead");
  }

  /// Render the operand to an MCInst as a GPR32
  /// Asserts if the wrong number of operands are requested, or the operand
  /// is not a k_RegisterIndex compatible with RegKind_GPR
  void addGPR32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getGPR32Reg()));
  }

  /// Render the operand to an MCInst as a GPR64
  /// Asserts if the wrong number of operands are requested, or the operand
  /// is not a k_RegisterIndex compatible with RegKind_GPR
  void addGPR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getGPR64Reg()));
  }

  void addAFGR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getAFGR64Reg()));
  }

  void addFGR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getFGR64Reg()));
  }

  void addFGR32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getFGR32Reg()));
    // FIXME: We ought to do this for -integrated-as without -via-file-asm too.
    if (!AsmParser.useOddSPReg() && RegIdx.Index & 1)
      AsmParser.Error(StartLoc, "-mno-odd-spreg prohibits the use of odd FPU "
                                "registers");
  }

  void addFGRH32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getFGRH32Reg()));
  }

  void addFCCAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getFCCReg()));
  }

  void addMSA128AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMSA128Reg()));
  }

  void addMSACtrlAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMSACtrlReg()));
  }

  void addCOP2AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getCOP2Reg()));
  }

  void addCOP3AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getCOP3Reg()));
  }

  void addACC64DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getACC64DSPReg()));
  }

  void addHI32DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getHI32DSPReg()));
  }

  void addLO32DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getLO32DSPReg()));
  }

  void addCCRAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getCCRReg()));
  }

  void addHWRegsAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getHWRegsReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()->getGPR32Reg()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  bool isReg() const override {
    // As a special case until we sort out the definition of div/divu, pretend
    // that $0/$zero are k_PhysRegister so that MCK_ZERO works correctly.
    if (isGPRAsmReg() && RegIdx.Index == 0)
      return true;

    return Kind == k_PhysRegister;
  }
  bool isRegIdx() const { return Kind == k_RegisterIndex; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isConstantImm() const {
    return isImm() && dyn_cast<MCConstantExpr>(getImm());
  }
  bool isToken() const override {
    // Note: It's not possible to pretend that other operand kinds are tokens.
    // The matcher emitter checks tokens first.
    return Kind == k_Token;
  }
  bool isMem() const override { return Kind == k_Memory; }
  bool isConstantMemOff() const {
    return isMem() && dyn_cast<MCConstantExpr>(getMemOff());
  }
  template <unsigned Bits> bool isMemWithSimmOffset() const {
    return isMem() && isConstantMemOff() && isInt<Bits>(getConstantMemOff());
  }
  bool isInvNum() const { return Kind == k_Immediate; }
  bool isLSAImm() const {
    if (!isConstantImm())
      return false;
    int64_t Val = getConstantImm();
    return 1 <= Val && Val <= 4;
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const override {
    // As a special case until we sort out the definition of div/divu, pretend
    // that $0/$zero are k_PhysRegister so that MCK_ZERO works correctly.
    if (Kind == k_RegisterIndex && RegIdx.Index == 0 &&
        RegIdx.Kind & RegKind_GPR)
      return getGPR32Reg(); // FIXME: GPR64 too

    assert(Kind == k_PhysRegister && "Invalid access!");
    return PhysReg.Num;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  int64_t getConstantImm() const {
    const MCExpr *Val = getImm();
    return static_cast<const MCConstantExpr *>(Val)->getValue();
  }

  MipsOperand *getMemBase() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Base;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Off;
  }

  int64_t getConstantMemOff() const {
    return static_cast<const MCConstantExpr *>(getMemOff())->getValue();
  }

  static std::unique_ptr<MipsOperand> CreateToken(StringRef Str, SMLoc S,
                                                  MipsAsmParser &Parser) {
    auto Op = make_unique<MipsOperand>(k_Token, Parser);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  /// Create a numeric register (e.g. $1). The exact register remains
  /// unresolved until an instruction successfully matches
  static std::unique_ptr<MipsOperand>
  CreateNumericReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
                   SMLoc E, MipsAsmParser &Parser) {
    DEBUG(dbgs() << "CreateNumericReg(" << Index << ", ...)\n");
    return CreateReg(Index, RegKind_Numeric, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely a GPR.
  /// This is typically only used for named registers such as $gp.
  static std::unique_ptr<MipsOperand>
  CreateGPRReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_GPR, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely a FGR.
  /// This is typically only used for named registers such as $f0.
  static std::unique_ptr<MipsOperand>
  CreateFGRReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_FGR, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an FCC.
  /// This is typically only used for named registers such as $fcc0.
  static std::unique_ptr<MipsOperand>
  CreateFCCReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_FCC, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an ACC.
  /// This is typically only used for named registers such as $ac0.
  static std::unique_ptr<MipsOperand>
  CreateACCReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_ACC, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an MSA128.
  /// This is typically only used for named registers such as $w0.
  static std::unique_ptr<MipsOperand>
  CreateMSA128Reg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
                  SMLoc E, MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_MSA128, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an MSACtrl.
  /// This is typically only used for named registers such as $msaaccess.
  static std::unique_ptr<MipsOperand>
  CreateMSACtrlReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
                   SMLoc E, MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_MSACtrl, RegInfo, S, E, Parser);
  }

  static std::unique_ptr<MipsOperand>
  CreateImm(const MCExpr *Val, SMLoc S, SMLoc E, MipsAsmParser &Parser) {
    auto Op = make_unique<MipsOperand>(k_Immediate, Parser);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<MipsOperand>
  CreateMem(std::unique_ptr<MipsOperand> Base, const MCExpr *Off, SMLoc S,
            SMLoc E, MipsAsmParser &Parser) {
    auto Op = make_unique<MipsOperand>(k_Memory, Parser);
    Op->Mem.Base = Base.release();
    Op->Mem.Off = Off;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  bool isGPRAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_GPR && RegIdx.Index <= 31;
  }
  bool isFGRAsmReg() const {
    // AFGR64 is $0-$15 but we handle this in getAFGR64()
    return isRegIdx() && RegIdx.Kind & RegKind_FGR && RegIdx.Index <= 31;
  }
  bool isHWRegsAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_HWRegs && RegIdx.Index <= 31;
  }
  bool isCCRAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_CCR && RegIdx.Index <= 31;
  }
  bool isFCCAsmReg() const {
    if (!(isRegIdx() && RegIdx.Kind & RegKind_FCC))
      return false;
    if (!AsmParser.hasEightFccRegisters())
      return RegIdx.Index == 0;
    return RegIdx.Index <= 7;
  }
  bool isACCAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_ACC && RegIdx.Index <= 3;
  }
  bool isCOP2AsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_COP2 && RegIdx.Index <= 31;
  }
  bool isCOP3AsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_COP3 && RegIdx.Index <= 31;
  }
  bool isMSA128AsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_MSA128 && RegIdx.Index <= 31;
  }
  bool isMSACtrlAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_MSACtrl && RegIdx.Index <= 7;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  virtual ~MipsOperand() {
    switch (Kind) {
    case k_Immediate:
      break;
    case k_Memory:
      delete Mem.Base;
      break;
    case k_PhysRegister:
    case k_RegisterIndex:
    case k_Token:
      break;
    }
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case k_Immediate:
      OS << "Imm<";
      Imm.Val->print(OS);
      OS << ">";
      break;
    case k_Memory:
      OS << "Mem<";
      Mem.Base->print(OS);
      OS << ", ";
      Mem.Off->print(OS);
      OS << ">";
      break;
    case k_PhysRegister:
      OS << "PhysReg<" << PhysReg.Num << ">";
      break;
    case k_RegisterIndex:
      OS << "RegIdx<" << RegIdx.Index << ":" << RegIdx.Kind << ">";
      break;
    case k_Token:
      OS << Tok.Data;
      break;
    }
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

  if (MCID.isBranch() || MCID.isCall()) {
    const unsigned Opcode = Inst.getOpcode();
    MCOperand Offset;

    switch (Opcode) {
    default:
      break;
    case Mips::BEQ:
    case Mips::BNE:
    case Mips::BEQ_MM:
    case Mips::BNE_MM:
      assert(MCID.getNumOperands() == 3 && "unexpected number of operands");
      Offset = Inst.getOperand(2);
      if (!Offset.isImm())
        break; // We'll deal with this situation later on when applying fixups.
      if (!isIntN(inMicroMipsMode() ? 17 : 18, Offset.getImm()))
        return Error(IDLoc, "branch target out of range");
      if (OffsetToAlignment(Offset.getImm(),
                            1LL << (inMicroMipsMode() ? 1 : 2)))
        return Error(IDLoc, "branch to misaligned address");
      break;
    case Mips::BGEZ:
    case Mips::BGTZ:
    case Mips::BLEZ:
    case Mips::BLTZ:
    case Mips::BGEZAL:
    case Mips::BLTZAL:
    case Mips::BC1F:
    case Mips::BC1T:
    case Mips::BGEZ_MM:
    case Mips::BGTZ_MM:
    case Mips::BLEZ_MM:
    case Mips::BLTZ_MM:
    case Mips::BGEZAL_MM:
    case Mips::BLTZAL_MM:
    case Mips::BC1F_MM:
    case Mips::BC1T_MM:
      assert(MCID.getNumOperands() == 2 && "unexpected number of operands");
      Offset = Inst.getOperand(1);
      if (!Offset.isImm())
        break; // We'll deal with this situation later on when applying fixups.
      if (!isIntN(inMicroMipsMode() ? 17 : 18, Offset.getImm()))
        return Error(IDLoc, "branch target out of range");
      if (OffsetToAlignment(Offset.getImm(),
                            1LL << (inMicroMipsMode() ? 1 : 2)))
        return Error(IDLoc, "branch to misaligned address");
      break;
    }
  }

  // SSNOP is deprecated on MIPS32r6/MIPS64r6
  // We still accept it but it is a normal nop.
  if (hasMips32r6() && Inst.getOpcode() == Mips::SSNOP) {
    std::string ISA = hasMips64r6() ? "MIPS64r6" : "MIPS32r6";
    Warning(IDLoc, "ssnop is deprecated for " + ISA + " and is equivalent to a "
                                                      "nop instruction");
  }

  if (MCID.hasDelaySlot() && AssemblerOptions.back()->isReorder()) {
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
    return expandInstruction(Inst, IDLoc, Instructions);
  else
    Instructions.push_back(Inst);

  return false;
}

bool MipsAsmParser::needsExpansion(MCInst &Inst) {

  switch (Inst.getOpcode()) {
  case Mips::LoadImm32Reg:
  case Mips::LoadAddr32Imm:
  case Mips::LoadAddr32Reg:
  case Mips::LoadImm64Reg:
    return true;
  default:
    return false;
  }
}

bool MipsAsmParser::expandInstruction(MCInst &Inst, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions) {
  switch (Inst.getOpcode()) {
  default:
    assert(0 && "unimplemented expansion");
    return true;
  case Mips::LoadImm32Reg:
    return expandLoadImm(Inst, IDLoc, Instructions);
  case Mips::LoadImm64Reg:
    if (!isGP64bit()) {
      Error(IDLoc, "instruction requires a CPU feature not currently enabled");
      return true;
    }
    return expandLoadImm(Inst, IDLoc, Instructions);
  case Mips::LoadAddr32Imm:
    return expandLoadAddressImm(Inst, IDLoc, Instructions);
  case Mips::LoadAddr32Reg:
    return expandLoadAddressReg(Inst, IDLoc, Instructions);
  }
}

namespace {
template <int Shift, bool PerformShift>
void createShiftOr(int64_t Value, unsigned RegNo, SMLoc IDLoc,
                   SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  if (PerformShift) {
    tmpInst.setOpcode(Mips::DSLL);
    tmpInst.addOperand(MCOperand::CreateReg(RegNo));
    tmpInst.addOperand(MCOperand::CreateReg(RegNo));
    tmpInst.addOperand(MCOperand::CreateImm(16));
    tmpInst.setLoc(IDLoc);
    Instructions.push_back(tmpInst);
    tmpInst.clear();
  }
  tmpInst.setOpcode(Mips::ORi);
  tmpInst.addOperand(MCOperand::CreateReg(RegNo));
  tmpInst.addOperand(MCOperand::CreateReg(RegNo));
  tmpInst.addOperand(
      MCOperand::CreateImm(((Value & (0xffffLL << Shift)) >> Shift)));
  tmpInst.setLoc(IDLoc);
  Instructions.push_back(tmpInst);
}
}

bool MipsAsmParser::expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");

  int64_t ImmValue = ImmOp.getImm();
  tmpInst.setLoc(IDLoc);
  // FIXME: gas has a special case for values that are 000...1111, which
  // becomes a li -1 and then a dsrl
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
  } else if ((ImmValue & 0xffffffff) == ImmValue) {
    // For any value of j that is representable as a 32-bit integer, create
    // a sequence of:
    // li d,j => lui d,hi16(j)
    //           ori d,d,lo16(j)
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::CreateImm((ImmValue & 0xffff0000) >> 16));
    Instructions.push_back(tmpInst);
    createShiftOr<0, false>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
  } else if ((ImmValue & (0xffffLL << 48)) == 0) {
    if (!isGP64bit()) {
      Error(IDLoc, "instruction requires a CPU feature not currently enabled");
      return true;
    }

    //            <-------  lo32 ------>
    // <-------  hi32 ------>
    // <- hi16 ->             <- lo16 ->
    //  _________________________________
    // |          |          |          |
    // | 16-bytes | 16-bytes | 16-bytes |
    // |__________|__________|__________|
    //
    // For any value of j that is representable as a 48-bit integer, create
    // a sequence of:
    // li d,j => lui d,hi16(j)
    //           ori d,d,hi16(lo32(j))
    //           dsll d,d,16
    //           ori d,d,lo16(lo32(j))
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(
        MCOperand::CreateImm((ImmValue & (0xffffLL << 32)) >> 32));
    Instructions.push_back(tmpInst);
    createShiftOr<16, false>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
    createShiftOr<0, true>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
  } else {
    if (!isGP64bit()) {
      Error(IDLoc, "instruction requires a CPU feature not currently enabled");
      return true;
    }

    // <-------  hi32 ------> <-------  lo32 ------>
    // <- hi16 ->                        <- lo16 ->
    //  ___________________________________________
    // |          |          |          |          |
    // | 16-bytes | 16-bytes | 16-bytes | 16-bytes |
    // |__________|__________|__________|__________|
    //
    // For any value of j that isn't representable as a 48-bit integer.
    // li d,j => lui d,hi16(j)
    //           ori d,d,lo16(hi32(j))
    //           dsll d,d,16
    //           ori d,d,hi16(lo32(j))
    //           dsll d,d,16
    //           ori d,d,lo16(lo32(j))
    tmpInst.setOpcode(Mips::LUi);
    tmpInst.addOperand(MCOperand::CreateReg(RegOp.getReg()));
    tmpInst.addOperand(
        MCOperand::CreateImm((ImmValue & (0xffffLL << 48)) >> 48));
    Instructions.push_back(tmpInst);
    createShiftOr<32, false>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
    createShiftOr<16, true>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
    createShiftOr<0, true>(ImmValue, RegOp.getReg(), IDLoc, Instructions);
  }
  return false;
}

bool
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
  return false;
}

bool
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
  return false;
}

void MipsAsmParser::expandMemInst(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions,
                                  bool isLoad, bool isImmOpnd) {
  const MCSymbolRefExpr *SR;
  MCInst TempInst;
  unsigned ImmOffset, HiOffset, LoOffset;
  const MCExpr *ExprOffset;
  unsigned TmpRegNum;
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
  // These are some of the types of expansions we perform here:
  // 1) lw $8, sym        => lui $8, %hi(sym)
  //                         lw $8, %lo(sym)($8)
  // 2) lw $8, offset($9) => lui $8, %hi(offset)
  //                         add $8, $8, $9
  //                         lw $8, %lo(offset)($9)
  // 3) lw $8, offset($8) => lui $at, %hi(offset)
  //                         add $at, $at, $8
  //                         lw $8, %lo(offset)($at)
  // 4) sw $8, sym        => lui $at, %hi(sym)
  //                         sw $8, %lo(sym)($at)
  // 5) sw $8, offset($8) => lui $at, %hi(offset)
  //                         add $at, $at, $8
  //                         sw $8, %lo(offset)($at)
  // 6) ldc1 $f0, sym     => lui $at, %hi(sym)
  //                         ldc1 $f0, %lo(sym)($at)
  //
  // For load instructions we can use the destination register as a temporary
  // if base and dst are different (examples 1 and 2) and if the base register
  // is general purpose otherwise we must use $at (example 6) and error if it's
  // not available. For stores we must use $at (examples 4 and 5) because we
  // must not clobber the source register setting up the offset.
  const MCInstrDesc &Desc = getInstDesc(Inst.getOpcode());
  int16_t RegClassOp0 = Desc.OpInfo[0].RegClass;
  unsigned RegClassIDOp0 =
      getContext().getRegisterInfo()->getRegClass(RegClassOp0).getID();
  bool IsGPR = (RegClassIDOp0 == Mips::GPR32RegClassID) ||
               (RegClassIDOp0 == Mips::GPR64RegClassID);
  if (isLoad && IsGPR && (BaseRegNum != RegOpNum))
    TmpRegNum = RegOpNum;
  else {
    int AT = getATReg(IDLoc);
    // At this point we need AT to perform the expansions and we exit if it is
    // not available.
    if (!AT)
      return;
    TmpRegNum = getReg(
        (isGP64bit()) ? Mips::GPR64RegClassID : Mips::GPR32RegClassID, AT);
  }

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
  // And finally, create original instruction with low part
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

unsigned MipsAsmParser::checkTargetMatchPredicate(MCInst &Inst) {
  // As described by the Mips32r2 spec, the registers Rd and Rs for
  // jalr.hb must be different.
  unsigned Opcode = Inst.getOpcode();

  if (Opcode == Mips::JALR_HB &&
      (Inst.getOperand(0).getReg() == Inst.getOperand(1).getReg()))
    return Match_RequiresDifferentSrcAndDst;

  return Match_Success;
}

bool MipsAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                            OperandVector &Operands,
                                            MCStreamer &Out,
                                            unsigned &ErrorInfo,
                                            bool MatchingInlineAsm) {

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
      Out.EmitInstruction(Instructions[i], STI);
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

      ErrorLoc = ((MipsOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  case Match_RequiresDifferentSrcAndDst:
    return Error(IDLoc, "source and destination must be different");
  }
  return true;
}

void MipsAsmParser::WarnIfAssemblerTemporary(int RegIndex, SMLoc Loc) {
  if ((RegIndex != 0) && 
      ((int)AssemblerOptions.back()->getATRegNum() == RegIndex)) {
    if (RegIndex == 1)
      Warning(Loc, "Used $at without \".set noat\"");
    else
      Warning(Loc, Twine("Used $") + Twine(RegIndex) + " with \".set at=$" +
                       Twine(RegIndex) + "\"");
  }
}

int MipsAsmParser::matchCPURegisterName(StringRef Name) {
  int CC;

  CC = StringSwitch<unsigned>(Name)
           .Case("zero", 0)
           .Case("at", 1)
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
           .Case("gp", 28)
           .Case("sp", 29)
           .Case("fp", 30)
           .Case("s8", 30)
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

  if (isABI_N32() || isABI_N64()) {
    // Although SGI documentation just cuts out t0-t3 for n32/n64,
    // GNU pushes the values of t0-t3 to override the o32/o64 values for t4-t7
    // We are supporting both cases, so for t0-t3 we'll just push them to t4-t7.
    if (8 <= CC && CC <= 11)
      CC += 4;

    if (CC == -1)
      CC = StringSwitch<unsigned>(Name)
               .Case("a4", 8)
               .Case("a5", 9)
               .Case("a6", 10)
               .Case("a7", 11)
               .Case("kt0", 26)
               .Case("kt1", 27)
               .Default(-1);
  }

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

bool MipsAssemblerOptions::setATReg(unsigned Reg) {
  if (Reg > 31)
    return false;

  ATReg = Reg;
  return true;
}

int MipsAsmParser::getATReg(SMLoc Loc) {
  int AT = AssemblerOptions.back()->getATRegNum();
  if (AT == 0)
    reportParseError(Loc,
                     "Pseudo instruction requires $at, which is not available");
  return AT;
}

unsigned MipsAsmParser::getReg(int RC, int RegNo) {
  return *(getContext().getRegisterInfo()->getRegClass(RC).begin() + RegNo);
}

unsigned MipsAsmParser::getGPR(int RegNo) {
  return getReg(isGP64bit() ? Mips::GPR64RegClassID : Mips::GPR32RegClassID,
                RegNo);
}

int MipsAsmParser::matchRegisterByNumber(unsigned RegNum, unsigned RegClass) {
  if (RegNum >
      getContext().getRegisterInfo()->getRegClass(RegClass).getNumRegs() - 1)
    return -1;

  return getReg(RegClass, RegNum);
}

bool MipsAsmParser::ParseOperand(OperandVector &Operands, StringRef Mnemonic) {
  DEBUG(dbgs() << "ParseOperand\n");

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

  DEBUG(dbgs() << ".. Generic Parser\n");

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return true;
  case AsmToken::Dollar: {
    // Parse the register.
    SMLoc S = Parser.getTok().getLoc();

    // Almost all registers have been parsed by custom parsers. There is only
    // one exception to this. $zero (and it's alias $0) will reach this point
    // for div, divu, and similar instructions because it is not an operand
    // to the instruction definition but an explicit register. Special case
    // this situation for now.
    if (ParseAnyRegister(Operands) != MatchOperand_NoMatch)
      return false;

    // Maybe it is a symbol reference.
    StringRef Identifier;
    if (Parser.parseIdentifier(Identifier))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    MCSymbol *Sym = getContext().GetOrCreateSymbol("$" + Identifier);
    // Otherwise create a symbol reference.
    const MCExpr *Res =
        MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());

    Operands.push_back(MipsOperand::CreateImm(Res, S, E, *this));
    return false;
  }
  // Else drop to expression parsing.
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::Tilde:
  case AsmToken::String: {
    DEBUG(dbgs() << ".. generic integer\n");
    OperandMatchResultTy ResTy = ParseImm(Operands);
    return ResTy != MatchOperand_Success;
  }
  case AsmToken::Percent: {
    // It is a symbol reference or constant expression.
    const MCExpr *IdVal;
    SMLoc S = Parser.getTok().getLoc(); // Start location of the operand.
    if (parseRelocOperand(IdVal))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

    Operands.push_back(MipsOperand::CreateImm(IdVal, S, E, *this));
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
    // It's a constant, evaluate reloc value.
    int16_t Val;
    switch (getVariantKind(RelocStr)) {
    case MCSymbolRefExpr::VK_Mips_ABS_LO:
      // Get the 1st 16-bits.
      Val = MCE->getValue() & 0xffff;
      break;
    case MCSymbolRefExpr::VK_Mips_ABS_HI:
      // Get the 2nd 16-bits. Also add 1 if bit 15 is 1, to compensate for low
      // 16 bits being negative.
      Val = ((MCE->getValue() + 0x8000) >> 16) & 0xffff;
      break;
    case MCSymbolRefExpr::VK_Mips_HIGHER:
      // Get the 3rd 16-bits.
      Val = ((MCE->getValue() + 0x80008000LL) >> 32) & 0xffff;
      break;
    case MCSymbolRefExpr::VK_Mips_HIGHEST:
      // Get the 4th 16-bits.
      Val = ((MCE->getValue() + 0x800080008000LL) >> 48) & 0xffff;
      break;
    default:
      report_fatal_error("Unsupported reloc value!");
    }
    return MCConstantExpr::Create(Val, getContext());
  }

  if (const MCSymbolRefExpr *MSRE = dyn_cast<MCSymbolRefExpr>(Expr)) {
    // It's a symbol, create a symbolic expression from the symbol.
    StringRef Symbol = MSRE->getSymbol().getName();
    MCSymbolRefExpr::VariantKind VK = getVariantKind(RelocStr);
    Res = MCSymbolRefExpr::Create(Symbol, VK, getContext());
    return Res;
  }

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    MCSymbolRefExpr::VariantKind VK = getVariantKind(RelocStr);

    // Try to create target expression.
    if (MipsMCExpr::isSupportedBinaryExpr(VK, BE))
      return MipsMCExpr::Create(VK, Expr, getContext());

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
  case MCExpr::Target:
    return true;
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
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> Operands;
  OperandMatchResultTy ResTy = ParseAnyRegister(Operands);
  if (ResTy == MatchOperand_Success) {
    assert(Operands.size() == 1);
    MipsOperand &Operand = static_cast<MipsOperand &>(*Operands.front());
    StartLoc = Operand.getStartLoc();
    EndLoc = Operand.getEndLoc();

    // AFAIK, we only support numeric registers and named GPR's in CFI
    // directives.
    // Don't worry about eating tokens before failing. Using an unrecognised
    // register is a parse error.
    if (Operand.isGPRAsmReg()) {
      // Resolve to GPR32 or GPR64 appropriately.
      RegNo = isGP64bit() ? Operand.getGPR64Reg() : Operand.getGPR32Reg();
    }

    return (RegNo == (unsigned)-1);
  }

  assert(Operands.size() == 0);
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

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseMemOperand(OperandVector &Operands) {
  DEBUG(dbgs() << "parseMemOperand\n");
  const MCExpr *IdVal = nullptr;
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
      MipsOperand &Mnemonic = static_cast<MipsOperand &>(*Operands[0]);
      if (Mnemonic.getToken() == "la") {
        SMLoc E =
            SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
        Operands.push_back(MipsOperand::CreateImm(IdVal, S, E, *this));
        return MatchOperand_Success;
      }
      if (Tok.is(AsmToken::EndOfStatement)) {
        SMLoc E =
            SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

        // Zero register assumed, add a memory operand with ZERO as its base.
        // "Base" will be managed by k_Memory.
        auto Base = MipsOperand::CreateGPRReg(0, getContext().getRegisterInfo(),
                                              S, E, *this);
        Operands.push_back(
            MipsOperand::CreateMem(std::move(Base), IdVal, S, E, *this));
        return MatchOperand_Success;
      }
      Error(Parser.getTok().getLoc(), "'(' expected");
      return MatchOperand_ParseFail;
    }

    Parser.Lex(); // Eat the '(' token.
  }

  Res = ParseAnyRegister(Operands);
  if (Res != MatchOperand_Success)
    return Res;

  if (Parser.getTok().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "')' expected");
    return MatchOperand_ParseFail;
  }

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  Parser.Lex(); // Eat the ')' token.

  if (!IdVal)
    IdVal = MCConstantExpr::Create(0, getContext());

  // Replace the register operand with the memory operand.
  std::unique_ptr<MipsOperand> op(
      static_cast<MipsOperand *>(Operands.back().release()));
  // Remove the register from the operands.
  // "op" will be managed by k_Memory.
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

  Operands.push_back(MipsOperand::CreateMem(std::move(op), IdVal, S, E, *this));
  return MatchOperand_Success;
}

bool MipsAsmParser::searchSymbolAlias(OperandVector &Operands) {

  MCSymbol *Sym = getContext().LookupSymbol(Parser.getTok().getIdentifier());
  if (Sym) {
    SMLoc S = Parser.getTok().getLoc();
    const MCExpr *Expr;
    if (Sym->isVariable())
      Expr = Sym->getVariableValue();
    else
      return false;
    if (Expr->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);
      const StringRef DefSymbol = Ref->getSymbol().getName();
      if (DefSymbol.startswith("$")) {
        OperandMatchResultTy ResTy =
            MatchAnyRegisterNameWithoutDollar(Operands, DefSymbol.substr(1), S);
        if (ResTy == MatchOperand_Success) {
          Parser.Lex();
          return true;
        } else if (ResTy == MatchOperand_ParseFail)
          llvm_unreachable("Should never ParseFail");
        return false;
      }
    } else if (Expr->getKind() == MCExpr::Constant) {
      Parser.Lex();
      const MCConstantExpr *Const = static_cast<const MCConstantExpr *>(Expr);
      Operands.push_back(
          MipsOperand::CreateImm(Const, S, Parser.getTok().getLoc(), *this));
      return true;
    }
  }
  return false;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::MatchAnyRegisterNameWithoutDollar(OperandVector &Operands,
                                                 StringRef Identifier,
                                                 SMLoc S) {
  int Index = matchCPURegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateGPRReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchFPURegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateFGRReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchFCCRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateFCCReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchACRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateACCReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchMSA128RegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateMSA128Reg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchMSA128CtrlRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::CreateMSACtrlReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  return MatchOperand_NoMatch;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::MatchAnyRegisterWithoutDollar(OperandVector &Operands, SMLoc S) {
  auto Token = Parser.getLexer().peekTok(false);

  if (Token.is(AsmToken::Identifier)) {
    DEBUG(dbgs() << ".. identifier\n");
    StringRef Identifier = Token.getIdentifier();
    OperandMatchResultTy ResTy =
        MatchAnyRegisterNameWithoutDollar(Operands, Identifier, S);
    return ResTy;
  } else if (Token.is(AsmToken::Integer)) {
    DEBUG(dbgs() << ".. integer\n");
    Operands.push_back(MipsOperand::CreateNumericReg(
        Token.getIntVal(), getContext().getRegisterInfo(), S, Token.getLoc(),
        *this));
    return MatchOperand_Success;
  }

  DEBUG(dbgs() << Parser.getTok().getKind() << "\n");

  return MatchOperand_NoMatch;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::ParseAnyRegister(OperandVector &Operands) {
  DEBUG(dbgs() << "ParseAnyRegister\n");

  auto Token = Parser.getTok();

  SMLoc S = Token.getLoc();

  if (Token.isNot(AsmToken::Dollar)) {
    DEBUG(dbgs() << ".. !$ -> try sym aliasing\n");
    if (Token.is(AsmToken::Identifier)) {
      if (searchSymbolAlias(Operands))
        return MatchOperand_Success;
    }
    DEBUG(dbgs() << ".. !symalias -> NoMatch\n");
    return MatchOperand_NoMatch;
  }
  DEBUG(dbgs() << ".. $\n");

  OperandMatchResultTy ResTy = MatchAnyRegisterWithoutDollar(Operands, S);
  if (ResTy == MatchOperand_Success) {
    Parser.Lex(); // $
    Parser.Lex(); // identifier
  }
  return ResTy;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::ParseImm(OperandVector &Operands) {
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::Tilde:
  case AsmToken::String:
    break;
  }

  const MCExpr *IdVal;
  SMLoc S = Parser.getTok().getLoc();
  if (getParser().parseExpression(IdVal))
    return MatchOperand_ParseFail;

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(MipsOperand::CreateImm(IdVal, S, E, *this));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::ParseJumpTarget(OperandVector &Operands) {
  DEBUG(dbgs() << "ParseJumpTarget\n");

  SMLoc S = getLexer().getLoc();

  // Integers and expressions are acceptable
  OperandMatchResultTy ResTy = ParseImm(Operands);
  if (ResTy != MatchOperand_NoMatch)
    return ResTy;

  // Registers are a valid target and have priority over symbols.
  ResTy = ParseAnyRegister(Operands);
  if (ResTy != MatchOperand_NoMatch)
    return ResTy;

  const MCExpr *Expr = nullptr;
  if (Parser.parseExpression(Expr)) {
    // We have no way of knowing if a symbol was consumed so we must ParseFail
    return MatchOperand_ParseFail;
  }
  Operands.push_back(
      MipsOperand::CreateImm(Expr, S, getLexer().getLoc(), *this));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseInvNum(OperandVector &Operands) {
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
      MCConstantExpr::Create(0 - Val, getContext()), S, E, *this));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::ParseLSAImm(OperandVector &Operands) {
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

  Operands.push_back(
      MipsOperand::CreateImm(Expr, S, Parser.getTok().getLoc(), *this));
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
          .Case("got_hi", MCSymbolRefExpr::VK_Mips_GOT_HI16)
          .Case("got_lo", MCSymbolRefExpr::VK_Mips_GOT_LO16)
          .Case("call_hi", MCSymbolRefExpr::VK_Mips_CALL_HI16)
          .Case("call_lo", MCSymbolRefExpr::VK_Mips_CALL_LO16)
          .Case("higher", MCSymbolRefExpr::VK_Mips_HIGHER)
          .Case("highest", MCSymbolRefExpr::VK_Mips_HIGHEST)
          .Case("pcrel_hi", MCSymbolRefExpr::VK_Mips_PCREL_HI16)
          .Case("pcrel_lo", MCSymbolRefExpr::VK_Mips_PCREL_LO16)
          .Default(MCSymbolRefExpr::VK_None);

  assert(VK != MCSymbolRefExpr::VK_None);

  return VK;
}

/// Sometimes (i.e. load/stores) the operand may be followed immediately by
/// either this.
/// ::= '(', register, ')'
/// handle it before we iterate so we don't get tripped up by the lack of
/// a comma.
bool MipsAsmParser::ParseParenSuffix(StringRef Name, OperandVector &Operands) {
  if (getLexer().is(AsmToken::LParen)) {
    Operands.push_back(
        MipsOperand::CreateToken("(", getLexer().getLoc(), *this));
    Parser.Lex();
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
    if (Parser.getTok().isNot(AsmToken::RParen)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token, expected ')'");
    }
    Operands.push_back(
        MipsOperand::CreateToken(")", getLexer().getLoc(), *this));
    Parser.Lex();
  }
  return false;
}

/// Sometimes (i.e. in MSA) the operand may be followed immediately by
/// either one of these.
/// ::= '[', register, ']'
/// ::= '[', integer, ']'
/// handle it before we iterate so we don't get tripped up by the lack of
/// a comma.
bool MipsAsmParser::ParseBracketSuffix(StringRef Name,
                                       OperandVector &Operands) {
  if (getLexer().is(AsmToken::LBrac)) {
    Operands.push_back(
        MipsOperand::CreateToken("[", getLexer().getLoc(), *this));
    Parser.Lex();
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
    if (Parser.getTok().isNot(AsmToken::RBrac)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token, expected ']'");
    }
    Operands.push_back(
        MipsOperand::CreateToken("]", getLexer().getLoc(), *this));
    Parser.Lex();
  }
  return false;
}

bool MipsAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                     SMLoc NameLoc, OperandVector &Operands) {
  DEBUG(dbgs() << "ParseInstruction\n");
  // We have reached first instruction, module directive after
  // this is forbidden.
  getTargetStreamer().setCanHaveModuleDir(false);
  // Check if we have valid mnemonic
  if (!mnemonicIsValid(Name, 0)) {
    Parser.eatToEndOfStatement();
    return Error(NameLoc, "Unknown instruction");
  }
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(MipsOperand::CreateToken(Name, NameLoc, *this));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
    if (getLexer().is(AsmToken::LBrac) && ParseBracketSuffix(Name, Operands))
      return true;
    // AFAIK, parenthesis suffixes are never on the first operand

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      // Parse and remember the operand.
      if (ParseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
      // Parse bracket and parenthesis suffixes before we iterate
      if (getLexer().is(AsmToken::LBrac)) {
        if (ParseBracketSuffix(Name, Operands))
          return true;
      } else if (getLexer().is(AsmToken::LParen) &&
                 ParseParenSuffix(Name, Operands))
        return true;
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

bool MipsAsmParser::reportParseError(Twine ErrorMsg) {
  SMLoc Loc = getLexer().getLoc();
  Parser.eatToEndOfStatement();
  return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::reportParseError(SMLoc Loc, Twine ErrorMsg) {
  return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::parseSetNoAtDirective() {
  // Line should look like: ".set noat".
  // set at reg to 0.
  AssemblerOptions.back()->setATReg(0);
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
    AssemblerOptions.back()->setATReg(1);
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

    if (AtRegNo < 0 || AtRegNo > 31) {
      reportParseError("unexpected token in statement");
      return false;
    }

    if (!AssemblerOptions.back()->setATReg(AtRegNo)) {
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
  AssemblerOptions.back()->setReorder();
  getTargetStreamer().emitDirectiveSetReorder();
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
  AssemblerOptions.back()->setNoReorder();
  getTargetStreamer().emitDirectiveSetNoReorder();
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
  AssemblerOptions.back()->setMacro();
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
  if (AssemblerOptions.back()->isReorder()) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  AssemblerOptions.back()->setNoMacro();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoMips16Directive() {
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  // For now do nothing.
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetFpDirective() {
  MipsABIFlagsSection::FpABIKind FpAbiVal;
  // Line can be: .set fp=32
  //              .set fp=xx
  //              .set fp=64
  Parser.Lex(); // Eat fp token
  AsmToken Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Equal)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Parser.Lex(); // Eat '=' token.
  Tok = Parser.getTok();

  if (!parseFpABIValue(FpAbiVal, ".set"))
    return false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  getTargetStreamer().emitDirectiveSetFp(FpAbiVal);
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetPopDirective() {
  SMLoc Loc = getLexer().getLoc();

  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  // Always keep an element on the options "stack" to prevent the user
  // from changing the initial options. This is how we remember them.
  if (AssemblerOptions.size() == 2)
    return reportParseError(Loc, ".set pop with no .set push");

  AssemblerOptions.pop_back();
  setAvailableFeatures(AssemblerOptions.back()->getFeatures());

  getTargetStreamer().emitDirectiveSetPop();
  return false;
}

bool MipsAsmParser::parseSetPushDirective() {
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  // Create a copy of the current assembler options environment and push it.
  MipsAssemblerOptions *TmpAOPtr = 
    new MipsAssemblerOptions(AssemblerOptions.back().get());
  AssemblerOptions.push_back(std::unique_ptr<MipsAssemblerOptions>(TmpAOPtr));

  getTargetStreamer().emitDirectiveSetPush();
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

  if (Parser.parseExpression(Value))
    return reportParseError("expected valid expression after comma");

  // Check if the Name already exists as a symbol.
  MCSymbol *Sym = getContext().LookupSymbol(Name);
  if (Sym)
    return reportParseError("symbol already defined");
  Sym = getContext().GetOrCreateSymbol(Name);
  Sym->setVariableValue(Value);

  return false;
}

bool MipsAsmParser::parseSetFeature(uint64_t Feature) {
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token in .set directive");

  switch (Feature) {
  default:
    llvm_unreachable("Unimplemented feature");
  case Mips::FeatureDSP:
    setFeatureBits(Mips::FeatureDSP, "dsp");
    getTargetStreamer().emitDirectiveSetDsp();
    break;
  case Mips::FeatureMicroMips:
    getTargetStreamer().emitDirectiveSetMicroMips();
    break;
  case Mips::FeatureMips16:
    getTargetStreamer().emitDirectiveSetMips16();
    break;
  case Mips::FeatureMips32r2:
    setFeatureBits(Mips::FeatureMips32r2, "mips32r2");
    getTargetStreamer().emitDirectiveSetMips32R2();
    break;
  case Mips::FeatureMips64:
    setFeatureBits(Mips::FeatureMips64, "mips64");
    getTargetStreamer().emitDirectiveSetMips64();
    break;
  case Mips::FeatureMips64r2:
    setFeatureBits(Mips::FeatureMips64r2, "mips64r2");
    getTargetStreamer().emitDirectiveSetMips64R2();
    break;
  }
  return false;
}

bool MipsAsmParser::eatComma(StringRef ErrorStr) {
  if (getLexer().isNot(AsmToken::Comma)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, ErrorStr);
  }

  Parser.Lex(); // Eat the comma.
  return true;
}

bool MipsAsmParser::parseDirectiveCPLoad(SMLoc Loc) {
  if (AssemblerOptions.back()->isReorder())
    Warning(Loc, ".cpload in reorder section");

  // FIXME: Warn if cpload is used in Mips16 mode.

  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> Reg;
  OperandMatchResultTy ResTy = ParseAnyRegister(Reg);
  if (ResTy == MatchOperand_NoMatch || ResTy == MatchOperand_ParseFail) {
    reportParseError("expected register containing function address");
    return false;
  }

  MipsOperand &RegOpnd = static_cast<MipsOperand &>(*Reg[0]);
  if (!RegOpnd.isGPRAsmReg()) {
    reportParseError(RegOpnd.getStartLoc(), "invalid register");
    return false;
  }

  getTargetStreamer().emitDirectiveCpload(RegOpnd.getGPR32Reg());
  return false;
}

bool MipsAsmParser::parseDirectiveCPSetup() {
  unsigned FuncReg;
  unsigned Save;
  bool SaveIsReg = true;

  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> TmpReg;
  OperandMatchResultTy ResTy = ParseAnyRegister(TmpReg);
  if (ResTy == MatchOperand_NoMatch) {
    reportParseError("expected register containing function address");
    Parser.eatToEndOfStatement();
    return false;
  }

  MipsOperand &FuncRegOpnd = static_cast<MipsOperand &>(*TmpReg[0]);
  if (!FuncRegOpnd.isGPRAsmReg()) {
    reportParseError(FuncRegOpnd.getStartLoc(), "invalid register");
    Parser.eatToEndOfStatement();
    return false;
  }

  FuncReg = FuncRegOpnd.getGPR32Reg();
  TmpReg.clear();

  if (!eatComma("expected comma parsing directive"))
    return true;

  ResTy = ParseAnyRegister(TmpReg);
  if (ResTy == MatchOperand_NoMatch) {
    const AsmToken &Tok = Parser.getTok();
    if (Tok.is(AsmToken::Integer)) {
      Save = Tok.getIntVal();
      SaveIsReg = false;
      Parser.Lex();
    } else {
      reportParseError("expected save register or stack offset");
      Parser.eatToEndOfStatement();
      return false;
    }
  } else {
    MipsOperand &SaveOpnd = static_cast<MipsOperand &>(*TmpReg[0]);
    if (!SaveOpnd.isGPRAsmReg()) {
      reportParseError(SaveOpnd.getStartLoc(), "invalid register");
      Parser.eatToEndOfStatement();
      return false;
    }
    Save = SaveOpnd.getGPR32Reg();
  }

  if (!eatComma("expected comma parsing directive"))
    return true;

  StringRef Name;
  if (Parser.parseIdentifier(Name))
    reportParseError("expected identifier");
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  getTargetStreamer().emitDirectiveCpsetup(FuncReg, Save, *Sym, SaveIsReg);
  return false;
}

bool MipsAsmParser::parseDirectiveNaN() {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    const AsmToken &Tok = Parser.getTok();

    if (Tok.getString() == "2008") {
      Parser.Lex();
      getTargetStreamer().emitDirectiveNaN2008();
      return false;
    } else if (Tok.getString() == "legacy") {
      Parser.Lex();
      getTargetStreamer().emitDirectiveNaNLegacy();
      return false;
    }
  }
  // If we don't recognize the option passed to the .nan
  // directive (e.g. no option or unknown option), emit an error.
  reportParseError("invalid option in .nan directive");
  return false;
}

bool MipsAsmParser::parseDirectiveSet() {

  // Get the next token.
  const AsmToken &Tok = Parser.getTok();

  if (Tok.getString() == "noat") {
    return parseSetNoAtDirective();
  } else if (Tok.getString() == "at") {
    return parseSetAtDirective();
  } else if (Tok.getString() == "fp") {
    return parseSetFpDirective();
  } else if (Tok.getString() == "pop") {
    return parseSetPopDirective();
  } else if (Tok.getString() == "push") {
    return parseSetPushDirective();
  } else if (Tok.getString() == "reorder") {
    return parseSetReorderDirective();
  } else if (Tok.getString() == "noreorder") {
    return parseSetNoReorderDirective();
  } else if (Tok.getString() == "macro") {
    return parseSetMacroDirective();
  } else if (Tok.getString() == "nomacro") {
    return parseSetNoMacroDirective();
  } else if (Tok.getString() == "mips16") {
    return parseSetFeature(Mips::FeatureMips16);
  } else if (Tok.getString() == "nomips16") {
    return parseSetNoMips16Directive();
  } else if (Tok.getString() == "nomicromips") {
    getTargetStreamer().emitDirectiveSetNoMicroMips();
    Parser.eatToEndOfStatement();
    return false;
  } else if (Tok.getString() == "micromips") {
    return parseSetFeature(Mips::FeatureMicroMips);
  } else if (Tok.getString() == "mips32r2") {
    return parseSetFeature(Mips::FeatureMips32r2);
  } else if (Tok.getString() == "mips64") {
    return parseSetFeature(Mips::FeatureMips64);
  } else if (Tok.getString() == "mips64r2") {
    return parseSetFeature(Mips::FeatureMips64r2);
  } else if (Tok.getString() == "dsp") {
    return parseSetFeature(Mips::FeatureDSP);
  } else {
    // It is just an identifier, look for an assignment.
    parseSetAssignment();
    return false;
  }

  return true;
}

/// parseDataDirective
///  ::= .word [ expression (, expression)* ]
bool MipsAsmParser::parseDataDirective(unsigned Size, SMLoc L) {
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

/// parseDirectiveGpDWord
///  ::= .gpdword local_sym
bool MipsAsmParser::parseDirectiveGpDWord() {
  const MCExpr *Value;
  // EmitGPRel64Value requires an expression, so we are using base class
  // method to evaluate the expression.
  if (getParser().parseExpression(Value))
    return true;
  getParser().getStreamer().EmitGPRel64Value(Value);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(), "unexpected token in directive");
  Parser.Lex(); // Eat EndOfStatement token.
  return false;
}

bool MipsAsmParser::parseDirectiveOption() {
  // Get the option token.
  AsmToken Tok = Parser.getTok();
  // At the moment only identifiers are supported.
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(), "unexpected token in .option directive");
    Parser.eatToEndOfStatement();
    return false;
  }

  StringRef Option = Tok.getIdentifier();

  if (Option == "pic0") {
    getTargetStreamer().emitDirectiveOptionPic0();
    Parser.Lex();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(),
            "unexpected token in .option pic0 directive");
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  if (Option == "pic2") {
    getTargetStreamer().emitDirectiveOptionPic2();
    Parser.Lex();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(),
            "unexpected token in .option pic2 directive");
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  // Unknown option.
  Warning(Parser.getTok().getLoc(), "unknown option in .option directive");
  Parser.eatToEndOfStatement();
  return false;
}

/// parseDirectiveModule
///  ::= .module oddspreg
///  ::= .module nooddspreg
///  ::= .module fp=value
bool MipsAsmParser::parseDirectiveModule() {
  MCAsmLexer &Lexer = getLexer();
  SMLoc L = Lexer.getLoc();

  if (!getTargetStreamer().getCanHaveModuleDir()) {
    // TODO : get a better message.
    reportParseError(".module directive must appear before any code");
    return false;
  }

  if (Lexer.is(AsmToken::Identifier)) {
    StringRef Option = Parser.getTok().getString();
    Parser.Lex();

    if (Option == "oddspreg") {
      getTargetStreamer().emitDirectiveModuleOddSPReg(true, isABI_O32());
      clearFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");

      if (getLexer().isNot(AsmToken::EndOfStatement)) {
        reportParseError("Expected end of statement");
        return false;
      }

      return false;
    } else if (Option == "nooddspreg") {
      if (!isABI_O32()) {
        Error(L, "'.module nooddspreg' requires the O32 ABI");
        return false;
      }

      getTargetStreamer().emitDirectiveModuleOddSPReg(false, isABI_O32());
      setFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");

      if (getLexer().isNot(AsmToken::EndOfStatement)) {
        reportParseError("Expected end of statement");
        return false;
      }

      return false;
    } else if (Option == "fp") {
      return parseDirectiveModuleFP();
    }

    return Error(L, "'" + Twine(Option) + "' is not a valid .module option.");
  }

  return false;
}

/// parseDirectiveModuleFP
///  ::= =32
///  ::= =xx
///  ::= =64
bool MipsAsmParser::parseDirectiveModuleFP() {
  MCAsmLexer &Lexer = getLexer();

  if (Lexer.isNot(AsmToken::Equal)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Parser.Lex(); // Eat '=' token.

  MipsABIFlagsSection::FpABIKind FpABI;
  if (!parseFpABIValue(FpABI, ".module"))
    return false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }

  // Emit appropriate flags.
  getTargetStreamer().emitDirectiveModuleFP(FpABI, isABI_O32());
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseFpABIValue(MipsABIFlagsSection::FpABIKind &FpABI,
                                    StringRef Directive) {
  MCAsmLexer &Lexer = getLexer();

  if (Lexer.is(AsmToken::Identifier)) {
    StringRef Value = Parser.getTok().getString();
    Parser.Lex();

    if (Value != "xx") {
      reportParseError("unsupported value, expected 'xx', '32' or '64'");
      return false;
    }

    if (!isABI_O32()) {
      reportParseError("'" + Directive + " fp=xx' requires the O32 ABI");
      return false;
    }

    FpABI = MipsABIFlagsSection::FpABIKind::XX;
    return true;
  }

  if (Lexer.is(AsmToken::Integer)) {
    unsigned Value = Parser.getTok().getIntVal();
    Parser.Lex();

    if (Value != 32 && Value != 64) {
      reportParseError("unsupported value, expected 'xx', '32' or '64'");
      return false;
    }

    if (Value == 32) {
      if (!isABI_O32()) {
        reportParseError("'" + Directive + " fp=32' requires the O32 ABI");
        return false;
      }

      FpABI = MipsABIFlagsSection::FpABIKind::S32;
    } else
      FpABI = MipsABIFlagsSection::FpABIKind::S64;

    return true;
  }

  return false;
}

bool MipsAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".cpload")
    return parseDirectiveCPLoad(DirectiveID.getLoc());
  if (IDVal == ".dword") {
    parseDataDirective(8, DirectiveID.getLoc());
    return false;
  }

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

  if (IDVal == ".nan")
    return parseDirectiveNaN();

  if (IDVal == ".gpword") {
    parseDirectiveGpWord();
    return false;
  }

  if (IDVal == ".gpdword") {
    parseDirectiveGpDWord();
    return false;
  }

  if (IDVal == ".word") {
    parseDataDirective(4, DirectiveID.getLoc());
    return false;
  }

  if (IDVal == ".option")
    return parseDirectiveOption();

  if (IDVal == ".abicalls") {
    getTargetStreamer().emitDirectiveAbiCalls();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(), "unexpected token in directive");
      // Clear line
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  if (IDVal == ".cpsetup")
    return parseDirectiveCPSetup();

  if (IDVal == ".module")
    return parseDirectiveModule();

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
