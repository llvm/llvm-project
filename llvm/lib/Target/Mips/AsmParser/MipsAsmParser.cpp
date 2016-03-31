//===-- MipsAsmParser.cpp - Parse Mips assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MipsABIInfo.h"
#include "MCTargetDesc/MipsMCExpr.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "MipsRegisterInfo.h"
#include "MipsTargetObjectFile.h"
#include "MipsTargetStreamer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "mips-asm-parser"

namespace llvm {
class MCInstrInfo;
}

namespace {
class MipsAssemblerOptions {
public:
  MipsAssemblerOptions(const FeatureBitset &Features_) :
    ATReg(1), Reorder(true), Macro(true), Features(Features_) {}

  MipsAssemblerOptions(const MipsAssemblerOptions *Opts) {
    ATReg = Opts->getATRegIndex();
    Reorder = Opts->isReorder();
    Macro = Opts->isMacro();
    Features = Opts->getFeatures();
  }

  unsigned getATRegIndex() const { return ATReg; }
  bool setATRegIndex(unsigned Reg) {
    if (Reg > 31)
      return false;

    ATReg = Reg;
    return true;
  }

  bool isReorder() const { return Reorder; }
  void setReorder() { Reorder = true; }
  void setNoReorder() { Reorder = false; }

  bool isMacro() const { return Macro; }
  void setMacro() { Macro = true; }
  void setNoMacro() { Macro = false; }

  const FeatureBitset &getFeatures() const { return Features; }
  void setFeatures(const FeatureBitset &Features_) { Features = Features_; }

  // Set of features that are either architecture features or referenced
  // by them (e.g.: FeatureNaN2008 implied by FeatureMips32r6).
  // The full table can be found in MipsGenSubtargetInfo.inc (MipsFeatureKV[]).
  // The reason we need this mask is explained in the selectArch function.
  // FIXME: Ideally we would like TableGen to generate this information.
  static const FeatureBitset AllArchRelatedMask;

private:
  unsigned ATReg;
  bool Reorder;
  bool Macro;
  FeatureBitset Features;
};
}

const FeatureBitset MipsAssemblerOptions::AllArchRelatedMask = {
    Mips::FeatureMips1, Mips::FeatureMips2, Mips::FeatureMips3,
    Mips::FeatureMips3_32, Mips::FeatureMips3_32r2, Mips::FeatureMips4,
    Mips::FeatureMips4_32, Mips::FeatureMips4_32r2, Mips::FeatureMips5,
    Mips::FeatureMips5_32r2, Mips::FeatureMips32, Mips::FeatureMips32r2,
    Mips::FeatureMips32r3, Mips::FeatureMips32r5, Mips::FeatureMips32r6,
    Mips::FeatureMips64, Mips::FeatureMips64r2, Mips::FeatureMips64r3,
    Mips::FeatureMips64r5, Mips::FeatureMips64r6, Mips::FeatureCnMips,
    Mips::FeatureFP64Bit, Mips::FeatureGP64Bit, Mips::FeatureNaN2008
};

namespace {
class MipsAsmParser : public MCTargetAsmParser {
  MipsTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<MipsTargetStreamer &>(TS);
  }

  MipsABIInfo ABI;
  SmallVector<std::unique_ptr<MipsAssemblerOptions>, 2> AssemblerOptions;
  MCSymbol *CurrentFn; // Pointer to the function being parsed. It may be a
                       // nullptr, which indicates that no function is currently
                       // selected. This usually happens after an '.end func'
                       // directive.
  bool IsLittleEndian;
  bool IsPicEnabled;
  bool IsCpRestoreSet;
  int CpRestoreOffset;
  unsigned CpSaveLocation;
  /// If true, then CpSaveLocation is a register, otherwise it's an offset.
  bool     CpSaveLocationIsRegister;

  // Print a warning along with its fix-it message at the given range.
  void printWarningWithFixIt(const Twine &Msg, const Twine &FixMsg,
                             SMRange Range, bool ShowColors = true);

#define GET_ASSEMBLER_HEADER
#include "MipsGenAsmMatcher.inc"

  unsigned checkTargetMatchPredicate(MCInst &Inst) override;

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  /// Parse a register as used in CFI directives
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  bool parseParenSuffix(StringRef Name, OperandVector &Operands);

  bool parseBracketSuffix(StringRef Name, OperandVector &Operands);

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  OperandMatchResultTy parseMemOperand(OperandVector &Operands);
  OperandMatchResultTy
  matchAnyRegisterNameWithoutDollar(OperandVector &Operands,
                                    StringRef Identifier, SMLoc S);
  OperandMatchResultTy matchAnyRegisterWithoutDollar(OperandVector &Operands,
                                                     SMLoc S);
  OperandMatchResultTy parseAnyRegister(OperandVector &Operands);
  OperandMatchResultTy parseImm(OperandVector &Operands);
  OperandMatchResultTy parseJumpTarget(OperandVector &Operands);
  OperandMatchResultTy parseInvNum(OperandVector &Operands);
  OperandMatchResultTy parseLSAImm(OperandVector &Operands);
  OperandMatchResultTy parseRegisterPair(OperandVector &Operands);
  OperandMatchResultTy parseMovePRegPair(OperandVector &Operands);
  OperandMatchResultTy parseRegisterList(OperandVector &Operands);

  bool searchSymbolAlias(OperandVector &Operands);

  bool parseOperand(OperandVector &, StringRef Mnemonic);

  enum MacroExpanderResultTy {
    MER_NotAMacro,
    MER_Success,
    MER_Fail,
  };

  // Expands assembly pseudo instructions.
  MacroExpanderResultTy
  tryExpandInstruction(MCInst &Inst, SMLoc IDLoc,
                       SmallVectorImpl<MCInst> &Instructions);

  bool expandJalWithRegs(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions);

  bool loadImmediate(int64_t ImmValue, unsigned DstReg, unsigned SrcReg,
                     bool Is32BitImm, bool IsAddress, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);

  bool loadAndAddSymbolAddress(const MCExpr *SymExpr, unsigned DstReg,
                               unsigned SrcReg, bool Is32BitSym, SMLoc IDLoc,
                               SmallVectorImpl<MCInst> &Instructions);

  bool expandLoadImm(MCInst &Inst, bool Is32BitImm, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);

  bool expandLoadAddress(unsigned DstReg, unsigned BaseReg,
                         const MCOperand &Offset, bool Is32BitAddress,
                         SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions);

  bool expandUncondBranchMMPseudo(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions);

  void expandMemInst(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions, bool isLoad,
                     bool isImmOpnd);

  bool expandLoadStoreMultiple(MCInst &Inst, SMLoc IDLoc,
                               SmallVectorImpl<MCInst> &Instructions);

  bool expandAliasImmediate(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  bool expandBranchImm(MCInst &Inst, SMLoc IDLoc,
                       SmallVectorImpl<MCInst> &Instructions);

  bool expandCondBranches(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  bool expandDiv(MCInst &Inst, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions, const bool IsMips64,
                 const bool Signed);

  bool expandTrunc(MCInst &Inst, bool IsDouble, bool Is64FPU, SMLoc IDLoc,
                   SmallVectorImpl<MCInst> &Instructions);

  bool expandUlh(MCInst &Inst, bool Signed, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions);

  bool expandUlw(MCInst &Inst, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions);

  bool expandRotation(MCInst &Inst, SMLoc IDLoc,
                      SmallVectorImpl<MCInst> &Instructions);
  bool expandRotationImm(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions);
  bool expandDRotation(MCInst &Inst, SMLoc IDLoc,
                       SmallVectorImpl<MCInst> &Instructions);
  bool expandDRotationImm(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  bool expandAbs(MCInst &Inst, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions);

  void createNop(bool hasShortDelaySlot, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions);

  void createAddu(unsigned DstReg, unsigned SrcReg, unsigned TrgReg,
                  bool Is64Bit, SmallVectorImpl<MCInst> &Instructions);

  void createCpRestoreMemOp(bool IsLoad, int StackOffset, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions);

  bool reportParseError(Twine ErrorMsg);
  bool reportParseError(SMLoc Loc, Twine ErrorMsg);

  bool parseMemOffset(const MCExpr *&Res, bool isParenExpr);
  bool parseRelocOperand(const MCExpr *&Res);

  const MCExpr *evaluateRelocExpr(const MCExpr *Expr, StringRef RelocStr);

  bool isEvaluated(const MCExpr *Expr);
  bool parseSetMips0Directive();
  bool parseSetArchDirective();
  bool parseSetFeature(uint64_t Feature);
  bool isPicAndNotNxxAbi(); // Used by .cpload, .cprestore, and .cpsetup.
  bool parseDirectiveCpLoad(SMLoc Loc);
  bool parseDirectiveCpRestore(SMLoc Loc);
  bool parseDirectiveCPSetup();
  bool parseDirectiveCPReturn();
  bool parseDirectiveNaN();
  bool parseDirectiveSet();
  bool parseDirectiveOption();
  bool parseInsnDirective();
  bool parseSSectionDirective(StringRef Section, unsigned Type);

  bool parseSetAtDirective();
  bool parseSetNoAtDirective();
  bool parseSetMacroDirective();
  bool parseSetNoMacroDirective();
  bool parseSetMsaDirective();
  bool parseSetNoMsaDirective();
  bool parseSetNoDspDirective();
  bool parseSetReorderDirective();
  bool parseSetNoReorderDirective();
  bool parseSetMips16Directive();
  bool parseSetNoMips16Directive();
  bool parseSetFpDirective();
  bool parseSetOddSPRegDirective();
  bool parseSetNoOddSPRegDirective();
  bool parseSetPopDirective();
  bool parseSetPushDirective();
  bool parseSetSoftFloatDirective();
  bool parseSetHardFloatDirective();

  bool parseSetAssignment();

  bool parseDataDirective(unsigned Size, SMLoc L);
  bool parseDirectiveGpWord();
  bool parseDirectiveGpDWord();
  bool parseDirectiveModule();
  bool parseDirectiveModuleFP();
  bool parseFpABIValue(MipsABIFlagsSection::FpABIKind &FpABI,
                       StringRef Directive);

  bool parseInternalDirectiveReallowModule();

  MCSymbolRefExpr::VariantKind getVariantKind(StringRef Symbol);

  bool eatComma(StringRef ErrorStr);

  int matchCPURegisterName(StringRef Symbol);

  int matchHWRegsRegisterName(StringRef Symbol);

  int matchRegisterByNumber(unsigned RegNum, unsigned RegClass);

  int matchFPURegisterName(StringRef Name);

  int matchFCCRegisterName(StringRef Name);

  int matchACRegisterName(StringRef Name);

  int matchMSA128RegisterName(StringRef Name);

  int matchMSA128CtrlRegisterName(StringRef Name);

  unsigned getReg(int RC, int RegNo);

  unsigned getGPR(int RegNo);

  /// Returns the internal register number for the current AT. Also checks if
  /// the current AT is unavailable (set to $0) and gives an error if it is.
  /// This should be used in pseudo-instruction expansions which need AT.
  unsigned getATReg(SMLoc Loc);

  bool processInstruction(MCInst &Inst, SMLoc IDLoc,
                          SmallVectorImpl<MCInst> &Instructions);

  // Helper function that checks if the value of a vector index is within the
  // boundaries of accepted values for each RegisterKind
  // Example: INSERT.B $w0[n], $1 => 16 > n >= 0
  bool validateMSAIndex(int Val, int RegKind);

  // Selects a new architecture by updating the FeatureBits with the necessary
  // info including implied dependencies.
  // Internally, it clears all the feature bits related to *any* architecture
  // and selects the new one using the ToggleFeature functionality of the
  // MCSubtargetInfo object that handles implied dependencies. The reason we
  // clear all the arch related bits manually is because ToggleFeature only
  // clears the features that imply the feature being cleared and not the
  // features implied by the feature being cleared. This is easier to see
  // with an example:
  //  --------------------------------------------------
  // | Feature         | Implies                        |
  // | -------------------------------------------------|
  // | FeatureMips1    | None                           |
  // | FeatureMips2    | FeatureMips1                   |
  // | FeatureMips3    | FeatureMips2 | FeatureMipsGP64 |
  // | FeatureMips4    | FeatureMips3                   |
  // | ...             |                                |
  //  --------------------------------------------------
  //
  // Setting Mips3 is equivalent to set: (FeatureMips3 | FeatureMips2 |
  // FeatureMipsGP64 | FeatureMips1)
  // Clearing Mips3 is equivalent to clear (FeatureMips3 | FeatureMips4).
  void selectArch(StringRef ArchFeature) {
    MCSubtargetInfo &STI = copySTI();
    FeatureBitset FeatureBits = STI.getFeatureBits();
    FeatureBits &= ~MipsAssemblerOptions::AllArchRelatedMask;
    STI.setFeatureBits(FeatureBits);
    setAvailableFeatures(
        ComputeAvailableFeatures(STI.ToggleFeature(ArchFeature)));
    AssemblerOptions.back()->setFeatures(STI.getFeatureBits());
  }

  void setFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (!(getSTI().getFeatureBits()[Feature])) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
      AssemblerOptions.back()->setFeatures(STI.getFeatureBits());
    }
  }

  void clearFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (getSTI().getFeatureBits()[Feature]) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
      AssemblerOptions.back()->setFeatures(STI.getFeatureBits());
    }
  }

  void setModuleFeatureBits(uint64_t Feature, StringRef FeatureString) {
    setFeatureBits(Feature, FeatureString);
    AssemblerOptions.front()->setFeatures(getSTI().getFeatureBits());
  }

  void clearModuleFeatureBits(uint64_t Feature, StringRef FeatureString) {
    clearFeatureBits(Feature, FeatureString);
    AssemblerOptions.front()->setFeatures(getSTI().getFeatureBits());
  }

public:
  enum MipsMatchResultTy {
    Match_RequiresDifferentSrcAndDst = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "MipsGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  MipsAsmParser(const MCSubtargetInfo &sti, MCAsmParser &parser,
                const MCInstrInfo &MII, const MCTargetOptions &Options)
    : MCTargetAsmParser(Options, sti),
        ABI(MipsABIInfo::computeTargetABI(Triple(sti.getTargetTriple()),
                                          sti.getCPU(), Options)) {
    MCAsmParserExtension::Initialize(parser);

    parser.addAliasForDirective(".asciiz", ".asciz");

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));

    // Remember the initial assembler options. The user can not modify these.
    AssemblerOptions.push_back(
        llvm::make_unique<MipsAssemblerOptions>(getSTI().getFeatureBits()));

    // Create an assembler options environment for the user to modify.
    AssemblerOptions.push_back(
        llvm::make_unique<MipsAssemblerOptions>(getSTI().getFeatureBits()));

    getTargetStreamer().updateABIInfo(*this);

    if (!isABI_O32() && !useOddSPReg() != 0)
      report_fatal_error("-mno-odd-spreg requires the O32 ABI");

    CurrentFn = nullptr;

    IsPicEnabled =
        (getContext().getObjectFileInfo()->getRelocM() == Reloc::PIC_);

    IsCpRestoreSet = false;
    CpRestoreOffset = -1;

    Triple TheTriple(sti.getTargetTriple());
    if ((TheTriple.getArch() == Triple::mips) ||
        (TheTriple.getArch() == Triple::mips64))
      IsLittleEndian = false;
    else
      IsLittleEndian = true;
  }

  /// True if all of $fcc0 - $fcc7 exist for the current ISA.
  bool hasEightFccRegisters() const { return hasMips4() || hasMips32(); }

  bool isGP64bit() const {
    return getSTI().getFeatureBits()[Mips::FeatureGP64Bit];
  }
  bool isFP64bit() const {
    return getSTI().getFeatureBits()[Mips::FeatureFP64Bit];
  }
  const MipsABIInfo &getABI() const { return ABI; }
  bool isABI_N32() const { return ABI.IsN32(); }
  bool isABI_N64() const { return ABI.IsN64(); }
  bool isABI_O32() const { return ABI.IsO32(); }
  bool isABI_FPXX() const {
    return getSTI().getFeatureBits()[Mips::FeatureFPXX];
  }

  bool useOddSPReg() const {
    return !(getSTI().getFeatureBits()[Mips::FeatureNoOddSPReg]);
  }

  bool inMicroMipsMode() const {
    return getSTI().getFeatureBits()[Mips::FeatureMicroMips];
  }
  bool hasMips1() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips1];
  }
  bool hasMips2() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips2];
  }
  bool hasMips3() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips3];
  }
  bool hasMips4() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips4];
  }
  bool hasMips5() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips5];
  }
  bool hasMips32() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips32];
  }
  bool hasMips64() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips64];
  }
  bool hasMips32r2() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips32r2];
  }
  bool hasMips64r2() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips64r2];
  }
  bool hasMips32r3() const {
    return (getSTI().getFeatureBits()[Mips::FeatureMips32r3]);
  }
  bool hasMips64r3() const {
    return (getSTI().getFeatureBits()[Mips::FeatureMips64r3]);
  }
  bool hasMips32r5() const {
    return (getSTI().getFeatureBits()[Mips::FeatureMips32r5]);
  }
  bool hasMips64r5() const {
    return (getSTI().getFeatureBits()[Mips::FeatureMips64r5]);
  }
  bool hasMips32r6() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips32r6];
  }
  bool hasMips64r6() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips64r6];
  }

  bool hasDSP() const {
    return getSTI().getFeatureBits()[Mips::FeatureDSP];
  }
  bool hasDSPR2() const {
    return getSTI().getFeatureBits()[Mips::FeatureDSPR2];
  }
  bool hasDSPR3() const {
    return getSTI().getFeatureBits()[Mips::FeatureDSPR3];
  }
  bool hasMSA() const {
    return getSTI().getFeatureBits()[Mips::FeatureMSA];
  }
  bool hasCnMips() const {
    return (getSTI().getFeatureBits()[Mips::FeatureCnMips]);
  }

  bool inPicMode() {
    return IsPicEnabled;
  }

  bool inMips16Mode() const {
    return getSTI().getFeatureBits()[Mips::FeatureMips16];
  }

  bool useTraps() const {
    return getSTI().getFeatureBits()[Mips::FeatureUseTCCInDIV];
  }

  bool useSoftFloat() const {
    return getSTI().getFeatureBits()[Mips::FeatureSoftFloat];
  }

  /// Warn if RegIndex is the same as the current AT.
  void warnIfRegIndexIsAT(unsigned RegIndex, SMLoc Loc);

  void warnIfNoMacro(SMLoc Loc);

  bool isLittle() const { return IsLittleEndian; }
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
    RegKind_COP0 = 1024,  /// COP0
    /// Potentially any (e.g. $1)
    RegKind_Numeric = RegKind_GPR | RegKind_FGR | RegKind_FCC | RegKind_MSA128 |
                      RegKind_MSACtrl | RegKind_COP2 | RegKind_ACC |
                      RegKind_CCR | RegKind_HWRegs | RegKind_COP3 | RegKind_COP0
  };

private:
  enum KindTy {
    k_Immediate,     /// An immediate (possibly involving symbol references)
    k_Memory,        /// Base + Offset Memory Address
    k_PhysRegister,  /// A physical register from the Mips namespace
    k_RegisterIndex, /// A register index in one or more RegKind.
    k_Token,         /// A simple token
    k_RegList,       /// A physical register list
    k_RegPair        /// A pair of physical register
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

  struct RegListOp {
    SmallVector<unsigned, 10> *List;
  };

  union {
    struct Token Tok;
    struct PhysRegOp PhysReg;
    struct RegIdxOp RegIdx;
    struct ImmOp Imm;
    struct MemOp Mem;
    struct RegListOp RegList;
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
    AsmParser.warnIfRegIndexIsAT(RegIdx.Index, StartLoc);
    unsigned ClassID = Mips::GPR32RegClassID;
    return RegIdx.RegInfo->getRegClass(ClassID).getRegister(RegIdx.Index);
  }

  /// Coerce the register to GPR32 and return the real register for the current
  /// target.
  unsigned getGPRMM16Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_GPR) && "Invalid access!");
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

  /// Coerce the register to COP0 and return the real register for the
  /// current target.
  unsigned getCOP0Reg() const {
    assert(isRegIdx() && (RegIdx.Kind & RegKind_COP0) && "Invalid access!");
    unsigned ClassID = Mips::COP0RegClassID;
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
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    llvm_unreachable("Use a custom parser instead");
  }

  /// Render the operand to an MCInst as a GPR32
  /// Asserts if the wrong number of operands are requested, or the operand
  /// is not a k_RegisterIndex compatible with RegKind_GPR
  void addGPR32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getGPR32Reg()));
  }

  void addGPRMM16AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getGPRMM16Reg()));
  }

  void addGPRMM16AsmRegZeroOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getGPRMM16Reg()));
  }

  void addGPRMM16AsmRegMovePOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getGPRMM16Reg()));
  }

  /// Render the operand to an MCInst as a GPR64
  /// Asserts if the wrong number of operands are requested, or the operand
  /// is not a k_RegisterIndex compatible with RegKind_GPR
  void addGPR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getGPR64Reg()));
  }

  void addAFGR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getAFGR64Reg()));
  }

  void addFGR64AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getFGR64Reg()));
  }

  void addFGR32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getFGR32Reg()));
    // FIXME: We ought to do this for -integrated-as without -via-file-asm too.
    if (!AsmParser.useOddSPReg() && RegIdx.Index & 1)
      AsmParser.Error(StartLoc, "-mno-odd-spreg prohibits the use of odd FPU "
                                "registers");
  }

  void addFGRH32AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getFGRH32Reg()));
  }

  void addFCCAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getFCCReg()));
  }

  void addMSA128AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getMSA128Reg()));
  }

  void addMSACtrlAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getMSACtrlReg()));
  }

  void addCOP0AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getCOP0Reg()));
  }

  void addCOP2AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getCOP2Reg()));
  }

  void addCOP3AsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getCOP3Reg()));
  }

  void addACC64DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getACC64DSPReg()));
  }

  void addHI32DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getHI32DSPReg()));
  }

  void addLO32DSPAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getLO32DSPReg()));
  }

  void addCCRAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getCCRReg()));
  }

  void addHWRegsAsmRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getHWRegsReg()));
  }

  template <unsigned Bits, int Offset = 0, int AdjustOffset = 0>
  void addConstantUImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    uint64_t Imm = getConstantImm() - Offset;
    Imm &= (1 << Bits) - 1;
    Imm += Offset;
    Imm += AdjustOffset;
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  template <unsigned Bits>
  void addSImmOperands(MCInst &Inst, unsigned N) const {
    if (isImm() && !isConstantImm()) {
      addExpr(Inst, getImm());
      return;
    }
    addConstantSImmOperands<Bits, 0, 0>(Inst, N);
  }

  template <unsigned Bits>
  void addUImmOperands(MCInst &Inst, unsigned N) const {
    if (isImm() && !isConstantImm()) {
      addExpr(Inst, getImm());
      return;
    }
    addConstantUImmOperands<Bits, 0, 0>(Inst, N);
  }

  template <unsigned Bits, int Offset = 0, int AdjustOffset = 0>
  void addConstantSImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    int64_t Imm = getConstantImm() - Offset;
    Imm = SignExtend64<Bits>(Imm);
    Imm += Offset;
    Imm += AdjustOffset;
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(AsmParser.getABI().ArePtrs64bit()
                                             ? getMemBase()->getGPR64Reg()
                                             : getMemBase()->getGPR32Reg()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  void addMicroMipsMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()->getGPRMM16Reg()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  void addRegListOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    for (auto RegNo : getRegList())
      Inst.addOperand(MCOperand::createReg(RegNo));
  }

  void addRegPairOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    unsigned RegNo = getRegPair();
    Inst.addOperand(MCOperand::createReg(RegNo++));
    Inst.addOperand(MCOperand::createReg(RegNo));
  }

  void addMovePRegPairOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    for (auto RegNo : getRegList())
      Inst.addOperand(MCOperand::createReg(RegNo));
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
    return isImm() && isa<MCConstantExpr>(getImm());
  }
  bool isConstantImmz() const {
    return isConstantImm() && getConstantImm() == 0;
  }
  template <unsigned Bits, int Offset = 0> bool isConstantUImm() const {
    return isConstantImm() && isUInt<Bits>(getConstantImm() - Offset);
  }
  template <unsigned Bits> bool isSImm() const {
    return isConstantImm() ? isInt<Bits>(getConstantImm()) : isImm();
  }
  template <unsigned Bits> bool isUImm() const {
    return isConstantImm() ? isUInt<Bits>(getConstantImm()) : isImm();
  }
  template <unsigned Bits> bool isAnyImm() const {
    return isConstantImm() ? (isInt<Bits>(getConstantImm()) ||
                              isUInt<Bits>(getConstantImm()))
                           : isImm();
  }
  template <unsigned Bits, int Offset = 0> bool isConstantSImm() const {
    return isConstantImm() && isInt<Bits>(getConstantImm() - Offset);
  }
  template <unsigned Bottom, unsigned Top> bool isConstantUImmRange() const {
    return isConstantImm() && getConstantImm() >= Bottom &&
           getConstantImm() <= Top;
  }
  bool isToken() const override {
    // Note: It's not possible to pretend that other operand kinds are tokens.
    // The matcher emitter checks tokens first.
    return Kind == k_Token;
  }
  bool isMem() const override { return Kind == k_Memory; }
  bool isConstantMemOff() const {
    return isMem() && isa<MCConstantExpr>(getMemOff());
  }
  template <unsigned Bits, unsigned ShiftAmount = 0>
  bool isMemWithSimmOffset() const {
    return isMem() && isConstantMemOff() &&
           isShiftedInt<Bits, ShiftAmount>(getConstantMemOff()) &&
           getMemBase()->isGPRAsmReg();
  }
  template <unsigned Bits> bool isMemWithSimmOffsetGPR() const {
    return isMem() && isConstantMemOff() && isInt<Bits>(getConstantMemOff()) &&
           getMemBase()->isGPRAsmReg();
  }
  bool isMemWithGRPMM16Base() const {
    return isMem() && getMemBase()->isMM16AsmReg();
  }
  template <unsigned Bits> bool isMemWithUimmOffsetSP() const {
    return isMem() && isConstantMemOff() && isUInt<Bits>(getConstantMemOff())
      && getMemBase()->isRegIdx() && (getMemBase()->getGPR32Reg() == Mips::SP);
  }
  template <unsigned Bits> bool isMemWithUimmWordAlignedOffsetSP() const {
    return isMem() && isConstantMemOff() && isUInt<Bits>(getConstantMemOff())
      && (getConstantMemOff() % 4 == 0) && getMemBase()->isRegIdx()
      && (getMemBase()->getGPR32Reg() == Mips::SP);
  }
  template <unsigned Bits, unsigned ShiftLeftAmount>
  bool isScaledUImm() const {
    return isConstantImm() &&
           isShiftedUInt<Bits, ShiftLeftAmount>(getConstantImm());
  }
  template <unsigned Bits, unsigned ShiftLeftAmount>
  bool isScaledSImm() const {
    return isConstantImm() &&
           isShiftedInt<Bits, ShiftLeftAmount>(getConstantImm());
  }
  bool isRegList16() const {
    if (!isRegList())
      return false;

    int Size = RegList.List->size();
    if (Size < 2 || Size > 5)
      return false;

    unsigned R0 = RegList.List->front();
    unsigned R1 = RegList.List->back();
    if (!((R0 == Mips::S0 && R1 == Mips::RA) ||
          (R0 == Mips::S0_64 && R1 == Mips::RA_64)))
      return false;

    int PrevReg = *RegList.List->begin();
    for (int i = 1; i < Size - 1; i++) {
      int Reg = (*(RegList.List))[i];
      if ( Reg != PrevReg + 1)
        return false;
      PrevReg = Reg;
    }

    return true;
  }
  bool isInvNum() const { return Kind == k_Immediate; }
  bool isLSAImm() const {
    if (!isConstantImm())
      return false;
    int64_t Val = getConstantImm();
    return 1 <= Val && Val <= 4;
  }
  bool isRegList() const { return Kind == k_RegList; }
  bool isMovePRegPair() const {
    if (Kind != k_RegList || RegList.List->size() != 2)
      return false;

    unsigned R0 = RegList.List->front();
    unsigned R1 = RegList.List->back();

    if ((R0 == Mips::A1 && R1 == Mips::A2) ||
        (R0 == Mips::A1 && R1 == Mips::A3) ||
        (R0 == Mips::A2 && R1 == Mips::A3) ||
        (R0 == Mips::A0 && R1 == Mips::S5) ||
        (R0 == Mips::A0 && R1 == Mips::S6) ||
        (R0 == Mips::A0 && R1 == Mips::A1) ||
        (R0 == Mips::A0 && R1 == Mips::A2) ||
        (R0 == Mips::A0 && R1 == Mips::A3))
      return true;

    return false;
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }
  bool isRegPair() const { return Kind == k_RegPair; }

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

  const SmallVectorImpl<unsigned> &getRegList() const {
    assert((Kind == k_RegList) && "Invalid access!");
    return *(RegList.List);
  }

  unsigned getRegPair() const {
    assert((Kind == k_RegPair) && "Invalid access!");
    return RegIdx.Index;
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
  createNumericReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
                   SMLoc E, MipsAsmParser &Parser) {
    DEBUG(dbgs() << "createNumericReg(" << Index << ", ...)\n");
    return CreateReg(Index, RegKind_Numeric, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely a GPR.
  /// This is typically only used for named registers such as $gp.
  static std::unique_ptr<MipsOperand>
  createGPRReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_GPR, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely a FGR.
  /// This is typically only used for named registers such as $f0.
  static std::unique_ptr<MipsOperand>
  createFGRReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_FGR, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely a HWReg.
  /// This is typically only used for named registers such as $hwr_cpunum.
  static std::unique_ptr<MipsOperand>
  createHWRegsReg(unsigned Index, const MCRegisterInfo *RegInfo,
                  SMLoc S, SMLoc E, MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_HWRegs, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an FCC.
  /// This is typically only used for named registers such as $fcc0.
  static std::unique_ptr<MipsOperand>
  createFCCReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_FCC, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an ACC.
  /// This is typically only used for named registers such as $ac0.
  static std::unique_ptr<MipsOperand>
  createACCReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S, SMLoc E,
               MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_ACC, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an MSA128.
  /// This is typically only used for named registers such as $w0.
  static std::unique_ptr<MipsOperand>
  createMSA128Reg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
                  SMLoc E, MipsAsmParser &Parser) {
    return CreateReg(Index, RegKind_MSA128, RegInfo, S, E, Parser);
  }

  /// Create a register that is definitely an MSACtrl.
  /// This is typically only used for named registers such as $msaaccess.
  static std::unique_ptr<MipsOperand>
  createMSACtrlReg(unsigned Index, const MCRegisterInfo *RegInfo, SMLoc S,
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

  static std::unique_ptr<MipsOperand>
  CreateRegList(SmallVectorImpl<unsigned> &Regs, SMLoc StartLoc, SMLoc EndLoc,
                MipsAsmParser &Parser) {
    assert (Regs.size() > 0 && "Empty list not allowed");

    auto Op = make_unique<MipsOperand>(k_RegList, Parser);
    Op->RegList.List = new SmallVector<unsigned, 10>(Regs.begin(), Regs.end());
    Op->StartLoc = StartLoc;
    Op->EndLoc = EndLoc;
    return Op;
  }

  static std::unique_ptr<MipsOperand>
  CreateRegPair(unsigned RegNo, SMLoc S, SMLoc E, MipsAsmParser &Parser) {
    auto Op = make_unique<MipsOperand>(k_RegPair, Parser);
    Op->RegIdx.Index = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  bool isGPRAsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_GPR && RegIdx.Index <= 31;
  }
  bool isMM16AsmReg() const {
    if (!(isRegIdx() && RegIdx.Kind))
      return false;
    return ((RegIdx.Index >= 2 && RegIdx.Index <= 7)
            || RegIdx.Index == 16 || RegIdx.Index == 17);
  }
  bool isMM16AsmRegZero() const {
    if (!(isRegIdx() && RegIdx.Kind))
      return false;
    return (RegIdx.Index == 0 ||
            (RegIdx.Index >= 2 && RegIdx.Index <= 7) ||
            RegIdx.Index == 17);
  }
  bool isMM16AsmRegMoveP() const {
    if (!(isRegIdx() && RegIdx.Kind))
      return false;
    return (RegIdx.Index == 0 || (RegIdx.Index >= 2 && RegIdx.Index <= 3) ||
      (RegIdx.Index >= 16 && RegIdx.Index <= 20));
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
  bool isCOP0AsmReg() const {
    return isRegIdx() && RegIdx.Kind & RegKind_COP0 && RegIdx.Index <= 31;
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
    case k_RegList:
      delete RegList.List;
    case k_PhysRegister:
    case k_RegisterIndex:
    case k_Token:
    case k_RegPair:
      break;
    }
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case k_Immediate:
      OS << "Imm<";
      OS << *Imm.Val;
      OS << ">";
      break;
    case k_Memory:
      OS << "Mem<";
      Mem.Base->print(OS);
      OS << ", ";
      OS << *Mem.Off;
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
    case k_RegList:
      OS << "RegList< ";
      for (auto Reg : (*RegList.List))
        OS << Reg << " ";
      OS <<  ">";
      break;
    case k_RegPair:
      OS << "RegPair<" << RegIdx.Index << "," << RegIdx.Index + 1 << ">";
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

static bool hasShortDelaySlot(unsigned Opcode) {
  switch (Opcode) {
    case Mips::JALS_MM:
    case Mips::JALRS_MM:
    case Mips::JALRS16_MM:
    case Mips::BGEZALS_MM:
    case Mips::BLTZALS_MM:
      return true;
    default:
      return false;
  }
}

static const MCSymbol *getSingleMCSymbol(const MCExpr *Expr) {
  if (const MCSymbolRefExpr *SRExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
    return &SRExpr->getSymbol();
  }

  if (const MCBinaryExpr *BExpr = dyn_cast<MCBinaryExpr>(Expr)) {
    const MCSymbol *LHSSym = getSingleMCSymbol(BExpr->getLHS());
    const MCSymbol *RHSSym = getSingleMCSymbol(BExpr->getRHS());

    if (LHSSym)
      return LHSSym;

    if (RHSSym)
      return RHSSym;

    return nullptr;
  }

  if (const MCUnaryExpr *UExpr = dyn_cast<MCUnaryExpr>(Expr))
    return getSingleMCSymbol(UExpr->getSubExpr());

  return nullptr;
}

static unsigned countMCSymbolRefExpr(const MCExpr *Expr) {
  if (isa<MCSymbolRefExpr>(Expr))
    return 1;

  if (const MCBinaryExpr *BExpr = dyn_cast<MCBinaryExpr>(Expr))
    return countMCSymbolRefExpr(BExpr->getLHS()) +
           countMCSymbolRefExpr(BExpr->getRHS());

  if (const MCUnaryExpr *UExpr = dyn_cast<MCUnaryExpr>(Expr))
    return countMCSymbolRefExpr(UExpr->getSubExpr());

  return 0;
}

namespace {
void emitRX(unsigned Opcode, unsigned Reg0, MCOperand Op1, SMLoc IDLoc,
            SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  tmpInst.setOpcode(Opcode);
  tmpInst.addOperand(MCOperand::createReg(Reg0));
  tmpInst.addOperand(Op1);
  tmpInst.setLoc(IDLoc);
  Instructions.push_back(tmpInst);
}

void emitRI(unsigned Opcode, unsigned Reg0, int32_t Imm, SMLoc IDLoc,
            SmallVectorImpl<MCInst> &Instructions) {
  emitRX(Opcode, Reg0, MCOperand::createImm(Imm), IDLoc, Instructions);
}

void emitRR(unsigned Opcode, unsigned Reg0, unsigned Reg1, SMLoc IDLoc,
            SmallVectorImpl<MCInst> &Instructions) {
  emitRX(Opcode, Reg0, MCOperand::createReg(Reg1), IDLoc, Instructions);
}

void emitII(unsigned Opcode, int16_t Imm1, int16_t Imm2, SMLoc IDLoc,
            SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  tmpInst.setOpcode(Opcode);
  tmpInst.addOperand(MCOperand::createImm(Imm1));
  tmpInst.addOperand(MCOperand::createImm(Imm2));
  tmpInst.setLoc(IDLoc);
  Instructions.push_back(tmpInst);
}

void emitR(unsigned Opcode, unsigned Reg0, SMLoc IDLoc,
           SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  tmpInst.setOpcode(Opcode);
  tmpInst.addOperand(MCOperand::createReg(Reg0));
  tmpInst.setLoc(IDLoc);
  Instructions.push_back(tmpInst);
}

void emitRRX(unsigned Opcode, unsigned Reg0, unsigned Reg1, MCOperand Op2,
             SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  MCInst tmpInst;
  tmpInst.setOpcode(Opcode);
  tmpInst.addOperand(MCOperand::createReg(Reg0));
  tmpInst.addOperand(MCOperand::createReg(Reg1));
  tmpInst.addOperand(Op2);
  tmpInst.setLoc(IDLoc);
  Instructions.push_back(tmpInst);
}

void emitRRR(unsigned Opcode, unsigned Reg0, unsigned Reg1, unsigned Reg2,
             SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  emitRRX(Opcode, Reg0, Reg1, MCOperand::createReg(Reg2), IDLoc,
          Instructions);
}

void emitRRI(unsigned Opcode, unsigned Reg0, unsigned Reg1, int16_t Imm,
             SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  emitRRX(Opcode, Reg0, Reg1, MCOperand::createImm(Imm), IDLoc,
          Instructions);
}

void emitAppropriateDSLL(unsigned DstReg, unsigned SrcReg, int16_t ShiftAmount,
                         SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  if (ShiftAmount >= 32) {
    emitRRI(Mips::DSLL32, DstReg, SrcReg, ShiftAmount - 32, IDLoc,
            Instructions);
    return;
  }

  emitRRI(Mips::DSLL, DstReg, SrcReg, ShiftAmount, IDLoc, Instructions);
}
} // end anonymous namespace.

bool MipsAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                       SmallVectorImpl<MCInst> &Instructions) {
  const MCInstrDesc &MCID = getInstDesc(Inst.getOpcode());
  bool ExpandedJalSym = false;

  Inst.setLoc(IDLoc);

  if (MCID.isBranch() || MCID.isCall()) {
    const unsigned Opcode = Inst.getOpcode();
    MCOperand Offset;

    switch (Opcode) {
    default:
      break;
    case Mips::BBIT0:
    case Mips::BBIT032:
    case Mips::BBIT1:
    case Mips::BBIT132:
      assert(hasCnMips() && "instruction only valid for octeon cpus");
      // Fall through

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
    case Mips::BEQZ16_MM:
    case Mips::BEQZC16_MMR6:
    case Mips::BNEZ16_MM:
    case Mips::BNEZC16_MMR6:
      assert(MCID.getNumOperands() == 2 && "unexpected number of operands");
      Offset = Inst.getOperand(1);
      if (!Offset.isImm())
        break; // We'll deal with this situation later on when applying fixups.
      if (!isInt<8>(Offset.getImm()))
        return Error(IDLoc, "branch target out of range");
      if (OffsetToAlignment(Offset.getImm(), 2LL))
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

  if (hasCnMips()) {
    const unsigned Opcode = Inst.getOpcode();
    MCOperand Opnd;
    int Imm;

    switch (Opcode) {
      default:
        break;

      case Mips::BBIT0:
      case Mips::BBIT032:
      case Mips::BBIT1:
      case Mips::BBIT132:
        assert(MCID.getNumOperands() == 3 && "unexpected number of operands");
        // The offset is handled above
        Opnd = Inst.getOperand(1);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < 0 || Imm > (Opcode == Mips::BBIT0 ||
                              Opcode == Mips::BBIT1 ? 63 : 31))
          return Error(IDLoc, "immediate operand value out of range");
        if (Imm > 31) {
          Inst.setOpcode(Opcode == Mips::BBIT0 ? Mips::BBIT032
                                               : Mips::BBIT132);
          Inst.getOperand(1).setImm(Imm - 32);
        }
        break;

      case Mips::SEQi:
      case Mips::SNEi:
        assert(MCID.getNumOperands() == 3 && "unexpected number of operands");
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (!isInt<10>(Imm))
          return Error(IDLoc, "immediate operand value out of range");
        break;
    }
  }

  // This expansion is not in a function called by tryExpandInstruction()
  // because the pseudo-instruction doesn't have a distinct opcode.
  if ((Inst.getOpcode() == Mips::JAL || Inst.getOpcode() == Mips::JAL_MM) &&
      inPicMode()) {
    warnIfNoMacro(IDLoc);

    const MCExpr *JalExpr = Inst.getOperand(0).getExpr();

    // We can do this expansion if there's only 1 symbol in the argument
    // expression.
    if (countMCSymbolRefExpr(JalExpr) > 1)
      return Error(IDLoc, "jal doesn't support multiple symbols in PIC mode");

    // FIXME: This is checking the expression can be handled by the later stages
    //        of the assembler. We ought to leave it to those later stages but
    //        we can't do that until we stop evaluateRelocExpr() rewriting the
    //        expressions into non-equivalent forms.
    const MCSymbol *JalSym = getSingleMCSymbol(JalExpr);

    // FIXME: Add support for label+offset operands (currently causes an error).
    // FIXME: Add support for forward-declared local symbols.
    // FIXME: Add expansion for when the LargeGOT option is enabled.
    if (JalSym->isInSection() || JalSym->isTemporary()) {
      if (isABI_O32()) {
        // If it's a local symbol and the O32 ABI is being used, we expand to:
        //  lw $25, 0($gp)
        //    R_(MICRO)MIPS_GOT16  label
        //  addiu $25, $25, 0
        //    R_(MICRO)MIPS_LO16   label
        //  jalr  $25
        const MCExpr *Got16RelocExpr = evaluateRelocExpr(JalExpr, "got");
        const MCExpr *Lo16RelocExpr = evaluateRelocExpr(JalExpr, "lo");

        emitRRX(Mips::LW, Mips::T9, Mips::GP,
                MCOperand::createExpr(Got16RelocExpr), IDLoc, Instructions);
        emitRRX(Mips::ADDiu, Mips::T9, Mips::T9,
                MCOperand::createExpr(Lo16RelocExpr), IDLoc, Instructions);
      } else if (isABI_N32() || isABI_N64()) {
        // If it's a local symbol and the N32/N64 ABIs are being used,
        // we expand to:
        //  lw/ld $25, 0($gp)
        //    R_(MICRO)MIPS_GOT_DISP  label
        //  jalr  $25
        const MCExpr *GotDispRelocExpr = evaluateRelocExpr(JalExpr, "got_disp");

        emitRRX(ABI.ArePtrs64bit() ? Mips::LD : Mips::LW, Mips::T9, Mips::GP,
                MCOperand::createExpr(GotDispRelocExpr), IDLoc, Instructions);
      }
    } else {
      // If it's an external/weak symbol, we expand to:
      //  lw/ld    $25, 0($gp)
      //    R_(MICRO)MIPS_CALL16  label
      //  jalr  $25
      const MCExpr *Call16RelocExpr = evaluateRelocExpr(JalExpr, "call16");

      emitRRX(ABI.ArePtrs64bit() ? Mips::LD : Mips::LW, Mips::T9, Mips::GP,
              MCOperand::createExpr(Call16RelocExpr), IDLoc, Instructions);
    }

    MCInst JalrInst;
    if (IsCpRestoreSet && inMicroMipsMode())
      JalrInst.setOpcode(Mips::JALRS_MM);
    else
      JalrInst.setOpcode(inMicroMipsMode() ? Mips::JALR_MM : Mips::JALR);
    JalrInst.addOperand(MCOperand::createReg(Mips::RA));
    JalrInst.addOperand(MCOperand::createReg(Mips::T9));

    // FIXME: Add an R_(MICRO)MIPS_JALR relocation after the JALR.
    // This relocation is supposed to be an optimization hint for the linker
    // and is not necessary for correctness.

    Inst = JalrInst;
    ExpandedJalSym = true;
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

  if (inMicroMipsMode()) {
    if (MCID.mayLoad()) {
      // Try to create 16-bit GP relative load instruction.
      for (unsigned i = 0; i < MCID.getNumOperands(); i++) {
        const MCOperandInfo &OpInfo = MCID.OpInfo[i];
        if ((OpInfo.OperandType == MCOI::OPERAND_MEMORY) ||
            (OpInfo.OperandType == MCOI::OPERAND_UNKNOWN)) {
          MCOperand &Op = Inst.getOperand(i);
          if (Op.isImm()) {
            int MemOffset = Op.getImm();
            MCOperand &DstReg = Inst.getOperand(0);
            MCOperand &BaseReg = Inst.getOperand(1);
            if (isInt<9>(MemOffset) && (MemOffset % 4 == 0) &&
                getContext().getRegisterInfo()->getRegClass(
                  Mips::GPRMM16RegClassID).contains(DstReg.getReg()) &&
                (BaseReg.getReg() == Mips::GP ||
                BaseReg.getReg() == Mips::GP_64)) {

              emitRRI(Mips::LWGP_MM, DstReg.getReg(), Mips::GP, MemOffset,
                      IDLoc, Instructions);
              return false;
            }
          }
        }
      } // for
    }   // if load

    // TODO: Handle this with the AsmOperandClass.PredicateMethod.

    MCOperand Opnd;
    int Imm;

    switch (Inst.getOpcode()) {
      default:
        break;
      case Mips::ADDIUSP_MM:
        Opnd = Inst.getOperand(0);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < -1032 || Imm > 1028 || (Imm < 8 && Imm > -12) ||
            Imm % 4 != 0)
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::SLL16_MM:
      case Mips::SRL16_MM:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < 1 || Imm > 8)
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::LI16_MM:
        Opnd = Inst.getOperand(1);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < -1 || Imm > 126)
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::ADDIUR2_MM:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (!(Imm == 1 || Imm == -1 ||
              ((Imm % 4 == 0) && Imm < 28 && Imm > 0)))
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::ANDI16_MM:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (!(Imm == 128 || (Imm >= 1 && Imm <= 4) || Imm == 7 || Imm == 8 ||
              Imm == 15 || Imm == 16 || Imm == 31 || Imm == 32 || Imm == 63 ||
              Imm == 64 || Imm == 255 || Imm == 32768 || Imm == 65535))
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::LBU16_MM:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < -1 || Imm > 14)
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::SB16_MM:
      case Mips::SB16_MMR6:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < 0 || Imm > 15)
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::LHU16_MM:
      case Mips::SH16_MM:
      case Mips::SH16_MMR6:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < 0 || Imm > 30 || (Imm % 2 != 0))
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::LW16_MM:
      case Mips::SW16_MM:
      case Mips::SW16_MMR6:
        Opnd = Inst.getOperand(2);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        Imm = Opnd.getImm();
        if (Imm < 0 || Imm > 60 || (Imm % 4 != 0))
          return Error(IDLoc, "immediate operand value out of range");
        break;
      case Mips::ADDIUPC_MM:
        MCOperand Opnd = Inst.getOperand(1);
        if (!Opnd.isImm())
          return Error(IDLoc, "expected immediate operand kind");
        int Imm = Opnd.getImm();
        if ((Imm % 4 != 0) || !isInt<25>(Imm))
          return Error(IDLoc, "immediate operand value out of range");
        break;
    }
  }

  MacroExpanderResultTy ExpandResult =
      tryExpandInstruction(Inst, IDLoc, Instructions);
  switch (ExpandResult) {
  case MER_NotAMacro:
    Instructions.push_back(Inst);
    break;
  case MER_Success:
    break;
  case MER_Fail:
    return true;
  }

  // If this instruction has a delay slot and .set reorder is active,
  // emit a NOP after it.
  if (MCID.hasDelaySlot() && AssemblerOptions.back()->isReorder())
    createNop(hasShortDelaySlot(Inst.getOpcode()), IDLoc, Instructions);

  if ((Inst.getOpcode() == Mips::JalOneReg ||
       Inst.getOpcode() == Mips::JalTwoReg || ExpandedJalSym) &&
      isPicAndNotNxxAbi()) {
    if (IsCpRestoreSet) {
      // We need a NOP between the JALR and the LW:
      // If .set reorder has been used, we've already emitted a NOP.
      // If .set noreorder has been used, we need to emit a NOP at this point.
      if (!AssemblerOptions.back()->isReorder())
        createNop(hasShortDelaySlot(Inst.getOpcode()), IDLoc, Instructions);

      // Load the $gp from the stack.
      SmallVector<MCInst, 3> LoadInsts;
      createCpRestoreMemOp(true /*IsLoad*/, CpRestoreOffset /*StackOffset*/,
                           IDLoc, LoadInsts);

      for (const MCInst &Inst : LoadInsts)
        Instructions.push_back(Inst);

    } else
      Warning(IDLoc, "no .cprestore used in PIC mode");
  }

  return false;
}

MipsAsmParser::MacroExpanderResultTy
MipsAsmParser::tryExpandInstruction(MCInst &Inst, SMLoc IDLoc,
                                    SmallVectorImpl<MCInst> &Instructions) {
  switch (Inst.getOpcode()) {
  default:
    return MER_NotAMacro;
  case Mips::LoadImm32:
    return expandLoadImm(Inst, true, IDLoc, Instructions) ? MER_Fail
                                                          : MER_Success;
  case Mips::LoadImm64:
    return expandLoadImm(Inst, false, IDLoc, Instructions) ? MER_Fail
                                                           : MER_Success;
  case Mips::LoadAddrImm32:
  case Mips::LoadAddrImm64:
    assert(Inst.getOperand(0).isReg() && "expected register operand kind");
    assert((Inst.getOperand(1).isImm() || Inst.getOperand(1).isExpr()) &&
           "expected immediate operand kind");

    return expandLoadAddress(Inst.getOperand(0).getReg(), Mips::NoRegister,
                             Inst.getOperand(1),
                             Inst.getOpcode() == Mips::LoadAddrImm32, IDLoc,
                             Instructions)
               ? MER_Fail
               : MER_Success;
  case Mips::LoadAddrReg32:
  case Mips::LoadAddrReg64:
    assert(Inst.getOperand(0).isReg() && "expected register operand kind");
    assert(Inst.getOperand(1).isReg() && "expected register operand kind");
    assert((Inst.getOperand(2).isImm() || Inst.getOperand(2).isExpr()) &&
           "expected immediate operand kind");

    return expandLoadAddress(Inst.getOperand(0).getReg(),
                             Inst.getOperand(1).getReg(), Inst.getOperand(2),
                             Inst.getOpcode() == Mips::LoadAddrReg32, IDLoc,
                             Instructions)
               ? MER_Fail
               : MER_Success;
  case Mips::B_MM_Pseudo:
  case Mips::B_MMR6_Pseudo:
    return expandUncondBranchMMPseudo(Inst, IDLoc, Instructions) ? MER_Fail
                                                                 : MER_Success;
  case Mips::SWM_MM:
  case Mips::LWM_MM:
    return expandLoadStoreMultiple(Inst, IDLoc, Instructions) ? MER_Fail
                                                              : MER_Success;
  case Mips::JalOneReg:
  case Mips::JalTwoReg:
    return expandJalWithRegs(Inst, IDLoc, Instructions) ? MER_Fail
                                                        : MER_Success;
  case Mips::BneImm:
  case Mips::BeqImm:
    return expandBranchImm(Inst, IDLoc, Instructions) ? MER_Fail : MER_Success;
  case Mips::BLT:
  case Mips::BLE:
  case Mips::BGE:
  case Mips::BGT:
  case Mips::BLTU:
  case Mips::BLEU:
  case Mips::BGEU:
  case Mips::BGTU:
  case Mips::BLTL:
  case Mips::BLEL:
  case Mips::BGEL:
  case Mips::BGTL:
  case Mips::BLTUL:
  case Mips::BLEUL:
  case Mips::BGEUL:
  case Mips::BGTUL:
  case Mips::BLTImmMacro:
  case Mips::BLEImmMacro:
  case Mips::BGEImmMacro:
  case Mips::BGTImmMacro:
  case Mips::BLTUImmMacro:
  case Mips::BLEUImmMacro:
  case Mips::BGEUImmMacro:
  case Mips::BGTUImmMacro:
  case Mips::BLTLImmMacro:
  case Mips::BLELImmMacro:
  case Mips::BGELImmMacro:
  case Mips::BGTLImmMacro:
  case Mips::BLTULImmMacro:
  case Mips::BLEULImmMacro:
  case Mips::BGEULImmMacro:
  case Mips::BGTULImmMacro:
    return expandCondBranches(Inst, IDLoc, Instructions) ? MER_Fail
                                                         : MER_Success;
  case Mips::SDivMacro:
    return expandDiv(Inst, IDLoc, Instructions, false, true) ? MER_Fail
                                                             : MER_Success;
  case Mips::DSDivMacro:
    return expandDiv(Inst, IDLoc, Instructions, true, true) ? MER_Fail
                                                            : MER_Success;
  case Mips::UDivMacro:
    return expandDiv(Inst, IDLoc, Instructions, false, false) ? MER_Fail
                                                              : MER_Success;
  case Mips::DUDivMacro:
    return expandDiv(Inst, IDLoc, Instructions, true, false) ? MER_Fail
                                                             : MER_Success;
  case Mips::PseudoTRUNC_W_S:
    return expandTrunc(Inst, false, false, IDLoc, Instructions) ? MER_Fail
                                                                : MER_Success;
  case Mips::PseudoTRUNC_W_D32:
    return expandTrunc(Inst, true, false, IDLoc, Instructions) ? MER_Fail
                                                               : MER_Success;
  case Mips::PseudoTRUNC_W_D:
    return expandTrunc(Inst, true, true, IDLoc, Instructions) ? MER_Fail
                                                              : MER_Success;
  case Mips::Ulh:
    return expandUlh(Inst, true, IDLoc, Instructions) ? MER_Fail : MER_Success;
  case Mips::Ulhu:
    return expandUlh(Inst, false, IDLoc, Instructions) ? MER_Fail : MER_Success;
  case Mips::Ulw:
    return expandUlw(Inst, IDLoc, Instructions) ? MER_Fail : MER_Success;
  case Mips::NORImm:
    return expandAliasImmediate(Inst, IDLoc, Instructions) ? MER_Fail
                                                           : MER_Success;
  case Mips::ADDi:
  case Mips::ADDiu:
  case Mips::SLTi:
  case Mips::SLTiu:
    if ((Inst.getNumOperands() == 3) && Inst.getOperand(0).isReg() &&
        Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
      int64_t ImmValue = Inst.getOperand(2).getImm();
      if (isInt<16>(ImmValue))
        return MER_NotAMacro;
      return expandAliasImmediate(Inst, IDLoc, Instructions) ? MER_Fail
                                                             : MER_Success;
    }
    return MER_NotAMacro;
  case Mips::ANDi:
  case Mips::ORi:
  case Mips::XORi:
    if ((Inst.getNumOperands() == 3) && Inst.getOperand(0).isReg() &&
        Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
      int64_t ImmValue = Inst.getOperand(2).getImm();
      if (isUInt<16>(ImmValue))
        return MER_NotAMacro;
      return expandAliasImmediate(Inst, IDLoc, Instructions) ? MER_Fail
                                                             : MER_Success;
    }
    return MER_NotAMacro;
  case Mips::ROL:
  case Mips::ROR:
    return expandRotation(Inst, IDLoc, Instructions) ? MER_Fail
                                                     : MER_Success;
  case Mips::ROLImm:
  case Mips::RORImm:
    return expandRotationImm(Inst, IDLoc, Instructions) ? MER_Fail
                                                        : MER_Success;
  case Mips::DROL:
  case Mips::DROR:
    return expandDRotation(Inst, IDLoc, Instructions) ? MER_Fail
                                                      : MER_Success;
  case Mips::DROLImm:
  case Mips::DRORImm:
    return expandDRotationImm(Inst, IDLoc, Instructions) ? MER_Fail
                                                         : MER_Success;
  case Mips::ABSMacro:
    return expandAbs(Inst, IDLoc, Instructions) ? MER_Fail
                                                : MER_Success;
  }
}

bool MipsAsmParser::expandJalWithRegs(MCInst &Inst, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions) {
  // Create a JALR instruction which is going to replace the pseudo-JAL.
  MCInst JalrInst;
  JalrInst.setLoc(IDLoc);
  const MCOperand FirstRegOp = Inst.getOperand(0);
  const unsigned Opcode = Inst.getOpcode();

  if (Opcode == Mips::JalOneReg) {
    // jal $rs => jalr $rs
    if (IsCpRestoreSet && inMicroMipsMode()) {
      JalrInst.setOpcode(Mips::JALRS16_MM);
      JalrInst.addOperand(FirstRegOp);
    } else if (inMicroMipsMode()) {
      JalrInst.setOpcode(hasMips32r6() ? Mips::JALRC16_MMR6 : Mips::JALR16_MM);
      JalrInst.addOperand(FirstRegOp);
    } else {
      JalrInst.setOpcode(Mips::JALR);
      JalrInst.addOperand(MCOperand::createReg(Mips::RA));
      JalrInst.addOperand(FirstRegOp);
    }
  } else if (Opcode == Mips::JalTwoReg) {
    // jal $rd, $rs => jalr $rd, $rs
    if (IsCpRestoreSet && inMicroMipsMode())
      JalrInst.setOpcode(Mips::JALRS_MM);
    else
      JalrInst.setOpcode(inMicroMipsMode() ? Mips::JALR_MM : Mips::JALR);
    JalrInst.addOperand(FirstRegOp);
    const MCOperand SecondRegOp = Inst.getOperand(1);
    JalrInst.addOperand(SecondRegOp);
  }
  Instructions.push_back(JalrInst);

  // If .set reorder is active and branch instruction has a delay slot,
  // emit a NOP after it.
  const MCInstrDesc &MCID = getInstDesc(JalrInst.getOpcode());
  if (MCID.hasDelaySlot() && AssemblerOptions.back()->isReorder()) {
    createNop(hasShortDelaySlot(JalrInst.getOpcode()), IDLoc, Instructions);
  }

  return false;
}

/// Can the value be represented by a unsigned N-bit value and a shift left?
template <unsigned N> static bool isShiftedUIntAtAnyPosition(uint64_t x) {
  unsigned BitNum = findFirstSet(x);

  return (x == x >> BitNum << BitNum) && isUInt<N>(x >> BitNum);
}

/// Load (or add) an immediate into a register.
///
/// @param ImmValue     The immediate to load.
/// @param DstReg       The register that will hold the immediate.
/// @param SrcReg       A register to add to the immediate or Mips::NoRegister
///                     for a simple initialization.
/// @param Is32BitImm   Is ImmValue 32-bit or 64-bit?
/// @param IsAddress    True if the immediate represents an address. False if it
///                     is an integer.
/// @param IDLoc        Location of the immediate in the source file.
/// @param Instructions The instructions emitted by this expansion.
bool MipsAsmParser::loadImmediate(int64_t ImmValue, unsigned DstReg,
                                  unsigned SrcReg, bool Is32BitImm,
                                  bool IsAddress, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions) {
  if (!Is32BitImm && !isGP64bit()) {
    Error(IDLoc, "instruction requires a 64-bit architecture");
    return true;
  }

  if (Is32BitImm) {
    if (isInt<32>(ImmValue) || isUInt<32>(ImmValue)) {
      // Sign extend up to 64-bit so that the predicates match the hardware
      // behaviour. In particular, isInt<16>(0xffff8000) and similar should be
      // true.
      ImmValue = SignExtend64<32>(ImmValue);
    } else {
      Error(IDLoc, "instruction requires a 32-bit immediate");
      return true;
    }
  }

  unsigned ZeroReg = IsAddress ? ABI.GetNullPtr() : ABI.GetZeroReg();
  unsigned AdduOp = !Is32BitImm ? Mips::DADDu : Mips::ADDu;

  bool UseSrcReg = false;
  if (SrcReg != Mips::NoRegister)
    UseSrcReg = true;

  unsigned TmpReg = DstReg;
  if (UseSrcReg &&
      getContext().getRegisterInfo()->isSuperOrSubRegisterEq(DstReg, SrcReg)) {
    // At this point we need AT to perform the expansions and we exit if it is
    // not available.
    unsigned ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;
    TmpReg = ATReg;
  }

  if (isInt<16>(ImmValue)) {
    if (!UseSrcReg)
      SrcReg = ZeroReg;

    // This doesn't quite follow the usual ABI expectations for N32 but matches
    // traditional assembler behaviour. N32 would normally use addiu for both
    // integers and addresses.
    if (IsAddress && !Is32BitImm) {
      emitRRI(Mips::DADDiu, DstReg, SrcReg, ImmValue, IDLoc, Instructions);
      return false;
    }

    emitRRI(Mips::ADDiu, DstReg, SrcReg, ImmValue, IDLoc, Instructions);
    return false;
  }

  if (isUInt<16>(ImmValue)) {
    unsigned TmpReg = DstReg;
    if (SrcReg == DstReg) {
      TmpReg = getATReg(IDLoc);
      if (!TmpReg)
        return true;
    }

    emitRRI(Mips::ORi, TmpReg, ZeroReg, ImmValue, IDLoc, Instructions);
    if (UseSrcReg)
      emitRRR(ABI.GetPtrAdduOp(), DstReg, TmpReg, SrcReg, IDLoc, Instructions);
    return false;
  }

  if (isInt<32>(ImmValue) || isUInt<32>(ImmValue)) {
    warnIfNoMacro(IDLoc);

    uint16_t Bits31To16 = (ImmValue >> 16) & 0xffff;
    uint16_t Bits15To0 = ImmValue & 0xffff;

    if (!Is32BitImm && !isInt<32>(ImmValue)) {
      // Traditional behaviour seems to special case this particular value. It's
      // not clear why other masks are handled differently.
      if (ImmValue == 0xffffffff) {
        emitRI(Mips::LUi, TmpReg, 0xffff, IDLoc, Instructions);
        emitRRI(Mips::DSRL32, TmpReg, TmpReg, 0, IDLoc, Instructions);
        if (UseSrcReg)
          emitRRR(AdduOp, DstReg, TmpReg, SrcReg, IDLoc, Instructions);
        return false;
      }

      // Expand to an ORi instead of a LUi to avoid sign-extending into the
      // upper 32 bits.
      emitRRI(Mips::ORi, TmpReg, ZeroReg, Bits31To16, IDLoc, Instructions);
      emitRRI(Mips::DSLL, TmpReg, TmpReg, 16, IDLoc, Instructions);
      if (Bits15To0)
        emitRRI(Mips::ORi, TmpReg, TmpReg, Bits15To0, IDLoc, Instructions);
      if (UseSrcReg)
        emitRRR(AdduOp, DstReg, TmpReg, SrcReg, IDLoc, Instructions);
      return false;
    }

    emitRI(Mips::LUi, TmpReg, Bits31To16, IDLoc, Instructions);
    if (Bits15To0)
      emitRRI(Mips::ORi, TmpReg, TmpReg, Bits15To0, IDLoc, Instructions);
    if (UseSrcReg)
      emitRRR(AdduOp, DstReg, TmpReg, SrcReg, IDLoc, Instructions);
    return false;
  }

  if (isShiftedUIntAtAnyPosition<16>(ImmValue)) {
    if (Is32BitImm) {
      Error(IDLoc, "instruction requires a 32-bit immediate");
      return true;
    }

    // Traditionally, these immediates are shifted as little as possible and as
    // such we align the most significant bit to bit 15 of our temporary.
    unsigned FirstSet = findFirstSet((uint64_t)ImmValue);
    unsigned LastSet = findLastSet((uint64_t)ImmValue);
    unsigned ShiftAmount = FirstSet - (15 - (LastSet - FirstSet));
    uint16_t Bits = (ImmValue >> ShiftAmount) & 0xffff;
    emitRRI(Mips::ORi, TmpReg, ZeroReg, Bits, IDLoc, Instructions);
    emitRRI(Mips::DSLL, TmpReg, TmpReg, ShiftAmount, IDLoc, Instructions);

    if (UseSrcReg)
      emitRRR(AdduOp, DstReg, TmpReg, SrcReg, IDLoc, Instructions);

    return false;
  }

  warnIfNoMacro(IDLoc);

  // The remaining case is packed with a sequence of dsll and ori with zeros
  // being omitted and any neighbouring dsll's being coalesced.
  // The highest 32-bit's are equivalent to a 32-bit immediate load.

  // Load bits 32-63 of ImmValue into bits 0-31 of the temporary register.
  if (loadImmediate(ImmValue >> 32, TmpReg, Mips::NoRegister, true, false,
                    IDLoc, Instructions))
    return false;

  // Shift and accumulate into the register. If a 16-bit chunk is zero, then
  // skip it and defer the shift to the next chunk.
  unsigned ShiftCarriedForwards = 16;
  for (int BitNum = 16; BitNum >= 0; BitNum -= 16) {
    uint16_t ImmChunk = (ImmValue >> BitNum) & 0xffff;

    if (ImmChunk != 0) {
      emitAppropriateDSLL(TmpReg, TmpReg, ShiftCarriedForwards, IDLoc,
                          Instructions);
      emitRRI(Mips::ORi, TmpReg, TmpReg, ImmChunk, IDLoc, Instructions);
      ShiftCarriedForwards = 0;
    }

    ShiftCarriedForwards += 16;
  }
  ShiftCarriedForwards -= 16;

  // Finish any remaining shifts left by trailing zeros.
  if (ShiftCarriedForwards)
    emitAppropriateDSLL(TmpReg, TmpReg, ShiftCarriedForwards, IDLoc,
                        Instructions);

  if (UseSrcReg)
    emitRRR(AdduOp, DstReg, TmpReg, SrcReg, IDLoc, Instructions);

  return false;
}

bool MipsAsmParser::expandLoadImm(MCInst &Inst, bool Is32BitImm, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions) {
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");

  if (loadImmediate(ImmOp.getImm(), DstRegOp.getReg(), Mips::NoRegister,
                    Is32BitImm, false, IDLoc, Instructions))
    return true;

  return false;
}

bool MipsAsmParser::expandLoadAddress(unsigned DstReg, unsigned BaseReg,
                                      const MCOperand &Offset,
                                      bool Is32BitAddress, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions) {
  // la can't produce a usable address when addresses are 64-bit.
  if (Is32BitAddress && ABI.ArePtrs64bit()) {
    // FIXME: Demote this to a warning and continue as if we had 'dla' instead.
    //        We currently can't do this because we depend on the equality
    //        operator and N64 can end up with a GPR32/GPR64 mismatch.
    Error(IDLoc, "la used to load 64-bit address");
    // Continue as if we had 'dla' instead.
    Is32BitAddress = false;
  }

  // dla requires 64-bit addresses.
  if (!Is32BitAddress && !hasMips3()) {
    Error(IDLoc, "instruction requires a 64-bit architecture");
    return true;
  }

  if (!Offset.isImm())
    return loadAndAddSymbolAddress(Offset.getExpr(), DstReg, BaseReg,
                                   Is32BitAddress, IDLoc, Instructions);

  if (!ABI.ArePtrs64bit()) {
    // Continue as if we had 'la' whether we had 'la' or 'dla'.
    Is32BitAddress = true;
  }

  return loadImmediate(Offset.getImm(), DstReg, BaseReg, Is32BitAddress, true,
                       IDLoc, Instructions);
}

bool MipsAsmParser::loadAndAddSymbolAddress(
    const MCExpr *SymExpr, unsigned DstReg, unsigned SrcReg, bool Is32BitSym,
    SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  warnIfNoMacro(IDLoc);

  const MCExpr *Symbol = cast<MCExpr>(SymExpr);
  const MipsMCExpr *HiExpr = MipsMCExpr::create(
      MCSymbolRefExpr::VK_Mips_ABS_HI, Symbol, getContext());
  const MipsMCExpr *LoExpr = MipsMCExpr::create(
      MCSymbolRefExpr::VK_Mips_ABS_LO, Symbol, getContext());

  bool UseSrcReg = SrcReg != Mips::NoRegister;

  // This is the 64-bit symbol address expansion.
  if (ABI.ArePtrs64bit() && isGP64bit()) {
    // We always need AT for the 64-bit expansion.
    // If it is not available we exit.
    unsigned ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;

    const MipsMCExpr *HighestExpr = MipsMCExpr::create(
        MCSymbolRefExpr::VK_Mips_HIGHEST, Symbol, getContext());
    const MipsMCExpr *HigherExpr = MipsMCExpr::create(
        MCSymbolRefExpr::VK_Mips_HIGHER, Symbol, getContext());

    if (UseSrcReg &&
        getContext().getRegisterInfo()->isSuperOrSubRegisterEq(DstReg,
                                                               SrcReg)) {
      // If $rs is the same as $rd:
      // (d)la $rd, sym($rd) => lui    $at, %highest(sym)
      //                        daddiu $at, $at, %higher(sym)
      //                        dsll   $at, $at, 16
      //                        daddiu $at, $at, %hi(sym)
      //                        dsll   $at, $at, 16
      //                        daddiu $at, $at, %lo(sym)
      //                        daddu  $rd, $at, $rd
      emitRX(Mips::LUi, ATReg, MCOperand::createExpr(HighestExpr), IDLoc,
             Instructions);
      emitRRX(Mips::DADDiu, ATReg, ATReg, MCOperand::createExpr(HigherExpr),
              IDLoc, Instructions);
      emitRRI(Mips::DSLL, ATReg, ATReg, 16, IDLoc, Instructions);
      emitRRX(Mips::DADDiu, ATReg, ATReg, MCOperand::createExpr(HiExpr), IDLoc,
              Instructions);
      emitRRI(Mips::DSLL, ATReg, ATReg, 16, IDLoc, Instructions);
      emitRRX(Mips::DADDiu, ATReg, ATReg, MCOperand::createExpr(LoExpr), IDLoc,
              Instructions);
      emitRRR(Mips::DADDu, DstReg, ATReg, SrcReg, IDLoc, Instructions);

      return false;
    }

    // Otherwise, if the $rs is different from $rd or if $rs isn't specified:
    // (d)la $rd, sym/sym($rs) => lui    $rd, %highest(sym)
    //                            lui    $at, %hi(sym)
    //                            daddiu $rd, $rd, %higher(sym)
    //                            daddiu $at, $at, %lo(sym)
    //                            dsll32 $rd, $rd, 0
    //                            daddu  $rd, $rd, $at
    //                            (daddu  $rd, $rd, $rs)
    emitRX(Mips::LUi, DstReg, MCOperand::createExpr(HighestExpr), IDLoc,
           Instructions);
    emitRX(Mips::LUi, ATReg, MCOperand::createExpr(HiExpr), IDLoc,
           Instructions);
    emitRRX(Mips::DADDiu, DstReg, DstReg, MCOperand::createExpr(HigherExpr),
            IDLoc, Instructions);
    emitRRX(Mips::DADDiu, ATReg, ATReg, MCOperand::createExpr(LoExpr), IDLoc,
            Instructions);
    emitRRI(Mips::DSLL32, DstReg, DstReg, 0, IDLoc, Instructions);
    emitRRR(Mips::DADDu, DstReg, DstReg, ATReg, IDLoc, Instructions);
    if (UseSrcReg)
      emitRRR(Mips::DADDu, DstReg, DstReg, SrcReg, IDLoc, Instructions);

    return false;
  }

  // And now, the 32-bit symbol address expansion:
  // If $rs is the same as $rd:
  // (d)la $rd, sym($rd)     => lui   $at, %hi(sym)
  //                            ori   $at, $at, %lo(sym)
  //                            addu  $rd, $at, $rd
  // Otherwise, if the $rs is different from $rd or if $rs isn't specified:
  // (d)la $rd, sym/sym($rs) => lui   $rd, %hi(sym)
  //                            ori   $rd, $rd, %lo(sym)
  //                            (addu $rd, $rd, $rs)
  unsigned TmpReg = DstReg;
  if (UseSrcReg &&
      getContext().getRegisterInfo()->isSuperOrSubRegisterEq(DstReg, SrcReg)) {
    // If $rs is the same as $rd, we need to use AT.
    // If it is not available we exit.
    unsigned ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;
    TmpReg = ATReg;
  }

  emitRX(Mips::LUi, TmpReg, MCOperand::createExpr(HiExpr), IDLoc, Instructions);
  emitRRX(Mips::ADDiu, TmpReg, TmpReg, MCOperand::createExpr(LoExpr), IDLoc,
          Instructions);

  if (UseSrcReg)
    emitRRR(Mips::ADDu, DstReg, TmpReg, SrcReg, IDLoc, Instructions);
  else
    assert(
        getContext().getRegisterInfo()->isSuperOrSubRegisterEq(DstReg, TmpReg));

  return false;
}

bool MipsAsmParser::expandUncondBranchMMPseudo(
    MCInst &Inst, SMLoc IDLoc, SmallVectorImpl<MCInst> &Instructions) {
  assert(getInstDesc(Inst.getOpcode()).getNumOperands() == 1 &&
         "unexpected number of operands");

  MCOperand Offset = Inst.getOperand(0);
  if (Offset.isExpr()) {
    Inst.clear();
    Inst.setOpcode(Mips::BEQ_MM);
    Inst.addOperand(MCOperand::createReg(Mips::ZERO));
    Inst.addOperand(MCOperand::createReg(Mips::ZERO));
    Inst.addOperand(MCOperand::createExpr(Offset.getExpr()));
  } else {
    assert(Offset.isImm() && "expected immediate operand kind");
    if (isInt<11>(Offset.getImm())) {
      // If offset fits into 11 bits then this instruction becomes microMIPS
      // 16-bit unconditional branch instruction.
      if (inMicroMipsMode())
        Inst.setOpcode(hasMips32r6() ? Mips::BC16_MMR6 : Mips::B16_MM);
    } else {
      if (!isInt<17>(Offset.getImm()))
        Error(IDLoc, "branch target out of range");
      if (OffsetToAlignment(Offset.getImm(), 1LL << 1))
        Error(IDLoc, "branch to misaligned address");
      Inst.clear();
      Inst.setOpcode(Mips::BEQ_MM);
      Inst.addOperand(MCOperand::createReg(Mips::ZERO));
      Inst.addOperand(MCOperand::createReg(Mips::ZERO));
      Inst.addOperand(MCOperand::createImm(Offset.getImm()));
    }
  }
  Instructions.push_back(Inst);

  // If .set reorder is active and branch instruction has a delay slot,
  // emit a NOP after it.
  const MCInstrDesc &MCID = getInstDesc(Inst.getOpcode());
  if (MCID.hasDelaySlot() && AssemblerOptions.back()->isReorder())
    createNop(true, IDLoc, Instructions);

  return false;
}

bool MipsAsmParser::expandBranchImm(MCInst &Inst, SMLoc IDLoc,
                                    SmallVectorImpl<MCInst> &Instructions) {
  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");

  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");

  const MCOperand &MemOffsetOp = Inst.getOperand(2);
  assert((MemOffsetOp.isImm() || MemOffsetOp.isExpr()) &&
         "expected immediate or expression operand");

  unsigned OpCode = 0;
  switch(Inst.getOpcode()) {
    case Mips::BneImm:
      OpCode = Mips::BNE;
      break;
    case Mips::BeqImm:
      OpCode = Mips::BEQ;
      break;
    default:
      llvm_unreachable("Unknown immediate branch pseudo-instruction.");
      break;
  }

  int64_t ImmValue = ImmOp.getImm();
  if (ImmValue == 0)
    emitRRX(OpCode, DstRegOp.getReg(), Mips::ZERO, MemOffsetOp, IDLoc,
            Instructions);
  else {
    warnIfNoMacro(IDLoc);

    unsigned ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;

    if (loadImmediate(ImmValue, ATReg, Mips::NoRegister, !isGP64bit(), true,
                      IDLoc, Instructions))
      return true;

    emitRRX(OpCode, DstRegOp.getReg(), ATReg, MemOffsetOp, IDLoc, Instructions);
  }
  return false;
}

void MipsAsmParser::expandMemInst(MCInst &Inst, SMLoc IDLoc,
                                  SmallVectorImpl<MCInst> &Instructions,
                                  bool isLoad, bool isImmOpnd) {
  MCOperand HiOperand, LoOperand;
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
    unsigned ImmOffset = Inst.getOperand(2).getImm();
    unsigned LoOffset = ImmOffset & 0x0000ffff;
    unsigned HiOffset = (ImmOffset & 0xffff0000) >> 16;
    // If msb of LoOffset is 1(negative number) we must increment HiOffset.
    if (LoOffset & 0x8000)
      HiOffset++;
    LoOperand = MCOperand::createImm(LoOffset);
    HiOperand = MCOperand::createImm(HiOffset);
  } else {
    const MCExpr *ExprOffset = Inst.getOperand(2).getExpr();
    LoOperand = MCOperand::createExpr(evaluateRelocExpr(ExprOffset, "lo"));
    HiOperand = MCOperand::createExpr(evaluateRelocExpr(ExprOffset, "hi"));
  }
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
    // At this point we need AT to perform the expansions and we exit if it is
    // not available.
    TmpRegNum = getATReg(IDLoc);
    if (!TmpRegNum)
      return;
  }

  emitRX(Mips::LUi, TmpRegNum, HiOperand, IDLoc, Instructions);
  // Add temp register to base.
  if (BaseRegNum != Mips::ZERO)
    emitRRR(Mips::ADDu, TmpRegNum, TmpRegNum, BaseRegNum, IDLoc, Instructions);
  // And finally, create original instruction with low part
  // of offset and new base.
  emitRRX(Inst.getOpcode(), RegOpNum, TmpRegNum, LoOperand, IDLoc, Instructions);
}

bool
MipsAsmParser::expandLoadStoreMultiple(MCInst &Inst, SMLoc IDLoc,
                                       SmallVectorImpl<MCInst> &Instructions) {
  unsigned OpNum = Inst.getNumOperands();
  unsigned Opcode = Inst.getOpcode();
  unsigned NewOpcode = Opcode == Mips::SWM_MM ? Mips::SWM32_MM : Mips::LWM32_MM;

  assert (Inst.getOperand(OpNum - 1).isImm() &&
          Inst.getOperand(OpNum - 2).isReg() &&
          Inst.getOperand(OpNum - 3).isReg() && "Invalid instruction operand.");

  if (OpNum < 8 && Inst.getOperand(OpNum - 1).getImm() <= 60 &&
      Inst.getOperand(OpNum - 1).getImm() >= 0 &&
      (Inst.getOperand(OpNum - 2).getReg() == Mips::SP ||
       Inst.getOperand(OpNum - 2).getReg() == Mips::SP_64) &&
      (Inst.getOperand(OpNum - 3).getReg() == Mips::RA ||
       Inst.getOperand(OpNum - 3).getReg() == Mips::RA_64)) {
    // It can be implemented as SWM16 or LWM16 instruction.
    if (inMicroMipsMode() && hasMips32r6())
      NewOpcode = Opcode == Mips::SWM_MM ? Mips::SWM16_MMR6 : Mips::LWM16_MMR6;
    else
      NewOpcode = Opcode == Mips::SWM_MM ? Mips::SWM16_MM : Mips::LWM16_MM;
  }

  Inst.setOpcode(NewOpcode);
  Instructions.push_back(Inst);
  return false;
}

bool MipsAsmParser::expandCondBranches(MCInst &Inst, SMLoc IDLoc,
                                       SmallVectorImpl<MCInst> &Instructions) {
  bool EmittedNoMacroWarning = false;
  unsigned PseudoOpcode = Inst.getOpcode();
  unsigned SrcReg = Inst.getOperand(0).getReg();
  const MCOperand &TrgOp = Inst.getOperand(1);
  const MCExpr *OffsetExpr = Inst.getOperand(2).getExpr();

  unsigned ZeroSrcOpcode, ZeroTrgOpcode;
  bool ReverseOrderSLT, IsUnsigned, IsLikely, AcceptsEquality;

  unsigned TrgReg;
  if (TrgOp.isReg())
    TrgReg = TrgOp.getReg();
  else if (TrgOp.isImm()) {
    warnIfNoMacro(IDLoc);
    EmittedNoMacroWarning = true;

    TrgReg = getATReg(IDLoc);
    if (!TrgReg)
      return true;

    switch(PseudoOpcode) {
    default:
      llvm_unreachable("unknown opcode for branch pseudo-instruction");
    case Mips::BLTImmMacro:
      PseudoOpcode = Mips::BLT;
      break;
    case Mips::BLEImmMacro:
      PseudoOpcode = Mips::BLE;
      break;
    case Mips::BGEImmMacro:
      PseudoOpcode = Mips::BGE;
      break;
    case Mips::BGTImmMacro:
      PseudoOpcode = Mips::BGT;
      break;
    case Mips::BLTUImmMacro:
      PseudoOpcode = Mips::BLTU;
      break;
    case Mips::BLEUImmMacro:
      PseudoOpcode = Mips::BLEU;
      break;
    case Mips::BGEUImmMacro:
      PseudoOpcode = Mips::BGEU;
      break;
    case Mips::BGTUImmMacro:
      PseudoOpcode = Mips::BGTU;
      break;
    case Mips::BLTLImmMacro:
      PseudoOpcode = Mips::BLTL;
      break;
    case Mips::BLELImmMacro:
      PseudoOpcode = Mips::BLEL;
      break;
    case Mips::BGELImmMacro:
      PseudoOpcode = Mips::BGEL;
      break;
    case Mips::BGTLImmMacro:
      PseudoOpcode = Mips::BGTL;
      break;
    case Mips::BLTULImmMacro:
      PseudoOpcode = Mips::BLTUL;
      break;
    case Mips::BLEULImmMacro:
      PseudoOpcode = Mips::BLEUL;
      break;
    case Mips::BGEULImmMacro:
      PseudoOpcode = Mips::BGEUL;
      break;
    case Mips::BGTULImmMacro:
      PseudoOpcode = Mips::BGTUL;
      break;
    }

    if (loadImmediate(TrgOp.getImm(), TrgReg, Mips::NoRegister, !isGP64bit(),
                      false, IDLoc, Instructions))
      return true;
  }

  switch (PseudoOpcode) {
  case Mips::BLT:
  case Mips::BLTU:
  case Mips::BLTL:
  case Mips::BLTUL:
    AcceptsEquality = false;
    ReverseOrderSLT = false;
    IsUnsigned = ((PseudoOpcode == Mips::BLTU) || (PseudoOpcode == Mips::BLTUL));
    IsLikely = ((PseudoOpcode == Mips::BLTL) || (PseudoOpcode == Mips::BLTUL));
    ZeroSrcOpcode = Mips::BGTZ;
    ZeroTrgOpcode = Mips::BLTZ;
    break;
  case Mips::BLE:
  case Mips::BLEU:
  case Mips::BLEL:
  case Mips::BLEUL:
    AcceptsEquality = true;
    ReverseOrderSLT = true;
    IsUnsigned = ((PseudoOpcode == Mips::BLEU) || (PseudoOpcode == Mips::BLEUL));
    IsLikely = ((PseudoOpcode == Mips::BLEL) || (PseudoOpcode == Mips::BLEUL));
    ZeroSrcOpcode = Mips::BGEZ;
    ZeroTrgOpcode = Mips::BLEZ;
    break;
  case Mips::BGE:
  case Mips::BGEU:
  case Mips::BGEL:
  case Mips::BGEUL:
    AcceptsEquality = true;
    ReverseOrderSLT = false;
    IsUnsigned = ((PseudoOpcode == Mips::BGEU) || (PseudoOpcode == Mips::BGEUL));
    IsLikely = ((PseudoOpcode == Mips::BGEL) || (PseudoOpcode == Mips::BGEUL));
    ZeroSrcOpcode = Mips::BLEZ;
    ZeroTrgOpcode = Mips::BGEZ;
    break;
  case Mips::BGT:
  case Mips::BGTU:
  case Mips::BGTL:
  case Mips::BGTUL:
    AcceptsEquality = false;
    ReverseOrderSLT = true;
    IsUnsigned = ((PseudoOpcode == Mips::BGTU) || (PseudoOpcode == Mips::BGTUL));
    IsLikely = ((PseudoOpcode == Mips::BGTL) || (PseudoOpcode == Mips::BGTUL));
    ZeroSrcOpcode = Mips::BLTZ;
    ZeroTrgOpcode = Mips::BGTZ;
    break;
  default:
    llvm_unreachable("unknown opcode for branch pseudo-instruction");
  }

  bool IsTrgRegZero = (TrgReg == Mips::ZERO);
  bool IsSrcRegZero = (SrcReg == Mips::ZERO);
  if (IsSrcRegZero && IsTrgRegZero) {
    // FIXME: All of these Opcode-specific if's are needed for compatibility
    // with GAS' behaviour. However, they may not generate the most efficient
    // code in some circumstances.
    if (PseudoOpcode == Mips::BLT) {
      emitRX(Mips::BLTZ, Mips::ZERO, MCOperand::createExpr(OffsetExpr), IDLoc,
             Instructions);
      return false;
    }
    if (PseudoOpcode == Mips::BLE) {
      emitRX(Mips::BLEZ, Mips::ZERO, MCOperand::createExpr(OffsetExpr), IDLoc,
             Instructions);
      Warning(IDLoc, "branch is always taken");
      return false;
    }
    if (PseudoOpcode == Mips::BGE) {
      emitRX(Mips::BGEZ, Mips::ZERO, MCOperand::createExpr(OffsetExpr), IDLoc,
             Instructions);
      Warning(IDLoc, "branch is always taken");
      return false;
    }
    if (PseudoOpcode == Mips::BGT) {
      emitRX(Mips::BGTZ, Mips::ZERO, MCOperand::createExpr(OffsetExpr), IDLoc,
             Instructions);
      return false;
    }
    if (PseudoOpcode == Mips::BGTU) {
      emitRRX(Mips::BNE, Mips::ZERO, Mips::ZERO,
              MCOperand::createExpr(OffsetExpr), IDLoc, Instructions);
      return false;
    }
    if (AcceptsEquality) {
      // If both registers are $0 and the pseudo-branch accepts equality, it
      // will always be taken, so we emit an unconditional branch.
      emitRRX(Mips::BEQ, Mips::ZERO, Mips::ZERO,
              MCOperand::createExpr(OffsetExpr), IDLoc, Instructions);
      Warning(IDLoc, "branch is always taken");
      return false;
    }
    // If both registers are $0 and the pseudo-branch does not accept
    // equality, it will never be taken, so we don't have to emit anything.
    return false;
  }
  if (IsSrcRegZero || IsTrgRegZero) {
    if ((IsSrcRegZero && PseudoOpcode == Mips::BGTU) ||
        (IsTrgRegZero && PseudoOpcode == Mips::BLTU)) {
      // If the $rs is $0 and the pseudo-branch is BGTU (0 > x) or
      // if the $rt is $0 and the pseudo-branch is BLTU (x < 0),
      // the pseudo-branch will never be taken, so we don't emit anything.
      // This only applies to unsigned pseudo-branches.
      return false;
    }
    if ((IsSrcRegZero && PseudoOpcode == Mips::BLEU) ||
        (IsTrgRegZero && PseudoOpcode == Mips::BGEU)) {
      // If the $rs is $0 and the pseudo-branch is BLEU (0 <= x) or
      // if the $rt is $0 and the pseudo-branch is BGEU (x >= 0),
      // the pseudo-branch will always be taken, so we emit an unconditional
      // branch.
      // This only applies to unsigned pseudo-branches.
      emitRRX(Mips::BEQ, Mips::ZERO, Mips::ZERO,
              MCOperand::createExpr(OffsetExpr), IDLoc, Instructions);
      Warning(IDLoc, "branch is always taken");
      return false;
    }
    if (IsUnsigned) {
      // If the $rs is $0 and the pseudo-branch is BLTU (0 < x) or
      // if the $rt is $0 and the pseudo-branch is BGTU (x > 0),
      // the pseudo-branch will be taken only when the non-zero register is
      // different from 0, so we emit a BNEZ.
      //
      // If the $rs is $0 and the pseudo-branch is BGEU (0 >= x) or
      // if the $rt is $0 and the pseudo-branch is BLEU (x <= 0),
      // the pseudo-branch will be taken only when the non-zero register is
      // equal to 0, so we emit a BEQZ.
      //
      // Because only BLEU and BGEU branch on equality, we can use the
      // AcceptsEquality variable to decide when to emit the BEQZ.
      emitRRX(AcceptsEquality ? Mips::BEQ : Mips::BNE,
              IsSrcRegZero ? TrgReg : SrcReg, Mips::ZERO,
              MCOperand::createExpr(OffsetExpr), IDLoc, Instructions);
      return false;
    }
    // If we have a signed pseudo-branch and one of the registers is $0,
    // we can use an appropriate compare-to-zero branch. We select which one
    // to use in the switch statement above.
    emitRX(IsSrcRegZero ? ZeroSrcOpcode : ZeroTrgOpcode,
           IsSrcRegZero ? TrgReg : SrcReg, MCOperand::createExpr(OffsetExpr),
           IDLoc, Instructions);
    return false;
  }

  // If neither the SrcReg nor the TrgReg are $0, we need AT to perform the
  // expansions. If it is not available, we return.
  unsigned ATRegNum = getATReg(IDLoc);
  if (!ATRegNum)
    return true;

  if (!EmittedNoMacroWarning)
    warnIfNoMacro(IDLoc);

  // SLT fits well with 2 of our 4 pseudo-branches:
  //   BLT, where $rs < $rt, translates into "slt $at, $rs, $rt" and
  //   BGT, where $rs > $rt, translates into "slt $at, $rt, $rs".
  // If the result of the SLT is 1, we branch, and if it's 0, we don't.
  // This is accomplished by using a BNEZ with the result of the SLT.
  //
  // The other 2 pseudo-branches are opposites of the above 2 (BGE with BLT
  // and BLE with BGT), so we change the BNEZ into a a BEQZ.
  // Because only BGE and BLE branch on equality, we can use the
  // AcceptsEquality variable to decide when to emit the BEQZ.
  // Note that the order of the SLT arguments doesn't change between
  // opposites.
  //
  // The same applies to the unsigned variants, except that SLTu is used
  // instead of SLT.
  emitRRR(IsUnsigned ? Mips::SLTu : Mips::SLT, ATRegNum,
          ReverseOrderSLT ? TrgReg : SrcReg, ReverseOrderSLT ? SrcReg : TrgReg,
          IDLoc, Instructions);

  emitRRX(IsLikely ? (AcceptsEquality ? Mips::BEQL : Mips::BNEL)
                   : (AcceptsEquality ? Mips::BEQ : Mips::BNE),
          ATRegNum, Mips::ZERO, MCOperand::createExpr(OffsetExpr), IDLoc,
          Instructions);
  return false;
}

bool MipsAsmParser::expandDiv(MCInst &Inst, SMLoc IDLoc,
                              SmallVectorImpl<MCInst> &Instructions,
                              const bool IsMips64, const bool Signed) {
  if (hasMips32r6()) {
    Error(IDLoc, "instruction not supported on mips32r6 or mips64r6");
    return false;
  }

  warnIfNoMacro(IDLoc);

  const MCOperand &RsRegOp = Inst.getOperand(0);
  assert(RsRegOp.isReg() && "expected register operand kind");
  unsigned RsReg = RsRegOp.getReg();

  const MCOperand &RtRegOp = Inst.getOperand(1);
  assert(RtRegOp.isReg() && "expected register operand kind");
  unsigned RtReg = RtRegOp.getReg();
  unsigned DivOp;
  unsigned ZeroReg;

  if (IsMips64) {
    DivOp = Signed ? Mips::DSDIV : Mips::DUDIV;
    ZeroReg = Mips::ZERO_64;
  } else {
    DivOp = Signed ? Mips::SDIV : Mips::UDIV;
    ZeroReg = Mips::ZERO;
  }

  bool UseTraps = useTraps();

  if (RsReg == Mips::ZERO || RsReg == Mips::ZERO_64) {
    if (RtReg == Mips::ZERO || RtReg == Mips::ZERO_64)
      Warning(IDLoc, "dividing zero by zero");
    if (IsMips64) {
      if (Signed && (RtReg == Mips::ZERO || RtReg == Mips::ZERO_64)) {
        if (UseTraps) {
          emitRRI(Mips::TEQ, RtReg, ZeroReg, 0x7, IDLoc, Instructions);
          return false;
        }

        emitII(Mips::BREAK, 0x7, 0, IDLoc, Instructions);
        return false;
      }
    } else {
      emitRR(DivOp, RsReg, RtReg, IDLoc, Instructions);
      return false;
    }
  }

  if (RtReg == Mips::ZERO || RtReg == Mips::ZERO_64) {
    Warning(IDLoc, "division by zero");
    if (Signed) {
      if (UseTraps) {
        emitRRI(Mips::TEQ, RtReg, ZeroReg, 0x7, IDLoc, Instructions);
        return false;
      }

      emitII(Mips::BREAK, 0x7, 0, IDLoc, Instructions);
      return false;
    }
  }

  // FIXME: The values for these two BranchTarget variables may be different in
  // micromips. These magic numbers need to be removed.
  unsigned BranchTargetNoTraps;
  unsigned BranchTarget;

  if (UseTraps) {
    BranchTarget = IsMips64 ? 12 : 8;
    emitRRI(Mips::TEQ, RtReg, ZeroReg, 0x7, IDLoc, Instructions);
  } else {
    BranchTarget = IsMips64 ? 20 : 16;
    BranchTargetNoTraps = 8;
    // Branch to the li instruction.
    emitRRI(Mips::BNE, RtReg, ZeroReg, BranchTargetNoTraps, IDLoc,
            Instructions);
  }

  emitRR(DivOp, RsReg, RtReg, IDLoc, Instructions);

  if (!UseTraps)
    emitII(Mips::BREAK, 0x7, 0, IDLoc, Instructions);

  if (!Signed) {
    emitR(Mips::MFLO, RsReg, IDLoc, Instructions);
    return false;
  }

  unsigned ATReg = getATReg(IDLoc);
  if (!ATReg)
    return true;

  emitRRI(Mips::ADDiu, ATReg, ZeroReg, -1, IDLoc, Instructions);
  if (IsMips64) {
    // Branch to the mflo instruction.
    emitRRI(Mips::BNE, RtReg, ATReg, BranchTarget, IDLoc, Instructions);
    emitRRI(Mips::ADDiu, ATReg, ZeroReg, 1, IDLoc, Instructions);
    emitRRI(Mips::DSLL32, ATReg, ATReg, 0x1f, IDLoc, Instructions);
  } else {
    // Branch to the mflo instruction.
    emitRRI(Mips::BNE, RtReg, ATReg, BranchTarget, IDLoc, Instructions);
    emitRI(Mips::LUi, ATReg, (uint16_t)0x8000, IDLoc, Instructions);
  }

  if (UseTraps)
    emitRRI(Mips::TEQ, RsReg, ATReg, 0x6, IDLoc, Instructions);
  else {
    // Branch to the mflo instruction.
    emitRRI(Mips::BNE, RsReg, ATReg, BranchTargetNoTraps, IDLoc, Instructions);
    emitRRI(Mips::SLL, ZeroReg, ZeroReg, 0, IDLoc, Instructions);
    emitII(Mips::BREAK, 0x6, 0, IDLoc, Instructions);
  }
  emitR(Mips::MFLO, RsReg, IDLoc, Instructions);
  return false;
}

bool MipsAsmParser::expandTrunc(MCInst &Inst, bool IsDouble, bool Is64FPU,
                                SMLoc IDLoc,
                                SmallVectorImpl<MCInst> &Instructions) {

  assert(Inst.getNumOperands() == 3 && "Invalid operand count");
  assert(Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg() &&
         Inst.getOperand(2).isReg() && "Invalid instruction operand.");

  unsigned FirstReg = Inst.getOperand(0).getReg();
  unsigned SecondReg = Inst.getOperand(1).getReg();
  unsigned ThirdReg = Inst.getOperand(2).getReg();

  if (hasMips1() && !hasMips2()) {
    unsigned ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;
    emitRR(Mips::CFC1, ThirdReg, Mips::RA, IDLoc, Instructions);
    emitRR(Mips::CFC1, ThirdReg, Mips::RA, IDLoc, Instructions);
    createNop(false, IDLoc, Instructions);
    emitRRI(Mips::ORi, ATReg, ThirdReg, 0x3, IDLoc, Instructions);
    emitRRI(Mips::XORi, ATReg, ATReg, 0x2, IDLoc, Instructions);
    emitRR(Mips::CTC1, Mips::RA, ATReg, IDLoc, Instructions);
    createNop(false, IDLoc, Instructions);
    emitRR(IsDouble ? (Is64FPU ? Mips::CVT_W_D64 : Mips::CVT_W_D32)
                    : Mips::CVT_W_S,
           FirstReg, SecondReg, IDLoc, Instructions);
    emitRR(Mips::CTC1, Mips::RA, ThirdReg, IDLoc, Instructions);
    createNop(false, IDLoc, Instructions);
    return false;
  }

  emitRR(IsDouble ? (Is64FPU ? Mips::TRUNC_W_D64 : Mips::TRUNC_W_D32)
                  : Mips::TRUNC_W_S,
         FirstReg, SecondReg, IDLoc, Instructions);

  return false;
}

bool MipsAsmParser::expandUlh(MCInst &Inst, bool Signed, SMLoc IDLoc,
                              SmallVectorImpl<MCInst> &Instructions) {
  if (hasMips32r6() || hasMips64r6()) {
    Error(IDLoc, "instruction not supported on mips32r6 or mips64r6");
    return false;
  }

  warnIfNoMacro(IDLoc);

  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");

  const MCOperand &SrcRegOp = Inst.getOperand(1);
  assert(SrcRegOp.isReg() && "expected register operand kind");

  const MCOperand &OffsetImmOp = Inst.getOperand(2);
  assert(OffsetImmOp.isImm() && "expected immediate operand kind");

  unsigned DstReg = DstRegOp.getReg();
  unsigned SrcReg = SrcRegOp.getReg();
  int64_t OffsetValue = OffsetImmOp.getImm();

  // NOTE: We always need AT for ULHU, as it is always used as the source
  // register for one of the LBu's.
  unsigned ATReg = getATReg(IDLoc);
  if (!ATReg)
    return true;

  // When the value of offset+1 does not fit in 16 bits, we have to load the
  // offset in AT, (D)ADDu the original source register (if there was one), and
  // then use AT as the source register for the 2 generated LBu's.
  bool LoadedOffsetInAT = false;
  if (!isInt<16>(OffsetValue + 1) || !isInt<16>(OffsetValue)) {
    LoadedOffsetInAT = true;

    if (loadImmediate(OffsetValue, ATReg, Mips::NoRegister, !ABI.ArePtrs64bit(),
                      true, IDLoc, Instructions))
      return true;

    // NOTE: We do this (D)ADDu here instead of doing it in loadImmediate()
    // because it will make our output more similar to GAS'. For example,
    // generating an "ori $1, $zero, 32768" followed by an "addu $1, $1, $9",
    // instead of just an "ori $1, $9, 32768".
    // NOTE: If there is no source register specified in the ULHU, the parser
    // will interpret it as $0.
    if (SrcReg != Mips::ZERO && SrcReg != Mips::ZERO_64)
      createAddu(ATReg, ATReg, SrcReg, ABI.ArePtrs64bit(), Instructions);
  }

  unsigned FirstLbuDstReg = LoadedOffsetInAT ? DstReg : ATReg;
  unsigned SecondLbuDstReg = LoadedOffsetInAT ? ATReg : DstReg;
  unsigned LbuSrcReg = LoadedOffsetInAT ? ATReg : SrcReg;

  int64_t FirstLbuOffset = 0, SecondLbuOffset = 0;
  if (isLittle()) {
    FirstLbuOffset = LoadedOffsetInAT ? 1 : (OffsetValue + 1);
    SecondLbuOffset = LoadedOffsetInAT ? 0 : OffsetValue;
  } else {
    FirstLbuOffset = LoadedOffsetInAT ? 0 : OffsetValue;
    SecondLbuOffset = LoadedOffsetInAT ? 1 : (OffsetValue + 1);
  }

  unsigned SllReg = LoadedOffsetInAT ? DstReg : ATReg;

  emitRRI(Signed ? Mips::LB : Mips::LBu, FirstLbuDstReg, LbuSrcReg,
          FirstLbuOffset, IDLoc, Instructions);

  emitRRI(Mips::LBu, SecondLbuDstReg, LbuSrcReg, SecondLbuOffset, IDLoc,
          Instructions);

  emitRRI(Mips::SLL, SllReg, SllReg, 8, IDLoc, Instructions);

  emitRRR(Mips::OR, DstReg, DstReg, ATReg, IDLoc, Instructions);

  return false;
}

bool MipsAsmParser::expandUlw(MCInst &Inst, SMLoc IDLoc,
                              SmallVectorImpl<MCInst> &Instructions) {
  if (hasMips32r6() || hasMips64r6()) {
    Error(IDLoc, "instruction not supported on mips32r6 or mips64r6");
    return false;
  }

  const MCOperand &DstRegOp = Inst.getOperand(0);
  assert(DstRegOp.isReg() && "expected register operand kind");

  const MCOperand &SrcRegOp = Inst.getOperand(1);
  assert(SrcRegOp.isReg() && "expected register operand kind");

  const MCOperand &OffsetImmOp = Inst.getOperand(2);
  assert(OffsetImmOp.isImm() && "expected immediate operand kind");

  unsigned SrcReg = SrcRegOp.getReg();
  int64_t OffsetValue = OffsetImmOp.getImm();
  unsigned ATReg = 0;

  // When the value of offset+3 does not fit in 16 bits, we have to load the
  // offset in AT, (D)ADDu the original source register (if there was one), and
  // then use AT as the source register for the generated LWL and LWR.
  bool LoadedOffsetInAT = false;
  if (!isInt<16>(OffsetValue + 3) || !isInt<16>(OffsetValue)) {
    ATReg = getATReg(IDLoc);
    if (!ATReg)
      return true;
    LoadedOffsetInAT = true;

    warnIfNoMacro(IDLoc);

    if (loadImmediate(OffsetValue, ATReg, Mips::NoRegister, !ABI.ArePtrs64bit(),
                      true, IDLoc, Instructions))
      return true;

    // NOTE: We do this (D)ADDu here instead of doing it in loadImmediate()
    // because it will make our output more similar to GAS'. For example,
    // generating an "ori $1, $zero, 32768" followed by an "addu $1, $1, $9",
    // instead of just an "ori $1, $9, 32768".
    // NOTE: If there is no source register specified in the ULW, the parser
    // will interpret it as $0.
    if (SrcReg != Mips::ZERO && SrcReg != Mips::ZERO_64)
      createAddu(ATReg, ATReg, SrcReg, ABI.ArePtrs64bit(), Instructions);
  }

  unsigned FinalSrcReg = LoadedOffsetInAT ? ATReg : SrcReg;
  int64_t LeftLoadOffset = 0, RightLoadOffset  = 0;
  if (isLittle()) {
    LeftLoadOffset = LoadedOffsetInAT ? 3 : (OffsetValue + 3);
    RightLoadOffset  = LoadedOffsetInAT ? 0 : OffsetValue;
  } else {
    LeftLoadOffset = LoadedOffsetInAT ? 0 : OffsetValue;
    RightLoadOffset  = LoadedOffsetInAT ? 3 : (OffsetValue + 3);
  }

  emitRRI(Mips::LWL, DstRegOp.getReg(), FinalSrcReg, LeftLoadOffset, IDLoc,
          Instructions);

  emitRRI(Mips::LWR, DstRegOp.getReg(), FinalSrcReg, RightLoadOffset, IDLoc,
          Instructions);

  return false;
}

bool MipsAsmParser::expandAliasImmediate(MCInst &Inst, SMLoc IDLoc,
                                         SmallVectorImpl<MCInst> &Instructions) {

  assert (Inst.getNumOperands() == 3 && "Invalid operand count");
  assert (Inst.getOperand(0).isReg() &&
          Inst.getOperand(1).isReg() &&
          Inst.getOperand(2).isImm() && "Invalid instruction operand.");

  unsigned ATReg = Mips::NoRegister;
  unsigned FinalDstReg = Mips::NoRegister;
  unsigned DstReg = Inst.getOperand(0).getReg();
  unsigned SrcReg = Inst.getOperand(1).getReg();
  int64_t ImmValue = Inst.getOperand(2).getImm();

  bool Is32Bit = isInt<32>(ImmValue) || isUInt<32>(ImmValue);

  unsigned FinalOpcode = Inst.getOpcode();

  if (DstReg == SrcReg) {
    ATReg = getATReg(Inst.getLoc());
    if (!ATReg)
      return true;
    FinalDstReg = DstReg;
    DstReg = ATReg;
  }

  if (!loadImmediate(ImmValue, DstReg, Mips::NoRegister, Is32Bit, false, Inst.getLoc(), Instructions)) {
    switch (FinalOpcode) {
    default:
      llvm_unreachable("unimplemented expansion");
    case (Mips::ADDi):
      FinalOpcode = Mips::ADD;
      break;
    case (Mips::ADDiu):
      FinalOpcode = Mips::ADDu;
      break;
    case (Mips::ANDi):
      FinalOpcode = Mips::AND;
      break;
    case (Mips::NORImm):
      FinalOpcode = Mips::NOR;
      break;
    case (Mips::ORi):
      FinalOpcode = Mips::OR;
      break;
    case (Mips::SLTi):
      FinalOpcode = Mips::SLT;
      break;
    case (Mips::SLTiu):
      FinalOpcode = Mips::SLTu;
      break;
    case (Mips::XORi):
      FinalOpcode = Mips::XOR;
      break;
    }

    if (FinalDstReg == Mips::NoRegister)
      emitRRR(FinalOpcode, DstReg, DstReg, SrcReg, IDLoc, Instructions);
    else
      emitRRR(FinalOpcode, FinalDstReg, FinalDstReg, DstReg, IDLoc,
              Instructions);
    return false;
  }
  return true;
}

bool MipsAsmParser::expandRotation(MCInst &Inst, SMLoc IDLoc,
                                   SmallVectorImpl<MCInst> &Instructions) {
  unsigned ATReg = Mips::NoRegister;
  unsigned DReg = Inst.getOperand(0).getReg();
  unsigned SReg = Inst.getOperand(1).getReg();
  unsigned TReg = Inst.getOperand(2).getReg();
  unsigned TmpReg = DReg;

  unsigned FirstShift = Mips::NOP;
  unsigned SecondShift = Mips::NOP;

  if (hasMips32r2()) {

    if (DReg == SReg) {
      TmpReg = getATReg(Inst.getLoc());
      if (!TmpReg)
        return true;
    }

    if (Inst.getOpcode() == Mips::ROL) {
      emitRRR(Mips::SUBu, TmpReg, Mips::ZERO, TReg, Inst.getLoc(), Instructions);
      emitRRR(Mips::ROTRV, DReg, SReg, TmpReg, Inst.getLoc(), Instructions);
      return false;
    }

    if (Inst.getOpcode() == Mips::ROR) {
      emitRRR(Mips::ROTRV, DReg, SReg, TReg, Inst.getLoc(), Instructions);
      return false;
    }

    return true;
  }

  if (hasMips32()) {

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unexpected instruction opcode");
    case Mips::ROL:
      FirstShift = Mips::SRLV;
      SecondShift = Mips::SLLV;
      break;
    case Mips::ROR:
      FirstShift = Mips::SLLV;
      SecondShift = Mips::SRLV;
      break;
    }

    ATReg = getATReg(Inst.getLoc());
    if (!ATReg)
      return true;

    emitRRR(Mips::SUBu, ATReg, Mips::ZERO, TReg, Inst.getLoc(), Instructions);
    emitRRR(FirstShift, ATReg, SReg, ATReg, Inst.getLoc(), Instructions);
    emitRRR(SecondShift, DReg, SReg, TReg, Inst.getLoc(), Instructions);
    emitRRR(Mips::OR, DReg, DReg, ATReg, Inst.getLoc(), Instructions);

    return false;
  }

  return true;
}

bool MipsAsmParser::expandRotationImm(MCInst &Inst, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions) {

  unsigned ATReg = Mips::NoRegister;
  unsigned DReg = Inst.getOperand(0).getReg();
  unsigned SReg = Inst.getOperand(1).getReg();
  int64_t ImmValue = Inst.getOperand(2).getImm();

  unsigned FirstShift = Mips::NOP;
  unsigned SecondShift = Mips::NOP;

  if (hasMips32r2()) {

    if (Inst.getOpcode() == Mips::ROLImm) {
      uint64_t MaxShift = 32;
      uint64_t ShiftValue = ImmValue;
      if (ImmValue != 0)
        ShiftValue = MaxShift - ImmValue;
      emitRRI(Mips::ROTR, DReg, SReg, ShiftValue, Inst.getLoc(), Instructions);
      return false;
    }

    if (Inst.getOpcode() == Mips::RORImm) {
      emitRRI(Mips::ROTR, DReg, SReg, ImmValue, Inst.getLoc(), Instructions);
      return false;
    }

    return true;
  }

  if (hasMips32()) {

    if (ImmValue == 0) {
      emitRRI(Mips::SRL, DReg, SReg, 0, Inst.getLoc(), Instructions);
      return false;
    }

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unexpected instruction opcode");
    case Mips::ROLImm:
      FirstShift = Mips::SLL;
      SecondShift = Mips::SRL;
      break;
    case Mips::RORImm:
      FirstShift = Mips::SRL;
      SecondShift = Mips::SLL;
      break;
    }

    ATReg = getATReg(Inst.getLoc());
    if (!ATReg)
      return true;

    emitRRI(FirstShift, ATReg, SReg, ImmValue, Inst.getLoc(), Instructions);
    emitRRI(SecondShift, DReg, SReg, 32 - ImmValue, Inst.getLoc(), Instructions);
    emitRRR(Mips::OR, DReg, DReg, ATReg, Inst.getLoc(), Instructions);

    return false;
  }

  return true;
}

bool MipsAsmParser::expandDRotation(MCInst &Inst, SMLoc IDLoc,
                                    SmallVectorImpl<MCInst> &Instructions) {

  unsigned ATReg = Mips::NoRegister;
  unsigned DReg = Inst.getOperand(0).getReg();
  unsigned SReg = Inst.getOperand(1).getReg();
  unsigned TReg = Inst.getOperand(2).getReg();
  unsigned TmpReg = DReg;

  unsigned FirstShift = Mips::NOP;
  unsigned SecondShift = Mips::NOP;

  if (hasMips64r2()) {

    if (TmpReg == SReg) {
      TmpReg = getATReg(Inst.getLoc());
      if (!TmpReg)
        return true;
    }

    if (Inst.getOpcode() == Mips::DROL) {
      emitRRR(Mips::DSUBu, TmpReg, Mips::ZERO, TReg, Inst.getLoc(), Instructions);
      emitRRR(Mips::DROTRV, DReg, SReg, TmpReg, Inst.getLoc(), Instructions);
      return false;
    }

    if (Inst.getOpcode() == Mips::DROR) {
      emitRRR(Mips::DROTRV, DReg, SReg, TReg, Inst.getLoc(), Instructions);
      return false;
    }

    return true;
  }

  if (hasMips64()) {

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unexpected instruction opcode");
    case Mips::DROL:
      FirstShift = Mips::DSRLV;
      SecondShift = Mips::DSLLV;
      break;
    case Mips::DROR:
      FirstShift = Mips::DSLLV;
      SecondShift = Mips::DSRLV;
      break;
    }

    ATReg = getATReg(Inst.getLoc());
    if (!ATReg)
      return true;

    emitRRR(Mips::DSUBu, ATReg, Mips::ZERO, TReg, Inst.getLoc(), Instructions);
    emitRRR(FirstShift, ATReg, SReg, ATReg, Inst.getLoc(), Instructions);
    emitRRR(SecondShift, DReg, SReg, TReg, Inst.getLoc(), Instructions);
    emitRRR(Mips::OR, DReg, DReg, ATReg, Inst.getLoc(), Instructions);

    return false;
  }

  return true;
}

bool MipsAsmParser::expandDRotationImm(MCInst &Inst, SMLoc IDLoc,
                                       SmallVectorImpl<MCInst> &Instructions) {

  unsigned ATReg = Mips::NoRegister;
  unsigned DReg = Inst.getOperand(0).getReg();
  unsigned SReg = Inst.getOperand(1).getReg();
  int64_t ImmValue = Inst.getOperand(2).getImm() % 64;

  unsigned FirstShift = Mips::NOP;
  unsigned SecondShift = Mips::NOP;

  MCInst TmpInst;

  if (hasMips64r2()) {

    unsigned FinalOpcode = Mips::NOP;
    if (ImmValue == 0)
      FinalOpcode = Mips::DROTR;
    else if (ImmValue % 32 == 0)
      FinalOpcode = Mips::DROTR32;
    else if ((ImmValue >= 1) && (ImmValue <= 32)) {
      if (Inst.getOpcode() == Mips::DROLImm)
        FinalOpcode = Mips::DROTR32;
      else
        FinalOpcode = Mips::DROTR;
    } else if (ImmValue >= 33) {
      if (Inst.getOpcode() == Mips::DROLImm)
        FinalOpcode = Mips::DROTR;
      else
        FinalOpcode = Mips::DROTR32;
    }

    uint64_t ShiftValue = ImmValue % 32;
    if (Inst.getOpcode() == Mips::DROLImm)
      ShiftValue = (32 - ImmValue % 32) % 32;

    emitRRI(FinalOpcode, DReg, SReg, ShiftValue, Inst.getLoc(), Instructions);

    return false;
  }

  if (hasMips64()) {

    if (ImmValue == 0) {
      emitRRI(Mips::DSRL, DReg, SReg, 0, Inst.getLoc(), Instructions);
      return false;
    }

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unexpected instruction opcode");
    case Mips::DROLImm:
      if ((ImmValue >= 1) && (ImmValue <= 31)) {
        FirstShift = Mips::DSLL;
        SecondShift = Mips::DSRL32;
      }
      if (ImmValue == 32) {
        FirstShift = Mips::DSLL32;
        SecondShift = Mips::DSRL32;
      }
      if ((ImmValue >= 33) && (ImmValue <= 63)) {
        FirstShift = Mips::DSLL32;
        SecondShift = Mips::DSRL;
      }
      break;
    case Mips::DRORImm:
      if ((ImmValue >= 1) && (ImmValue <= 31)) {
        FirstShift = Mips::DSRL;
        SecondShift = Mips::DSLL32;
      }
      if (ImmValue == 32) {
        FirstShift = Mips::DSRL32;
        SecondShift = Mips::DSLL32;
      }
      if ((ImmValue >= 33) && (ImmValue <= 63)) {
        FirstShift = Mips::DSRL32;
        SecondShift = Mips::DSLL;
      }
      break;
    }

    ATReg = getATReg(Inst.getLoc());
    if (!ATReg)
      return true;

    emitRRI(FirstShift, ATReg, SReg, ImmValue % 32, Inst.getLoc(), Instructions);
    emitRRI(SecondShift, DReg, SReg, (32 - ImmValue % 32) % 32, Inst.getLoc(), Instructions);
    emitRRR(Mips::OR, DReg, DReg, ATReg, Inst.getLoc(), Instructions);

    return false;
  }

  return true;
}

bool MipsAsmParser::expandAbs(MCInst &Inst, SMLoc IDLoc,
                              SmallVectorImpl<MCInst> &Instructions) {

  unsigned FirstRegOp = Inst.getOperand(0).getReg();
  unsigned SecondRegOp = Inst.getOperand(1).getReg();

  emitRI(Mips::BGEZ, SecondRegOp, 8, IDLoc, Instructions);
  if (FirstRegOp != SecondRegOp)
    emitRRR(Mips::ADDu, FirstRegOp, SecondRegOp, Mips::ZERO, IDLoc, Instructions);
  else
    createNop(false, IDLoc, Instructions);
  emitRRR(Mips::SUB, FirstRegOp, Mips::ZERO, SecondRegOp, IDLoc, Instructions);

  return false;
}

void MipsAsmParser::createNop(bool hasShortDelaySlot, SMLoc IDLoc,
                              SmallVectorImpl<MCInst> &Instructions) {
  if (hasShortDelaySlot)
    emitRR(Mips::MOVE16_MM, Mips::ZERO, Mips::ZERO, IDLoc, Instructions);
  else
    emitRRI(Mips::SLL, Mips::ZERO, Mips::ZERO, 0, IDLoc, Instructions);
}

void MipsAsmParser::createAddu(unsigned DstReg, unsigned SrcReg,
                               unsigned TrgReg, bool Is64Bit,
                               SmallVectorImpl<MCInst> &Instructions) {
  emitRRR(Is64Bit ? Mips::DADDu : Mips::ADDu, DstReg, SrcReg, TrgReg, SMLoc(),
          Instructions);
}

void MipsAsmParser::createCpRestoreMemOp(
    bool IsLoad, int StackOffset, SMLoc IDLoc,
    SmallVectorImpl<MCInst> &Instructions) {
  // If the offset can not fit into 16 bits, we need to expand.
  if (!isInt<16>(StackOffset)) {
    MCInst MemInst;
    MemInst.setOpcode(IsLoad ? Mips::LW : Mips::SW);
    MemInst.addOperand(MCOperand::createReg(Mips::GP));
    MemInst.addOperand(MCOperand::createReg(Mips::SP));
    MemInst.addOperand(MCOperand::createImm(StackOffset));
    expandMemInst(MemInst, IDLoc, Instructions, IsLoad, true /*HasImmOpnd*/);
    return;
  }

  emitRRI(IsLoad ? Mips::LW : Mips::SW, Mips::GP, Mips::SP, StackOffset, IDLoc,
          Instructions);
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

static SMLoc RefineErrorLoc(const SMLoc Loc, const OperandVector &Operands,
                            uint64_t ErrorInfo) {
  if (ErrorInfo != ~0ULL && ErrorInfo < Operands.size()) {
    SMLoc ErrorLoc = Operands[ErrorInfo]->getStartLoc();
    if (ErrorLoc == SMLoc())
      return Loc;
    return ErrorLoc;
  }
  return Loc;
}

bool MipsAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                            OperandVector &Operands,
                                            MCStreamer &Out,
                                            uint64_t &ErrorInfo,
                                            bool MatchingInlineAsm) {

  MCInst Inst;
  SmallVector<MCInst, 8> Instructions;
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);

  switch (MatchResult) {
  case Match_Success: {
    if (processInstruction(Inst, IDLoc, Instructions))
      return true;
    for (unsigned i = 0; i < Instructions.size(); i++)
      Out.EmitInstruction(Instructions[i], getSTI());
    return false;
  }
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
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
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  case Match_RequiresDifferentSrcAndDst:
    return Error(IDLoc, "source and destination must be different");
  case Match_Immz:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo), "expected '0'");
  case Match_UImm1_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 1-bit unsigned immediate");
  case Match_UImm2_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 2-bit unsigned immediate");
  case Match_UImm2_1:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range 1 .. 4");
  case Match_UImm3_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 3-bit unsigned immediate");
  case Match_UImm4_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 4-bit unsigned immediate");
  case Match_SImm4_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 4-bit signed immediate");
  case Match_UImm5_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 5-bit unsigned immediate");
  case Match_SImm5_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 5-bit signed immediate");
  case Match_UImm5_1:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range 1 .. 32");
  case Match_UImm5_32:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range 32 .. 63");
  case Match_UImm5_33:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range 33 .. 64");
  case Match_UImm5_0_Report_UImm6:
    // This is used on UImm5 operands that have a corresponding UImm5_32
    // operand to avoid confusing the user.
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 6-bit unsigned immediate");
  case Match_UImm5_Lsl2:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected both 7-bit unsigned immediate and multiple of 4");
  case Match_UImmRange2_64:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range 2 .. 64");
  case Match_UImm6_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 6-bit unsigned immediate");
  case Match_UImm6_Lsl2:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected both 8-bit unsigned immediate and multiple of 4");
  case Match_SImm6_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 6-bit signed immediate");
  case Match_UImm7_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 7-bit unsigned immediate");
  case Match_UImm7_N1:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected immediate in range -1 .. 126");
  case Match_SImm7_Lsl2:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected both 9-bit signed immediate and multiple of 4");
  case Match_UImm8_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 8-bit unsigned immediate");
  case Match_UImm10_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 10-bit unsigned immediate");
  case Match_SImm10_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 10-bit signed immediate");
  case Match_SImm11_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 11-bit signed immediate");
  case Match_UImm16:
  case Match_UImm16_Relaxed:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 16-bit unsigned immediate");
  case Match_SImm16:
  case Match_SImm16_Relaxed:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 16-bit signed immediate");
  case Match_UImm20_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 20-bit unsigned immediate");
  case Match_UImm26_0:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected 26-bit unsigned immediate");
  case Match_MemSImm9:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 9-bit signed offset");
  case Match_MemGPSImm9:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with $gp and 9-bit signed offset");
  case Match_MemSImm10:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 10-bit signed offset");
  case Match_MemSImm10Lsl1:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 11-bit signed offset and multiple of 2");
  case Match_MemSImm10Lsl2:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 12-bit signed offset and multiple of 4");
  case Match_MemSImm10Lsl3:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 13-bit signed offset and multiple of 8");
  case Match_MemSImm11:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 11-bit signed offset");
  case Match_MemSImm16:
    return Error(RefineErrorLoc(IDLoc, Operands, ErrorInfo),
                 "expected memory with 16-bit signed offset");
  }

  llvm_unreachable("Implement any new match types added!");
}

void MipsAsmParser::warnIfRegIndexIsAT(unsigned RegIndex, SMLoc Loc) {
  if (RegIndex != 0 && AssemblerOptions.back()->getATRegIndex() == RegIndex)
    Warning(Loc, "used $at (currently $" + Twine(RegIndex) +
                     ") without \".set noat\"");
}

void MipsAsmParser::warnIfNoMacro(SMLoc Loc) {
  if (!AssemblerOptions.back()->isMacro())
    Warning(Loc, "macro instruction expanded into multiple instructions");
}

void
MipsAsmParser::printWarningWithFixIt(const Twine &Msg, const Twine &FixMsg,
                                     SMRange Range, bool ShowColors) {
  getSourceManager().PrintMessage(Range.Start, SourceMgr::DK_Warning, Msg,
                                  Range, SMFixIt(Range, FixMsg),
                                  ShowColors);
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

  if (!(isABI_N32() || isABI_N64()))
    return CC;

  if (12 <= CC && CC <= 15) {
    // Name is one of t4-t7
    AsmToken RegTok = getLexer().peekTok();
    SMRange RegRange = RegTok.getLocRange();

    StringRef FixedName = StringSwitch<StringRef>(Name)
                              .Case("t4", "t0")
                              .Case("t5", "t1")
                              .Case("t6", "t2")
                              .Case("t7", "t3")
                              .Default("");
    assert(FixedName != "" &&  "Register name is not one of t4-t7.");

    printWarningWithFixIt("register names $t4-$t7 are only available in O32.",
                          "Did you mean $" + FixedName + "?", RegRange);
  }

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

  return CC;
}

int MipsAsmParser::matchHWRegsRegisterName(StringRef Name) {
  int CC;

  CC = StringSwitch<unsigned>(Name)
            .Case("hwr_cpunum", 0)
            .Case("hwr_synci_step", 1)
            .Case("hwr_cc", 2)
            .Case("hwr_ccres", 3)
            .Case("hwr_ulr", 29)
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

unsigned MipsAsmParser::getATReg(SMLoc Loc) {
  unsigned ATIndex = AssemblerOptions.back()->getATRegIndex();
  if (ATIndex == 0) {
    reportParseError(Loc,
                     "pseudo-instruction requires $at, which is not available");
    return 0;
  }
  unsigned AT = getReg(
      (isGP64bit()) ? Mips::GPR64RegClassID : Mips::GPR32RegClassID, ATIndex);
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

bool MipsAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  MCAsmParser &Parser = getParser();
  DEBUG(dbgs() << "parseOperand\n");

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
    if (parseAnyRegister(Operands) != MatchOperand_NoMatch)
      return false;

    // Maybe it is a symbol reference.
    StringRef Identifier;
    if (Parser.parseIdentifier(Identifier))
      return true;

    SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    MCSymbol *Sym = getContext().getOrCreateSymbol("$" + Identifier);
    // Otherwise create a symbol reference.
    const MCExpr *Res =
        MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());

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
    OperandMatchResultTy ResTy = parseImm(Operands);
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
    case MCSymbolRefExpr::VK_Mips_GOT:
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
      report_fatal_error("unsupported reloc value");
    }
    return MCConstantExpr::create(Val, getContext());
  }

  if (const MCSymbolRefExpr *MSRE = dyn_cast<MCSymbolRefExpr>(Expr)) {
    // It's a symbol, create a symbolic expression from the symbol.
    const MCSymbol *Symbol = &MSRE->getSymbol();
    MCSymbolRefExpr::VariantKind VK = getVariantKind(RelocStr);
    Res = MCSymbolRefExpr::create(Symbol, VK, getContext());
    return Res;
  }

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    MCSymbolRefExpr::VariantKind VK = getVariantKind(RelocStr);

    // Try to create target expression.
    if (MipsMCExpr::isSupportedBinaryExpr(VK, BE))
      return MipsMCExpr::create(VK, Expr, getContext());

    const MCExpr *LExp = evaluateRelocExpr(BE->getLHS(), RelocStr);
    const MCExpr *RExp = evaluateRelocExpr(BE->getRHS(), RelocStr);
    Res = MCBinaryExpr::create(BE->getOpcode(), LExp, RExp, getContext());
    return Res;
  }

  if (const MCUnaryExpr *UN = dyn_cast<MCUnaryExpr>(Expr)) {
    const MCExpr *UnExp = evaluateRelocExpr(UN->getSubExpr(), RelocStr);
    Res = MCUnaryExpr::create(UN->getOpcode(), UnExp, getContext());
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
  MCAsmParser &Parser = getParser();
  Parser.Lex();                          // Eat the % token.
  const AsmToken &Tok = Parser.getTok(); // Get next token, operation.
  if (Tok.isNot(AsmToken::Identifier))
    return true;

  std::string Str = Tok.getIdentifier();

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
  OperandMatchResultTy ResTy = parseAnyRegister(Operands);
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
  MCAsmParser &Parser = getParser();
  SMLoc S;
  bool Result = true;
  unsigned NumOfLParen = 0;

  while (getLexer().getKind() == AsmToken::LParen) {
    Parser.Lex();
    ++NumOfLParen;
  }

  switch (getLexer().getKind()) {
  default:
    return true;
  case AsmToken::Identifier:
  case AsmToken::LParen:
  case AsmToken::Integer:
  case AsmToken::Minus:
  case AsmToken::Plus:
    if (isParenExpr)
      Result = getParser().parseParenExprOfDepth(NumOfLParen, Res, S);
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
  MCAsmParser &Parser = getParser();
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
      if (Mnemonic.getToken() == "la" || Mnemonic.getToken() == "dla") {
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
        auto Base = MipsOperand::createGPRReg(0, getContext().getRegisterInfo(),
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

  Res = parseAnyRegister(Operands);
  if (Res != MatchOperand_Success)
    return Res;

  if (Parser.getTok().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "')' expected");
    return MatchOperand_ParseFail;
  }

  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  Parser.Lex(); // Eat the ')' token.

  if (!IdVal)
    IdVal = MCConstantExpr::create(0, getContext());

  // Replace the register operand with the memory operand.
  std::unique_ptr<MipsOperand> op(
      static_cast<MipsOperand *>(Operands.back().release()));
  // Remove the register from the operands.
  // "op" will be managed by k_Memory.
  Operands.pop_back();
  // Add the memory operand.
  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(IdVal)) {
    int64_t Imm;
    if (IdVal->evaluateAsAbsolute(Imm))
      IdVal = MCConstantExpr::create(Imm, getContext());
    else if (BE->getLHS()->getKind() != MCExpr::SymbolRef)
      IdVal = MCBinaryExpr::create(BE->getOpcode(), BE->getRHS(), BE->getLHS(),
                                   getContext());
  }

  Operands.push_back(MipsOperand::CreateMem(std::move(op), IdVal, S, E, *this));
  return MatchOperand_Success;
}

bool MipsAsmParser::searchSymbolAlias(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  MCSymbol *Sym = getContext().lookupSymbol(Parser.getTok().getIdentifier());
  if (Sym) {
    SMLoc S = Parser.getTok().getLoc();
    const MCExpr *Expr;
    if (Sym->isVariable())
      Expr = Sym->getVariableValue();
    else
      return false;
    if (Expr->getKind() == MCExpr::SymbolRef) {
      const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);
      StringRef DefSymbol = Ref->getSymbol().getName();
      if (DefSymbol.startswith("$")) {
        OperandMatchResultTy ResTy =
            matchAnyRegisterNameWithoutDollar(Operands, DefSymbol.substr(1), S);
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
MipsAsmParser::matchAnyRegisterNameWithoutDollar(OperandVector &Operands,
                                                 StringRef Identifier,
                                                 SMLoc S) {
  int Index = matchCPURegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createGPRReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchHWRegsRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createHWRegsReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchFPURegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createFGRReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchFCCRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createFCCReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchACRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createACCReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchMSA128RegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createMSA128Reg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  Index = matchMSA128CtrlRegisterName(Identifier);
  if (Index != -1) {
    Operands.push_back(MipsOperand::createMSACtrlReg(
        Index, getContext().getRegisterInfo(), S, getLexer().getLoc(), *this));
    return MatchOperand_Success;
  }

  return MatchOperand_NoMatch;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::matchAnyRegisterWithoutDollar(OperandVector &Operands, SMLoc S) {
  MCAsmParser &Parser = getParser();
  auto Token = Parser.getLexer().peekTok(false);

  if (Token.is(AsmToken::Identifier)) {
    DEBUG(dbgs() << ".. identifier\n");
    StringRef Identifier = Token.getIdentifier();
    OperandMatchResultTy ResTy =
        matchAnyRegisterNameWithoutDollar(Operands, Identifier, S);
    return ResTy;
  } else if (Token.is(AsmToken::Integer)) {
    DEBUG(dbgs() << ".. integer\n");
    Operands.push_back(MipsOperand::createNumericReg(
        Token.getIntVal(), getContext().getRegisterInfo(), S, Token.getLoc(),
        *this));
    return MatchOperand_Success;
  }

  DEBUG(dbgs() << Parser.getTok().getKind() << "\n");

  return MatchOperand_NoMatch;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseAnyRegister(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  DEBUG(dbgs() << "parseAnyRegister\n");

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

  OperandMatchResultTy ResTy = matchAnyRegisterWithoutDollar(Operands, S);
  if (ResTy == MatchOperand_Success) {
    Parser.Lex(); // $
    Parser.Lex(); // identifier
  }
  return ResTy;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseImm(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
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
MipsAsmParser::parseJumpTarget(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  DEBUG(dbgs() << "parseJumpTarget\n");

  SMLoc S = getLexer().getLoc();

  // Integers and expressions are acceptable
  OperandMatchResultTy ResTy = parseImm(Operands);
  if (ResTy != MatchOperand_NoMatch)
    return ResTy;

  // Registers are a valid target and have priority over symbols.
  ResTy = parseAnyRegister(Operands);
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
  MCAsmParser &Parser = getParser();
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
      MCConstantExpr::create(0 - Val, getContext()), S, E, *this));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseLSAImm(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
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
  if (!Expr->evaluateAsAbsolute(Val)) {
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

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseRegisterList(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  SmallVector<unsigned, 10> Regs;
  unsigned RegNo;
  unsigned PrevReg = Mips::NoRegister;
  bool RegRange = false;
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 8> TmpOperands;

  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_ParseFail;

  SMLoc S = Parser.getTok().getLoc();
  while (parseAnyRegister(TmpOperands) == MatchOperand_Success) {
    SMLoc E = getLexer().getLoc();
    MipsOperand &Reg = static_cast<MipsOperand &>(*TmpOperands.back());
    RegNo = isGP64bit() ? Reg.getGPR64Reg() : Reg.getGPR32Reg();
    if (RegRange) {
      // Remove last register operand because registers from register range
      // should be inserted first.
      if ((isGP64bit() && RegNo == Mips::RA_64) ||
          (!isGP64bit() && RegNo == Mips::RA)) {
        Regs.push_back(RegNo);
      } else {
        unsigned TmpReg = PrevReg + 1;
        while (TmpReg <= RegNo) {
          if ((((TmpReg < Mips::S0) || (TmpReg > Mips::S7)) && !isGP64bit()) ||
              (((TmpReg < Mips::S0_64) || (TmpReg > Mips::S7_64)) &&
               isGP64bit())) {
            Error(E, "invalid register operand");
            return MatchOperand_ParseFail;
          }

          PrevReg = TmpReg;
          Regs.push_back(TmpReg++);
        }
      }

      RegRange = false;
    } else {
      if ((PrevReg == Mips::NoRegister) &&
          ((isGP64bit() && (RegNo != Mips::S0_64) && (RegNo != Mips::RA_64)) ||
          (!isGP64bit() && (RegNo != Mips::S0) && (RegNo != Mips::RA)))) {
        Error(E, "$16 or $31 expected");
        return MatchOperand_ParseFail;
      } else if (!(((RegNo == Mips::FP || RegNo == Mips::RA ||
                    (RegNo >= Mips::S0 && RegNo <= Mips::S7)) &&
                    !isGP64bit()) ||
                   ((RegNo == Mips::FP_64 || RegNo == Mips::RA_64 ||
                    (RegNo >= Mips::S0_64 && RegNo <= Mips::S7_64)) &&
                    isGP64bit()))) {
        Error(E, "invalid register operand");
        return MatchOperand_ParseFail;
      } else if ((PrevReg != Mips::NoRegister) && (RegNo != PrevReg + 1) &&
                 ((RegNo != Mips::FP && RegNo != Mips::RA && !isGP64bit()) ||
                  (RegNo != Mips::FP_64 && RegNo != Mips::RA_64 &&
                   isGP64bit()))) {
        Error(E, "consecutive register numbers expected");
        return MatchOperand_ParseFail;
      }

      Regs.push_back(RegNo);
    }

    if (Parser.getTok().is(AsmToken::Minus))
      RegRange = true;

    if (!Parser.getTok().isNot(AsmToken::Minus) &&
        !Parser.getTok().isNot(AsmToken::Comma)) {
      Error(E, "',' or '-' expected");
      return MatchOperand_ParseFail;
    }

    Lex(); // Consume comma or minus
    if (Parser.getTok().isNot(AsmToken::Dollar))
      break;

    PrevReg = RegNo;
  }

  SMLoc E = Parser.getTok().getLoc();
  Operands.push_back(MipsOperand::CreateRegList(Regs, S, E, *this));
  parseMemOperand(Operands);
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseRegisterPair(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();

  SMLoc S = Parser.getTok().getLoc();
  if (parseAnyRegister(Operands) != MatchOperand_Success)
    return MatchOperand_ParseFail;

  SMLoc E = Parser.getTok().getLoc();
  MipsOperand &Op = static_cast<MipsOperand &>(*Operands.back());
  unsigned Reg = Op.getGPR32Reg();
  Operands.pop_back();
  Operands.push_back(MipsOperand::CreateRegPair(Reg, S, E, *this));
  return MatchOperand_Success;
}

MipsAsmParser::OperandMatchResultTy
MipsAsmParser::parseMovePRegPair(OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 8> TmpOperands;
  SmallVector<unsigned, 10> Regs;

  if (Parser.getTok().isNot(AsmToken::Dollar))
    return MatchOperand_ParseFail;

  SMLoc S = Parser.getTok().getLoc();

  if (parseAnyRegister(TmpOperands) != MatchOperand_Success)
    return MatchOperand_ParseFail;

  MipsOperand *Reg = &static_cast<MipsOperand &>(*TmpOperands.back());
  unsigned RegNo = isGP64bit() ? Reg->getGPR64Reg() : Reg->getGPR32Reg();
  Regs.push_back(RegNo);

  SMLoc E = Parser.getTok().getLoc();
  if (Parser.getTok().isNot(AsmToken::Comma)) {
    Error(E, "',' expected");
    return MatchOperand_ParseFail;
  }

  // Remove comma.
  Parser.Lex();

  if (parseAnyRegister(TmpOperands) != MatchOperand_Success)
    return MatchOperand_ParseFail;

  Reg = &static_cast<MipsOperand &>(*TmpOperands.back());
  RegNo = isGP64bit() ? Reg->getGPR64Reg() : Reg->getGPR32Reg();
  Regs.push_back(RegNo);

  Operands.push_back(MipsOperand::CreateRegList(Regs, S, E, *this));

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
bool MipsAsmParser::parseParenSuffix(StringRef Name, OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  if (getLexer().is(AsmToken::LParen)) {
    Operands.push_back(
        MipsOperand::CreateToken("(", getLexer().getLoc(), *this));
    Parser.Lex();
    if (parseOperand(Operands, Name)) {
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
bool MipsAsmParser::parseBracketSuffix(StringRef Name,
                                       OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  if (getLexer().is(AsmToken::LBrac)) {
    Operands.push_back(
        MipsOperand::CreateToken("[", getLexer().getLoc(), *this));
    Parser.Lex();
    if (parseOperand(Operands, Name)) {
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
  MCAsmParser &Parser = getParser();
  DEBUG(dbgs() << "ParseInstruction\n");

  // We have reached first instruction, module directive are now forbidden.
  getTargetStreamer().forbidModuleDirective();

  // Check if we have valid mnemonic
  if (!mnemonicIsValid(Name, 0)) {
    Parser.eatToEndOfStatement();
    return Error(NameLoc, "unknown instruction");
  }
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(MipsOperand::CreateToken(Name, NameLoc, *this));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (parseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
    if (getLexer().is(AsmToken::LBrac) && parseBracketSuffix(Name, Operands))
      return true;
    // AFAIK, parenthesis suffixes are never on the first operand

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      // Parse and remember the operand.
      if (parseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
      // Parse bracket and parenthesis suffixes before we iterate
      if (getLexer().is(AsmToken::LBrac)) {
        if (parseBracketSuffix(Name, Operands))
          return true;
      } else if (getLexer().is(AsmToken::LParen) &&
                 parseParenSuffix(Name, Operands))
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
  MCAsmParser &Parser = getParser();
  SMLoc Loc = getLexer().getLoc();
  Parser.eatToEndOfStatement();
  return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::reportParseError(SMLoc Loc, Twine ErrorMsg) {
  return Error(Loc, ErrorMsg);
}

bool MipsAsmParser::parseSetNoAtDirective() {
  MCAsmParser &Parser = getParser();
  // Line should look like: ".set noat".

  // Set the $at register to $0.
  AssemblerOptions.back()->setATRegIndex(0);

  Parser.Lex(); // Eat "noat".

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  getTargetStreamer().emitDirectiveSetNoAt();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetAtDirective() {
  // Line can be: ".set at", which sets $at to $1
  //          or  ".set at=$reg", which sets $at to $reg.
  MCAsmParser &Parser = getParser();
  Parser.Lex(); // Eat "at".

  if (getLexer().is(AsmToken::EndOfStatement)) {
    // No register was specified, so we set $at to $1.
    AssemblerOptions.back()->setATRegIndex(1);

    getTargetStreamer().emitDirectiveSetAt();
    Parser.Lex(); // Consume the EndOfStatement.
    return false;
  }

  if (getLexer().isNot(AsmToken::Equal)) {
    reportParseError("unexpected token, expected equals sign");
    return false;
  }
  Parser.Lex(); // Eat "=".

  if (getLexer().isNot(AsmToken::Dollar)) {
    if (getLexer().is(AsmToken::EndOfStatement)) {
      reportParseError("no register specified");
      return false;
    } else {
      reportParseError("unexpected token, expected dollar sign '$'");
      return false;
    }
  }
  Parser.Lex(); // Eat "$".

  // Find out what "reg" is.
  unsigned AtRegNo;
  const AsmToken &Reg = Parser.getTok();
  if (Reg.is(AsmToken::Identifier)) {
    AtRegNo = matchCPURegisterName(Reg.getIdentifier());
  } else if (Reg.is(AsmToken::Integer)) {
    AtRegNo = Reg.getIntVal();
  } else {
    reportParseError("unexpected token, expected identifier or integer");
    return false;
  }

  // Check if $reg is a valid register. If it is, set $at to $reg.
  if (!AssemblerOptions.back()->setATRegIndex(AtRegNo)) {
    reportParseError("invalid register");
    return false;
  }
  Parser.Lex(); // Eat "reg".

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  getTargetStreamer().emitDirectiveSetAtWithArg(AtRegNo);

  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetReorderDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }
  AssemblerOptions.back()->setReorder();
  getTargetStreamer().emitDirectiveSetReorder();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoReorderDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }
  AssemblerOptions.back()->setNoReorder();
  getTargetStreamer().emitDirectiveSetNoReorder();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetMacroDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }
  AssemblerOptions.back()->setMacro();
  getTargetStreamer().emitDirectiveSetMacro();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoMacroDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }
  if (AssemblerOptions.back()->isReorder()) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  AssemblerOptions.back()->setNoMacro();
  getTargetStreamer().emitDirectiveSetNoMacro();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetMsaDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  setFeatureBits(Mips::FeatureMSA, "msa");
  getTargetStreamer().emitDirectiveSetMsa();
  return false;
}

bool MipsAsmParser::parseSetNoMsaDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  clearFeatureBits(Mips::FeatureMSA, "msa");
  getTargetStreamer().emitDirectiveSetNoMsa();
  return false;
}

bool MipsAsmParser::parseSetNoDspDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex(); // Eat "nodsp".

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  clearFeatureBits(Mips::FeatureDSP, "dsp");
  getTargetStreamer().emitDirectiveSetNoDsp();
  return false;
}

bool MipsAsmParser::parseSetMips16Directive() {
  MCAsmParser &Parser = getParser();
  Parser.Lex(); // Eat "mips16".

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  setFeatureBits(Mips::FeatureMips16, "mips16");
  getTargetStreamer().emitDirectiveSetMips16();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetNoMips16Directive() {
  MCAsmParser &Parser = getParser();
  Parser.Lex(); // Eat "nomips16".

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  clearFeatureBits(Mips::FeatureMips16, "mips16");
  getTargetStreamer().emitDirectiveSetNoMips16();
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetFpDirective() {
  MCAsmParser &Parser = getParser();
  MipsABIFlagsSection::FpABIKind FpAbiVal;
  // Line can be: .set fp=32
  //              .set fp=xx
  //              .set fp=64
  Parser.Lex(); // Eat fp token
  AsmToken Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Equal)) {
    reportParseError("unexpected token, expected equals sign '='");
    return false;
  }
  Parser.Lex(); // Eat '=' token.
  Tok = Parser.getTok();

  if (!parseFpABIValue(FpAbiVal, ".set"))
    return false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }
  getTargetStreamer().emitDirectiveSetFp(FpAbiVal);
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseSetOddSPRegDirective() {
  MCAsmParser &Parser = getParser();

  Parser.Lex(); // Eat "oddspreg".
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  clearFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");
  getTargetStreamer().emitDirectiveSetOddSPReg();
  return false;
}

bool MipsAsmParser::parseSetNoOddSPRegDirective() {
  MCAsmParser &Parser = getParser();

  Parser.Lex(); // Eat "nooddspreg".
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  setFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");
  getTargetStreamer().emitDirectiveSetNoOddSPReg();
  return false;
}

bool MipsAsmParser::parseSetPopDirective() {
  MCAsmParser &Parser = getParser();
  SMLoc Loc = getLexer().getLoc();

  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  // Always keep an element on the options "stack" to prevent the user
  // from changing the initial options. This is how we remember them.
  if (AssemblerOptions.size() == 2)
    return reportParseError(Loc, ".set pop with no .set push");

  MCSubtargetInfo &STI = copySTI();
  AssemblerOptions.pop_back();
  setAvailableFeatures(
      ComputeAvailableFeatures(AssemblerOptions.back()->getFeatures()));
  STI.setFeatureBits(AssemblerOptions.back()->getFeatures());

  getTargetStreamer().emitDirectiveSetPop();
  return false;
}

bool MipsAsmParser::parseSetPushDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  // Create a copy of the current assembler options environment and push it.
  AssemblerOptions.push_back(
              make_unique<MipsAssemblerOptions>(AssemblerOptions.back().get()));

  getTargetStreamer().emitDirectiveSetPush();
  return false;
}

bool MipsAsmParser::parseSetSoftFloatDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  setFeatureBits(Mips::FeatureSoftFloat, "soft-float");
  getTargetStreamer().emitDirectiveSetSoftFloat();
  return false;
}

bool MipsAsmParser::parseSetHardFloatDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  clearFeatureBits(Mips::FeatureSoftFloat, "soft-float");
  getTargetStreamer().emitDirectiveSetHardFloat();
  return false;
}

bool MipsAsmParser::parseSetAssignment() {
  StringRef Name;
  const MCExpr *Value;
  MCAsmParser &Parser = getParser();

  if (Parser.parseIdentifier(Name))
    reportParseError("expected identifier after .set");

  if (getLexer().isNot(AsmToken::Comma))
    return reportParseError("unexpected token, expected comma");
  Lex(); // Eat comma

  if (Parser.parseExpression(Value))
    return reportParseError("expected valid expression after comma");

  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);
  Sym->setVariableValue(Value);

  return false;
}

bool MipsAsmParser::parseSetMips0Directive() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

  // Reset assembler options to their initial values.
  MCSubtargetInfo &STI = copySTI();
  setAvailableFeatures(
      ComputeAvailableFeatures(AssemblerOptions.front()->getFeatures()));
  STI.setFeatureBits(AssemblerOptions.front()->getFeatures());
  AssemblerOptions.back()->setFeatures(AssemblerOptions.front()->getFeatures());

  getTargetStreamer().emitDirectiveSetMips0();
  return false;
}

bool MipsAsmParser::parseSetArchDirective() {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::Equal))
    return reportParseError("unexpected token, expected equals sign");

  Parser.Lex();
  StringRef Arch;
  if (Parser.parseIdentifier(Arch))
    return reportParseError("expected arch identifier");

  StringRef ArchFeatureName =
      StringSwitch<StringRef>(Arch)
          .Case("mips1", "mips1")
          .Case("mips2", "mips2")
          .Case("mips3", "mips3")
          .Case("mips4", "mips4")
          .Case("mips5", "mips5")
          .Case("mips32", "mips32")
          .Case("mips32r2", "mips32r2")
          .Case("mips32r3", "mips32r3")
          .Case("mips32r5", "mips32r5")
          .Case("mips32r6", "mips32r6")
          .Case("mips64", "mips64")
          .Case("mips64r2", "mips64r2")
          .Case("mips64r3", "mips64r3")
          .Case("mips64r5", "mips64r5")
          .Case("mips64r6", "mips64r6")
          .Case("cnmips", "cnmips")
          .Case("r4000", "mips3") // This is an implementation of Mips3.
          .Default("");

  if (ArchFeatureName.empty())
    return reportParseError("unsupported architecture");

  selectArch(ArchFeatureName);
  getTargetStreamer().emitDirectiveSetArch(Arch);
  return false;
}

bool MipsAsmParser::parseSetFeature(uint64_t Feature) {
  MCAsmParser &Parser = getParser();
  Parser.Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return reportParseError("unexpected token, expected end of statement");

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
  case Mips::FeatureMips1:
    selectArch("mips1");
    getTargetStreamer().emitDirectiveSetMips1();
    break;
  case Mips::FeatureMips2:
    selectArch("mips2");
    getTargetStreamer().emitDirectiveSetMips2();
    break;
  case Mips::FeatureMips3:
    selectArch("mips3");
    getTargetStreamer().emitDirectiveSetMips3();
    break;
  case Mips::FeatureMips4:
    selectArch("mips4");
    getTargetStreamer().emitDirectiveSetMips4();
    break;
  case Mips::FeatureMips5:
    selectArch("mips5");
    getTargetStreamer().emitDirectiveSetMips5();
    break;
  case Mips::FeatureMips32:
    selectArch("mips32");
    getTargetStreamer().emitDirectiveSetMips32();
    break;
  case Mips::FeatureMips32r2:
    selectArch("mips32r2");
    getTargetStreamer().emitDirectiveSetMips32R2();
    break;
  case Mips::FeatureMips32r3:
    selectArch("mips32r3");
    getTargetStreamer().emitDirectiveSetMips32R3();
    break;
  case Mips::FeatureMips32r5:
    selectArch("mips32r5");
    getTargetStreamer().emitDirectiveSetMips32R5();
    break;
  case Mips::FeatureMips32r6:
    selectArch("mips32r6");
    getTargetStreamer().emitDirectiveSetMips32R6();
    break;
  case Mips::FeatureMips64:
    selectArch("mips64");
    getTargetStreamer().emitDirectiveSetMips64();
    break;
  case Mips::FeatureMips64r2:
    selectArch("mips64r2");
    getTargetStreamer().emitDirectiveSetMips64R2();
    break;
  case Mips::FeatureMips64r3:
    selectArch("mips64r3");
    getTargetStreamer().emitDirectiveSetMips64R3();
    break;
  case Mips::FeatureMips64r5:
    selectArch("mips64r5");
    getTargetStreamer().emitDirectiveSetMips64R5();
    break;
  case Mips::FeatureMips64r6:
    selectArch("mips64r6");
    getTargetStreamer().emitDirectiveSetMips64R6();
    break;
  }
  return false;
}

bool MipsAsmParser::eatComma(StringRef ErrorStr) {
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::Comma)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, ErrorStr);
  }

  Parser.Lex(); // Eat the comma.
  return true;
}

// Used to determine if .cpload, .cprestore, and .cpsetup have any effect.
// In this class, it is only used for .cprestore.
// FIXME: Only keep track of IsPicEnabled in one place, instead of in both
// MipsTargetELFStreamer and MipsAsmParser.
bool MipsAsmParser::isPicAndNotNxxAbi() {
  return inPicMode() && !(isABI_N32() || isABI_N64());
}

bool MipsAsmParser::parseDirectiveCpLoad(SMLoc Loc) {
  if (AssemblerOptions.back()->isReorder())
    Warning(Loc, ".cpload should be inside a noreorder section");

  if (inMips16Mode()) {
    reportParseError(".cpload is not supported in Mips16 mode");
    return false;
  }

  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> Reg;
  OperandMatchResultTy ResTy = parseAnyRegister(Reg);
  if (ResTy == MatchOperand_NoMatch || ResTy == MatchOperand_ParseFail) {
    reportParseError("expected register containing function address");
    return false;
  }

  MipsOperand &RegOpnd = static_cast<MipsOperand &>(*Reg[0]);
  if (!RegOpnd.isGPRAsmReg()) {
    reportParseError(RegOpnd.getStartLoc(), "invalid register");
    return false;
  }

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  getTargetStreamer().emitDirectiveCpLoad(RegOpnd.getGPR32Reg());
  return false;
}

bool MipsAsmParser::parseDirectiveCpRestore(SMLoc Loc) {
  MCAsmParser &Parser = getParser();

  // Note that .cprestore is ignored if used with the N32 and N64 ABIs or if it
  // is used in non-PIC mode.

  if (inMips16Mode()) {
    reportParseError(".cprestore is not supported in Mips16 mode");
    return false;
  }

  // Get the stack offset value.
  const MCExpr *StackOffset;
  int64_t StackOffsetVal;
  if (Parser.parseExpression(StackOffset)) {
    reportParseError("expected stack offset value");
    return false;
  }

  if (!StackOffset->evaluateAsAbsolute(StackOffsetVal)) {
    reportParseError("stack offset is not an absolute expression");
    return false;
  }

  if (StackOffsetVal < 0) {
    Warning(Loc, ".cprestore with negative stack offset has no effect");
    IsCpRestoreSet = false;
  } else {
    IsCpRestoreSet = true;
    CpRestoreOffset = StackOffsetVal;
  }

  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  // Store the $gp on the stack.
  SmallVector<MCInst, 3> StoreInsts;
  createCpRestoreMemOp(false /*IsLoad*/, CpRestoreOffset /*StackOffset*/, Loc,
                       StoreInsts);

  getTargetStreamer().emitDirectiveCpRestore(StoreInsts, CpRestoreOffset);
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseDirectiveCPSetup() {
  MCAsmParser &Parser = getParser();
  unsigned FuncReg;
  unsigned Save;
  bool SaveIsReg = true;

  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> TmpReg;
  OperandMatchResultTy ResTy = parseAnyRegister(TmpReg);
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

  if (!eatComma("unexpected token, expected comma"))
    return true;

  ResTy = parseAnyRegister(TmpReg);
  if (ResTy == MatchOperand_NoMatch) {
    const MCExpr *OffsetExpr;
    int64_t OffsetVal;
    SMLoc ExprLoc = getLexer().getLoc();

    if (Parser.parseExpression(OffsetExpr) ||
        !OffsetExpr->evaluateAsAbsolute(OffsetVal)) {
      reportParseError(ExprLoc, "expected save register or stack offset");
      Parser.eatToEndOfStatement();
      return false;
    }

    Save = OffsetVal;
    SaveIsReg = false;
  } else {
    MipsOperand &SaveOpnd = static_cast<MipsOperand &>(*TmpReg[0]);
    if (!SaveOpnd.isGPRAsmReg()) {
      reportParseError(SaveOpnd.getStartLoc(), "invalid register");
      Parser.eatToEndOfStatement();
      return false;
    }
    Save = SaveOpnd.getGPR32Reg();
  }

  if (!eatComma("unexpected token, expected comma"))
    return true;

  const MCExpr *Expr;
  if (Parser.parseExpression(Expr)) {
    reportParseError("expected expression");
    return false;
  }

  if (Expr->getKind() != MCExpr::SymbolRef) {
    reportParseError("expected symbol");
    return false;
  }
  const MCSymbolRefExpr *Ref = static_cast<const MCSymbolRefExpr *>(Expr);

  CpSaveLocation = Save;
  CpSaveLocationIsRegister = SaveIsReg;

  getTargetStreamer().emitDirectiveCpsetup(FuncReg, Save, Ref->getSymbol(),
                                           SaveIsReg);
  return false;
}

bool MipsAsmParser::parseDirectiveCPReturn() {
  getTargetStreamer().emitDirectiveCpreturn(CpSaveLocation,
                                            CpSaveLocationIsRegister);
  return false;
}

bool MipsAsmParser::parseDirectiveNaN() {
  MCAsmParser &Parser = getParser();
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
  MCAsmParser &Parser = getParser();
  // Get the next token.
  const AsmToken &Tok = Parser.getTok();

  if (Tok.getString() == "noat") {
    return parseSetNoAtDirective();
  } else if (Tok.getString() == "at") {
    return parseSetAtDirective();
  } else if (Tok.getString() == "arch") {
    return parseSetArchDirective();
  } else if (Tok.getString() == "fp") {
    return parseSetFpDirective();
  } else if (Tok.getString() == "oddspreg") {
    return parseSetOddSPRegDirective();
  } else if (Tok.getString() == "nooddspreg") {
    return parseSetNoOddSPRegDirective();
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
    return parseSetMips16Directive();
  } else if (Tok.getString() == "nomips16") {
    return parseSetNoMips16Directive();
  } else if (Tok.getString() == "nomicromips") {
    getTargetStreamer().emitDirectiveSetNoMicroMips();
    Parser.eatToEndOfStatement();
    return false;
  } else if (Tok.getString() == "micromips") {
    return parseSetFeature(Mips::FeatureMicroMips);
  } else if (Tok.getString() == "mips0") {
    return parseSetMips0Directive();
  } else if (Tok.getString() == "mips1") {
    return parseSetFeature(Mips::FeatureMips1);
  } else if (Tok.getString() == "mips2") {
    return parseSetFeature(Mips::FeatureMips2);
  } else if (Tok.getString() == "mips3") {
    return parseSetFeature(Mips::FeatureMips3);
  } else if (Tok.getString() == "mips4") {
    return parseSetFeature(Mips::FeatureMips4);
  } else if (Tok.getString() == "mips5") {
    return parseSetFeature(Mips::FeatureMips5);
  } else if (Tok.getString() == "mips32") {
    return parseSetFeature(Mips::FeatureMips32);
  } else if (Tok.getString() == "mips32r2") {
    return parseSetFeature(Mips::FeatureMips32r2);
  } else if (Tok.getString() == "mips32r3") {
    return parseSetFeature(Mips::FeatureMips32r3);
  } else if (Tok.getString() == "mips32r5") {
    return parseSetFeature(Mips::FeatureMips32r5);
  } else if (Tok.getString() == "mips32r6") {
    return parseSetFeature(Mips::FeatureMips32r6);
  } else if (Tok.getString() == "mips64") {
    return parseSetFeature(Mips::FeatureMips64);
  } else if (Tok.getString() == "mips64r2") {
    return parseSetFeature(Mips::FeatureMips64r2);
  } else if (Tok.getString() == "mips64r3") {
    return parseSetFeature(Mips::FeatureMips64r3);
  } else if (Tok.getString() == "mips64r5") {
    return parseSetFeature(Mips::FeatureMips64r5);
  } else if (Tok.getString() == "mips64r6") {
    return parseSetFeature(Mips::FeatureMips64r6);
  } else if (Tok.getString() == "dsp") {
    return parseSetFeature(Mips::FeatureDSP);
  } else if (Tok.getString() == "nodsp") {
    return parseSetNoDspDirective();
  } else if (Tok.getString() == "msa") {
    return parseSetMsaDirective();
  } else if (Tok.getString() == "nomsa") {
    return parseSetNoMsaDirective();
  } else if (Tok.getString() == "softfloat") {
    return parseSetSoftFloatDirective();
  } else if (Tok.getString() == "hardfloat") {
    return parseSetHardFloatDirective();
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
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token, expected comma");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// parseDirectiveGpWord
///  ::= .gpword local_sym
bool MipsAsmParser::parseDirectiveGpWord() {
  MCAsmParser &Parser = getParser();
  const MCExpr *Value;
  // EmitGPRel32Value requires an expression, so we are using base class
  // method to evaluate the expression.
  if (getParser().parseExpression(Value))
    return true;
  getParser().getStreamer().EmitGPRel32Value(Value);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(), 
                "unexpected token, expected end of statement");
  Parser.Lex(); // Eat EndOfStatement token.
  return false;
}

/// parseDirectiveGpDWord
///  ::= .gpdword local_sym
bool MipsAsmParser::parseDirectiveGpDWord() {
  MCAsmParser &Parser = getParser();
  const MCExpr *Value;
  // EmitGPRel64Value requires an expression, so we are using base class
  // method to evaluate the expression.
  if (getParser().parseExpression(Value))
    return true;
  getParser().getStreamer().EmitGPRel64Value(Value);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(), 
                "unexpected token, expected end of statement");
  Parser.Lex(); // Eat EndOfStatement token.
  return false;
}

bool MipsAsmParser::parseDirectiveOption() {
  MCAsmParser &Parser = getParser();
  // Get the option token.
  AsmToken Tok = Parser.getTok();
  // At the moment only identifiers are supported.
  if (Tok.isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(), "unexpected token, expected identifier");
    Parser.eatToEndOfStatement();
    return false;
  }

  StringRef Option = Tok.getIdentifier();

  if (Option == "pic0") {
    // MipsAsmParser needs to know if the current PIC mode changes.
    IsPicEnabled = false;

    getTargetStreamer().emitDirectiveOptionPic0();
    Parser.Lex();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(),
            "unexpected token, expected end of statement");
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  if (Option == "pic2") {
    // MipsAsmParser needs to know if the current PIC mode changes.
    IsPicEnabled = true;

    getTargetStreamer().emitDirectiveOptionPic2();
    Parser.Lex();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(),
            "unexpected token, expected end of statement");
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  // Unknown option.
  Warning(Parser.getTok().getLoc(), 
          "unknown option, expected 'pic0' or 'pic2'");
  Parser.eatToEndOfStatement();
  return false;
}

/// parseInsnDirective
///  ::= .insn
bool MipsAsmParser::parseInsnDirective() {
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  // The actual label marking happens in
  // MipsELFStreamer::createPendingLabelRelocs().
  getTargetStreamer().emitDirectiveInsn();

  getParser().Lex(); // Eat EndOfStatement token.
  return false;
}

/// parseSSectionDirective
///  ::= .sbss
///  ::= .sdata
bool MipsAsmParser::parseSSectionDirective(StringRef Section, unsigned Type) {
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  MCSection *ELFSection = getContext().getELFSection(
      Section, Type, ELF::SHF_WRITE | ELF::SHF_ALLOC | ELF::SHF_MIPS_GPREL);
  getParser().getStreamer().SwitchSection(ELFSection);

  getParser().Lex(); // Eat EndOfStatement token.
  return false;
}

/// parseDirectiveModule
///  ::= .module oddspreg
///  ::= .module nooddspreg
///  ::= .module fp=value
///  ::= .module softfloat
///  ::= .module hardfloat
bool MipsAsmParser::parseDirectiveModule() {
  MCAsmParser &Parser = getParser();
  MCAsmLexer &Lexer = getLexer();
  SMLoc L = Lexer.getLoc();

  if (!getTargetStreamer().isModuleDirectiveAllowed()) {
    // TODO : get a better message.
    reportParseError(".module directive must appear before any code");
    return false;
  }

  StringRef Option;
  if (Parser.parseIdentifier(Option)) {
    reportParseError("expected .module option identifier");
    return false;
  }

  if (Option == "oddspreg") {
    clearModuleFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");

    // Synchronize the abiflags information with the FeatureBits information we
    // changed above.
    getTargetStreamer().updateABIInfo(*this);

    // If printing assembly, use the recently updated abiflags information.
    // If generating ELF, don't do anything (the .MIPS.abiflags section gets
    // emitted at the end).
    getTargetStreamer().emitDirectiveModuleOddSPReg();

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    return false; // parseDirectiveModule has finished successfully.
  } else if (Option == "nooddspreg") {
    if (!isABI_O32()) {
      Error(L, "'.module nooddspreg' requires the O32 ABI");
      return false;
    }

    setModuleFeatureBits(Mips::FeatureNoOddSPReg, "nooddspreg");

    // Synchronize the abiflags information with the FeatureBits information we
    // changed above.
    getTargetStreamer().updateABIInfo(*this);

    // If printing assembly, use the recently updated abiflags information.
    // If generating ELF, don't do anything (the .MIPS.abiflags section gets
    // emitted at the end).
    getTargetStreamer().emitDirectiveModuleOddSPReg();

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    return false; // parseDirectiveModule has finished successfully.
  } else if (Option == "fp") {
    return parseDirectiveModuleFP();
  } else if (Option == "softfloat") {
    setModuleFeatureBits(Mips::FeatureSoftFloat, "soft-float");

    // Synchronize the ABI Flags information with the FeatureBits information we
    // updated above.
    getTargetStreamer().updateABIInfo(*this);

    // If printing assembly, use the recently updated ABI Flags information.
    // If generating ELF, don't do anything (the .MIPS.abiflags section gets
    // emitted later).
    getTargetStreamer().emitDirectiveModuleSoftFloat();

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    return false; // parseDirectiveModule has finished successfully.
  } else if (Option == "hardfloat") {
    clearModuleFeatureBits(Mips::FeatureSoftFloat, "soft-float");

    // Synchronize the ABI Flags information with the FeatureBits information we
    // updated above.
    getTargetStreamer().updateABIInfo(*this);

    // If printing assembly, use the recently updated ABI Flags information.
    // If generating ELF, don't do anything (the .MIPS.abiflags section gets
    // emitted later).
    getTargetStreamer().emitDirectiveModuleHardFloat();

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    return false; // parseDirectiveModule has finished successfully.
  } else {
    return Error(L, "'" + Twine(Option) + "' is not a valid .module option.");
  }
}

/// parseDirectiveModuleFP
///  ::= =32
///  ::= =xx
///  ::= =64
bool MipsAsmParser::parseDirectiveModuleFP() {
  MCAsmParser &Parser = getParser();
  MCAsmLexer &Lexer = getLexer();

  if (Lexer.isNot(AsmToken::Equal)) {
    reportParseError("unexpected token, expected equals sign '='");
    return false;
  }
  Parser.Lex(); // Eat '=' token.

  MipsABIFlagsSection::FpABIKind FpABI;
  if (!parseFpABIValue(FpABI, ".module"))
    return false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  // Synchronize the abiflags information with the FeatureBits information we
  // changed above.
  getTargetStreamer().updateABIInfo(*this);

  // If printing assembly, use the recently updated abiflags information.
  // If generating ELF, don't do anything (the .MIPS.abiflags section gets
  // emitted at the end).
  getTargetStreamer().emitDirectiveModuleFP();

  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool MipsAsmParser::parseFpABIValue(MipsABIFlagsSection::FpABIKind &FpABI,
                                    StringRef Directive) {
  MCAsmParser &Parser = getParser();
  MCAsmLexer &Lexer = getLexer();
  bool ModuleLevelOptions = Directive == ".module";

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
    if (ModuleLevelOptions) {
      setModuleFeatureBits(Mips::FeatureFPXX, "fpxx");
      clearModuleFeatureBits(Mips::FeatureFP64Bit, "fp64");
    } else {
      setFeatureBits(Mips::FeatureFPXX, "fpxx");
      clearFeatureBits(Mips::FeatureFP64Bit, "fp64");
    }
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
      if (ModuleLevelOptions) {
        clearModuleFeatureBits(Mips::FeatureFPXX, "fpxx");
        clearModuleFeatureBits(Mips::FeatureFP64Bit, "fp64");
      } else {
        clearFeatureBits(Mips::FeatureFPXX, "fpxx");
        clearFeatureBits(Mips::FeatureFP64Bit, "fp64");
      }
    } else {
      FpABI = MipsABIFlagsSection::FpABIKind::S64;
      if (ModuleLevelOptions) {
        clearModuleFeatureBits(Mips::FeatureFPXX, "fpxx");
        setModuleFeatureBits(Mips::FeatureFP64Bit, "fp64");
      } else {
        clearFeatureBits(Mips::FeatureFPXX, "fpxx");
        setFeatureBits(Mips::FeatureFP64Bit, "fp64");
      }
    }

    return true;
  }

  return false;
}

bool MipsAsmParser::ParseDirective(AsmToken DirectiveID) {
  MCAsmParser &Parser = getParser();
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".cpload")
    return parseDirectiveCpLoad(DirectiveID.getLoc());
  if (IDVal == ".cprestore")
    return parseDirectiveCpRestore(DirectiveID.getLoc());
  if (IDVal == ".dword") {
    parseDataDirective(8, DirectiveID.getLoc());
    return false;
  }
  if (IDVal == ".ent") {
    StringRef SymbolName;

    if (Parser.parseIdentifier(SymbolName)) {
      reportParseError("expected identifier after .ent");
      return false;
    }

    // There's an undocumented extension that allows an integer to
    // follow the name of the procedure which AFAICS is ignored by GAS.
    // Example: .ent foo,2
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma)) {
        // Even though we accept this undocumented extension for compatibility
        // reasons, the additional integer argument does not actually change
        // the behaviour of the '.ent' directive, so we would like to discourage
        // its use. We do this by not referring to the extended version in
        // error messages which are not directly related to its use.
        reportParseError("unexpected token, expected end of statement");
        return false;
      }
      Parser.Lex(); // Eat the comma.
      const MCExpr *DummyNumber;
      int64_t DummyNumberVal;
      // If the user was explicitly trying to use the extended version,
      // we still give helpful extension-related error messages.
      if (Parser.parseExpression(DummyNumber)) {
        reportParseError("expected number after comma");
        return false;
      }
      if (!DummyNumber->evaluateAsAbsolute(DummyNumberVal)) {
        reportParseError("expected an absolute expression after comma");
        return false;
      }
    }

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    MCSymbol *Sym = getContext().getOrCreateSymbol(SymbolName);

    getTargetStreamer().emitDirectiveEnt(*Sym);
    CurrentFn = Sym;
    IsCpRestoreSet = false;
    return false;
  }

  if (IDVal == ".end") {
    StringRef SymbolName;

    if (Parser.parseIdentifier(SymbolName)) {
      reportParseError("expected identifier after .end");
      return false;
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    if (CurrentFn == nullptr) {
      reportParseError(".end used without .ent");
      return false;
    }

    if ((SymbolName != CurrentFn->getName())) {
      reportParseError(".end symbol does not match .ent symbol");
      return false;
    }

    getTargetStreamer().emitDirectiveEnd(SymbolName);
    CurrentFn = nullptr;
    IsCpRestoreSet = false;
    return false;
  }

  if (IDVal == ".frame") {
    // .frame $stack_reg, frame_size_in_bytes, $return_reg
    SmallVector<std::unique_ptr<MCParsedAsmOperand>, 1> TmpReg;
    OperandMatchResultTy ResTy = parseAnyRegister(TmpReg);
    if (ResTy == MatchOperand_NoMatch || ResTy == MatchOperand_ParseFail) {
      reportParseError("expected stack register");
      return false;
    }

    MipsOperand &StackRegOpnd = static_cast<MipsOperand &>(*TmpReg[0]);
    if (!StackRegOpnd.isGPRAsmReg()) {
      reportParseError(StackRegOpnd.getStartLoc(),
                       "expected general purpose register");
      return false;
    }
    unsigned StackReg = StackRegOpnd.getGPR32Reg();

    if (Parser.getTok().is(AsmToken::Comma))
      Parser.Lex();
    else {
      reportParseError("unexpected token, expected comma");
      return false;
    }

    // Parse the frame size.
    const MCExpr *FrameSize;
    int64_t FrameSizeVal;

    if (Parser.parseExpression(FrameSize)) {
      reportParseError("expected frame size value");
      return false;
    }

    if (!FrameSize->evaluateAsAbsolute(FrameSizeVal)) {
      reportParseError("frame size not an absolute expression");
      return false;
    }

    if (Parser.getTok().is(AsmToken::Comma))
      Parser.Lex();
    else {
      reportParseError("unexpected token, expected comma");
      return false;
    }

    // Parse the return register.
    TmpReg.clear();
    ResTy = parseAnyRegister(TmpReg);
    if (ResTy == MatchOperand_NoMatch || ResTy == MatchOperand_ParseFail) {
      reportParseError("expected return register");
      return false;
    }

    MipsOperand &ReturnRegOpnd = static_cast<MipsOperand &>(*TmpReg[0]);
    if (!ReturnRegOpnd.isGPRAsmReg()) {
      reportParseError(ReturnRegOpnd.getStartLoc(),
                       "expected general purpose register");
      return false;
    }

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    getTargetStreamer().emitFrame(StackReg, FrameSizeVal,
                                  ReturnRegOpnd.getGPR32Reg());
    IsCpRestoreSet = false;
    return false;
  }

  if (IDVal == ".set") {
    return parseDirectiveSet();
  }

  if (IDVal == ".mask" || IDVal == ".fmask") {
    // .mask bitmask, frame_offset
    // bitmask: One bit for each register used.
    // frame_offset: Offset from Canonical Frame Address ($sp on entry) where
    //               first register is expected to be saved.
    // Examples:
    //   .mask 0x80000000, -4
    //   .fmask 0x80000000, -4
    //

    // Parse the bitmask
    const MCExpr *BitMask;
    int64_t BitMaskVal;

    if (Parser.parseExpression(BitMask)) {
      reportParseError("expected bitmask value");
      return false;
    }

    if (!BitMask->evaluateAsAbsolute(BitMaskVal)) {
      reportParseError("bitmask not an absolute expression");
      return false;
    }

    if (Parser.getTok().is(AsmToken::Comma))
      Parser.Lex();
    else {
      reportParseError("unexpected token, expected comma");
      return false;
    }

    // Parse the frame_offset
    const MCExpr *FrameOffset;
    int64_t FrameOffsetVal;

    if (Parser.parseExpression(FrameOffset)) {
      reportParseError("expected frame offset value");
      return false;
    }

    if (!FrameOffset->evaluateAsAbsolute(FrameOffsetVal)) {
      reportParseError("frame offset not an absolute expression");
      return false;
    }

    // If this is not the end of the statement, report an error.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      reportParseError("unexpected token, expected end of statement");
      return false;
    }

    if (IDVal == ".mask")
      getTargetStreamer().emitMask(BitMaskVal, FrameOffsetVal);
    else
      getTargetStreamer().emitFMask(BitMaskVal, FrameOffsetVal);
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

  if (IDVal == ".hword") {
    parseDataDirective(2, DirectiveID.getLoc());
    return false;
  }

  if (IDVal == ".option")
    return parseDirectiveOption();

  if (IDVal == ".abicalls") {
    getTargetStreamer().emitDirectiveAbiCalls();
    if (Parser.getTok().isNot(AsmToken::EndOfStatement)) {
      Error(Parser.getTok().getLoc(), 
            "unexpected token, expected end of statement");
      // Clear line
      Parser.eatToEndOfStatement();
    }
    return false;
  }

  if (IDVal == ".cpsetup")
    return parseDirectiveCPSetup();

  if (IDVal == ".cpreturn")
    return parseDirectiveCPReturn();

  if (IDVal == ".module")
    return parseDirectiveModule();

  if (IDVal == ".llvm_internal_mips_reallow_module_directive")
    return parseInternalDirectiveReallowModule();

  if (IDVal == ".insn")
    return parseInsnDirective();

  if (IDVal == ".sbss")
    return parseSSectionDirective(IDVal, ELF::SHT_NOBITS);
  if (IDVal == ".sdata")
    return parseSSectionDirective(IDVal, ELF::SHT_PROGBITS);

  return true;
}

bool MipsAsmParser::parseInternalDirectiveReallowModule() {
  // If this is not the end of the statement, report an error.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token, expected end of statement");
    return false;
  }

  getTargetStreamer().reallowModuleDirective();

  getParser().Lex(); // Eat EndOfStatement token.
  return false;
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
