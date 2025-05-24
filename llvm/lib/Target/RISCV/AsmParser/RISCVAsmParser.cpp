//===-- RISCVAsmParser.cpp - Parse RISC-V assembly to MCInst instructions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVAsmBackend.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVInstPrinter.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "MCTargetDesc/RISCVTargetStreamer.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/RISCVAttributes.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

#include <limits>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "riscv-asm-parser"

STATISTIC(RISCVNumInstrsCompressed,
          "Number of RISC-V Compressed instructions emitted");

static cl::opt<bool> AddBuildAttributes("riscv-add-build-attributes",
                                        cl::init(false));

namespace llvm {
extern const SubtargetFeatureKV RISCVFeatureKV[RISCV::NumSubtargetFeatures];
} // namespace llvm

namespace {
struct RISCVOperand;

struct ParserOptionsSet {
  bool IsPicEnabled;
};

class RISCVAsmParser : public MCTargetAsmParser {
  // This tracks the parsing of the 4 operands that make up the vtype portion
  // of vset(i)vli instructions which are separated by commas. The state names
  // represent the next expected operand with Done meaning no other operands are
  // expected.
  enum VTypeState {
    VTypeState_SEW,
    VTypeState_LMUL,
    VTypeState_TailPolicy,
    VTypeState_MaskPolicy,
    VTypeState_Done,
  };

  SmallVector<FeatureBitset, 4> FeatureBitStack;

  SmallVector<ParserOptionsSet, 4> ParserOptionsStack;
  ParserOptionsSet ParserOptions;

  SMLoc getLoc() const { return getParser().getTok().getLoc(); }
  bool isRV64() const { return getSTI().hasFeature(RISCV::Feature64Bit); }
  bool isRVE() const { return getSTI().hasFeature(RISCV::FeatureStdExtE); }
  bool enableExperimentalExtension() const {
    return getSTI().hasFeature(RISCV::Experimental);
  }

  RISCVTargetStreamer &getTargetStreamer() {
    assert(getParser().getStreamer().getTargetStreamer() &&
           "do not have a target streamer");
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<RISCVTargetStreamer &>(TS);
  }

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  bool generateImmOutOfRangeError(OperandVector &Operands, uint64_t ErrorInfo,
                                  int64_t Lower, int64_t Upper,
                                  const Twine &Msg);
  bool generateImmOutOfRangeError(SMLoc ErrorLoc, int64_t Lower, int64_t Upper,
                                  const Twine &Msg);

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  MCRegister matchRegisterNameHelper(StringRef Name) const;
  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  ParseStatus parseDirective(AsmToken DirectiveID) override;

  bool parseVTypeToken(const AsmToken &Tok, VTypeState &State, unsigned &Sew,
                       unsigned &Lmul, bool &Fractional, bool &TailAgnostic,
                       bool &MaskAgnostic);
  bool generateVTypeError(SMLoc ErrorLoc);

  bool generateXSfmmVTypeError(SMLoc ErrorLoc);
  // Helper to actually emit an instruction to the MCStreamer. Also, when
  // possible, compression of the instruction is performed.
  void emitToStreamer(MCStreamer &S, const MCInst &Inst);

  // Helper to emit a combination of LUI, ADDI(W), and SLLI instructions that
  // synthesize the desired immediate value into the destination register.
  void emitLoadImm(MCRegister DestReg, int64_t Value, MCStreamer &Out);

  // Helper to emit a combination of AUIPC and SecondOpcode. Used to implement
  // helpers such as emitLoadLocalAddress and emitLoadAddress.
  void emitAuipcInstPair(MCRegister DestReg, MCRegister TmpReg,
                         const MCExpr *Symbol, RISCVMCExpr::Specifier VKHi,
                         unsigned SecondOpcode, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "lla" used in PC-rel addressing.
  void emitLoadLocalAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "lga" used in GOT-rel addressing.
  void emitLoadGlobalAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la" used in GOT/PC-rel addressing.
  void emitLoadAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la.tls.ie" used in initial-exec TLS
  // addressing.
  void emitLoadTLSIEAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo instruction "la.tls.gd" used in global-dynamic TLS
  // addressing.
  void emitLoadTLSGDAddress(MCInst &Inst, SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo load/store instruction with a symbol.
  void emitLoadStoreSymbol(MCInst &Inst, unsigned Opcode, SMLoc IDLoc,
                           MCStreamer &Out, bool HasTmpReg);

  // Helper to emit pseudo sign/zero extend instruction.
  void emitPseudoExtend(MCInst &Inst, bool SignExtend, int64_t Width,
                        SMLoc IDLoc, MCStreamer &Out);

  // Helper to emit pseudo vmsge{u}.vx instruction.
  void emitVMSGE(MCInst &Inst, unsigned Opcode, SMLoc IDLoc, MCStreamer &Out);

  // Checks that a PseudoAddTPRel is using x4/tp in its second input operand.
  // Enforcing this using a restricted register class for the second input
  // operand of PseudoAddTPRel results in a poor diagnostic due to the fact
  // 'add' is an overloaded mnemonic.
  bool checkPseudoAddTPRel(MCInst &Inst, OperandVector &Operands);

  // Checks that a PseudoTLSDESCCall is using x5/t0 in its output operand.
  // Enforcing this using a restricted register class for the output
  // operand of PseudoTLSDESCCall results in a poor diagnostic due to the fact
  // 'jalr' is an overloaded mnemonic.
  bool checkPseudoTLSDESCCall(MCInst &Inst, OperandVector &Operands);

  // Check instruction constraints.
  bool validateInstruction(MCInst &Inst, OperandVector &Operands);

  /// Helper for processing MC instructions that have been successfully matched
  /// by matchAndEmitInstruction. Modifications to the emitted instructions,
  /// like the expansion of pseudo instructions (e.g., "li"), can be performed
  /// in this method.
  bool processInstruction(MCInst &Inst, SMLoc IDLoc, OperandVector &Operands,
                          MCStreamer &Out);

// Auto-generated instruction matching functions
#define GET_ASSEMBLER_HEADER
#include "RISCVGenAsmMatcher.inc"

  ParseStatus parseCSRSystemRegister(OperandVector &Operands);
  ParseStatus parseFPImm(OperandVector &Operands);
  ParseStatus parseImmediate(OperandVector &Operands);
  ParseStatus parseRegister(OperandVector &Operands, bool AllowParens = false);
  ParseStatus parseMemOpBaseReg(OperandVector &Operands);
  ParseStatus parseZeroOffsetMemOp(OperandVector &Operands);
  ParseStatus parseOperandWithSpecifier(OperandVector &Operands);
  ParseStatus parseBareSymbol(OperandVector &Operands);
  ParseStatus parseCallSymbol(OperandVector &Operands);
  ParseStatus parsePseudoJumpSymbol(OperandVector &Operands);
  ParseStatus parseJALOffset(OperandVector &Operands);
  ParseStatus parseVTypeI(OperandVector &Operands);
  ParseStatus parseMaskReg(OperandVector &Operands);
  ParseStatus parseInsnDirectiveOpcode(OperandVector &Operands);
  ParseStatus parseInsnCDirectiveOpcode(OperandVector &Operands);
  ParseStatus parseGPRAsFPR(OperandVector &Operands);
  ParseStatus parseGPRAsFPR64(OperandVector &Operands);
  ParseStatus parseGPRPairAsFPR64(OperandVector &Operands);
  template <bool IsRV64Inst> ParseStatus parseGPRPair(OperandVector &Operands);
  ParseStatus parseGPRPair(OperandVector &Operands, bool IsRV64Inst);
  ParseStatus parseFRMArg(OperandVector &Operands);
  ParseStatus parseFenceArg(OperandVector &Operands);
  ParseStatus parseRegList(OperandVector &Operands, bool MustIncludeS0 = false);
  ParseStatus parseRegListS0(OperandVector &Operands) {
    return parseRegList(Operands, /*MustIncludeS0=*/true);
  }

  ParseStatus parseRegReg(OperandVector &Operands);
  ParseStatus parseXSfmmVType(OperandVector &Operands);
  ParseStatus parseRetval(OperandVector &Operands);
  ParseStatus parseZcmpStackAdj(OperandVector &Operands,
                                bool ExpectNegative = false);
  ParseStatus parseZcmpNegStackAdj(OperandVector &Operands) {
    return parseZcmpStackAdj(Operands, /*ExpectNegative*/ true);
  }

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);
  bool parseExprWithSpecifier(const MCExpr *&Res, SMLoc &E);
  bool parseDataExpr(const MCExpr *&Res) override;

  bool parseDirectiveOption();
  bool parseDirectiveAttribute();
  bool parseDirectiveInsn(SMLoc L);
  bool parseDirectiveVariantCC();

  /// Helper to reset target features for a new arch string. It
  /// also records the new arch string that is expanded by RISCVISAInfo
  /// and reports error for invalid arch string.
  bool resetToArch(StringRef Arch, SMLoc Loc, std::string &Result,
                   bool FromOptionDirective);

  void setFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (!(getSTI().hasFeature(Feature))) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
  }

  void clearFeatureBits(uint64_t Feature, StringRef FeatureString) {
    if (getSTI().hasFeature(Feature)) {
      MCSubtargetInfo &STI = copySTI();
      setAvailableFeatures(
          ComputeAvailableFeatures(STI.ToggleFeature(FeatureString)));
    }
  }

  void pushFeatureBits() {
    assert(FeatureBitStack.size() == ParserOptionsStack.size() &&
           "These two stacks must be kept synchronized");
    FeatureBitStack.push_back(getSTI().getFeatureBits());
    ParserOptionsStack.push_back(ParserOptions);
  }

  bool popFeatureBits() {
    assert(FeatureBitStack.size() == ParserOptionsStack.size() &&
           "These two stacks must be kept synchronized");
    if (FeatureBitStack.empty())
      return true;

    FeatureBitset FeatureBits = FeatureBitStack.pop_back_val();
    copySTI().setFeatureBits(FeatureBits);
    setAvailableFeatures(ComputeAvailableFeatures(FeatureBits));

    ParserOptions = ParserOptionsStack.pop_back_val();

    return false;
  }

  std::unique_ptr<RISCVOperand> defaultMaskRegOp() const;
  std::unique_ptr<RISCVOperand> defaultFRMArgOp() const;
  std::unique_ptr<RISCVOperand> defaultFRMArgLegacyOp() const;

public:
  enum RISCVMatchResultTy : unsigned {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "RISCVGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  static bool classifySymbolRef(const MCExpr *Expr,
                                RISCVMCExpr::Specifier &Kind);
  static bool isSymbolDiff(const MCExpr *Expr);

  RISCVAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                 const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    MCAsmParserExtension::Initialize(Parser);

    Parser.addAliasForDirective(".half", ".2byte");
    Parser.addAliasForDirective(".hword", ".2byte");
    Parser.addAliasForDirective(".word", ".4byte");
    Parser.addAliasForDirective(".dword", ".8byte");
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));

    auto ABIName = StringRef(Options.ABIName);
    if (ABIName.ends_with("f") && !getSTI().hasFeature(RISCV::FeatureStdExtF)) {
      errs() << "Hard-float 'f' ABI can't be used for a target that "
                "doesn't support the F instruction set extension (ignoring "
                "target-abi)\n";
    } else if (ABIName.ends_with("d") &&
               !getSTI().hasFeature(RISCV::FeatureStdExtD)) {
      errs() << "Hard-float 'd' ABI can't be used for a target that "
                "doesn't support the D instruction set extension (ignoring "
                "target-abi)\n";
    }

    // Use computeTargetABI to check if ABIName is valid. If invalid, output
    // error message.
    RISCVABI::computeTargetABI(STI.getTargetTriple(), STI.getFeatureBits(),
                               ABIName);

    const MCObjectFileInfo *MOFI = Parser.getContext().getObjectFileInfo();
    ParserOptions.IsPicEnabled = MOFI->isPositionIndependent();

    if (AddBuildAttributes)
      getTargetStreamer().emitTargetAttributes(STI, /*EmitStackAlign*/ false);
  }
};

/// RISCVOperand - Instances of this class represent a parsed machine
/// instruction
struct RISCVOperand final : public MCParsedAsmOperand {

  enum class KindTy {
    Token,
    Register,
    Immediate,
    FPImmediate,
    SystemRegister,
    VType,
    FRM,
    Fence,
    RegList,
    StackAdj,
    RegReg,
  } Kind;

  struct RegOp {
    MCRegister RegNum;
    bool IsGPRAsFPR;
  };

  struct ImmOp {
    const MCExpr *Val;
    bool IsRV64;
  };

  struct FPImmOp {
    uint64_t Val;
  };

  struct SysRegOp {
    const char *Data;
    unsigned Length;
    unsigned Encoding;
    // FIXME: Add the Encoding parsed fields as needed for checks,
    // e.g.: read/write or user/supervisor/machine privileges.
  };

  struct VTypeOp {
    unsigned Val;
  };

  struct FRMOp {
    RISCVFPRndMode::RoundingMode FRM;
  };

  struct FenceOp {
    unsigned Val;
  };

  struct RegListOp {
    unsigned Encoding;
  };

  struct StackAdjOp {
    unsigned Val;
  };

  struct RegRegOp {
    MCRegister BaseReg;
    MCRegister OffsetReg;
  };

  SMLoc StartLoc, EndLoc;
  union {
    StringRef Tok;
    RegOp Reg;
    ImmOp Imm;
    FPImmOp FPImm;
    SysRegOp SysReg;
    VTypeOp VType;
    FRMOp FRM;
    FenceOp Fence;
    RegListOp RegList;
    StackAdjOp StackAdj;
    RegRegOp RegReg;
  };

  RISCVOperand(KindTy K) : Kind(K) {}

public:
  RISCVOperand(const RISCVOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case KindTy::Register:
      Reg = o.Reg;
      break;
    case KindTy::Immediate:
      Imm = o.Imm;
      break;
    case KindTy::FPImmediate:
      FPImm = o.FPImm;
      break;
    case KindTy::Token:
      Tok = o.Tok;
      break;
    case KindTy::SystemRegister:
      SysReg = o.SysReg;
      break;
    case KindTy::VType:
      VType = o.VType;
      break;
    case KindTy::FRM:
      FRM = o.FRM;
      break;
    case KindTy::Fence:
      Fence = o.Fence;
      break;
    case KindTy::RegList:
      RegList = o.RegList;
      break;
    case KindTy::StackAdj:
      StackAdj = o.StackAdj;
      break;
    case KindTy::RegReg:
      RegReg = o.RegReg;
      break;
    }
  }

  bool isToken() const override { return Kind == KindTy::Token; }
  bool isReg() const override { return Kind == KindTy::Register; }
  bool isV0Reg() const {
    return Kind == KindTy::Register && Reg.RegNum == RISCV::V0;
  }
  bool isAnyReg() const {
    return Kind == KindTy::Register &&
           (RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(Reg.RegNum) ||
            RISCVMCRegisterClasses[RISCV::FPR64RegClassID].contains(Reg.RegNum) ||
            RISCVMCRegisterClasses[RISCV::VRRegClassID].contains(Reg.RegNum));
  }
  bool isAnyRegC() const {
    return Kind == KindTy::Register &&
           (RISCVMCRegisterClasses[RISCV::GPRCRegClassID].contains(
                Reg.RegNum) ||
            RISCVMCRegisterClasses[RISCV::FPR64CRegClassID].contains(
                Reg.RegNum));
  }
  bool isImm() const override { return Kind == KindTy::Immediate; }
  bool isMem() const override { return false; }
  bool isSystemRegister() const { return Kind == KindTy::SystemRegister; }
  bool isRegReg() const { return Kind == KindTy::RegReg; }
  bool isRegList() const { return Kind == KindTy::RegList; }
  bool isRegListS0() const {
    return Kind == KindTy::RegList && RegList.Encoding != RISCVZC::RA;
  }
  bool isStackAdj() const { return Kind == KindTy::StackAdj; }

  bool isGPR() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(Reg.RegNum);
  }

  bool isGPRPair() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRPairRegClassID].contains(
               Reg.RegNum);
  }

  bool isGPRPairC() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRPairCRegClassID].contains(
               Reg.RegNum);
  }

  bool isGPRPairNoX0() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRPairNoX0RegClassID].contains(
               Reg.RegNum);
  }

  bool isGPRF16() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRF16RegClassID].contains(Reg.RegNum);
  }

  bool isGPRF32() const {
    return Kind == KindTy::Register &&
           RISCVMCRegisterClasses[RISCV::GPRF32RegClassID].contains(Reg.RegNum);
  }

  bool isGPRAsFPR() const { return isGPR() && Reg.IsGPRAsFPR; }
  bool isGPRAsFPR16() const { return isGPRF16() && Reg.IsGPRAsFPR; }
  bool isGPRAsFPR32() const { return isGPRF32() && Reg.IsGPRAsFPR; }
  bool isGPRPairAsFPR64() const { return isGPRPair() && Reg.IsGPRAsFPR; }

  static bool evaluateConstantImm(const MCExpr *Expr, int64_t &Imm) {
    if (auto CE = dyn_cast<MCConstantExpr>(Expr)) {
      Imm = CE->getValue();
      return true;
    }

    return false;
  }

  // True if operand is a symbol with no modifiers, or a constant with no
  // modifiers and isShiftedInt<N-1, 1>(Op).
  template <int N> bool isBareSimmNLsb0() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isShiftedInt<N - 1, 1>(fixImmediateForRV32(Imm, isRV64Imm()));

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RISCVMCExpr::VK_None;
  }

  // True if operand is a symbol with no modifiers, or a constant with no
  // modifiers and isInt<N>(Op).
  template <int N> bool isBareSimmN() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isInt<N>(fixImmediateForRV32(Imm, isRV64Imm()));

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RISCVMCExpr::VK_None;
  }

  // Predicate methods for AsmOperands defined in RISCVInstrInfo.td

  bool isBareSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm))
      return false;

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RISCVMCExpr::VK_None;
  }

  bool isCallSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm))
      return false;

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == ELF::R_RISCV_CALL_PLT;
  }

  bool isPseudoJumpSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm))
      return false;

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == ELF::R_RISCV_CALL_PLT;
  }

  bool isTPRelAddSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm))
      return false;

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == ELF::R_RISCV_TPREL_ADD;
  }

  bool isTLSDESCCallSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    if (!isImm() || evaluateConstantImm(getImm(), Imm))
      return false;

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == ELF::R_RISCV_TLSDESC_CALL;
  }

  bool isCSRSystemRegister() const { return isSystemRegister(); }

  // If the last operand of the vsetvli/vsetvli instruction is a constant
  // expression, KindTy is Immediate.
  bool isVTypeI10() const {
    if (Kind == KindTy::VType)
      return true;
    return isUImm<10>();
  }
  bool isVTypeI11() const {
    if (Kind == KindTy::VType)
      return true;
    return isUImm<11>();
  }

  bool isXSfmmVType() const {
    return Kind == KindTy::VType && RISCVVType::isValidXSfmmVType(VType.Val);
  }

  /// Return true if the operand is a valid for the fence instruction e.g.
  /// ('iorw').
  bool isFenceArg() const { return Kind == KindTy::Fence; }

  /// Return true if the operand is a valid floating point rounding mode.
  bool isFRMArg() const { return Kind == KindTy::FRM; }
  bool isFRMArgLegacy() const { return Kind == KindTy::FRM; }
  bool isRTZArg() const { return isFRMArg() && FRM.FRM == RISCVFPRndMode::RTZ; }

  /// Return true if the operand is a valid fli.s floating-point immediate.
  bool isLoadFPImm() const {
    if (isImm())
      return isUImm5();
    if (Kind != KindTy::FPImmediate)
      return false;
    int Idx = RISCVLoadFPImm::getLoadFPImm(
        APFloat(APFloat::IEEEdouble(), APInt(64, getFPConst())));
    // Don't allow decimal version of the minimum value. It is a different value
    // for each supported data type.
    return Idx >= 0 && Idx != 1;
  }

  bool isImmXLenLI() const {
    int64_t Imm;
    if (!isImm())
      return false;
    // Given only Imm, ensuring that the actually specified constant is either
    // a signed or unsigned 64-bit number is unfortunately impossible.
    if (evaluateConstantImm(getImm(), Imm))
      return isRV64Imm() || (isInt<32>(Imm) || isUInt<32>(Imm));

    return RISCVAsmParser::isSymbolDiff(getImm());
  }

  bool isImmXLenLI_Restricted() const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    // 'la imm' supports constant immediates only.
    return IsConstantImm &&
           (isRV64Imm() || (isInt<32>(Imm) || isUInt<32>(Imm)));
  }

  template <unsigned N> bool isUImm() const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isUInt<N>(Imm);
  }

  template <unsigned N, unsigned S> bool isUImmShifted() const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isShiftedUInt<N, S>(Imm);
  }

  template <class Pred> bool isUImmPred(Pred p) const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && p(Imm);
  }

  bool isUImmLog2XLen() const {
    if (isImm() && isRV64Imm())
      return isUImm<6>();
    return isUImm<5>();
  }

  bool isUImmLog2XLenNonZero() const {
    if (isImm() && isRV64Imm())
      return isUImmPred([](int64_t Imm) { return Imm != 0 && isUInt<6>(Imm); });
    return isUImmPred([](int64_t Imm) { return Imm != 0 && isUInt<5>(Imm); });
  }

  bool isUImmLog2XLenHalf() const {
    if (isImm() && isRV64Imm())
      return isUImm<5>();
    return isUImm<4>();
  }

  bool isUImm1() const { return isUImm<1>(); }
  bool isUImm2() const { return isUImm<2>(); }
  bool isUImm3() const { return isUImm<3>(); }
  bool isUImm4() const { return isUImm<4>(); }
  bool isUImm5() const { return isUImm<5>(); }
  bool isUImm6() const { return isUImm<6>(); }
  bool isUImm7() const { return isUImm<7>(); }
  bool isUImm8() const { return isUImm<8>(); }
  bool isUImm10() const { return isUImm<10>(); }
  bool isUImm11() const { return isUImm<11>(); }
  bool isUImm16() const { return isUImm<16>(); }
  bool isUImm20() const { return isUImm<20>(); }
  bool isUImm32() const { return isUImm<32>(); }
  bool isUImm48() const { return isUImm<48>(); }
  bool isUImm64() const { return isUImm<64>(); }

  bool isUImm5NonZero() const {
    return isUImmPred([](int64_t Imm) { return Imm != 0 && isUInt<5>(Imm); });
  }

  bool isUImm5GT3() const {
    return isUImmPred([](int64_t Imm) { return isUInt<5>(Imm) && Imm > 3; });
  }

  bool isUImm5Plus1() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm > 0 && isUInt<5>(Imm - 1); });
  }

  bool isUImm5GE6Plus1() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm >= 6 && isUInt<5>(Imm - 1); });
  }

  bool isUImm5Slist() const {
    return isUImmPred([](int64_t Imm) {
      return (Imm == 0) || (Imm == 1) || (Imm == 2) || (Imm == 4) ||
             (Imm == 8) || (Imm == 16) || (Imm == 15) || (Imm == 31);
    });
  }

  bool isUImm8GE32() const {
    return isUImmPred([](int64_t Imm) { return isUInt<8>(Imm) && Imm >= 32; });
  }

  bool isRnumArg() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm >= INT64_C(0) && Imm <= INT64_C(10); });
  }

  bool isRnumArg_0_7() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm >= INT64_C(0) && Imm <= INT64_C(7); });
  }

  bool isRnumArg_1_10() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm >= INT64_C(1) && Imm <= INT64_C(10); });
  }

  bool isRnumArg_2_14() const {
    return isUImmPred(
        [](int64_t Imm) { return Imm >= INT64_C(2) && Imm <= INT64_C(14); });
  }

  template <unsigned N> bool isSImm() const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isInt<N>(fixImmediateForRV32(Imm, isRV64Imm()));
  }

  template <class Pred> bool isSImmPred(Pred p) const {
    int64_t Imm;
    if (!isImm())
      return false;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && p(fixImmediateForRV32(Imm, isRV64Imm()));
  }

  bool isSImm5() const { return isSImm<5>(); }
  bool isSImm6() const { return isSImm<6>(); }
  bool isSImm11() const { return isSImm<11>(); }
  bool isSImm16() const { return isSImm<16>(); }
  bool isSImm26() const { return isSImm<26>(); }

  bool isSImm5NonZero() const {
    return isSImmPred([](int64_t Imm) { return Imm != 0 && isInt<5>(Imm); });
  }

  bool isSImm6NonZero() const {
    return isSImmPred([](int64_t Imm) { return Imm != 0 && isInt<6>(Imm); });
  }

  bool isCLUIImm() const {
    return isUImmPred([](int64_t Imm) {
      return (isUInt<5>(Imm) && Imm != 0) || (Imm >= 0xfffe0 && Imm <= 0xfffff);
    });
  }

  bool isUImm2Lsb0() const { return isUImmShifted<1, 1>(); }

  bool isUImm5Lsb0() const { return isUImmShifted<4, 1>(); }

  bool isUImm6Lsb0() const { return isUImmShifted<5, 1>(); }

  bool isUImm7Lsb00() const { return isUImmShifted<5, 2>(); }

  bool isUImm7Lsb000() const { return isUImmShifted<4, 3>(); }

  bool isUImm8Lsb00() const { return isUImmShifted<6, 2>(); }

  bool isUImm8Lsb000() const { return isUImmShifted<5, 3>(); }

  bool isUImm9Lsb000() const { return isUImmShifted<6, 3>(); }

  bool isUImm14Lsb00() const { return isUImmShifted<12, 2>(); }

  bool isUImm10Lsb00NonZero() const {
    return isUImmPred(
        [](int64_t Imm) { return isShiftedUInt<8, 2>(Imm) && (Imm != 0); });
  }

  // If this a RV32 and the immediate is a uimm32, sign extend it to 32 bits.
  // This allows writing 'addi a0, a0, 0xffffffff'.
  static int64_t fixImmediateForRV32(int64_t Imm, bool IsRV64Imm) {
    if (IsRV64Imm || !isUInt<32>(Imm))
      return Imm;
    return SignExtend64<32>(Imm);
  }

  bool isSImm11Lsb0() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<10, 1>(Imm); });
  }

  bool isSImm12() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isInt<12>(fixImmediateForRV32(Imm, isRV64Imm()));

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           (VK == RISCVMCExpr::VK_LO || VK == RISCVMCExpr::VK_PCREL_LO ||
            VK == RISCVMCExpr::VK_TPREL_LO ||
            VK == ELF::R_RISCV_TLSDESC_LOAD_LO12 ||
            VK == ELF::R_RISCV_TLSDESC_ADD_LO12);
  }

  bool isSImm12Lsb00000() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<7, 5>(Imm); });
  }

  bool isSImm10Lsb0000NonZero() const {
    return isSImmPred(
        [](int64_t Imm) { return Imm != 0 && isShiftedInt<6, 4>(Imm); });
  }

  bool isSImm16NonZero() const {
    return isSImmPred([](int64_t Imm) { return Imm != 0 && isInt<16>(Imm); });
  }

  bool isUImm16NonZero() const {
    return isUImmPred([](int64_t Imm) { return isUInt<16>(Imm) && Imm != 0; });
  }

  bool isSImm20LI() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isInt<20>(fixImmediateForRV32(Imm, isRV64Imm()));

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           VK == RISCVMCExpr::VK_QC_ABS20;
  }

  bool isUImm20LUI() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isUInt<20>(Imm);

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           (VK == ELF::R_RISCV_HI20 || VK == ELF::R_RISCV_TPREL_HI20);
  }

  bool isUImm20AUIPC() const {
    if (!isImm())
      return false;

    int64_t Imm;
    if (evaluateConstantImm(getImm(), Imm))
      return isUInt<20>(Imm);

    RISCVMCExpr::Specifier VK = RISCVMCExpr::VK_None;
    return RISCVAsmParser::classifySymbolRef(getImm(), VK) &&
           (VK == ELF::R_RISCV_PCREL_HI20 || VK == ELF::R_RISCV_GOT_HI20 ||
            VK == ELF::R_RISCV_TLS_GOT_HI20 || VK == ELF::R_RISCV_TLS_GD_HI20 ||
            VK == ELF::R_RISCV_TLSDESC_HI20);
  }

  bool isImmZero() const {
    return isUImmPred([](int64_t Imm) { return 0 == Imm; });
  }

  bool isImmThree() const {
    return isUImmPred([](int64_t Imm) { return 3 == Imm; });
  }

  bool isImmFour() const {
    return isUImmPred([](int64_t Imm) { return 4 == Imm; });
  }

  bool isSImm5Plus1() const {
    return isSImmPred(
        [](int64_t Imm) { return Imm != INT64_MIN && isInt<5>(Imm - 1); });
  }

  bool isSImm18() const {
    return isSImmPred([](int64_t Imm) { return isInt<18>(Imm); });
  }

  bool isSImm18Lsb0() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<17, 1>(Imm); });
  }

  bool isSImm19Lsb00() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<17, 2>(Imm); });
  }

  bool isSImm20Lsb000() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<17, 3>(Imm); });
  }

  bool isSImm32Lsb0() const {
    return isSImmPred([](int64_t Imm) { return isShiftedInt<31, 1>(Imm); });
  }

  /// getStartLoc - Gets location of the first token of this operand
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Gets location of the last token of this operand
  SMLoc getEndLoc() const override { return EndLoc; }
  /// True if this operand is for an RV64 instruction
  bool isRV64Imm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.IsRV64;
  }

  MCRegister getReg() const override {
    assert(Kind == KindTy::Register && "Invalid type access!");
    return Reg.RegNum;
  }

  StringRef getSysReg() const {
    assert(Kind == KindTy::SystemRegister && "Invalid type access!");
    return StringRef(SysReg.Data, SysReg.Length);
  }

  const MCExpr *getImm() const {
    assert(Kind == KindTy::Immediate && "Invalid type access!");
    return Imm.Val;
  }

  uint64_t getFPConst() const {
    assert(Kind == KindTy::FPImmediate && "Invalid type access!");
    return FPImm.Val;
  }

  StringRef getToken() const {
    assert(Kind == KindTy::Token && "Invalid type access!");
    return Tok;
  }

  unsigned getVType() const {
    assert(Kind == KindTy::VType && "Invalid type access!");
    return VType.Val;
  }

  RISCVFPRndMode::RoundingMode getFRM() const {
    assert(Kind == KindTy::FRM && "Invalid type access!");
    return FRM.FRM;
  }

  unsigned getFence() const {
    assert(Kind == KindTy::Fence && "Invalid type access!");
    return Fence.Val;
  }

  void print(raw_ostream &OS) const override {
    auto RegName = [](MCRegister Reg) {
      if (Reg)
        return RISCVInstPrinter::getRegisterName(Reg);
      else
        return "noreg";
    };

    switch (Kind) {
    case KindTy::Immediate:
      OS << "<imm: " << *Imm.Val << " " << (Imm.IsRV64 ? "rv64" : "rv32")
         << ">";
      break;
    case KindTy::FPImmediate:
      OS << "<fpimm: " << FPImm.Val << ">";
      break;
    case KindTy::Register:
      OS << "<reg: " << RegName(Reg.RegNum) << " (" << Reg.RegNum
         << (Reg.IsGPRAsFPR ? ") GPRasFPR>" : ")>");
      break;
    case KindTy::Token:
      OS << "'" << getToken() << "'";
      break;
    case KindTy::SystemRegister:
      OS << "<sysreg: " << getSysReg() << " (" << SysReg.Encoding << ")>";
      break;
    case KindTy::VType:
      OS << "<vtype: ";
      RISCVVType::printVType(getVType(), OS);
      OS << '>';
      break;
    case KindTy::FRM:
      OS << "<frm: ";
      roundingModeToString(getFRM());
      OS << '>';
      break;
    case KindTy::Fence:
      OS << "<fence: ";
      OS << getFence();
      OS << '>';
      break;
    case KindTy::RegList:
      OS << "<reglist: ";
      RISCVZC::printRegList(RegList.Encoding, OS);
      OS << '>';
      break;
    case KindTy::StackAdj:
      OS << "<stackadj: ";
      OS << StackAdj.Val;
      OS << '>';
      break;
    case KindTy::RegReg:
      OS << "<RegReg: BaseReg " << RegName(RegReg.BaseReg) << " OffsetReg "
         << RegName(RegReg.OffsetReg);
      break;
    }
  }

  static std::unique_ptr<RISCVOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand>
  createReg(MCRegister Reg, SMLoc S, SMLoc E, bool IsGPRAsFPR = false) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::Register);
    Op->Reg.RegNum = Reg;
    Op->Reg.IsGPRAsFPR = IsGPRAsFPR;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createImm(const MCExpr *Val, SMLoc S,
                                                 SMLoc E, bool IsRV64) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::Immediate);
    Op->Imm.Val = Val;
    Op->Imm.IsRV64 = IsRV64;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createFPImm(uint64_t Val, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::FPImmediate);
    Op->FPImm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createSysReg(StringRef Str, SMLoc S,
                                                    unsigned Encoding) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::SystemRegister);
    Op->SysReg.Data = Str.data();
    Op->SysReg.Length = Str.size();
    Op->SysReg.Encoding = Encoding;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand>
  createFRMArg(RISCVFPRndMode::RoundingMode FRM, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::FRM);
    Op->FRM.FRM = FRM;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createFenceArg(unsigned Val, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::Fence);
    Op->Fence.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createVType(unsigned VTypeI, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::VType);
    Op->VType.Val = VTypeI;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createRegList(unsigned RlistEncode,
                                                   SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::RegList);
    Op->RegList.Encoding = RlistEncode;
    Op->StartLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand>
  createRegReg(MCRegister BaseReg, MCRegister OffsetReg, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::RegReg);
    Op->RegReg.BaseReg = BaseReg;
    Op->RegReg.OffsetReg = OffsetReg;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<RISCVOperand> createStackAdj(unsigned StackAdj, SMLoc S) {
    auto Op = std::make_unique<RISCVOperand>(KindTy::StackAdj);
    Op->StackAdj.Val = StackAdj;
    Op->StartLoc = S;
    return Op;
  }

  static void addExpr(MCInst &Inst, const MCExpr *Expr, bool IsRV64Imm) {
    assert(Expr && "Expr shouldn't be null!");
    int64_t Imm = 0;
    bool IsConstant = evaluateConstantImm(Expr, Imm);

    if (IsConstant)
      Inst.addOperand(
          MCOperand::createImm(fixImmediateForRV32(Imm, IsRV64Imm)));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm(), isRV64Imm());
  }

  void addFPImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (isImm()) {
      addExpr(Inst, getImm(), isRV64Imm());
      return;
    }

    int Imm = RISCVLoadFPImm::getLoadFPImm(
        APFloat(APFloat::IEEEdouble(), APInt(64, getFPConst())));
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void addFenceArgOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(Fence.Val));
  }

  void addCSRSystemRegisterOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(SysReg.Encoding));
  }

  // Support non-canonical syntax:
  // "vsetivli rd, uimm, 0xabc" or "vsetvli rd, rs1, 0xabc"
  // "vsetivli rd, uimm, (0xc << N)" or "vsetvli rd, rs1, (0xc << N)"
  void addVTypeIOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    int64_t Imm = 0;
    if (Kind == KindTy::Immediate) {
      [[maybe_unused]] bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
      assert(IsConstantImm && "Invalid VTypeI Operand!");
    } else {
      Imm = getVType();
    }
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void addRegListOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(RegList.Encoding));
  }

  void addRegRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(RegReg.BaseReg));
    Inst.addOperand(MCOperand::createReg(RegReg.OffsetReg));
  }

  void addStackAdjOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(StackAdj.Val));
  }

  void addFRMArgOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createImm(getFRM()));
  }
};
} // end anonymous namespace.

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "RISCVGenAsmMatcher.inc"

static MCRegister convertFPR64ToFPR16(MCRegister Reg) {
  assert(Reg >= RISCV::F0_D && Reg <= RISCV::F31_D && "Invalid register");
  return Reg - RISCV::F0_D + RISCV::F0_H;
}

static MCRegister convertFPR64ToFPR32(MCRegister Reg) {
  assert(Reg >= RISCV::F0_D && Reg <= RISCV::F31_D && "Invalid register");
  return Reg - RISCV::F0_D + RISCV::F0_F;
}

static MCRegister convertFPR64ToFPR128(MCRegister Reg) {
  assert(Reg >= RISCV::F0_D && Reg <= RISCV::F31_D && "Invalid register");
  return Reg - RISCV::F0_D + RISCV::F0_Q;
}

static MCRegister convertVRToVRMx(const MCRegisterInfo &RI, MCRegister Reg,
                                  unsigned Kind) {
  unsigned RegClassID;
  if (Kind == MCK_VRM2)
    RegClassID = RISCV::VRM2RegClassID;
  else if (Kind == MCK_VRM4)
    RegClassID = RISCV::VRM4RegClassID;
  else if (Kind == MCK_VRM8)
    RegClassID = RISCV::VRM8RegClassID;
  else
    return MCRegister();
  return RI.getMatchingSuperReg(Reg, RISCV::sub_vrm1_0,
                                &RISCVMCRegisterClasses[RegClassID]);
}

unsigned RISCVAsmParser::validateTargetOperandClass(MCParsedAsmOperand &AsmOp,
                                                    unsigned Kind) {
  RISCVOperand &Op = static_cast<RISCVOperand &>(AsmOp);
  if (!Op.isReg())
    return Match_InvalidOperand;

  MCRegister Reg = Op.getReg();
  bool IsRegFPR64 =
      RISCVMCRegisterClasses[RISCV::FPR64RegClassID].contains(Reg);
  bool IsRegFPR64C =
      RISCVMCRegisterClasses[RISCV::FPR64CRegClassID].contains(Reg);
  bool IsRegVR = RISCVMCRegisterClasses[RISCV::VRRegClassID].contains(Reg);

  if (IsRegFPR64 && Kind == MCK_FPR128) {
    Op.Reg.RegNum = convertFPR64ToFPR128(Reg);
    return Match_Success;
  }
  // As the parser couldn't differentiate an FPR32 from an FPR64, coerce the
  // register from FPR64 to FPR32 or FPR64C to FPR32C if necessary.
  if ((IsRegFPR64 && Kind == MCK_FPR32) ||
      (IsRegFPR64C && Kind == MCK_FPR32C)) {
    Op.Reg.RegNum = convertFPR64ToFPR32(Reg);
    return Match_Success;
  }
  // As the parser couldn't differentiate an FPR16 from an FPR64, coerce the
  // register from FPR64 to FPR16 if necessary.
  if (IsRegFPR64 && Kind == MCK_FPR16) {
    Op.Reg.RegNum = convertFPR64ToFPR16(Reg);
    return Match_Success;
  }
  if (Kind == MCK_GPRAsFPR16 && Op.isGPRAsFPR()) {
    Op.Reg.RegNum = Reg - RISCV::X0 + RISCV::X0_H;
    return Match_Success;
  }
  if (Kind == MCK_GPRAsFPR32 && Op.isGPRAsFPR()) {
    Op.Reg.RegNum = Reg - RISCV::X0 + RISCV::X0_W;
    return Match_Success;
  }

  // There are some GPRF64AsFPR instructions that have no RV32 equivalent. We
  // reject them at parsing thinking we should match as GPRPairAsFPR for RV32.
  // So we explicitly accept them here for RV32 to allow the generic code to
  // report that the instruction requires RV64.
  if (RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(Reg) &&
      Kind == MCK_GPRF64AsFPR && STI->hasFeature(RISCV::FeatureStdExtZdinx) &&
      !isRV64())
    return Match_Success;

  // As the parser couldn't differentiate an VRM2/VRM4/VRM8 from an VR, coerce
  // the register from VR to VRM2/VRM4/VRM8 if necessary.
  if (IsRegVR && (Kind == MCK_VRM2 || Kind == MCK_VRM4 || Kind == MCK_VRM8)) {
    Op.Reg.RegNum = convertVRToVRMx(*getContext().getRegisterInfo(), Reg, Kind);
    if (!Op.Reg.RegNum)
      return Match_InvalidOperand;
    return Match_Success;
  }
  return Match_InvalidOperand;
}

bool RISCVAsmParser::generateImmOutOfRangeError(
    SMLoc ErrorLoc, int64_t Lower, int64_t Upper,
    const Twine &Msg = "immediate must be an integer in the range") {
  return Error(ErrorLoc, Msg + " [" + Twine(Lower) + ", " + Twine(Upper) + "]");
}

bool RISCVAsmParser::generateImmOutOfRangeError(
    OperandVector &Operands, uint64_t ErrorInfo, int64_t Lower, int64_t Upper,
    const Twine &Msg = "immediate must be an integer in the range") {
  SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
  return generateImmOutOfRangeError(ErrorLoc, Lower, Upper, Msg);
}

bool RISCVAsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
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
    if (validateInstruction(Inst, Operands))
      return true;
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
    std::string Suggestion = RISCVMnemonicSpellCheck(
        ((RISCVOperand &)*Operands[0]).getToken(), FBS, 0);
    return Error(IDLoc, "unrecognized instruction mnemonic" + Suggestion);
  }
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
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
  case Match_InvalidImmXLenLI:
    if (isRV64()) {
      SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
      return Error(ErrorLoc, "operand must be a constant 64-bit integer");
    }
    return generateImmOutOfRangeError(Operands, ErrorInfo,
                                      std::numeric_limits<int32_t>::min(),
                                      std::numeric_limits<uint32_t>::max());
  case Match_InvalidImmXLenLI_Restricted:
    if (isRV64()) {
      SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
      return Error(ErrorLoc, "operand either must be a constant 64-bit integer "
                             "or a bare symbol name");
    }
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, std::numeric_limits<int32_t>::min(),
        std::numeric_limits<uint32_t>::max(),
        "operand either must be a bare symbol name or an immediate integer in "
        "the range");
  case Match_InvalidUImmLog2XLen:
    if (isRV64())
      return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 6) - 1);
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 5) - 1);
  case Match_InvalidUImmLog2XLenNonZero:
    if (isRV64())
      return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 6) - 1);
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 5) - 1);
  case Match_InvalidUImm1:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 1) - 1);
  case Match_InvalidUImm2:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 2) - 1);
  case Match_InvalidUImm2Lsb0:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, 2,
                                      "immediate must be one of");
  case Match_InvalidUImm3:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 3) - 1);
  case Match_InvalidUImm4:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 4) - 1);
  case Match_InvalidUImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 5) - 1);
  case Match_InvalidUImm5NonZero:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 5) - 1);
  case Match_InvalidUImm5GT3:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 4, (1 << 5) - 1);
  case Match_InvalidUImm5Plus1:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 5));
  case Match_InvalidUImm5GE6Plus1:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 6, (1 << 5));
  case Match_InvalidUImm5Slist: {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc,
                 "immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31");
  }
  case Match_InvalidUImm6:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 6) - 1);
  case Match_InvalidUImm7:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 7) - 1);
  case Match_InvalidUImm8:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 8) - 1);
  case Match_InvalidUImm8GE32:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 32, (1 << 8) - 1);
  case Match_InvalidSImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 4),
                                      (1 << 4) - 1);
  case Match_InvalidSImm5NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 4), (1 << 4) - 1,
        "immediate must be non-zero in the range");
  case Match_InvalidSImm6:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 5),
                                      (1 << 5) - 1);
  case Match_InvalidSImm6NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 5), (1 << 5) - 1,
        "immediate must be non-zero in the range");
  case Match_InvalidCLUIImm:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 1, (1 << 5) - 1,
        "immediate must be in [0xfffe0, 0xfffff] or");
  case Match_InvalidUImm5Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 5) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm6Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 6) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm7Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 7) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm8Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 8) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm8Lsb000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 8) - 8,
        "immediate must be a multiple of 8 bytes in the range");
  case Match_InvalidBareSImm9Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 8), (1 << 8) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm9Lsb000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 9) - 8,
        "immediate must be a multiple of 8 bytes in the range");
  case Match_InvalidUImm10Lsb00NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 4, (1 << 10) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidSImm10Lsb0000NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 9), (1 << 9) - 16,
        "immediate must be a multiple of 16 bytes and non-zero in the range");
  case Match_InvalidSImm11:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 10),
                                      (1 << 10) - 1);
  case Match_InvalidSImm11Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 10), (1 << 10) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm10:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 10) - 1);
  case Match_InvalidUImm11:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 11) - 1);
  case Match_InvalidUImm14Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 14) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm16NonZero:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 16) - 1);
  case Match_InvalidSImm12:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 1,
        "operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an "
        "integer in the range");
  case Match_InvalidBareSImm12Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidSImm12Lsb00000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 11), (1 << 11) - 32,
        "immediate must be a multiple of 32 bytes in the range");
  case Match_InvalidBareSImm13Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 12), (1 << 12) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidSImm16:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 15),
                                      (1 << 15) - 1);
  case Match_InvalidSImm16NonZero:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 15), (1 << 15) - 1,
        "immediate must be non-zero in the range");
  case Match_InvalidSImm20LI:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 19), (1 << 19) - 1,
        "operand must be a symbol with a %qc.abs20 specifier or an integer "
        " in the range");
  case Match_InvalidUImm20LUI:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 20) - 1,
        "operand must be a symbol with "
        "%hi/%tprel_hi specifier or an integer in "
        "the range");
  case Match_InvalidUImm20:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 20) - 1);
  case Match_InvalidUImm20AUIPC:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 20) - 1,
        "operand must be a symbol with a "
        "%pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier "
        "or "
        "an integer in the range");
  case Match_InvalidBareSImm21Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 20), (1 << 20) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidCSRSystemRegister: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 12) - 1,
                                      "operand must be a valid system register "
                                      "name or an integer in the range");
  }
  case Match_InvalidVTypeI: {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
    return generateVTypeError(ErrorLoc);
  }
  case Match_InvalidSImm5Plus1: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 4) + 1,
                                      (1 << 4),
                                      "immediate must be in the range");
  }
  case Match_InvalidSImm18:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 17),
                                      (1 << 17) - 1);
  case Match_InvalidSImm18Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 17), (1 << 17) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidSImm19Lsb00:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 18), (1 << 18) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidSImm20Lsb000:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, -(1 << 19), (1 << 19) - 8,
        "immediate must be a multiple of 8 bytes in the range");
  case Match_InvalidSImm26:
    return generateImmOutOfRangeError(Operands, ErrorInfo, -(1 << 25),
                                      (1 << 25) - 1);
  case Match_InvalidBareSImm32:
    return generateImmOutOfRangeError(Operands, ErrorInfo,
                                      std::numeric_limits<int32_t>::min(),
                                      std::numeric_limits<uint32_t>::max());
  case Match_InvalidBareSImm32Lsb0:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max() - 1,
        "operand must be a multiple of 2 bytes in the range");
  case Match_InvalidRnumArg: {
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, 10);
  }
  case Match_InvalidStackAdj: {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(
        ErrorLoc,
        "stack adjustment is invalid for this instruction and register list");
  }
  }

  if (const char *MatchDiag = getMatchKindDiag((RISCVMatchResultTy)Result)) {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, MatchDiag);
  }

  llvm_unreachable("Unknown match type detected!");
}

// Attempts to match Name as a register (either using the default name or
// alternative ABI names), returning the matching register. Upon failure,
// returns a non-valid MCRegister. If IsRVE, then registers x16-x31 will be
// rejected.
MCRegister RISCVAsmParser::matchRegisterNameHelper(StringRef Name) const {
  MCRegister Reg = MatchRegisterName(Name);
  // The 16-/32-/128- and 64-bit FPRs have the same asm name. Check
  // that the initial match always matches the 64-bit variant, and
  // not the 16/32/128-bit one.
  assert(!(Reg >= RISCV::F0_H && Reg <= RISCV::F31_H));
  assert(!(Reg >= RISCV::F0_F && Reg <= RISCV::F31_F));
  assert(!(Reg >= RISCV::F0_Q && Reg <= RISCV::F31_Q));
  // The default FPR register class is based on the tablegen enum ordering.
  static_assert(RISCV::F0_D < RISCV::F0_H, "FPR matching must be updated");
  static_assert(RISCV::F0_D < RISCV::F0_F, "FPR matching must be updated");
  static_assert(RISCV::F0_D < RISCV::F0_Q, "FPR matching must be updated");
  if (!Reg)
    Reg = MatchRegisterAltName(Name);
  if (isRVE() && Reg >= RISCV::X16 && Reg <= RISCV::X31)
    Reg = MCRegister();
  return Reg;
}

bool RISCVAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                   SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus RISCVAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
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

ParseStatus RISCVAsmParser::parseRegister(OperandVector &Operands,
                                          bool AllowParens) {
  SMLoc FirstS = getLoc();
  bool HadParens = false;
  AsmToken LParen;

  // If this is an LParen and a parenthesised register name is allowed, parse it
  // atomically.
  if (AllowParens && getLexer().is(AsmToken::LParen)) {
    AsmToken Buf[2];
    size_t ReadCount = getLexer().peekTokens(Buf);
    if (ReadCount == 2 && Buf[1].getKind() == AsmToken::RParen) {
      HadParens = true;
      LParen = getParser().getTok();
      getParser().Lex(); // Eat '('
    }
  }

  switch (getLexer().getKind()) {
  default:
    if (HadParens)
      getLexer().UnLex(LParen);
    return ParseStatus::NoMatch;
  case AsmToken::Identifier:
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister Reg = matchRegisterNameHelper(Name);

    if (!Reg) {
      if (HadParens)
        getLexer().UnLex(LParen);
      return ParseStatus::NoMatch;
    }
    if (HadParens)
      Operands.push_back(RISCVOperand::createToken("(", FirstS));
    SMLoc S = getLoc();
    SMLoc E = getTok().getEndLoc();
    getLexer().Lex();
    Operands.push_back(RISCVOperand::createReg(Reg, S, E));
  }

  if (HadParens) {
    getParser().Lex(); // Eat ')'
    Operands.push_back(RISCVOperand::createToken(")", getLoc()));
  }

  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseInsnDirectiveOpcode(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
  case AsmToken::Integer:
  case AsmToken::String: {
    if (getParser().parseExpression(Res, E))
      return ParseStatus::Failure;

    auto *CE = dyn_cast<MCConstantExpr>(Res);
    if (CE) {
      int64_t Imm = CE->getValue();
      if (isUInt<7>(Imm)) {
        Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
        return ParseStatus::Success;
      }
    }

    break;
  }
  case AsmToken::Identifier: {
    StringRef Identifier;
    if (getParser().parseIdentifier(Identifier))
      return ParseStatus::Failure;

    auto Opcode = RISCVInsnOpcode::lookupRISCVOpcodeByName(Identifier);
    if (Opcode) {
      assert(isUInt<7>(Opcode->Value) && (Opcode->Value & 0x3) == 3 &&
             "Unexpected opcode");
      Res = MCConstantExpr::create(Opcode->Value, getContext());
      E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());
      Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
      return ParseStatus::Success;
    }

    break;
  }
  case AsmToken::Percent:
    break;
  }

  return generateImmOutOfRangeError(
      S, 0, 127,
      "opcode must be a valid opcode name or an immediate in the range");
}

ParseStatus RISCVAsmParser::parseInsnCDirectiveOpcode(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
  case AsmToken::Integer:
  case AsmToken::String: {
    if (getParser().parseExpression(Res, E))
      return ParseStatus::Failure;

    auto *CE = dyn_cast<MCConstantExpr>(Res);
    if (CE) {
      int64_t Imm = CE->getValue();
      if (Imm >= 0 && Imm <= 2) {
        Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
        return ParseStatus::Success;
      }
    }

    break;
  }
  case AsmToken::Identifier: {
    StringRef Identifier;
    if (getParser().parseIdentifier(Identifier))
      return ParseStatus::Failure;

    unsigned Opcode;
    if (Identifier == "C0")
      Opcode = 0;
    else if (Identifier == "C1")
      Opcode = 1;
    else if (Identifier == "C2")
      Opcode = 2;
    else
      break;

    Res = MCConstantExpr::create(Opcode, getContext());
    E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());
    Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
    return ParseStatus::Success;
  }
  case AsmToken::Percent: {
    // Discard operand with modifier.
    break;
  }
  }

  return generateImmOutOfRangeError(
      S, 0, 2,
      "opcode must be a valid opcode name or an immediate in the range");
}

ParseStatus RISCVAsmParser::parseCSRSystemRegister(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Res;

  auto SysRegFromConstantInt = [this](const MCExpr *E, SMLoc S) {
    if (auto *CE = dyn_cast<MCConstantExpr>(E)) {
      int64_t Imm = CE->getValue();
      if (isUInt<12>(Imm)) {
        auto Range = RISCVSysReg::lookupSysRegByEncoding(Imm);
        // Accept an immediate representing a named Sys Reg if it satisfies the
        // the required features.
        for (auto &Reg : Range) {
          if (Reg.IsAltName || Reg.IsDeprecatedName)
            continue;
          if (Reg.haveRequiredFeatures(STI->getFeatureBits()))
            return RISCVOperand::createSysReg(Reg.Name, S, Imm);
        }
        // Accept an immediate representing an un-named Sys Reg if the range is
        // valid, regardless of the required features.
        return RISCVOperand::createSysReg("", S, Imm);
      }
    }
    return std::unique_ptr<RISCVOperand>();
  };

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
  case AsmToken::Integer:
  case AsmToken::String: {
    if (getParser().parseExpression(Res))
      return ParseStatus::Failure;

    if (auto SysOpnd = SysRegFromConstantInt(Res, S)) {
      Operands.push_back(std::move(SysOpnd));
      return ParseStatus::Success;
    }

    return generateImmOutOfRangeError(S, 0, (1 << 12) - 1);
  }
  case AsmToken::Identifier: {
    StringRef Identifier;
    if (getParser().parseIdentifier(Identifier))
      return ParseStatus::Failure;

    const auto *SysReg = RISCVSysReg::lookupSysRegByName(Identifier);

    if (SysReg) {
      if (SysReg->IsDeprecatedName) {
        // Lookup the undeprecated name.
        auto Range = RISCVSysReg::lookupSysRegByEncoding(SysReg->Encoding);
        for (auto &Reg : Range) {
          if (Reg.IsAltName || Reg.IsDeprecatedName)
            continue;
          Warning(S, "'" + Identifier + "' is a deprecated alias for '" +
                         Reg.Name + "'");
        }
      }

      // Accept a named Sys Reg if the required features are present.
      const auto &FeatureBits = getSTI().getFeatureBits();
      if (!SysReg->haveRequiredFeatures(FeatureBits)) {
        const auto *Feature = llvm::find_if(RISCVFeatureKV, [&](auto Feature) {
          return SysReg->FeaturesRequired[Feature.Value];
        });
        auto ErrorMsg = std::string("system register '") + SysReg->Name + "' ";
        if (SysReg->IsRV32Only && FeatureBits[RISCV::Feature64Bit]) {
          ErrorMsg += "is RV32 only";
          if (Feature != std::end(RISCVFeatureKV))
            ErrorMsg += " and ";
        }
        if (Feature != std::end(RISCVFeatureKV)) {
          ErrorMsg +=
              "requires '" + std::string(Feature->Key) + "' to be enabled";
        }

        return Error(S, ErrorMsg);
      }
      Operands.push_back(
          RISCVOperand::createSysReg(Identifier, S, SysReg->Encoding));
      return ParseStatus::Success;
    }

    // Accept a symbol name that evaluates to an absolute value.
    MCSymbol *Sym = getContext().lookupSymbol(Identifier);
    if (Sym && Sym->isVariable()) {
      // Pass false for SetUsed, since redefining the value later does not
      // affect this instruction.
      if (auto SysOpnd = SysRegFromConstantInt(
              Sym->getVariableValue(/*SetUsed=*/false), S)) {
        Operands.push_back(std::move(SysOpnd));
        return ParseStatus::Success;
      }
    }

    return generateImmOutOfRangeError(S, 0, (1 << 12) - 1,
                                      "operand must be a valid system register "
                                      "name or an integer in the range");
  }
  case AsmToken::Percent: {
    // Discard operand with modifier.
    return generateImmOutOfRangeError(S, 0, (1 << 12) - 1);
  }
  }

  return ParseStatus::NoMatch;
}

ParseStatus RISCVAsmParser::parseFPImm(OperandVector &Operands) {
  SMLoc S = getLoc();

  // Parse special floats (inf/nan/min) representation.
  if (getTok().is(AsmToken::Identifier)) {
    StringRef Identifier = getTok().getIdentifier();
    if (Identifier.compare_insensitive("inf") == 0) {
      Operands.push_back(
          RISCVOperand::createImm(MCConstantExpr::create(30, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else if (Identifier.compare_insensitive("nan") == 0) {
      Operands.push_back(
          RISCVOperand::createImm(MCConstantExpr::create(31, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else if (Identifier.compare_insensitive("min") == 0) {
      Operands.push_back(
          RISCVOperand::createImm(MCConstantExpr::create(1, getContext()), S,
                                  getTok().getEndLoc(), isRV64()));
    } else {
      return TokError("invalid floating point literal");
    }

    Lex(); // Eat the token.

    return ParseStatus::Success;
  }

  // Handle negation, as that still comes through as a separate token.
  bool IsNegative = parseOptionalToken(AsmToken::Minus);

  const AsmToken &Tok = getTok();
  if (!Tok.is(AsmToken::Real))
    return TokError("invalid floating point immediate");

  // Parse FP representation.
  APFloat RealVal(APFloat::IEEEdouble());
  auto StatusOrErr =
      RealVal.convertFromString(Tok.getString(), APFloat::rmTowardZero);
  if (errorToBool(StatusOrErr.takeError()))
    return TokError("invalid floating point representation");

  if (IsNegative)
    RealVal.changeSign();

  Operands.push_back(RISCVOperand::createFPImm(
      RealVal.bitcastToAPInt().getZExtValue(), S));

  Lex(); // Eat the token.

  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseImmediate(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;
  case AsmToken::LParen:
  case AsmToken::Dot:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Exclaim:
  case AsmToken::Tilde:
  case AsmToken::Integer:
  case AsmToken::String:
  case AsmToken::Identifier:
    if (getParser().parseExpression(Res, E))
      return ParseStatus::Failure;
    break;
  case AsmToken::Percent:
    return parseOperandWithSpecifier(Operands);
  }

  Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseOperandWithSpecifier(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;

  if (parseToken(AsmToken::Percent, "expected '%' relocation specifier"))
    return ParseStatus::Failure;
  const MCExpr *Expr = nullptr;
  bool Failed = parseExprWithSpecifier(Expr, E);
  if (!Failed)
    Operands.push_back(RISCVOperand::createImm(Expr, S, E, isRV64()));
  return Failed;
}

bool RISCVAsmParser::parseExprWithSpecifier(const MCExpr *&Res, SMLoc &E) {
  if (getLexer().getKind() != AsmToken::Identifier)
    return Error(getLoc(), "expected '%' relocation specifier");
  StringRef Identifier = getParser().getTok().getIdentifier();
  auto Spec = RISCVMCExpr::getSpecifierForName(Identifier);
  if (!Spec)
    return Error(getLoc(), "invalid relocation specifier");

  getParser().Lex(); // Eat the identifier
  if (parseToken(AsmToken::LParen, "expected '('"))
    return true;

  const MCExpr *SubExpr;
  if (getParser().parseParenExpression(SubExpr, E))
    return true;

  Res = RISCVMCExpr::create(SubExpr, *Spec, getContext());
  return false;
}

bool RISCVAsmParser::parseDataExpr(const MCExpr *&Res) {
  SMLoc E;
  if (parseOptionalToken(AsmToken::Percent))
    return parseExprWithSpecifier(Res, E);
  return getParser().parseExpression(Res);
}

ParseStatus RISCVAsmParser::parseBareSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Res;

  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;

  StringRef Identifier;
  AsmToken Tok = getLexer().getTok();

  if (getParser().parseIdentifier(Identifier))
    return ParseStatus::Failure;

  SMLoc E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);

  if (Sym->isVariable()) {
    const MCExpr *V = Sym->getVariableValue(/*SetUsed=*/false);
    if (!isa<MCSymbolRefExpr>(V)) {
      getLexer().UnLex(Tok); // Put back if it's not a bare symbol.
      return ParseStatus::NoMatch;
    }
    Res = V;
  } else
    Res = MCSymbolRefExpr::create(Sym, getContext());

  MCBinaryExpr::Opcode Opcode;
  switch (getLexer().getKind()) {
  default:
    Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
    return ParseStatus::Success;
  case AsmToken::Plus:
    Opcode = MCBinaryExpr::Add;
    getLexer().Lex();
    break;
  case AsmToken::Minus:
    Opcode = MCBinaryExpr::Sub;
    getLexer().Lex();
    break;
  }

  const MCExpr *Expr;
  if (getParser().parseExpression(Expr, E))
    return ParseStatus::Failure;
  Res = MCBinaryExpr::create(Opcode, Res, Expr, getContext());
  Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseCallSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  const MCExpr *Res;

  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;
  std::string Identifier(getTok().getIdentifier());

  if (getLexer().peekTok().is(AsmToken::At)) {
    Lex();
    Lex();
    StringRef PLT;
    SMLoc Loc = getLoc();
    if (getParser().parseIdentifier(PLT) || PLT != "plt")
      return Error(Loc, "@ (except the deprecated/ignored @plt) is disallowed");
  } else if (!getLexer().peekTok().is(AsmToken::EndOfStatement)) {
    // Avoid parsing the register in `call rd, foo` as a call symbol.
    return ParseStatus::NoMatch;
  } else {
    Lex();
  }

  SMLoc E = SMLoc::getFromPointer(S.getPointer() + Identifier.size());
  RISCVMCExpr::Specifier Kind = ELF::R_RISCV_CALL_PLT;

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);
  Res = MCSymbolRefExpr::create(Sym, getContext());
  Res = RISCVMCExpr::create(Res, Kind, getContext());
  Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parsePseudoJumpSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E;
  const MCExpr *Res;

  if (getParser().parseExpression(Res, E))
    return ParseStatus::Failure;

  if (Res->getKind() != MCExpr::ExprKind::SymbolRef)
    return Error(S, "operand must be a valid jump target");

  Res = RISCVMCExpr::create(Res, ELF::R_RISCV_CALL_PLT, getContext());
  Operands.push_back(RISCVOperand::createImm(Res, S, E, isRV64()));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseJALOffset(OperandVector &Operands) {
  // Parsing jal operands is fiddly due to the `jal foo` and `jal ra, foo`
  // both being acceptable forms. When parsing `jal ra, foo` this function
  // will be called for the `ra` register operand in an attempt to match the
  // single-operand alias. parseJALOffset must fail for this case. It would
  // seem logical to try parse the operand using parseImmediate and return
  // NoMatch if the next token is a comma (meaning we must be parsing a jal in
  // the second form rather than the first). We can't do this as there's no
  // way of rewinding the lexer state. Instead, return NoMatch if this operand
  // is an identifier and is followed by a comma.
  if (getLexer().is(AsmToken::Identifier) &&
      getLexer().peekTok().is(AsmToken::Comma))
    return ParseStatus::NoMatch;

  return parseImmediate(Operands);
}

bool RISCVAsmParser::parseVTypeToken(const AsmToken &Tok, VTypeState &State,
                                     unsigned &Sew, unsigned &Lmul,
                                     bool &Fractional, bool &TailAgnostic,
                                     bool &MaskAgnostic) {
  if (Tok.isNot(AsmToken::Identifier))
    return true;

  StringRef Identifier = Tok.getIdentifier();

  switch (State) {
  case VTypeState_SEW:
    if (!Identifier.consume_front("e"))
      break;
    if (Identifier.getAsInteger(10, Sew))
      break;
    if (!RISCVVType::isValidSEW(Sew))
      break;
    State = VTypeState_LMUL;
    return false;
  case VTypeState_LMUL: {
    if (!Identifier.consume_front("m"))
      break;
    Fractional = Identifier.consume_front("f");
    if (Identifier.getAsInteger(10, Lmul))
      break;
    if (!RISCVVType::isValidLMUL(Lmul, Fractional))
      break;

    if (Fractional) {
      unsigned ELEN = STI->hasFeature(RISCV::FeatureStdExtZve64x) ? 64 : 32;
      unsigned MinLMUL = ELEN / 8;
      if (Lmul > MinLMUL)
        Warning(Tok.getLoc(),
                "use of vtype encodings with LMUL < SEWMIN/ELEN == mf" +
                    Twine(MinLMUL) + " is reserved");
    }

    State = VTypeState_TailPolicy;
    return false;
  }
  case VTypeState_TailPolicy:
    if (Identifier == "ta")
      TailAgnostic = true;
    else if (Identifier == "tu")
      TailAgnostic = false;
    else
      break;
    State = VTypeState_MaskPolicy;
    return false;
  case VTypeState_MaskPolicy:
    if (Identifier == "ma")
      MaskAgnostic = true;
    else if (Identifier == "mu")
      MaskAgnostic = false;
    else
      break;
    State = VTypeState_Done;
    return false;
  case VTypeState_Done:
    // Extra token?
    break;
  }

  return true;
}

ParseStatus RISCVAsmParser::parseVTypeI(OperandVector &Operands) {
  SMLoc S = getLoc();

  unsigned Sew = 0;
  unsigned Lmul = 0;
  bool Fractional = false;
  bool TailAgnostic = false;
  bool MaskAgnostic = false;

  VTypeState State = VTypeState_SEW;
  SMLoc SEWLoc = S;

  if (parseVTypeToken(getTok(), State, Sew, Lmul, Fractional, TailAgnostic,
                      MaskAgnostic))
    return ParseStatus::NoMatch;

  getLexer().Lex();

  while (parseOptionalToken(AsmToken::Comma)) {
    if (parseVTypeToken(getTok(), State, Sew, Lmul, Fractional, TailAgnostic,
                        MaskAgnostic))
      break;

    getLexer().Lex();
  }

  if (getLexer().is(AsmToken::EndOfStatement) && State == VTypeState_Done) {
    RISCVVType::VLMUL VLMUL = RISCVVType::encodeLMUL(Lmul, Fractional);
    if (Fractional) {
      unsigned ELEN = STI->hasFeature(RISCV::FeatureStdExtZve64x) ? 64 : 32;
      unsigned MaxSEW = ELEN / Lmul;
      // If MaxSEW < 8, we should have printed warning about reserved LMUL.
      if (MaxSEW >= 8 && Sew > MaxSEW)
        Warning(SEWLoc,
                "use of vtype encodings with SEW > " + Twine(MaxSEW) +
                    " and LMUL == mf" + Twine(Lmul) +
                    " may not be compatible with all RVV implementations");
    }

    unsigned VTypeI =
        RISCVVType::encodeVTYPE(VLMUL, Sew, TailAgnostic, MaskAgnostic);
    Operands.push_back(RISCVOperand::createVType(VTypeI, S));
    return ParseStatus::Success;
  }

  return generateVTypeError(S);
}

bool RISCVAsmParser::generateVTypeError(SMLoc ErrorLoc) {
  return Error(
      ErrorLoc,
      "operand must be "
      "e[8|16|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]");
}

ParseStatus RISCVAsmParser::parseXSfmmVType(OperandVector &Operands) {
  SMLoc S = getLoc();

  unsigned Widen = 0;
  unsigned SEW = 0;
  bool AltFmt = false;
  StringRef Identifier;

  if (getTok().isNot(AsmToken::Identifier))
    goto Fail;

  Identifier = getTok().getIdentifier();

  if (!Identifier.consume_front("e"))
    goto Fail;

  if (Identifier.getAsInteger(10, SEW)) {
    if (Identifier != "16alt")
      goto Fail;

    AltFmt = true;
    SEW = 16;
  }
  if (!RISCVVType::isValidSEW(SEW))
    goto Fail;

  Lex();

  if (!parseOptionalToken(AsmToken::Comma))
    goto Fail;

  if (getTok().isNot(AsmToken::Identifier))
    goto Fail;

  Identifier = getTok().getIdentifier();

  if (!Identifier.consume_front("w"))
    goto Fail;
  if (Identifier.getAsInteger(10, Widen))
    goto Fail;
  if (Widen != 1 && Widen != 2 && Widen != 4)
    goto Fail;

  Lex();

  if (getLexer().is(AsmToken::EndOfStatement)) {
    Operands.push_back(RISCVOperand::createVType(
        RISCVVType::encodeXSfmmVType(SEW, Widen, AltFmt), S));
    return ParseStatus::Success;
  }

Fail:
  return generateXSfmmVTypeError(S);
}

bool RISCVAsmParser::generateXSfmmVTypeError(SMLoc ErrorLoc) {
  return Error(ErrorLoc, "operand must be e[8|16|16alt|32|64],w[1|2|4]");
}

ParseStatus RISCVAsmParser::parseMaskReg(OperandVector &Operands) {
  if (getLexer().isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = getLexer().getTok().getIdentifier();
  if (!Name.consume_back(".t"))
    return Error(getLoc(), "expected '.t' suffix");
  MCRegister Reg = matchRegisterNameHelper(Name);

  if (!Reg)
    return ParseStatus::NoMatch;
  if (Reg != RISCV::V0)
    return ParseStatus::NoMatch;
  SMLoc S = getLoc();
  SMLoc E = getTok().getEndLoc();
  getLexer().Lex();
  Operands.push_back(RISCVOperand::createReg(Reg, S, E));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseGPRAsFPR64(OperandVector &Operands) {
  if (!isRV64() || getSTI().hasFeature(RISCV::FeatureStdExtF))
    return ParseStatus::NoMatch;

  return parseGPRAsFPR(Operands);
}

ParseStatus RISCVAsmParser::parseGPRAsFPR(OperandVector &Operands) {
  if (getLexer().isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = getLexer().getTok().getIdentifier();
  MCRegister Reg = matchRegisterNameHelper(Name);

  if (!Reg)
    return ParseStatus::NoMatch;
  SMLoc S = getLoc();
  SMLoc E = getTok().getEndLoc();
  getLexer().Lex();
  Operands.push_back(RISCVOperand::createReg(
      Reg, S, E, !getSTI().hasFeature(RISCV::FeatureStdExtF)));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseGPRPairAsFPR64(OperandVector &Operands) {
  if (isRV64() || getSTI().hasFeature(RISCV::FeatureStdExtF))
    return ParseStatus::NoMatch;

  if (getLexer().isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = getLexer().getTok().getIdentifier();
  MCRegister Reg = matchRegisterNameHelper(Name);

  if (!Reg)
    return ParseStatus::NoMatch;

  if (!RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(Reg))
    return ParseStatus::NoMatch;

  if ((Reg - RISCV::X0) & 1) {
    // Only report the even register error if we have at least Zfinx so we know
    // some FP is enabled. We already checked F earlier.
    if (getSTI().hasFeature(RISCV::FeatureStdExtZfinx))
      return TokError("double precision floating point operands must use even "
                      "numbered X register");
    return ParseStatus::NoMatch;
  }

  SMLoc S = getLoc();
  SMLoc E = getTok().getEndLoc();
  getLexer().Lex();

  const MCRegisterInfo *RI = getContext().getRegisterInfo();
  MCRegister Pair = RI->getMatchingSuperReg(
      Reg, RISCV::sub_gpr_even,
      &RISCVMCRegisterClasses[RISCV::GPRPairRegClassID]);
  Operands.push_back(RISCVOperand::createReg(Pair, S, E, /*isGPRAsFPR=*/true));
  return ParseStatus::Success;
}

template <bool IsRV64>
ParseStatus RISCVAsmParser::parseGPRPair(OperandVector &Operands) {
  return parseGPRPair(Operands, IsRV64);
}

ParseStatus RISCVAsmParser::parseGPRPair(OperandVector &Operands,
                                         bool IsRV64Inst) {
  // If this is not an RV64 GPRPair instruction, don't parse as a GPRPair on
  // RV64 as it will prevent matching the RV64 version of the same instruction
  // that doesn't use a GPRPair.
  // If this is an RV64 GPRPair instruction, there is no RV32 version so we can
  // still parse as a pair.
  if (!IsRV64Inst && isRV64())
    return ParseStatus::NoMatch;

  if (getLexer().isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = getLexer().getTok().getIdentifier();
  MCRegister Reg = matchRegisterNameHelper(Name);

  if (!Reg)
    return ParseStatus::NoMatch;

  if (!RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(Reg))
    return ParseStatus::NoMatch;

  if ((Reg - RISCV::X0) & 1)
    return TokError("register must be even");

  SMLoc S = getLoc();
  SMLoc E = getTok().getEndLoc();
  getLexer().Lex();

  const MCRegisterInfo *RI = getContext().getRegisterInfo();
  MCRegister Pair = RI->getMatchingSuperReg(
      Reg, RISCV::sub_gpr_even,
      &RISCVMCRegisterClasses[RISCV::GPRPairRegClassID]);
  Operands.push_back(RISCVOperand::createReg(Pair, S, E));
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseFRMArg(OperandVector &Operands) {
  if (getLexer().isNot(AsmToken::Identifier))
    return TokError(
        "operand must be a valid floating point rounding mode mnemonic");

  StringRef Str = getLexer().getTok().getIdentifier();
  RISCVFPRndMode::RoundingMode FRM = RISCVFPRndMode::stringToRoundingMode(Str);

  if (FRM == RISCVFPRndMode::Invalid)
    return TokError(
        "operand must be a valid floating point rounding mode mnemonic");

  Operands.push_back(RISCVOperand::createFRMArg(FRM, getLoc()));
  Lex(); // Eat identifier token.
  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseFenceArg(OperandVector &Operands) {
  const AsmToken &Tok = getLexer().getTok();

  if (Tok.is(AsmToken::Integer)) {
    if (Tok.getIntVal() != 0)
      goto ParseFail;

    Operands.push_back(RISCVOperand::createFenceArg(0, getLoc()));
    Lex();
    return ParseStatus::Success;
  }

  if (Tok.is(AsmToken::Identifier)) {
    StringRef Str = Tok.getIdentifier();

    // Letters must be unique, taken from 'iorw', and in ascending order. This
    // holds as long as each individual character is one of 'iorw' and is
    // greater than the previous character.
    unsigned Imm = 0;
    bool Valid = true;
    char Prev = '\0';
    for (char c : Str) {
      switch (c) {
      default:
        Valid = false;
        break;
      case 'i':
        Imm |= RISCVFenceField::I;
        break;
      case 'o':
        Imm |= RISCVFenceField::O;
        break;
      case 'r':
        Imm |= RISCVFenceField::R;
        break;
      case 'w':
        Imm |= RISCVFenceField::W;
        break;
      }

      if (c <= Prev) {
        Valid = false;
        break;
      }
      Prev = c;
    }

    if (!Valid)
      goto ParseFail;

    Operands.push_back(RISCVOperand::createFenceArg(Imm, getLoc()));
    Lex();
    return ParseStatus::Success;
  }

ParseFail:
  return TokError("operand must be formed of letters selected in-order from "
                  "'iorw' or be 0");
}

ParseStatus RISCVAsmParser::parseMemOpBaseReg(OperandVector &Operands) {
  if (parseToken(AsmToken::LParen, "expected '('"))
    return ParseStatus::Failure;
  Operands.push_back(RISCVOperand::createToken("(", getLoc()));

  if (!parseRegister(Operands).isSuccess())
    return Error(getLoc(), "expected register");

  if (parseToken(AsmToken::RParen, "expected ')'"))
    return ParseStatus::Failure;
  Operands.push_back(RISCVOperand::createToken(")", getLoc()));

  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseZeroOffsetMemOp(OperandVector &Operands) {
  // Atomic operations such as lr.w, sc.w, and amo*.w accept a "memory operand"
  // as one of their register operands, such as `(a0)`. This just denotes that
  // the register (in this case `a0`) contains a memory address.
  //
  // Normally, we would be able to parse these by putting the parens into the
  // instruction string. However, GNU as also accepts a zero-offset memory
  // operand (such as `0(a0)`), and ignores the 0. Normally this would be parsed
  // with parseImmediate followed by parseMemOpBaseReg, but these instructions
  // do not accept an immediate operand, and we do not want to add a "dummy"
  // operand that is silently dropped.
  //
  // Instead, we use this custom parser. This will: allow (and discard) an
  // offset if it is zero; require (and discard) parentheses; and add only the
  // parsed register operand to `Operands`.
  //
  // These operands are printed with RISCVInstPrinter::printZeroOffsetMemOp,
  // which will only print the register surrounded by parentheses (which GNU as
  // also uses as its canonical representation for these operands).
  std::unique_ptr<RISCVOperand> OptionalImmOp;

  if (getLexer().isNot(AsmToken::LParen)) {
    // Parse an Integer token. We do not accept arbitrary constant expressions
    // in the offset field (because they may include parens, which complicates
    // parsing a lot).
    int64_t ImmVal;
    SMLoc ImmStart = getLoc();
    if (getParser().parseIntToken(ImmVal,
                                  "expected '(' or optional integer offset"))
      return ParseStatus::Failure;

    // Create a RISCVOperand for checking later (so the error messages are
    // nicer), but we don't add it to Operands.
    SMLoc ImmEnd = getLoc();
    OptionalImmOp =
        RISCVOperand::createImm(MCConstantExpr::create(ImmVal, getContext()),
                                ImmStart, ImmEnd, isRV64());
  }

  if (parseToken(AsmToken::LParen,
                 OptionalImmOp ? "expected '(' after optional integer offset"
                               : "expected '(' or optional integer offset"))
    return ParseStatus::Failure;

  if (!parseRegister(Operands).isSuccess())
    return Error(getLoc(), "expected register");

  if (parseToken(AsmToken::RParen, "expected ')'"))
    return ParseStatus::Failure;

  // Deferred Handling of non-zero offsets. This makes the error messages nicer.
  if (OptionalImmOp && !OptionalImmOp->isImmZero())
    return Error(
        OptionalImmOp->getStartLoc(), "optional integer offset must be 0",
        SMRange(OptionalImmOp->getStartLoc(), OptionalImmOp->getEndLoc()));

  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseRegReg(OperandVector &Operands) {
  // RR : a2(a1)
  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;

  SMLoc S = getLoc();
  StringRef OffsetRegName = getLexer().getTok().getIdentifier();
  MCRegister OffsetReg = matchRegisterNameHelper(OffsetRegName);
  if (!OffsetReg ||
      !RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(OffsetReg))
    return Error(getLoc(), "expected GPR register");
  getLexer().Lex();

  if (parseToken(AsmToken::LParen, "expected '(' or invalid operand"))
    return ParseStatus::Failure;

  if (getLexer().getKind() != AsmToken::Identifier)
    return Error(getLoc(), "expected GPR register");

  StringRef BaseRegName = getLexer().getTok().getIdentifier();
  MCRegister BaseReg = matchRegisterNameHelper(BaseRegName);
  if (!BaseReg ||
      !RISCVMCRegisterClasses[RISCV::GPRRegClassID].contains(BaseReg))
    return Error(getLoc(), "expected GPR register");
  getLexer().Lex();

  if (parseToken(AsmToken::RParen, "expected ')'"))
    return ParseStatus::Failure;

  Operands.push_back(RISCVOperand::createRegReg(BaseReg, OffsetReg, S));

  return ParseStatus::Success;
}

// RegList: {ra [, s0[-sN]]}
// XRegList: {x1 [, x8[-x9][, x18[-xN]]]}

// When MustIncludeS0 = true (not the default) (used for `qc.cm.pushfp`) which
// must include `fp`/`s0` in the list:
// RegList: {ra, s0[-sN]}
// XRegList: {x1, x8[-x9][, x18[-xN]]}
ParseStatus RISCVAsmParser::parseRegList(OperandVector &Operands,
                                         bool MustIncludeS0) {
  if (getTok().isNot(AsmToken::LCurly))
    return ParseStatus::NoMatch;

  SMLoc S = getLoc();

  Lex();

  bool UsesXRegs;
  MCRegister RegEnd;
  do {
    if (getTok().isNot(AsmToken::Identifier))
      return Error(getLoc(), "invalid register");

    StringRef RegName = getTok().getIdentifier();
    MCRegister Reg = matchRegisterNameHelper(RegName);
    if (!Reg)
      return Error(getLoc(), "invalid register");

    if (!RegEnd) {
      UsesXRegs = RegName[0] == 'x';
      if (Reg != RISCV::X1)
        return Error(getLoc(), "register list must start from 'ra' or 'x1'");
    } else if (RegEnd == RISCV::X1) {
      if (Reg != RISCV::X8 || (UsesXRegs != (RegName[0] == 'x')))
        return Error(getLoc(), Twine("register must be '") +
                                   (UsesXRegs ? "x8" : "s0") + "'");
    } else if (RegEnd == RISCV::X9 && UsesXRegs) {
      if (Reg != RISCV::X18 || (RegName[0] != 'x'))
        return Error(getLoc(), "register must be 'x18'");
    } else {
      return Error(getLoc(), "too many register ranges");
    }

    RegEnd = Reg;

    Lex();

    SMLoc MinusLoc = getLoc();
    if (parseOptionalToken(AsmToken::Minus)) {
      if (RegEnd == RISCV::X1)
        return Error(MinusLoc, Twine("register '") + (UsesXRegs ? "x1" : "ra") +
                                   "' cannot start a multiple register range");

      if (getTok().isNot(AsmToken::Identifier))
        return Error(getLoc(), "invalid register");

      StringRef RegName = getTok().getIdentifier();
      MCRegister Reg = matchRegisterNameHelper(RegName);
      if (!Reg)
        return Error(getLoc(), "invalid register");

      if (RegEnd == RISCV::X8) {
        if ((Reg != RISCV::X9 &&
             (UsesXRegs || Reg < RISCV::X18 || Reg > RISCV::X27)) ||
            (UsesXRegs != (RegName[0] == 'x'))) {
          if (UsesXRegs)
            return Error(getLoc(), "register must be 'x9'");
          return Error(getLoc(), "register must be in the range 's1' to 's11'");
        }
      } else if (RegEnd == RISCV::X18) {
        if (Reg < RISCV::X19 || Reg > RISCV::X27 || (RegName[0] != 'x'))
          return Error(getLoc(),
                       "register must be in the range 'x19' to 'x27'");
      } else
        llvm_unreachable("unexpected register");

      RegEnd = Reg;

      Lex();
    }
  } while (parseOptionalToken(AsmToken::Comma));

  if (parseToken(AsmToken::RCurly, "expected ',' or '}'"))
    return ParseStatus::Failure;

  if (RegEnd == RISCV::X26)
    return Error(S, "invalid register list, '{ra, s0-s10}' or '{x1, x8-x9, "
                    "x18-x26}' is not supported");

  auto Encode = RISCVZC::encodeRegList(RegEnd, isRVE());
  assert(Encode != RISCVZC::INVALID_RLIST);

  if (MustIncludeS0 && Encode == RISCVZC::RA)
    return Error(S, "register list must include 's0' or 'x8'");

  Operands.push_back(RISCVOperand::createRegList(Encode, S));

  return ParseStatus::Success;
}

ParseStatus RISCVAsmParser::parseZcmpStackAdj(OperandVector &Operands,
                                              bool ExpectNegative) {
  SMLoc S = getLoc();
  bool Negative = parseOptionalToken(AsmToken::Minus);

  if (getTok().isNot(AsmToken::Integer))
    return ParseStatus::NoMatch;

  int64_t StackAdjustment = getTok().getIntVal();

  auto *RegListOp = static_cast<RISCVOperand *>(Operands.back().get());
  if (!RegListOp->isRegList())
    return ParseStatus::NoMatch;

  unsigned RlistEncode = RegListOp->RegList.Encoding;

  assert(RlistEncode != RISCVZC::INVALID_RLIST);
  unsigned StackAdjBase = RISCVZC::getStackAdjBase(RlistEncode, isRV64());
  if (Negative != ExpectNegative || StackAdjustment % 16 != 0 ||
      StackAdjustment < StackAdjBase || (StackAdjustment - StackAdjBase) > 48) {
    int64_t Lower = StackAdjBase;
    int64_t Upper = StackAdjBase + 48;
    if (ExpectNegative) {
      Lower = -Lower;
      Upper = -Upper;
      std::swap(Lower, Upper);
    }
    return generateImmOutOfRangeError(S, Lower, Upper,
                                      "stack adjustment for register list must "
                                      "be a multiple of 16 bytes in the range");
  }

  unsigned StackAdj = (StackAdjustment - StackAdjBase);
  Operands.push_back(RISCVOperand::createStackAdj(StackAdj, S));
  Lex();
  return ParseStatus::Success;
}

/// Looks at a token type and creates the relevant operand from this
/// information, adding to Operands. If operand was parsed, returns false, else
/// true.
bool RISCVAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  ParseStatus Result =
      MatchOperandParserImpl(Operands, Mnemonic, /*ParseForAllFeatures=*/true);
  if (Result.isSuccess())
    return false;
  if (Result.isFailure())
    return true;

  // Attempt to parse token as a register.
  if (parseRegister(Operands, true).isSuccess())
    return false;

  // Attempt to parse token as an immediate
  if (parseImmediate(Operands).isSuccess()) {
    // Parse memory base register if present
    if (getLexer().is(AsmToken::LParen))
      return !parseMemOpBaseReg(Operands).isSuccess();
    return false;
  }

  // Finally we have exhausted all options and must declare defeat.
  Error(getLoc(), "unknown operand");
  return true;
}

bool RISCVAsmParser::parseInstruction(ParseInstructionInfo &Info,
                                      StringRef Name, SMLoc NameLoc,
                                      OperandVector &Operands) {
  // Apply mnemonic aliases because the destination mnemonic may have require
  // custom operand parsing. The generic tblgen'erated code does this later, at
  // the start of MatchInstructionImpl(), but that's too late for custom
  // operand parsing.
  const FeatureBitset &AvailableFeatures = getAvailableFeatures();
  applyMnemonicAliases(Name, AvailableFeatures, 0);

  // First operand is token for instruction
  Operands.push_back(RISCVOperand::createToken(Name, NameLoc));

  // If there are no more operands, then finish
  if (getLexer().is(AsmToken::EndOfStatement)) {
    getParser().Lex(); // Consume the EndOfStatement.
    return false;
  }

  // Parse first operand
  if (parseOperand(Operands, Name))
    return true;

  // Parse until end of statement, consuming commas between operands
  while (parseOptionalToken(AsmToken::Comma)) {
    // Parse next operand
    if (parseOperand(Operands, Name))
      return true;
  }

  if (getParser().parseEOL("unexpected token")) {
    getParser().eatToEndOfStatement();
    return true;
  }
  return false;
}

bool RISCVAsmParser::classifySymbolRef(const MCExpr *Expr,
                                       RISCVMCExpr::Specifier &Kind) {
  Kind = RISCVMCExpr::VK_None;

  if (const RISCVMCExpr *RE = dyn_cast<RISCVMCExpr>(Expr)) {
    Kind = RE->getSpecifier();
    Expr = RE->getSubExpr();
  }

  MCValue Res;
  if (Expr->evaluateAsRelocatable(Res, nullptr))
    return Res.getSpecifier() == RISCVMCExpr::VK_None;
  return false;
}

bool RISCVAsmParser::isSymbolDiff(const MCExpr *Expr) {
  MCValue Res;
  if (Expr->evaluateAsRelocatable(Res, nullptr)) {
    return Res.getSpecifier() == RISCVMCExpr::VK_None && Res.getAddSym() &&
           Res.getSubSym();
  }
  return false;
}

ParseStatus RISCVAsmParser::parseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".option")
    return parseDirectiveOption();
  if (IDVal == ".attribute")
    return parseDirectiveAttribute();
  if (IDVal == ".insn")
    return parseDirectiveInsn(DirectiveID.getLoc());
  if (IDVal == ".variant_cc")
    return parseDirectiveVariantCC();

  return ParseStatus::NoMatch;
}

bool RISCVAsmParser::resetToArch(StringRef Arch, SMLoc Loc, std::string &Result,
                                 bool FromOptionDirective) {
  for (auto &Feature : RISCVFeatureKV)
    if (llvm::RISCVISAInfo::isSupportedExtensionFeature(Feature.Key))
      clearFeatureBits(Feature.Value, Feature.Key);

  auto ParseResult = llvm::RISCVISAInfo::parseArchString(
      Arch, /*EnableExperimentalExtension=*/true,
      /*ExperimentalExtensionVersionCheck=*/true);
  if (!ParseResult) {
    std::string Buffer;
    raw_string_ostream OutputErrMsg(Buffer);
    handleAllErrors(ParseResult.takeError(), [&](llvm::StringError &ErrMsg) {
      OutputErrMsg << "invalid arch name '" << Arch << "', "
                   << ErrMsg.getMessage();
    });

    return Error(Loc, OutputErrMsg.str());
  }
  auto &ISAInfo = *ParseResult;

  for (auto &Feature : RISCVFeatureKV)
    if (ISAInfo->hasExtension(Feature.Key))
      setFeatureBits(Feature.Value, Feature.Key);

  if (FromOptionDirective) {
    if (ISAInfo->getXLen() == 32 && isRV64())
      return Error(Loc, "bad arch string switching from rv64 to rv32");
    else if (ISAInfo->getXLen() == 64 && !isRV64())
      return Error(Loc, "bad arch string switching from rv32 to rv64");
  }

  if (ISAInfo->getXLen() == 32)
    clearFeatureBits(RISCV::Feature64Bit, "64bit");
  else if (ISAInfo->getXLen() == 64)
    setFeatureBits(RISCV::Feature64Bit, "64bit");
  else
    return Error(Loc, "bad arch string " + Arch);

  Result = ISAInfo->toString();
  return false;
}

bool RISCVAsmParser::parseDirectiveOption() {
  MCAsmParser &Parser = getParser();
  // Get the option token.
  AsmToken Tok = Parser.getTok();

  // At the moment only identifiers are supported.
  if (parseToken(AsmToken::Identifier, "expected identifier"))
    return true;

  StringRef Option = Tok.getIdentifier();

  if (Option == "push") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionPush();
    pushFeatureBits();
    return false;
  }

  if (Option == "pop") {
    SMLoc StartLoc = Parser.getTok().getLoc();
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionPop();
    if (popFeatureBits())
      return Error(StartLoc, ".option pop with no .option push");

    return false;
  }

  if (Option == "arch") {
    SmallVector<RISCVOptionArchArg> Args;
    do {
      if (Parser.parseComma())
        return true;

      RISCVOptionArchArgType Type;
      if (parseOptionalToken(AsmToken::Plus))
        Type = RISCVOptionArchArgType::Plus;
      else if (parseOptionalToken(AsmToken::Minus))
        Type = RISCVOptionArchArgType::Minus;
      else if (!Args.empty())
        return Error(Parser.getTok().getLoc(),
                     "unexpected token, expected + or -");
      else
        Type = RISCVOptionArchArgType::Full;

      if (Parser.getTok().isNot(AsmToken::Identifier))
        return Error(Parser.getTok().getLoc(),
                     "unexpected token, expected identifier");

      StringRef Arch = Parser.getTok().getString();
      SMLoc Loc = Parser.getTok().getLoc();
      Parser.Lex();

      if (Type == RISCVOptionArchArgType::Full) {
        std::string Result;
        if (resetToArch(Arch, Loc, Result, true))
          return true;

        Args.emplace_back(Type, Result);
        break;
      }

      if (isDigit(Arch.back()))
        return Error(
            Loc, "extension version number parsing not currently implemented");

      std::string Feature = RISCVISAInfo::getTargetFeatureForExtension(Arch);
      if (!enableExperimentalExtension() &&
          StringRef(Feature).starts_with("experimental-"))
        return Error(Loc, "unexpected experimental extensions");
      auto Ext = llvm::lower_bound(RISCVFeatureKV, Feature);
      if (Ext == std::end(RISCVFeatureKV) || StringRef(Ext->Key) != Feature)
        return Error(Loc, "unknown extension feature");

      Args.emplace_back(Type, Arch.str());

      if (Type == RISCVOptionArchArgType::Plus) {
        FeatureBitset OldFeatureBits = STI->getFeatureBits();

        setFeatureBits(Ext->Value, Ext->Key);
        auto ParseResult = RISCVFeatures::parseFeatureBits(isRV64(), STI->getFeatureBits());
        if (!ParseResult) {
          copySTI().setFeatureBits(OldFeatureBits);
          setAvailableFeatures(ComputeAvailableFeatures(OldFeatureBits));

          std::string Buffer;
          raw_string_ostream OutputErrMsg(Buffer);
          handleAllErrors(ParseResult.takeError(), [&](llvm::StringError &ErrMsg) {
            OutputErrMsg << ErrMsg.getMessage();
          });

          return Error(Loc, OutputErrMsg.str());
        }
      } else {
        assert(Type == RISCVOptionArchArgType::Minus);
        // It is invalid to disable an extension that there are other enabled
        // extensions depend on it.
        // TODO: Make use of RISCVISAInfo to handle this
        for (auto &Feature : RISCVFeatureKV) {
          if (getSTI().hasFeature(Feature.Value) &&
              Feature.Implies.test(Ext->Value))
            return Error(Loc, Twine("can't disable ") + Ext->Key +
                                  " extension; " + Feature.Key +
                                  " extension requires " + Ext->Key +
                                  " extension");
        }

        clearFeatureBits(Ext->Value, Ext->Key);
      }
    } while (Parser.getTok().isNot(AsmToken::EndOfStatement));

    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionArch(Args);
    return false;
  }

  if (Option == "exact") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionExact();
    setFeatureBits(RISCV::FeatureExactAssembly, "exact-asm");
    clearFeatureBits(RISCV::FeatureRelax, "relax");
    return false;
  }

  if (Option == "noexact") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionNoExact();
    clearFeatureBits(RISCV::FeatureExactAssembly, "exact-asm");
    setFeatureBits(RISCV::FeatureRelax, "relax");
    return false;
  }

  if (Option == "rvc") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionRVC();
    setFeatureBits(RISCV::FeatureStdExtC, "c");
    return false;
  }

  if (Option == "norvc") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionNoRVC();
    clearFeatureBits(RISCV::FeatureStdExtC, "c");
    clearFeatureBits(RISCV::FeatureStdExtZca, "zca");
    return false;
  }

  if (Option == "pic") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionPIC();
    ParserOptions.IsPicEnabled = true;
    return false;
  }

  if (Option == "nopic") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionNoPIC();
    ParserOptions.IsPicEnabled = false;
    return false;
  }

  if (Option == "relax") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionRelax();
    setFeatureBits(RISCV::FeatureRelax, "relax");
    return false;
  }

  if (Option == "norelax") {
    if (Parser.parseEOL())
      return true;

    getTargetStreamer().emitDirectiveOptionNoRelax();
    clearFeatureBits(RISCV::FeatureRelax, "relax");
    return false;
  }

  // Unknown option.
  Warning(Parser.getTok().getLoc(),
          "unknown option, expected 'push', 'pop', "
          "'rvc', 'norvc', 'arch', 'relax', 'norelax', "
          "'exact', or 'noexact'");
  Parser.eatToEndOfStatement();
  return false;
}

/// parseDirectiveAttribute
///  ::= .attribute expression ',' ( expression | "string" )
///  ::= .attribute identifier ',' ( expression | "string" )
bool RISCVAsmParser::parseDirectiveAttribute() {
  MCAsmParser &Parser = getParser();
  int64_t Tag;
  SMLoc TagLoc;
  TagLoc = Parser.getTok().getLoc();
  if (Parser.getTok().is(AsmToken::Identifier)) {
    StringRef Name = Parser.getTok().getIdentifier();
    std::optional<unsigned> Ret =
        ELFAttrs::attrTypeFromString(Name, RISCVAttrs::getRISCVAttributeTags());
    if (!Ret)
      return Error(TagLoc, "attribute name not recognised: " + Name);
    Tag = *Ret;
    Parser.Lex();
  } else {
    const MCExpr *AttrExpr;

    TagLoc = Parser.getTok().getLoc();
    if (Parser.parseExpression(AttrExpr))
      return true;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(AttrExpr);
    if (check(!CE, TagLoc, "expected numeric constant"))
      return true;

    Tag = CE->getValue();
  }

  if (Parser.parseComma())
    return true;

  StringRef StringValue;
  int64_t IntegerValue = 0;
  bool IsIntegerValue = true;

  // RISC-V attributes have a string value if the tag number is odd
  // and an integer value if the tag number is even.
  if (Tag % 2)
    IsIntegerValue = false;

  SMLoc ValueExprLoc = Parser.getTok().getLoc();
  if (IsIntegerValue) {
    const MCExpr *ValueExpr;
    if (Parser.parseExpression(ValueExpr))
      return true;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ValueExpr);
    if (!CE)
      return Error(ValueExprLoc, "expected numeric constant");
    IntegerValue = CE->getValue();
  } else {
    if (Parser.getTok().isNot(AsmToken::String))
      return Error(Parser.getTok().getLoc(), "expected string constant");

    StringValue = Parser.getTok().getStringContents();
    Parser.Lex();
  }

  if (Parser.parseEOL())
    return true;

  if (IsIntegerValue)
    getTargetStreamer().emitAttribute(Tag, IntegerValue);
  else if (Tag != RISCVAttrs::ARCH)
    getTargetStreamer().emitTextAttribute(Tag, StringValue);
  else {
    std::string Result;
    if (resetToArch(StringValue, ValueExprLoc, Result, false))
      return true;

    // Then emit the arch string.
    getTargetStreamer().emitTextAttribute(Tag, Result);
  }

  return false;
}

bool isValidInsnFormat(StringRef Format, const MCSubtargetInfo &STI) {
  return StringSwitch<bool>(Format)
      .Cases("r", "r4", "i", "b", "sb", "u", "j", "uj", "s", true)
      .Cases("cr", "ci", "ciw", "css", "cl", "cs", "ca", "cb", "cj",
             STI.hasFeature(RISCV::FeatureStdExtZca))
      .Cases("qc.eai", "qc.ei", "qc.eb", "qc.ej", "qc.es",
             !STI.hasFeature(RISCV::Feature64Bit))
      .Default(false);
}

/// parseDirectiveInsn
/// ::= .insn [ format encoding, (operands (, operands)*) ]
/// ::= .insn [ length, value ]
/// ::= .insn [ value ]
bool RISCVAsmParser::parseDirectiveInsn(SMLoc L) {
  MCAsmParser &Parser = getParser();

  bool AllowC = getSTI().hasFeature(RISCV::FeatureStdExtC) ||
                getSTI().hasFeature(RISCV::FeatureStdExtZca);

  // Expect instruction format as identifier.
  StringRef Format;
  SMLoc ErrorLoc = Parser.getTok().getLoc();
  if (Parser.parseIdentifier(Format)) {
    // Try parsing .insn [ length , ] value
    std::optional<int64_t> Length;
    int64_t Value = 0;
    if (Parser.parseAbsoluteExpression(Value))
      return true;
    if (Parser.parseOptionalToken(AsmToken::Comma)) {
      Length = Value;
      if (Parser.parseAbsoluteExpression(Value))
        return true;

      if (*Length == 0 || (*Length % 2) != 0)
        return Error(ErrorLoc,
                     "instruction lengths must be a non-zero multiple of two");

      // TODO: Support Instructions > 64 bits.
      if (*Length > 8)
        return Error(ErrorLoc,
                     "instruction lengths over 64 bits are not supported");
    }

    // We only derive a length from the encoding for 16- and 32-bit
    // instructions, as the encodings for longer instructions are not frozen in
    // the spec.
    int64_t EncodingDerivedLength = ((Value & 0b11) == 0b11) ? 4 : 2;

    if (Length) {
      // Only check the length against the encoding if the length is present and
      // could match
      if ((*Length <= 4) && (*Length != EncodingDerivedLength))
        return Error(ErrorLoc,
                     "instruction length does not match the encoding");

      if (!isUIntN(*Length * 8, Value))
        return Error(ErrorLoc, "encoding value does not fit into instruction");
    } else {
      if (!isUIntN(EncodingDerivedLength * 8, Value))
        return Error(ErrorLoc, "encoding value does not fit into instruction");
    }

    if (!AllowC && (EncodingDerivedLength == 2))
      return Error(ErrorLoc, "compressed instructions are not allowed");

    if (getParser().parseEOL("invalid operand for instruction")) {
      getParser().eatToEndOfStatement();
      return true;
    }

    unsigned Opcode;
    if (Length) {
      switch (*Length) {
      case 2:
        Opcode = RISCV::Insn16;
        break;
      case 4:
        Opcode = RISCV::Insn32;
        break;
      case 6:
        Opcode = RISCV::Insn48;
        break;
      case 8:
        Opcode = RISCV::Insn64;
        break;
      default:
        llvm_unreachable("Error should have already been emitted");
      }
    } else
      Opcode = (EncodingDerivedLength == 2) ? RISCV::Insn16 : RISCV::Insn32;

    emitToStreamer(getStreamer(), MCInstBuilder(Opcode).addImm(Value));
    return false;
  }

  if (!isValidInsnFormat(Format, getSTI()))
    return Error(ErrorLoc, "invalid instruction format");

  std::string FormatName = (".insn_" + Format).str();

  ParseInstructionInfo Info;
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 8> Operands;

  if (parseInstruction(Info, FormatName, L, Operands))
    return true;

  unsigned Opcode;
  uint64_t ErrorInfo;
  return matchAndEmitInstruction(L, Opcode, Operands, Parser.getStreamer(),
                                 ErrorInfo,
                                 /*MatchingInlineAsm=*/false);
}

/// parseDirectiveVariantCC
///  ::= .variant_cc symbol
bool RISCVAsmParser::parseDirectiveVariantCC() {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected symbol name");
  if (parseEOL())
    return true;
  getTargetStreamer().emitDirectiveVariantCC(
      *getContext().getOrCreateSymbol(Name));
  return false;
}

void RISCVAsmParser::emitToStreamer(MCStreamer &S, const MCInst &Inst) {
  MCInst CInst;
  bool Res = false;
  const MCSubtargetInfo &STI = getSTI();
  if (!STI.hasFeature(RISCV::FeatureExactAssembly))
    Res = RISCVRVC::compress(CInst, Inst, STI);
  if (Res)
    ++RISCVNumInstrsCompressed;
  S.emitInstruction((Res ? CInst : Inst), STI);
}

void RISCVAsmParser::emitLoadImm(MCRegister DestReg, int64_t Value,
                                 MCStreamer &Out) {
  SmallVector<MCInst, 8> Seq;
  RISCVMatInt::generateMCInstSeq(Value, getSTI(), DestReg, Seq);

  for (MCInst &Inst : Seq) {
    emitToStreamer(Out, Inst);
  }
}

void RISCVAsmParser::emitAuipcInstPair(MCRegister DestReg, MCRegister TmpReg,
                                       const MCExpr *Symbol,
                                       RISCVMCExpr::Specifier VKHi,
                                       unsigned SecondOpcode, SMLoc IDLoc,
                                       MCStreamer &Out) {
  // A pair of instructions for PC-relative addressing; expands to
  //   TmpLabel: AUIPC TmpReg, VKHi(symbol)
  //             OP DestReg, TmpReg, %pcrel_lo(TmpLabel)
  MCContext &Ctx = getContext();

  MCSymbol *TmpLabel = Ctx.createNamedTempSymbol("pcrel_hi");
  Out.emitLabel(TmpLabel);

  const RISCVMCExpr *SymbolHi = RISCVMCExpr::create(Symbol, VKHi, Ctx);
  emitToStreamer(Out,
                 MCInstBuilder(RISCV::AUIPC).addReg(TmpReg).addExpr(SymbolHi));

  const MCExpr *RefToLinkTmpLabel = RISCVMCExpr::create(
      MCSymbolRefExpr::create(TmpLabel, Ctx), RISCVMCExpr::VK_PCREL_LO, Ctx);

  emitToStreamer(Out, MCInstBuilder(SecondOpcode)
                          .addReg(DestReg)
                          .addReg(TmpReg)
                          .addExpr(RefToLinkTmpLabel));
}

void RISCVAsmParser::emitLoadLocalAddress(MCInst &Inst, SMLoc IDLoc,
                                          MCStreamer &Out) {
  // The load local address pseudo-instruction "lla" is used in PC-relative
  // addressing of local symbols:
  //   lla rdest, symbol
  // expands to
  //   TmpLabel: AUIPC rdest, %pcrel_hi(symbol)
  //             ADDI rdest, rdest, %pcrel_lo(TmpLabel)
  MCRegister DestReg = Inst.getOperand(0).getReg();
  const MCExpr *Symbol = Inst.getOperand(1).getExpr();
  emitAuipcInstPair(DestReg, DestReg, Symbol, ELF::R_RISCV_PCREL_HI20,
                    RISCV::ADDI, IDLoc, Out);
}

void RISCVAsmParser::emitLoadGlobalAddress(MCInst &Inst, SMLoc IDLoc,
                                           MCStreamer &Out) {
  // The load global address pseudo-instruction "lga" is used in GOT-indirect
  // addressing of global symbols:
  //   lga rdest, symbol
  // expands to
  //   TmpLabel: AUIPC rdest, %got_pcrel_hi(symbol)
  //             Lx rdest, %pcrel_lo(TmpLabel)(rdest)
  MCRegister DestReg = Inst.getOperand(0).getReg();
  const MCExpr *Symbol = Inst.getOperand(1).getExpr();
  unsigned SecondOpcode = isRV64() ? RISCV::LD : RISCV::LW;
  emitAuipcInstPair(DestReg, DestReg, Symbol, ELF::R_RISCV_GOT_HI20,
                    SecondOpcode, IDLoc, Out);
}

void RISCVAsmParser::emitLoadAddress(MCInst &Inst, SMLoc IDLoc,
                                     MCStreamer &Out) {
  // The load address pseudo-instruction "la" is used in PC-relative and
  // GOT-indirect addressing of global symbols:
  //   la rdest, symbol
  // is an alias for either (for non-PIC)
  //   lla rdest, symbol
  // or (for PIC)
  //   lga rdest, symbol
  if (ParserOptions.IsPicEnabled)
    emitLoadGlobalAddress(Inst, IDLoc, Out);
  else
    emitLoadLocalAddress(Inst, IDLoc, Out);
}

void RISCVAsmParser::emitLoadTLSIEAddress(MCInst &Inst, SMLoc IDLoc,
                                          MCStreamer &Out) {
  // The load TLS IE address pseudo-instruction "la.tls.ie" is used in
  // initial-exec TLS model addressing of global symbols:
  //   la.tls.ie rdest, symbol
  // expands to
  //   TmpLabel: AUIPC rdest, %tls_ie_pcrel_hi(symbol)
  //             Lx rdest, %pcrel_lo(TmpLabel)(rdest)
  MCRegister DestReg = Inst.getOperand(0).getReg();
  const MCExpr *Symbol = Inst.getOperand(1).getExpr();
  unsigned SecondOpcode = isRV64() ? RISCV::LD : RISCV::LW;
  emitAuipcInstPair(DestReg, DestReg, Symbol, ELF::R_RISCV_TLS_GOT_HI20,
                    SecondOpcode, IDLoc, Out);
}

void RISCVAsmParser::emitLoadTLSGDAddress(MCInst &Inst, SMLoc IDLoc,
                                          MCStreamer &Out) {
  // The load TLS GD address pseudo-instruction "la.tls.gd" is used in
  // global-dynamic TLS model addressing of global symbols:
  //   la.tls.gd rdest, symbol
  // expands to
  //   TmpLabel: AUIPC rdest, %tls_gd_pcrel_hi(symbol)
  //             ADDI rdest, rdest, %pcrel_lo(TmpLabel)
  MCRegister DestReg = Inst.getOperand(0).getReg();
  const MCExpr *Symbol = Inst.getOperand(1).getExpr();
  emitAuipcInstPair(DestReg, DestReg, Symbol, ELF::R_RISCV_TLS_GD_HI20,
                    RISCV::ADDI, IDLoc, Out);
}

void RISCVAsmParser::emitLoadStoreSymbol(MCInst &Inst, unsigned Opcode,
                                         SMLoc IDLoc, MCStreamer &Out,
                                         bool HasTmpReg) {
  // The load/store pseudo-instruction does a pc-relative load with
  // a symbol.
  //
  // The expansion looks like this
  //
  //   TmpLabel: AUIPC tmp, %pcrel_hi(symbol)
  //             [S|L]X    rd, %pcrel_lo(TmpLabel)(tmp)
  unsigned DestRegOpIdx = HasTmpReg ? 1 : 0;
  MCRegister DestReg = Inst.getOperand(DestRegOpIdx).getReg();
  unsigned SymbolOpIdx = HasTmpReg ? 2 : 1;
  MCRegister TmpReg = Inst.getOperand(0).getReg();

  // If TmpReg is a GPR pair, get the even register.
  if (RISCVMCRegisterClasses[RISCV::GPRPairRegClassID].contains(TmpReg)) {
    const MCRegisterInfo *RI = getContext().getRegisterInfo();
    TmpReg = RI->getSubReg(TmpReg, RISCV::sub_gpr_even);
  }

  const MCExpr *Symbol = Inst.getOperand(SymbolOpIdx).getExpr();
  emitAuipcInstPair(DestReg, TmpReg, Symbol, ELF::R_RISCV_PCREL_HI20, Opcode,
                    IDLoc, Out);
}

void RISCVAsmParser::emitPseudoExtend(MCInst &Inst, bool SignExtend,
                                      int64_t Width, SMLoc IDLoc,
                                      MCStreamer &Out) {
  // The sign/zero extend pseudo-instruction does two shifts, with the shift
  // amounts dependent on the XLEN.
  //
  // The expansion looks like this
  //
  //    SLLI rd, rs, XLEN - Width
  //    SR[A|R]I rd, rd, XLEN - Width
  const MCOperand &DestReg = Inst.getOperand(0);
  const MCOperand &SourceReg = Inst.getOperand(1);

  unsigned SecondOpcode = SignExtend ? RISCV::SRAI : RISCV::SRLI;
  int64_t ShAmt = (isRV64() ? 64 : 32) - Width;

  assert(ShAmt > 0 && "Shift amount must be non-zero.");

  emitToStreamer(Out, MCInstBuilder(RISCV::SLLI)
                          .addOperand(DestReg)
                          .addOperand(SourceReg)
                          .addImm(ShAmt));

  emitToStreamer(Out, MCInstBuilder(SecondOpcode)
                          .addOperand(DestReg)
                          .addOperand(DestReg)
                          .addImm(ShAmt));
}

void RISCVAsmParser::emitVMSGE(MCInst &Inst, unsigned Opcode, SMLoc IDLoc,
                               MCStreamer &Out) {
  if (Inst.getNumOperands() == 3) {
    // unmasked va >= x
    //
    //  pseudoinstruction: vmsge{u}.vx vd, va, x
    //  expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd
    emitToStreamer(Out, MCInstBuilder(Opcode)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(1))
                            .addOperand(Inst.getOperand(2))
                            .addReg(MCRegister())
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMNAND_MM)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(0))
                            .setLoc(IDLoc));
  } else if (Inst.getNumOperands() == 4) {
    // masked va >= x, vd != v0
    //
    //  pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t
    //  expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0
    assert(Inst.getOperand(0).getReg() != RISCV::V0 &&
           "The destination register should not be V0.");
    emitToStreamer(Out, MCInstBuilder(Opcode)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(1))
                            .addOperand(Inst.getOperand(2))
                            .addOperand(Inst.getOperand(3))
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMXOR_MM)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(0))
                            .addReg(RISCV::V0)
                            .setLoc(IDLoc));
  } else if (Inst.getNumOperands() == 5 &&
             Inst.getOperand(0).getReg() == RISCV::V0) {
    // masked va >= x, vd == v0
    //
    //  pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt
    //  expansion: vmslt{u}.vx vt, va, x;  vmandn.mm vd, vd, vt
    assert(Inst.getOperand(0).getReg() == RISCV::V0 &&
           "The destination register should be V0.");
    assert(Inst.getOperand(1).getReg() != RISCV::V0 &&
           "The temporary vector register should not be V0.");
    emitToStreamer(Out, MCInstBuilder(Opcode)
                            .addOperand(Inst.getOperand(1))
                            .addOperand(Inst.getOperand(2))
                            .addOperand(Inst.getOperand(3))
                            .addReg(MCRegister())
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMANDN_MM)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(1))
                            .setLoc(IDLoc));
  } else if (Inst.getNumOperands() == 5) {
    // masked va >= x, any vd
    //
    // pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt
    // expansion: vmslt{u}.vx vt, va, x; vmandn.mm vt, v0, vt;
    //            vmandn.mm vd, vd, v0;  vmor.mm vd, vt, vd
    assert(Inst.getOperand(1).getReg() != RISCV::V0 &&
           "The temporary vector register should not be V0.");
    emitToStreamer(Out, MCInstBuilder(Opcode)
                            .addOperand(Inst.getOperand(1))
                            .addOperand(Inst.getOperand(2))
                            .addOperand(Inst.getOperand(3))
                            .addReg(MCRegister())
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMANDN_MM)
                            .addOperand(Inst.getOperand(1))
                            .addReg(RISCV::V0)
                            .addOperand(Inst.getOperand(1))
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMANDN_MM)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(0))
                            .addReg(RISCV::V0)
                            .setLoc(IDLoc));
    emitToStreamer(Out, MCInstBuilder(RISCV::VMOR_MM)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(1))
                            .addOperand(Inst.getOperand(0))
                            .setLoc(IDLoc));
  }
}

bool RISCVAsmParser::checkPseudoAddTPRel(MCInst &Inst,
                                         OperandVector &Operands) {
  assert(Inst.getOpcode() == RISCV::PseudoAddTPRel && "Invalid instruction");
  assert(Inst.getOperand(2).isReg() && "Unexpected second operand kind");
  if (Inst.getOperand(2).getReg() != RISCV::X4) {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[3]).getStartLoc();
    return Error(ErrorLoc, "the second input operand must be tp/x4 when using "
                           "%tprel_add specifier");
  }

  return false;
}

bool RISCVAsmParser::checkPseudoTLSDESCCall(MCInst &Inst,
                                            OperandVector &Operands) {
  assert(Inst.getOpcode() == RISCV::PseudoTLSDESCCall && "Invalid instruction");
  assert(Inst.getOperand(0).isReg() && "Unexpected operand kind");
  if (Inst.getOperand(0).getReg() != RISCV::X5) {
    SMLoc ErrorLoc = ((RISCVOperand &)*Operands[3]).getStartLoc();
    return Error(ErrorLoc, "the output operand must be t0/x5 when using "
                           "%tlsdesc_call specifier");
  }

  return false;
}

std::unique_ptr<RISCVOperand> RISCVAsmParser::defaultMaskRegOp() const {
  return RISCVOperand::createReg(MCRegister(), llvm::SMLoc(), llvm::SMLoc());
}

std::unique_ptr<RISCVOperand> RISCVAsmParser::defaultFRMArgOp() const {
  return RISCVOperand::createFRMArg(RISCVFPRndMode::RoundingMode::DYN,
                                    llvm::SMLoc());
}

std::unique_ptr<RISCVOperand> RISCVAsmParser::defaultFRMArgLegacyOp() const {
  return RISCVOperand::createFRMArg(RISCVFPRndMode::RoundingMode::RNE,
                                    llvm::SMLoc());
}

bool RISCVAsmParser::validateInstruction(MCInst &Inst,
                                         OperandVector &Operands) {
  unsigned Opcode = Inst.getOpcode();

  if (Opcode == RISCV::PseudoVMSGEU_VX_M_T ||
      Opcode == RISCV::PseudoVMSGE_VX_M_T) {
    MCRegister DestReg = Inst.getOperand(0).getReg();
    MCRegister TempReg = Inst.getOperand(1).getReg();
    if (DestReg == TempReg) {
      SMLoc Loc = Operands.back()->getStartLoc();
      return Error(Loc, "the temporary vector register cannot be the same as "
                        "the destination register");
    }
  }

  if (Opcode == RISCV::TH_LDD || Opcode == RISCV::TH_LWUD ||
      Opcode == RISCV::TH_LWD) {
    MCRegister Rd1 = Inst.getOperand(0).getReg();
    MCRegister Rd2 = Inst.getOperand(1).getReg();
    MCRegister Rs1 = Inst.getOperand(2).getReg();
    // The encoding with rd1 == rd2 == rs1 is reserved for XTHead load pair.
    if (Rs1 == Rd1 || Rs1 == Rd2 || Rd1 == Rd2) {
      SMLoc Loc = Operands[1]->getStartLoc();
      return Error(Loc, "rs1, rd1, and rd2 cannot overlap");
    }
  }

  if (Opcode == RISCV::CM_MVSA01 || Opcode == RISCV::QC_CM_MVSA01) {
    MCRegister Rd1 = Inst.getOperand(0).getReg();
    MCRegister Rd2 = Inst.getOperand(1).getReg();
    if (Rd1 == Rd2) {
      SMLoc Loc = Operands[1]->getStartLoc();
      return Error(Loc, "rs1 and rs2 must be different");
    }
  }

  const MCInstrDesc &MCID = MII.get(Opcode);
  if (!(MCID.TSFlags & RISCVII::ConstraintMask))
    return false;

  if (Opcode == RISCV::VC_V_XVW || Opcode == RISCV::VC_V_IVW ||
      Opcode == RISCV::VC_V_FVW || Opcode == RISCV::VC_V_VVW) {
    // Operands Opcode, Dst, uimm, Dst, Rs2, Rs1 for VC_V_XVW.
    MCRegister VCIXDst = Inst.getOperand(0).getReg();
    SMLoc VCIXDstLoc = Operands[2]->getStartLoc();
    if (MCID.TSFlags & RISCVII::VS1Constraint) {
      MCRegister VCIXRs1 = Inst.getOperand(Inst.getNumOperands() - 1).getReg();
      if (VCIXDst == VCIXRs1)
        return Error(VCIXDstLoc, "the destination vector register group cannot"
                                 " overlap the source vector register group");
    }
    if (MCID.TSFlags & RISCVII::VS2Constraint) {
      MCRegister VCIXRs2 = Inst.getOperand(Inst.getNumOperands() - 2).getReg();
      if (VCIXDst == VCIXRs2)
        return Error(VCIXDstLoc, "the destination vector register group cannot"
                                 " overlap the source vector register group");
    }
    return false;
  }

  MCRegister DestReg = Inst.getOperand(0).getReg();
  unsigned Offset = 0;
  int TiedOp = MCID.getOperandConstraint(1, MCOI::TIED_TO);
  if (TiedOp == 0)
    Offset = 1;

  // Operands[1] will be the first operand, DestReg.
  SMLoc Loc = Operands[1]->getStartLoc();
  if (MCID.TSFlags & RISCVII::VS2Constraint) {
    MCRegister CheckReg = Inst.getOperand(Offset + 1).getReg();
    if (DestReg == CheckReg)
      return Error(Loc, "the destination vector register group cannot overlap"
                        " the source vector register group");
  }
  if ((MCID.TSFlags & RISCVII::VS1Constraint) && Inst.getOperand(Offset + 2).isReg()) {
    MCRegister CheckReg = Inst.getOperand(Offset + 2).getReg();
    if (DestReg == CheckReg)
      return Error(Loc, "the destination vector register group cannot overlap"
                        " the source vector register group");
  }
  if ((MCID.TSFlags & RISCVII::VMConstraint) && (DestReg == RISCV::V0)) {
    // vadc, vsbc are special cases. These instructions have no mask register.
    // The destination register could not be V0.
    if (Opcode == RISCV::VADC_VVM || Opcode == RISCV::VADC_VXM ||
        Opcode == RISCV::VADC_VIM || Opcode == RISCV::VSBC_VVM ||
        Opcode == RISCV::VSBC_VXM || Opcode == RISCV::VFMERGE_VFM ||
        Opcode == RISCV::VMERGE_VIM || Opcode == RISCV::VMERGE_VVM ||
        Opcode == RISCV::VMERGE_VXM)
      return Error(Loc, "the destination vector register group cannot be V0");

    // Regardless masked or unmasked version, the number of operands is the
    // same. For example, "viota.m v0, v2" is "viota.m v0, v2, NoRegister"
    // actually. We need to check the last operand to ensure whether it is
    // masked or not.
    MCRegister CheckReg = Inst.getOperand(Inst.getNumOperands() - 1).getReg();
    assert((CheckReg == RISCV::V0 || !CheckReg) &&
           "Unexpected register for mask operand");

    if (DestReg == CheckReg)
      return Error(Loc, "the destination vector register group cannot overlap"
                        " the mask register");
  }
  return false;
}

bool RISCVAsmParser::processInstruction(MCInst &Inst, SMLoc IDLoc,
                                        OperandVector &Operands,
                                        MCStreamer &Out) {
  Inst.setLoc(IDLoc);

  switch (Inst.getOpcode()) {
  default:
    break;
  case RISCV::PseudoC_ADDI_NOP:
    emitToStreamer(Out, MCInstBuilder(RISCV::C_NOP));
    return false;
  case RISCV::PseudoLLAImm:
  case RISCV::PseudoLAImm:
  case RISCV::PseudoLI: {
    MCRegister Reg = Inst.getOperand(0).getReg();
    const MCOperand &Op1 = Inst.getOperand(1);
    if (Op1.isExpr()) {
      // We must have li reg, %lo(sym) or li reg, %pcrel_lo(sym) or similar.
      // Just convert to an addi. This allows compatibility with gas.
      emitToStreamer(Out, MCInstBuilder(RISCV::ADDI)
                              .addReg(Reg)
                              .addReg(RISCV::X0)
                              .addExpr(Op1.getExpr()));
      return false;
    }
    int64_t Imm = Inst.getOperand(1).getImm();
    // On RV32 the immediate here can either be a signed or an unsigned
    // 32-bit number. Sign extension has to be performed to ensure that Imm
    // represents the expected signed 64-bit number.
    if (!isRV64())
      Imm = SignExtend64<32>(Imm);
    emitLoadImm(Reg, Imm, Out);
    return false;
  }
  case RISCV::PseudoLLA:
    emitLoadLocalAddress(Inst, IDLoc, Out);
    return false;
  case RISCV::PseudoLGA:
    emitLoadGlobalAddress(Inst, IDLoc, Out);
    return false;
  case RISCV::PseudoLA:
    emitLoadAddress(Inst, IDLoc, Out);
    return false;
  case RISCV::PseudoLA_TLS_IE:
    emitLoadTLSIEAddress(Inst, IDLoc, Out);
    return false;
  case RISCV::PseudoLA_TLS_GD:
    emitLoadTLSGDAddress(Inst, IDLoc, Out);
    return false;
  case RISCV::PseudoLB:
  case RISCV::PseudoQC_E_LB:
    emitLoadStoreSymbol(Inst, RISCV::LB, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLBU:
  case RISCV::PseudoQC_E_LBU:
    emitLoadStoreSymbol(Inst, RISCV::LBU, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLH:
  case RISCV::PseudoQC_E_LH:
    emitLoadStoreSymbol(Inst, RISCV::LH, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLHU:
  case RISCV::PseudoQC_E_LHU:
    emitLoadStoreSymbol(Inst, RISCV::LHU, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLW:
  case RISCV::PseudoQC_E_LW:
    emitLoadStoreSymbol(Inst, RISCV::LW, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLWU:
    emitLoadStoreSymbol(Inst, RISCV::LWU, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLD:
    emitLoadStoreSymbol(Inst, RISCV::LD, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoLD_RV32:
    emitLoadStoreSymbol(Inst, RISCV::LD_RV32, IDLoc, Out, /*HasTmpReg=*/false);
    return false;
  case RISCV::PseudoFLH:
    emitLoadStoreSymbol(Inst, RISCV::FLH, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFLW:
    emitLoadStoreSymbol(Inst, RISCV::FLW, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFLD:
    emitLoadStoreSymbol(Inst, RISCV::FLD, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFLQ:
    emitLoadStoreSymbol(Inst, RISCV::FLQ, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoSB:
  case RISCV::PseudoQC_E_SB:
    emitLoadStoreSymbol(Inst, RISCV::SB, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoSH:
  case RISCV::PseudoQC_E_SH:
    emitLoadStoreSymbol(Inst, RISCV::SH, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoSW:
  case RISCV::PseudoQC_E_SW:
    emitLoadStoreSymbol(Inst, RISCV::SW, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoSD:
    emitLoadStoreSymbol(Inst, RISCV::SD, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoSD_RV32:
    emitLoadStoreSymbol(Inst, RISCV::SD_RV32, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFSH:
    emitLoadStoreSymbol(Inst, RISCV::FSH, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFSW:
    emitLoadStoreSymbol(Inst, RISCV::FSW, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFSD:
    emitLoadStoreSymbol(Inst, RISCV::FSD, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoFSQ:
    emitLoadStoreSymbol(Inst, RISCV::FSQ, IDLoc, Out, /*HasTmpReg=*/true);
    return false;
  case RISCV::PseudoAddTPRel:
    if (checkPseudoAddTPRel(Inst, Operands))
      return true;
    break;
  case RISCV::PseudoTLSDESCCall:
    if (checkPseudoTLSDESCCall(Inst, Operands))
      return true;
    break;
  case RISCV::PseudoSEXT_B:
    emitPseudoExtend(Inst, /*SignExtend=*/true, /*Width=*/8, IDLoc, Out);
    return false;
  case RISCV::PseudoSEXT_H:
    emitPseudoExtend(Inst, /*SignExtend=*/true, /*Width=*/16, IDLoc, Out);
    return false;
  case RISCV::PseudoZEXT_H:
    emitPseudoExtend(Inst, /*SignExtend=*/false, /*Width=*/16, IDLoc, Out);
    return false;
  case RISCV::PseudoZEXT_W:
    emitPseudoExtend(Inst, /*SignExtend=*/false, /*Width=*/32, IDLoc, Out);
    return false;
  case RISCV::PseudoVMSGEU_VX:
  case RISCV::PseudoVMSGEU_VX_M:
  case RISCV::PseudoVMSGEU_VX_M_T:
    emitVMSGE(Inst, RISCV::VMSLTU_VX, IDLoc, Out);
    return false;
  case RISCV::PseudoVMSGE_VX:
  case RISCV::PseudoVMSGE_VX_M:
  case RISCV::PseudoVMSGE_VX_M_T:
    emitVMSGE(Inst, RISCV::VMSLT_VX, IDLoc, Out);
    return false;
  case RISCV::PseudoVMSGE_VI:
  case RISCV::PseudoVMSLT_VI: {
    // These instructions are signed and so is immediate so we can subtract one
    // and change the opcode.
    int64_t Imm = Inst.getOperand(2).getImm();
    unsigned Opc = Inst.getOpcode() == RISCV::PseudoVMSGE_VI ? RISCV::VMSGT_VI
                                                             : RISCV::VMSLE_VI;
    emitToStreamer(Out, MCInstBuilder(Opc)
                            .addOperand(Inst.getOperand(0))
                            .addOperand(Inst.getOperand(1))
                            .addImm(Imm - 1)
                            .addOperand(Inst.getOperand(3))
                            .setLoc(IDLoc));
    return false;
  }
  case RISCV::PseudoVMSGEU_VI:
  case RISCV::PseudoVMSLTU_VI: {
    int64_t Imm = Inst.getOperand(2).getImm();
    // Unsigned comparisons are tricky because the immediate is signed. If the
    // immediate is 0 we can't just subtract one. vmsltu.vi v0, v1, 0 is always
    // false, but vmsle.vi v0, v1, -1 is always true. Instead we use
    // vmsne v0, v1, v1 which is always false.
    if (Imm == 0) {
      unsigned Opc = Inst.getOpcode() == RISCV::PseudoVMSGEU_VI
                         ? RISCV::VMSEQ_VV
                         : RISCV::VMSNE_VV;
      emitToStreamer(Out, MCInstBuilder(Opc)
                              .addOperand(Inst.getOperand(0))
                              .addOperand(Inst.getOperand(1))
                              .addOperand(Inst.getOperand(1))
                              .addOperand(Inst.getOperand(3))
                              .setLoc(IDLoc));
    } else {
      // Other immediate values can subtract one like signed.
      unsigned Opc = Inst.getOpcode() == RISCV::PseudoVMSGEU_VI
                         ? RISCV::VMSGTU_VI
                         : RISCV::VMSLEU_VI;
      emitToStreamer(Out, MCInstBuilder(Opc)
                              .addOperand(Inst.getOperand(0))
                              .addOperand(Inst.getOperand(1))
                              .addImm(Imm - 1)
                              .addOperand(Inst.getOperand(3))
                              .setLoc(IDLoc));
    }

    return false;
  }
  }

  emitToStreamer(Out, Inst);
  return false;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVAsmParser() {
  RegisterMCAsmParser<RISCVAsmParser> X(getTheRISCV32Target());
  RegisterMCAsmParser<RISCVAsmParser> Y(getTheRISCV64Target());
}
