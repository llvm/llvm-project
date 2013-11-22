//==- AArch64AsmParser.cpp - Parse AArch64 assembly to MCInst instructions -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the (GNU-style) assembly parser for the AArch64
// architecture.
//
//===----------------------------------------------------------------------===//


#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {

class AArch64Operand;

class AArch64AsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;

#define GET_ASSEMBLER_HEADER
#include "AArch64GenAsmMatcher.inc"

public:
  enum AArch64MatchResultTy {
    Match_FirstAArch64 = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "AArch64GenAsmMatcher.inc"
  };

  AArch64AsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser,
                   const MCInstrInfo &MII)
      : MCTargetAsmParser(), STI(_STI), Parser(_Parser) {
    MCAsmParserExtension::Initialize(_Parser);

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

  // These are the public interface of the MCTargetAsmParser
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool ParseDirective(AsmToken DirectiveID);
  bool ParseDirectiveTLSDescCall(SMLoc L);
  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer&Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  // The rest of the sub-parsers have more freedom over interface: they return
  // an OperandMatchResultTy because it's less ambiguous than true/false or
  // -1/0/1 even if it is more verbose
  OperandMatchResultTy
  ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
               StringRef Mnemonic);

  OperandMatchResultTy ParseImmediate(const MCExpr *&ExprVal);

  OperandMatchResultTy ParseRelocPrefix(AArch64MCExpr::VariantKind &RefKind);

  OperandMatchResultTy
  ParseNEONLane(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                uint32_t NumLanes);

  OperandMatchResultTy
  ParseRegister(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                uint32_t &NumLanes);

  OperandMatchResultTy
  ParseImmWithLSLOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseCondCodeOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseCRxOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseFPImmOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  template<typename SomeNamedImmMapper> OperandMatchResultTy
  ParseNamedImmOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
    return ParseNamedImmOperand(SomeNamedImmMapper(), Operands);
  }

  OperandMatchResultTy
  ParseNamedImmOperand(const NamedImmMapper &Mapper,
                       SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseLSXAddressOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseShiftExtend(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  OperandMatchResultTy
  ParseSysRegOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  bool TryParseVector(uint32_t &RegNum, SMLoc &RegEndLoc, StringRef &Layout,
                      SMLoc &LayoutLoc);

  OperandMatchResultTy ParseVectorList(SmallVectorImpl<MCParsedAsmOperand *> &);

  bool validateInstruction(MCInst &Inst,
                          const SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  /// Scan the next token (which had better be an identifier) and determine
  /// whether it represents a general-purpose or vector register. It returns
  /// true if an identifier was found and populates its reference arguments. It
  /// does not consume the token.
  bool
  IdentifyRegister(unsigned &RegNum, SMLoc &RegEndLoc, StringRef &LayoutSpec,
                   SMLoc &LayoutLoc) const;

};

}

namespace {

/// Instances of this class represent a parsed AArch64 machine instruction.
class AArch64Operand : public MCParsedAsmOperand {
private:
  enum KindTy {
    k_ImmWithLSL,     // #uimm {, LSL #amt }
    k_CondCode,       // eq/ne/...
    k_FPImmediate,    // Limited-precision floating-point imm
    k_Immediate,      // Including expressions referencing symbols
    k_Register,
    k_ShiftExtend,
    k_VectorList,     // A sequential list of 1 to 4 registers.
    k_SysReg,         // The register operand of MRS and MSR instructions
    k_Token,          // The mnemonic; other raw tokens the auto-generated
    k_WrappedRegister // Load/store exclusive permit a wrapped register.
  } Kind;

  SMLoc StartLoc, EndLoc;

  struct ImmWithLSLOp {
    const MCExpr *Val;
    unsigned ShiftAmount;
    bool ImplicitAmount;
  };

  struct CondCodeOp {
    A64CC::CondCodes Code;
  };

  struct FPImmOp {
    double Val;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct RegOp {
    unsigned RegNum;
  };

  struct ShiftExtendOp {
    A64SE::ShiftExtSpecifiers ShiftType;
    unsigned Amount;
    bool ImplicitAmount;
  };

  // A vector register list is a sequential list of 1 to 4 registers.
  struct VectorListOp {
    unsigned RegNum;
    unsigned Count;
    A64Layout::VectorLayout Layout;
  };

  struct SysRegOp {
    const char *Data;
    unsigned Length;
  };

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  union {
    struct ImmWithLSLOp ImmWithLSL;
    struct CondCodeOp CondCode;
    struct FPImmOp FPImm;
    struct ImmOp Imm;
    struct RegOp Reg;
    struct ShiftExtendOp ShiftExtend;
    struct VectorListOp VectorList;
    struct SysRegOp SysReg;
    struct TokOp Tok;
  };

  AArch64Operand(KindTy K, SMLoc S, SMLoc E)
    : MCParsedAsmOperand(), Kind(K), StartLoc(S), EndLoc(E) {}

public:
  AArch64Operand(const AArch64Operand &o) : MCParsedAsmOperand() {
  }

  SMLoc getStartLoc() const { return StartLoc; }
  SMLoc getEndLoc() const { return EndLoc; }
  void print(raw_ostream&) const;
  void dump() const;

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const {
    assert((Kind == k_Register || Kind == k_WrappedRegister)
           && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == k_Immediate && "Invalid access!");
    return Imm.Val;
  }

  A64CC::CondCodes getCondCode() const {
    assert(Kind == k_CondCode && "Invalid access!");
    return CondCode.Code;
  }

  static bool isNonConstantExpr(const MCExpr *E,
                                AArch64MCExpr::VariantKind &Variant) {
    if (const AArch64MCExpr *A64E = dyn_cast<AArch64MCExpr>(E)) {
      Variant = A64E->getKind();
      return true;
    } else if (!isa<MCConstantExpr>(E)) {
      Variant = AArch64MCExpr::VK_AARCH64_None;
      return true;
    }

    return false;
  }

  bool isCondCode() const { return Kind == k_CondCode; }
  bool isToken() const { return Kind == k_Token; }
  bool isReg() const { return Kind == k_Register; }
  bool isImm() const { return Kind == k_Immediate; }
  bool isMem() const { return false; }
  bool isFPImm() const { return Kind == k_FPImmediate; }
  bool isShiftOrExtend() const { return Kind == k_ShiftExtend; }
  bool isSysReg() const { return Kind == k_SysReg; }
  bool isImmWithLSL() const { return Kind == k_ImmWithLSL; }
  bool isWrappedReg() const { return Kind == k_WrappedRegister; }

  bool isAddSubImmLSL0() const {
    if (!isImmWithLSL()) return false;
    if (ImmWithLSL.ShiftAmount != 0) return false;

    AArch64MCExpr::VariantKind Variant;
    if (isNonConstantExpr(ImmWithLSL.Val, Variant)) {
      return Variant == AArch64MCExpr::VK_AARCH64_LO12
          || Variant == AArch64MCExpr::VK_AARCH64_DTPREL_LO12
          || Variant == AArch64MCExpr::VK_AARCH64_DTPREL_LO12_NC
          || Variant == AArch64MCExpr::VK_AARCH64_TPREL_LO12
          || Variant == AArch64MCExpr::VK_AARCH64_TPREL_LO12_NC
          || Variant == AArch64MCExpr::VK_AARCH64_TLSDESC_LO12;
    }

    // Otherwise it should be a real immediate in range:
    const MCConstantExpr *CE = cast<MCConstantExpr>(ImmWithLSL.Val);
    return CE->getValue() >= 0 && CE->getValue() <= 0xfff;
  }

  bool isAddSubImmLSL12() const {
    if (!isImmWithLSL()) return false;
    if (ImmWithLSL.ShiftAmount != 12) return false;

    AArch64MCExpr::VariantKind Variant;
    if (isNonConstantExpr(ImmWithLSL.Val, Variant)) {
      return Variant == AArch64MCExpr::VK_AARCH64_DTPREL_HI12
          || Variant == AArch64MCExpr::VK_AARCH64_TPREL_HI12;
    }

    // Otherwise it should be a real immediate in range:
    const MCConstantExpr *CE = cast<MCConstantExpr>(ImmWithLSL.Val);
    return CE->getValue() >= 0 && CE->getValue() <= 0xfff;
  }

  template<unsigned MemSize, unsigned RmSize> bool isAddrRegExtend() const {
    if (!isShiftOrExtend()) return false;

    A64SE::ShiftExtSpecifiers Ext = ShiftExtend.ShiftType;
    if (RmSize == 32 && !(Ext == A64SE::UXTW || Ext == A64SE::SXTW))
      return false;

    if (RmSize == 64 && !(Ext == A64SE::LSL || Ext == A64SE::SXTX))
      return false;

    return ShiftExtend.Amount == Log2_32(MemSize) || ShiftExtend.Amount == 0;
  }

  bool isAdrpLabel() const {
    if (!isImm()) return false;

    AArch64MCExpr::VariantKind Variant;
    if (isNonConstantExpr(getImm(), Variant)) {
      return Variant == AArch64MCExpr::VK_AARCH64_None
        || Variant == AArch64MCExpr::VK_AARCH64_GOT
        || Variant == AArch64MCExpr::VK_AARCH64_GOTTPREL
        || Variant == AArch64MCExpr::VK_AARCH64_TLSDESC;
    }

    return isLabel<21, 4096>();
  }

  template<unsigned RegWidth>  bool isBitfieldWidth() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    return CE->getValue() >= 1 && CE->getValue() <= RegWidth;
  }

  template<int RegWidth>
  bool isCVTFixedPos() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    return CE->getValue() >= 1 && CE->getValue() <= RegWidth;
  }

  bool isFMOVImm() const {
    if (!isFPImm()) return false;

    APFloat RealVal(FPImm.Val);
    uint32_t ImmVal;
    return A64Imms::isFPImm(RealVal, ImmVal);
  }

  bool isFPZero() const {
    if (!isFPImm()) return false;

    APFloat RealVal(FPImm.Val);
    return RealVal.isPosZero();
  }

  template<unsigned field_width, unsigned scale>
  bool isLabel() const {
    if (!isImm()) return false;

    if (dyn_cast<MCSymbolRefExpr>(Imm.Val)) {
      return true;
    } else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Imm.Val)) {
      int64_t Val = CE->getValue();
      int64_t Min = - (scale * (1LL << (field_width - 1)));
      int64_t Max = scale * ((1LL << (field_width - 1)) - 1);
      return (Val % scale) == 0 && Val >= Min && Val <= Max;
    }

    // N.b. this disallows explicit relocation specifications via an
    // AArch64MCExpr. Users needing that behaviour
    return false;
  }

  bool isLane1() const {
    if (!isImm()) return false;

    // Because it's come through custom assembly parsing, it must always be a
    // constant expression.
    return cast<MCConstantExpr>(getImm())->getValue() == 1;
  }

  bool isLoadLitLabel() const {
    if (!isImm()) return false;

    AArch64MCExpr::VariantKind Variant;
    if (isNonConstantExpr(getImm(), Variant)) {
      return Variant == AArch64MCExpr::VK_AARCH64_None
          || Variant == AArch64MCExpr::VK_AARCH64_GOTTPREL;
    }

    return isLabel<19, 4>();
  }

  template<unsigned RegWidth> bool isLogicalImm() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Imm.Val);
    if (!CE) return false;

    uint32_t Bits;
    return A64Imms::isLogicalImm(RegWidth, CE->getValue(), Bits);
  }

  template<unsigned RegWidth> bool isLogicalImmMOV() const {
    if (!isLogicalImm<RegWidth>()) return false;

    const MCConstantExpr *CE = cast<MCConstantExpr>(Imm.Val);

    // The move alias for ORR is only valid if the immediate cannot be
    // represented with a move (immediate) instruction; they take priority.
    int UImm16, Shift;
    return !A64Imms::isMOVZImm(RegWidth, CE->getValue(), UImm16, Shift)
      && !A64Imms::isMOVNImm(RegWidth, CE->getValue(), UImm16, Shift);
  }

  template<int MemSize>
  bool isOffsetUImm12() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());

    // Assume they know what they're doing for now if they've given us a
    // non-constant expression. In principle we could check for ridiculous
    // things that can't possibly work or relocations that would almost
    // certainly break resulting code.
    if (!CE)
      return true;

    int64_t Val = CE->getValue();

    // Must be a multiple of the access size in bytes.
    if ((Val & (MemSize - 1)) != 0) return false;

    // Must be 12-bit unsigned
    return Val >= 0 && Val <= 0xfff * MemSize;
  }

  template<A64SE::ShiftExtSpecifiers SHKind, bool is64Bit>
  bool isShift() const {
    if (!isShiftOrExtend()) return false;

    if (ShiftExtend.ShiftType != SHKind)
      return false;

    return is64Bit ? ShiftExtend.Amount <= 63 : ShiftExtend.Amount <= 31;
  }

  bool isMOVN32Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_SABS_G0,
      AArch64MCExpr::VK_AARCH64_SABS_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G0,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(32, PermittedModifiers, NumModifiers);
  }

  bool isMOVN64Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_SABS_G0,
      AArch64MCExpr::VK_AARCH64_SABS_G1,
      AArch64MCExpr::VK_AARCH64_SABS_G2,
      AArch64MCExpr::VK_AARCH64_DTPREL_G2,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G2,
      AArch64MCExpr::VK_AARCH64_TPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G0,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(64, PermittedModifiers, NumModifiers);
  }


  bool isMOVZ32Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_ABS_G0,
      AArch64MCExpr::VK_AARCH64_ABS_G1,
      AArch64MCExpr::VK_AARCH64_SABS_G0,
      AArch64MCExpr::VK_AARCH64_SABS_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G0,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(32, PermittedModifiers, NumModifiers);
  }

  bool isMOVZ64Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_ABS_G0,
      AArch64MCExpr::VK_AARCH64_ABS_G1,
      AArch64MCExpr::VK_AARCH64_ABS_G2,
      AArch64MCExpr::VK_AARCH64_ABS_G3,
      AArch64MCExpr::VK_AARCH64_SABS_G0,
      AArch64MCExpr::VK_AARCH64_SABS_G1,
      AArch64MCExpr::VK_AARCH64_SABS_G2,
      AArch64MCExpr::VK_AARCH64_DTPREL_G2,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G2,
      AArch64MCExpr::VK_AARCH64_TPREL_G1,
      AArch64MCExpr::VK_AARCH64_TPREL_G0,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(64, PermittedModifiers, NumModifiers);
  }

  bool isMOVK32Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_ABS_G0_NC,
      AArch64MCExpr::VK_AARCH64_ABS_G1_NC,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1_NC,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0_NC,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G0_NC,
      AArch64MCExpr::VK_AARCH64_TPREL_G1_NC,
      AArch64MCExpr::VK_AARCH64_TPREL_G0_NC,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(32, PermittedModifiers, NumModifiers);
  }

  bool isMOVK64Imm() const {
    static const AArch64MCExpr::VariantKind PermittedModifiers[] = {
      AArch64MCExpr::VK_AARCH64_ABS_G0_NC,
      AArch64MCExpr::VK_AARCH64_ABS_G1_NC,
      AArch64MCExpr::VK_AARCH64_ABS_G2_NC,
      AArch64MCExpr::VK_AARCH64_ABS_G3,
      AArch64MCExpr::VK_AARCH64_DTPREL_G1_NC,
      AArch64MCExpr::VK_AARCH64_DTPREL_G0_NC,
      AArch64MCExpr::VK_AARCH64_GOTTPREL_G0_NC,
      AArch64MCExpr::VK_AARCH64_TPREL_G1_NC,
      AArch64MCExpr::VK_AARCH64_TPREL_G0_NC,
    };
    const unsigned NumModifiers = llvm::array_lengthof(PermittedModifiers);

    return isMoveWideImm(64, PermittedModifiers, NumModifiers);
  }

  bool isMoveWideImm(unsigned RegWidth,
                     const AArch64MCExpr::VariantKind *PermittedModifiers,
                     unsigned NumModifiers) const {
    if (!isImmWithLSL()) return false;

    if (ImmWithLSL.ShiftAmount % 16 != 0) return false;
    if (ImmWithLSL.ShiftAmount >= RegWidth) return false;

    AArch64MCExpr::VariantKind Modifier;
    if (isNonConstantExpr(ImmWithLSL.Val, Modifier)) {
      // E.g. "#:abs_g0:sym, lsl #16" makes no sense.
      if (!ImmWithLSL.ImplicitAmount) return false;

      for (unsigned i = 0; i < NumModifiers; ++i)
        if (PermittedModifiers[i] == Modifier) return true;

      return false;
    }

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ImmWithLSL.Val);
    return CE && CE->getValue() >= 0  && CE->getValue() <= 0xffff;
  }

  template<int RegWidth, bool (*isValidImm)(int, uint64_t, int&, int&)>
  bool isMoveWideMovAlias() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    int UImm16, Shift;
    uint64_t Value = CE->getValue();

    // If this is a 32-bit instruction then all bits above 32 should be the
    // same: either of these is fine because signed/unsigned values should be
    // permitted.
    if (RegWidth == 32) {
      if ((Value >> 32) != 0 && (Value >> 32) != 0xffffffff)
        return false;

      Value &= 0xffffffffULL;
    }

    return isValidImm(RegWidth, Value, UImm16, Shift);
  }

  bool isMSRWithReg() const {
    if (!isSysReg()) return false;

    bool IsKnownRegister;
    StringRef Name(SysReg.Data, SysReg.Length);
    A64SysReg::MSRMapper().fromString(Name, IsKnownRegister);

    return IsKnownRegister;
  }

  bool isMSRPState() const {
    if (!isSysReg()) return false;

    bool IsKnownRegister;
    StringRef Name(SysReg.Data, SysReg.Length);
    A64PState::PStateMapper().fromString(Name, IsKnownRegister);

    return IsKnownRegister;
  }

  bool isMRS() const {
    if (!isSysReg()) return false;

    // First check against specific MSR-only (write-only) registers
    bool IsKnownRegister;
    StringRef Name(SysReg.Data, SysReg.Length);
    A64SysReg::MRSMapper().fromString(Name, IsKnownRegister);

    return IsKnownRegister;
  }

  bool isPRFM() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());

    if (!CE)
      return false;

    return CE->getValue() >= 0 && CE->getValue() <= 31;
  }

  template<A64SE::ShiftExtSpecifiers SHKind> bool isRegExtend() const {
    if (!isShiftOrExtend()) return false;

    if (ShiftExtend.ShiftType != SHKind)
      return false;

    return ShiftExtend.Amount <= 4;
  }

  bool isRegExtendLSL() const {
    if (!isShiftOrExtend()) return false;

    if (ShiftExtend.ShiftType != A64SE::LSL)
      return false;

    return !ShiftExtend.ImplicitAmount && ShiftExtend.Amount <= 4;
  }

  // if 0 < value <= w, return true
  bool isShrFixedWidth(int w) const {
    if (!isImm())
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return false;
    int64_t Value = CE->getValue();
    return Value > 0 && Value <= w;
  }

  bool isShrImm8() const { return isShrFixedWidth(8); }

  bool isShrImm16() const { return isShrFixedWidth(16); }

  bool isShrImm32() const { return isShrFixedWidth(32); }

  bool isShrImm64() const { return isShrFixedWidth(64); }

  // if 0 <= value < w, return true
  bool isShlFixedWidth(int w) const {
    if (!isImm())
      return false;
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return false;
    int64_t Value = CE->getValue();
    return Value >= 0 && Value < w;
  }

  bool isShlImm8() const { return isShlFixedWidth(8); }

  bool isShlImm16() const { return isShlFixedWidth(16); }

  bool isShlImm32() const { return isShlFixedWidth(32); }

  bool isShlImm64() const { return isShlFixedWidth(64); }

  bool isNeonMovImmShiftLSL() const {
    if (!isShiftOrExtend())
      return false;

    if (ShiftExtend.ShiftType != A64SE::LSL)
      return false;

    // Valid shift amount is 0, 8, 16 and 24.
    return ShiftExtend.Amount % 8 == 0 && ShiftExtend.Amount <= 24;
  }

  bool isNeonMovImmShiftLSLH() const {
    if (!isShiftOrExtend())
      return false;

    if (ShiftExtend.ShiftType != A64SE::LSL)
      return false;

    // Valid shift amount is 0 and 8.
    return ShiftExtend.Amount == 0 || ShiftExtend.Amount == 8;
  }

  bool isNeonMovImmShiftMSL() const {
    if (!isShiftOrExtend())
      return false;

    if (ShiftExtend.ShiftType != A64SE::MSL)
      return false;

    // Valid shift amount is 8 and 16.
    return ShiftExtend.Amount == 8 || ShiftExtend.Amount == 16;
  }

  template <A64Layout::VectorLayout Layout, unsigned Count>
  bool isVectorList() const {
    return Kind == k_VectorList && VectorList.Layout == Layout &&
           VectorList.Count == Count;
  }

  template <int MemSize> bool isSImm7Scaled() const {
    if (!isImm())
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    int64_t Val = CE->getValue();
    if (Val % MemSize != 0) return false;

    Val /= MemSize;

    return Val >= -64 && Val < 64;
  }

  template<int BitWidth>
  bool isSImm() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    return CE->getValue() >= -(1LL << (BitWidth - 1))
      && CE->getValue() < (1LL << (BitWidth - 1));
  }

  template<int bitWidth>
  bool isUImm() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    return CE->getValue() >= 0 && CE->getValue() < (1LL << bitWidth);
  }

  bool isUImm() const {
    if (!isImm()) return false;

    return isa<MCConstantExpr>(getImm());
  }

  bool isNeonUImm64Mask() const {
    if (!isImm())
      return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return false;

    uint64_t Value = CE->getValue();

    // i64 value with each byte being either 0x00 or 0xff.
    for (unsigned i = 0; i < 8; ++i, Value >>= 8)
      if ((Value & 0xff) != 0 && (Value & 0xff) != 0xff)
        return false;
    return true;
  }

  // if value == N, return true
  template<int N>
  bool isExactImm() const {
    if (!isImm()) return false;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE) return false;

    return CE->getValue() == N;
  }

  static AArch64Operand *CreateImmWithLSL(const MCExpr *Val,
                                          unsigned ShiftAmount,
                                          bool ImplicitAmount,
										  SMLoc S,SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_ImmWithLSL, S, E);
    Op->ImmWithLSL.Val = Val;
    Op->ImmWithLSL.ShiftAmount = ShiftAmount;
    Op->ImmWithLSL.ImplicitAmount = ImplicitAmount;
    return Op;
  }

  static AArch64Operand *CreateCondCode(A64CC::CondCodes Code,
                                        SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_CondCode, S, E);
    Op->CondCode.Code = Code;
    return Op;
  }

  static AArch64Operand *CreateFPImm(double Val,
                                     SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_FPImmediate, S, E);
    Op->FPImm.Val = Val;
    return Op;
  }

  static AArch64Operand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_Immediate, S, E);
    Op->Imm.Val = Val;
    return Op;
  }

  static AArch64Operand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_Register, S, E);
    Op->Reg.RegNum = RegNum;
    return Op;
  }

  static AArch64Operand *CreateWrappedReg(unsigned RegNum, SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_WrappedRegister, S, E);
    Op->Reg.RegNum = RegNum;
    return Op;
  }

  static AArch64Operand *CreateShiftExtend(A64SE::ShiftExtSpecifiers ShiftTyp,
                                           unsigned Amount,
                                           bool ImplicitAmount,
                                           SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_ShiftExtend, S, E);
    Op->ShiftExtend.ShiftType = ShiftTyp;
    Op->ShiftExtend.Amount = Amount;
    Op->ShiftExtend.ImplicitAmount = ImplicitAmount;
    return Op;
  }

  static AArch64Operand *CreateSysReg(StringRef Str, SMLoc S) {
    AArch64Operand *Op = new AArch64Operand(k_SysReg, S, S);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    return Op;
  }

  static AArch64Operand *CreateVectorList(unsigned RegNum, unsigned Count,
                                          A64Layout::VectorLayout Layout,
                                          SMLoc S, SMLoc E) {
    AArch64Operand *Op = new AArch64Operand(k_VectorList, S, E);
    Op->VectorList.RegNum = RegNum;
    Op->VectorList.Count = Count;
    Op->VectorList.Layout = Layout;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static AArch64Operand *CreateToken(StringRef Str, SMLoc S) {
    AArch64Operand *Op = new AArch64Operand(k_Token, S, S);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    return Op;
  }


  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  template<unsigned RegWidth>
  void addBFILSBOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    unsigned EncodedVal = (RegWidth - CE->getValue()) % RegWidth;
    Inst.addOperand(MCOperand::CreateImm(EncodedVal));
  }

  void addBFIWidthOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    Inst.addOperand(MCOperand::CreateImm(CE->getValue() - 1));
  }

  void addBFXWidthOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    uint64_t LSB = Inst.getOperand(Inst.getNumOperands()-1).getImm();
    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());

    Inst.addOperand(MCOperand::CreateImm(LSB + CE->getValue() - 1));
  }

  void addCondCodeOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateImm(getCondCode()));
  }

  void addCVTFixedPosOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    Inst.addOperand(MCOperand::CreateImm(64 - CE->getValue()));
  }

  void addFMOVImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    APFloat RealVal(FPImm.Val);
    uint32_t ImmVal;
    A64Imms::isFPImm(RealVal, ImmVal);

    Inst.addOperand(MCOperand::CreateImm(ImmVal));
  }

  void addFPZeroOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    Inst.addOperand(MCOperand::CreateImm(0));
  }

  void addInvCondCodeOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    unsigned Encoded = A64InvertCondCode(getCondCode());
    Inst.addOperand(MCOperand::CreateImm(Encoded));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  template<int MemSize>
  void addSImm7ScaledOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    uint64_t Val = CE->getValue() / MemSize;
    Inst.addOperand(MCOperand::CreateImm(Val  & 0x7f));
  }

  template<int BitWidth>
  void addSImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    uint64_t Val = CE->getValue();
    Inst.addOperand(MCOperand::CreateImm(Val  & ((1ULL << BitWidth) - 1)));
  }

  void addImmWithLSLOperands(MCInst &Inst, unsigned N) const {
    assert (N == 1 && "Invalid number of operands!");

    addExpr(Inst, ImmWithLSL.Val);
  }

  template<unsigned field_width, unsigned scale>
  void addLabelOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Imm.Val);

    if (!CE) {
      addExpr(Inst, Imm.Val);
      return;
    }

    int64_t Val = CE->getValue();
    assert(Val % scale == 0 && "Unaligned immediate in instruction");
    Val /= scale;

    Inst.addOperand(MCOperand::CreateImm(Val & ((1LL << field_width) - 1)));
  }

  template<int MemSize>
  void addOffsetUImm12Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm())) {
      Inst.addOperand(MCOperand::CreateImm(CE->getValue() / MemSize));
    } else {
      Inst.addOperand(MCOperand::CreateExpr(getImm()));
    }
  }

  template<unsigned RegWidth>
  void addLogicalImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands");
    const MCConstantExpr *CE = cast<MCConstantExpr>(Imm.Val);

    uint32_t Bits;
    A64Imms::isLogicalImm(RegWidth, CE->getValue(), Bits);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addMRSOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    StringRef Name(SysReg.Data, SysReg.Length);
    uint32_t Bits = A64SysReg::MRSMapper().fromString(Name, Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addMSRWithRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    StringRef Name(SysReg.Data, SysReg.Length);
    uint32_t Bits = A64SysReg::MSRMapper().fromString(Name, Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addMSRPStateOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    bool Valid;
    StringRef Name(SysReg.Data, SysReg.Length);
    uint32_t Bits = A64PState::PStateMapper().fromString(Name, Valid);

    Inst.addOperand(MCOperand::CreateImm(Bits));
  }

  void addMoveWideImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    addExpr(Inst, ImmWithLSL.Val);

    AArch64MCExpr::VariantKind Variant;
    if (!isNonConstantExpr(ImmWithLSL.Val, Variant)) {
      Inst.addOperand(MCOperand::CreateImm(ImmWithLSL.ShiftAmount / 16));
      return;
    }

    // We know it's relocated
    switch (Variant) {
    case AArch64MCExpr::VK_AARCH64_ABS_G0:
    case AArch64MCExpr::VK_AARCH64_ABS_G0_NC:
    case AArch64MCExpr::VK_AARCH64_SABS_G0:
    case AArch64MCExpr::VK_AARCH64_DTPREL_G0:
    case AArch64MCExpr::VK_AARCH64_DTPREL_G0_NC:
    case AArch64MCExpr::VK_AARCH64_GOTTPREL_G0_NC:
    case AArch64MCExpr::VK_AARCH64_TPREL_G0:
    case AArch64MCExpr::VK_AARCH64_TPREL_G0_NC:
      Inst.addOperand(MCOperand::CreateImm(0));
      break;
    case AArch64MCExpr::VK_AARCH64_ABS_G1:
    case AArch64MCExpr::VK_AARCH64_ABS_G1_NC:
    case AArch64MCExpr::VK_AARCH64_SABS_G1:
    case AArch64MCExpr::VK_AARCH64_DTPREL_G1:
    case AArch64MCExpr::VK_AARCH64_DTPREL_G1_NC:
    case AArch64MCExpr::VK_AARCH64_GOTTPREL_G1:
    case AArch64MCExpr::VK_AARCH64_TPREL_G1:
    case AArch64MCExpr::VK_AARCH64_TPREL_G1_NC:
      Inst.addOperand(MCOperand::CreateImm(1));
      break;
    case AArch64MCExpr::VK_AARCH64_ABS_G2:
    case AArch64MCExpr::VK_AARCH64_ABS_G2_NC:
    case AArch64MCExpr::VK_AARCH64_SABS_G2:
    case AArch64MCExpr::VK_AARCH64_DTPREL_G2:
    case AArch64MCExpr::VK_AARCH64_TPREL_G2:
      Inst.addOperand(MCOperand::CreateImm(2));
      break;
    case AArch64MCExpr::VK_AARCH64_ABS_G3:
      Inst.addOperand(MCOperand::CreateImm(3));
      break;
    default: llvm_unreachable("Inappropriate move wide relocation");
    }
  }

  template<int RegWidth, bool isValidImm(int, uint64_t, int&, int&)>
  void addMoveWideMovAliasOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    int UImm16, Shift;

    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    uint64_t Value = CE->getValue();

    if (RegWidth == 32) {
      Value &= 0xffffffffULL;
    }

    bool Valid = isValidImm(RegWidth, Value, UImm16, Shift);
    (void)Valid;
    assert(Valid && "Invalid immediates should have been weeded out by now");

    Inst.addOperand(MCOperand::CreateImm(UImm16));
    Inst.addOperand(MCOperand::CreateImm(Shift));
  }

  void addPRFMOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    const MCConstantExpr *CE = cast<MCConstantExpr>(getImm());
    assert(CE->getValue() >= 0 && CE->getValue() <= 31
           && "PRFM operand should be 5-bits");

    Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
  }

  // For Add-sub (extended register) operands.
  void addRegExtendOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateImm(ShiftExtend.Amount));
  }

  // For Vector Immediates shifted imm operands.
  void addNeonMovImmShiftLSLOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    if (ShiftExtend.Amount % 8 != 0 || ShiftExtend.Amount > 24)
      llvm_unreachable("Invalid shift amount for vector immediate inst.");

    // Encode LSL shift amount 0, 8, 16, 24 as 0, 1, 2, 3.
    int64_t Imm = ShiftExtend.Amount / 8;
    Inst.addOperand(MCOperand::CreateImm(Imm));
  }

  void addNeonMovImmShiftLSLHOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    if (ShiftExtend.Amount != 0 && ShiftExtend.Amount != 8)
      llvm_unreachable("Invalid shift amount for vector immediate inst.");

    // Encode LSLH shift amount 0, 8  as 0, 1.
    int64_t Imm = ShiftExtend.Amount / 8;
    Inst.addOperand(MCOperand::CreateImm(Imm));
  }

  void addNeonMovImmShiftMSLOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    if (ShiftExtend.Amount != 8 && ShiftExtend.Amount != 16)
      llvm_unreachable("Invalid shift amount for vector immediate inst.");

    // Encode MSL shift amount 8, 16  as 0, 1.
    int64_t Imm = ShiftExtend.Amount / 8 - 1;
    Inst.addOperand(MCOperand::CreateImm(Imm));
  }

  // For the extend in load-store (register offset) instructions.
  template<unsigned MemSize>
  void addAddrRegExtendOperands(MCInst &Inst, unsigned N) const {
    addAddrRegExtendOperands(Inst, N, MemSize);
  }

  void addAddrRegExtendOperands(MCInst &Inst, unsigned N,
                                unsigned MemSize) const {
    assert(N == 1 && "Invalid number of operands!");

    // First bit of Option is set in instruction classes, the high two bits are
    // as follows:
    unsigned OptionHi = 0;
    switch (ShiftExtend.ShiftType) {
    case A64SE::UXTW:
    case A64SE::LSL:
      OptionHi = 1;
      break;
    case A64SE::SXTW:
    case A64SE::SXTX:
      OptionHi = 3;
      break;
    default:
      llvm_unreachable("Invalid extend type for register offset");
    }

    unsigned S = 0;
    if (MemSize == 1 && !ShiftExtend.ImplicitAmount)
      S = 1;
    else if (MemSize != 1 && ShiftExtend.Amount != 0)
      S = 1;

    Inst.addOperand(MCOperand::CreateImm((OptionHi << 1) | S));
  }
  void addShiftOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateImm(ShiftExtend.Amount));
  }

  void addNeonUImm64MaskOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");

    // A bit from each byte in the constant forms the encoded immediate
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    uint64_t Value = CE->getValue();

    unsigned Imm = 0;
    for (unsigned i = 0; i < 8; ++i, Value >>= 8) {
      Imm |= (Value & 1) << i;
    }
    Inst.addOperand(MCOperand::CreateImm(Imm));
  }

  void addVectorListOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(VectorList.RegNum));
  }
};

} // end anonymous namespace.

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               StringRef Mnemonic) {

  // See if the operand has a custom parser
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // It could either succeed, fail or just not care.
  if (ResTy != MatchOperand_NoMatch)
    return ResTy;

  switch (getLexer().getKind()) {
  default:
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
    return MatchOperand_ParseFail;
  case AsmToken::Identifier: {
    // It might be in the LSL/UXTB family ...
    OperandMatchResultTy GotShift = ParseShiftExtend(Operands);

    // We can only continue if no tokens were eaten.
    if (GotShift != MatchOperand_NoMatch)
      return GotShift;

    // ... or it might be a register ...
    uint32_t NumLanes = 0;
    OperandMatchResultTy GotReg = ParseRegister(Operands, NumLanes);
    assert(GotReg != MatchOperand_ParseFail
           && "register parsing shouldn't partially succeed");

    if (GotReg == MatchOperand_Success) {
      if (Parser.getTok().is(AsmToken::LBrac))
        return ParseNEONLane(Operands, NumLanes);
      else
        return MatchOperand_Success;
    }
    // ... or it might be a symbolish thing
  }
    // Fall through
  case AsmToken::LParen:  // E.g. (strcmp-4)
  case AsmToken::Integer: // 1f, 2b labels
  case AsmToken::String:  // quoted labels
  case AsmToken::Dot:     // . is Current location
  case AsmToken::Dollar:  // $ is PC
  case AsmToken::Colon: {
    SMLoc StartLoc  = Parser.getTok().getLoc();
    SMLoc EndLoc;
    const MCExpr *ImmVal = 0;

    if (ParseImmediate(ImmVal) != MatchOperand_Success)
      return MatchOperand_ParseFail;

    EndLoc = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(AArch64Operand::CreateImm(ImmVal, StartLoc, EndLoc));
    return MatchOperand_Success;
  }
  case AsmToken::Hash: {   // Immediates
    SMLoc StartLoc = Parser.getTok().getLoc();
    SMLoc EndLoc;
    const MCExpr *ImmVal = 0;
    Parser.Lex();

    if (ParseImmediate(ImmVal) != MatchOperand_Success)
      return MatchOperand_ParseFail;

    EndLoc = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
    Operands.push_back(AArch64Operand::CreateImm(ImmVal, StartLoc, EndLoc));
    return MatchOperand_Success;
  }
  case AsmToken::LBrac: {
    SMLoc Loc = Parser.getTok().getLoc();
    Operands.push_back(AArch64Operand::CreateToken("[", Loc));
    Parser.Lex(); // Eat '['

    // There's no comma after a '[', so we can parse the next operand
    // immediately.
    return ParseOperand(Operands, Mnemonic);
  }
  // The following will likely be useful later, but not in very early cases
  case AsmToken::LCurly: // SIMD vector list is not parsed here
    llvm_unreachable("Don't know how to deal with '{' in operand");
    return MatchOperand_ParseFail;
  }
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseImmediate(const MCExpr *&ExprVal) {
  if (getLexer().is(AsmToken::Colon)) {
    AArch64MCExpr::VariantKind RefKind;

    OperandMatchResultTy ResTy = ParseRelocPrefix(RefKind);
    if (ResTy != MatchOperand_Success)
      return ResTy;

    const MCExpr *SubExprVal;
    if (getParser().parseExpression(SubExprVal))
      return MatchOperand_ParseFail;

    ExprVal = AArch64MCExpr::Create(RefKind, SubExprVal, getContext());
    return MatchOperand_Success;
  }

  // No weird AArch64MCExpr prefix
  return getParser().parseExpression(ExprVal)
    ? MatchOperand_ParseFail : MatchOperand_Success;
}

// A lane attached to a NEON register. "[N]", which should yield three tokens:
// '[', N, ']'. A hash is not allowed to precede the immediate here.
AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseNEONLane(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                uint32_t NumLanes) {
  SMLoc Loc = Parser.getTok().getLoc();

  assert(Parser.getTok().is(AsmToken::LBrac) && "inappropriate operand");
  Operands.push_back(AArch64Operand::CreateToken("[", Loc));
  Parser.Lex(); // Eat '['

  if (Parser.getTok().isNot(AsmToken::Integer)) {
    Error(Parser.getTok().getLoc(), "expected lane number");
    return MatchOperand_ParseFail;
  }

  if (Parser.getTok().getIntVal() >= NumLanes) {
    Error(Parser.getTok().getLoc(), "lane number incompatible with layout");
    return MatchOperand_ParseFail;
  }

  const MCExpr *Lane = MCConstantExpr::Create(Parser.getTok().getIntVal(),
                                              getContext());
  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat actual lane
  SMLoc E = Parser.getTok().getLoc();
  Operands.push_back(AArch64Operand::CreateImm(Lane, S, E));


  if (Parser.getTok().isNot(AsmToken::RBrac)) {
    Error(Parser.getTok().getLoc(), "expected ']' after lane");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(AArch64Operand::CreateToken("]", Loc));
  Parser.Lex(); // Eat ']'

  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseRelocPrefix(AArch64MCExpr::VariantKind &RefKind) {
  assert(getLexer().is(AsmToken::Colon) && "expected a ':'");
  Parser.Lex();

  if (getLexer().isNot(AsmToken::Identifier)) {
    Error(Parser.getTok().getLoc(),
          "expected relocation specifier in operand after ':'");
    return MatchOperand_ParseFail;
  }

  std::string LowerCase = Parser.getTok().getIdentifier().lower();
  RefKind = StringSwitch<AArch64MCExpr::VariantKind>(LowerCase)
    .Case("got",              AArch64MCExpr::VK_AARCH64_GOT)
    .Case("got_lo12",         AArch64MCExpr::VK_AARCH64_GOT_LO12)
    .Case("lo12",             AArch64MCExpr::VK_AARCH64_LO12)
    .Case("abs_g0",           AArch64MCExpr::VK_AARCH64_ABS_G0)
    .Case("abs_g0_nc",        AArch64MCExpr::VK_AARCH64_ABS_G0_NC)
    .Case("abs_g1",           AArch64MCExpr::VK_AARCH64_ABS_G1)
    .Case("abs_g1_nc",        AArch64MCExpr::VK_AARCH64_ABS_G1_NC)
    .Case("abs_g2",           AArch64MCExpr::VK_AARCH64_ABS_G2)
    .Case("abs_g2_nc",        AArch64MCExpr::VK_AARCH64_ABS_G2_NC)
    .Case("abs_g3",           AArch64MCExpr::VK_AARCH64_ABS_G3)
    .Case("abs_g0_s",         AArch64MCExpr::VK_AARCH64_SABS_G0)
    .Case("abs_g1_s",         AArch64MCExpr::VK_AARCH64_SABS_G1)
    .Case("abs_g2_s",         AArch64MCExpr::VK_AARCH64_SABS_G2)
    .Case("dtprel_g2",        AArch64MCExpr::VK_AARCH64_DTPREL_G2)
    .Case("dtprel_g1",        AArch64MCExpr::VK_AARCH64_DTPREL_G1)
    .Case("dtprel_g1_nc",     AArch64MCExpr::VK_AARCH64_DTPREL_G1_NC)
    .Case("dtprel_g0",        AArch64MCExpr::VK_AARCH64_DTPREL_G0)
    .Case("dtprel_g0_nc",     AArch64MCExpr::VK_AARCH64_DTPREL_G0_NC)
    .Case("dtprel_hi12",      AArch64MCExpr::VK_AARCH64_DTPREL_HI12)
    .Case("dtprel_lo12",      AArch64MCExpr::VK_AARCH64_DTPREL_LO12)
    .Case("dtprel_lo12_nc",   AArch64MCExpr::VK_AARCH64_DTPREL_LO12_NC)
    .Case("gottprel_g1",      AArch64MCExpr::VK_AARCH64_GOTTPREL_G1)
    .Case("gottprel_g0_nc",   AArch64MCExpr::VK_AARCH64_GOTTPREL_G0_NC)
    .Case("gottprel",         AArch64MCExpr::VK_AARCH64_GOTTPREL)
    .Case("gottprel_lo12",    AArch64MCExpr::VK_AARCH64_GOTTPREL_LO12)
    .Case("tprel_g2",         AArch64MCExpr::VK_AARCH64_TPREL_G2)
    .Case("tprel_g1",         AArch64MCExpr::VK_AARCH64_TPREL_G1)
    .Case("tprel_g1_nc",      AArch64MCExpr::VK_AARCH64_TPREL_G1_NC)
    .Case("tprel_g0",         AArch64MCExpr::VK_AARCH64_TPREL_G0)
    .Case("tprel_g0_nc",      AArch64MCExpr::VK_AARCH64_TPREL_G0_NC)
    .Case("tprel_hi12",       AArch64MCExpr::VK_AARCH64_TPREL_HI12)
    .Case("tprel_lo12",       AArch64MCExpr::VK_AARCH64_TPREL_LO12)
    .Case("tprel_lo12_nc",    AArch64MCExpr::VK_AARCH64_TPREL_LO12_NC)
    .Case("tlsdesc",          AArch64MCExpr::VK_AARCH64_TLSDESC)
    .Case("tlsdesc_lo12",     AArch64MCExpr::VK_AARCH64_TLSDESC_LO12)
    .Default(AArch64MCExpr::VK_AARCH64_None);

  if (RefKind == AArch64MCExpr::VK_AARCH64_None) {
    Error(Parser.getTok().getLoc(),
          "expected relocation specifier in operand after ':'");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat identifier

  if (getLexer().isNot(AsmToken::Colon)) {
    Error(Parser.getTok().getLoc(),
          "expected ':' after relocation specifier");
    return MatchOperand_ParseFail;
  }
  Parser.Lex();
  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseImmWithLSLOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // FIXME?: I want to live in a world where immediates must start with
  // #. Please don't dash my hopes (well, do if you have a good reason).
  if (Parser.getTok().isNot(AsmToken::Hash)) return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat '#'

  const MCExpr *Imm;
  if (ParseImmediate(Imm) != MatchOperand_Success)
    return MatchOperand_ParseFail;
  else if (Parser.getTok().isNot(AsmToken::Comma)) {
    SMLoc E = Parser.getTok().getLoc();
    Operands.push_back(AArch64Operand::CreateImmWithLSL(Imm, 0, true, S, E));
    return MatchOperand_Success;
  }

  // Eat ','
  Parser.Lex();

  // The optional operand must be "lsl #N" where N is non-negative.
  if (Parser.getTok().is(AsmToken::Identifier)
      && Parser.getTok().getIdentifier().equals_lower("lsl")) {
    Parser.Lex();

    if (Parser.getTok().is(AsmToken::Hash)) {
      Parser.Lex();

      if (Parser.getTok().isNot(AsmToken::Integer)) {
        Error(Parser.getTok().getLoc(), "only 'lsl #+N' valid after immediate");
        return MatchOperand_ParseFail;
      }
    }
  }

  int64_t ShiftAmount = Parser.getTok().getIntVal();

  if (ShiftAmount < 0) {
    Error(Parser.getTok().getLoc(), "positive shift amount required");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat the number

  SMLoc E = Parser.getTok().getLoc();
  Operands.push_back(AArch64Operand::CreateImmWithLSL(Imm, ShiftAmount,
                                                      false, S, E));
  return MatchOperand_Success;
}


AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseCondCodeOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  if (Parser.getTok().isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;

  StringRef Tok = Parser.getTok().getIdentifier();
  A64CC::CondCodes CondCode = A64StringToCondCode(Tok);

  if (CondCode == A64CC::Invalid)
    return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat condition code
  SMLoc E = Parser.getTok().getLoc();

  Operands.push_back(AArch64Operand::CreateCondCode(CondCode, S, E));
  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseCRxOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  if (Parser.getTok().isNot(AsmToken::Identifier)) {
    Error(S, "Expected cN operand where 0 <= N <= 15");
    return MatchOperand_ParseFail;
  }

  StringRef Tok = Parser.getTok().getIdentifier();
  if (Tok[0] != 'c' && Tok[0] != 'C') {
    Error(S, "Expected cN operand where 0 <= N <= 15");
    return MatchOperand_ParseFail;
  }

  uint32_t CRNum;
  bool BadNum = Tok.drop_front().getAsInteger(10, CRNum);
  if (BadNum || CRNum > 15) {
    Error(S, "Expected cN operand where 0 <= N <= 15");
    return MatchOperand_ParseFail;
  }

  const MCExpr *CRImm = MCConstantExpr::Create(CRNum, getContext());

  Parser.Lex();
  SMLoc E = Parser.getTok().getLoc();

  Operands.push_back(AArch64Operand::CreateImm(CRImm, S, E));
  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseFPImmOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {

  // FIXME?: I want to live in a world where immediates must start with
  // #. Please don't dash my hopes (well, do if you have a good reason).
  if (Parser.getTok().isNot(AsmToken::Hash)) return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex(); // Eat '#'

  bool Negative = false;
  if (Parser.getTok().is(AsmToken::Minus)) {
    Negative = true;
    Parser.Lex(); // Eat '-'
  } else if (Parser.getTok().is(AsmToken::Plus)) {
    Parser.Lex(); // Eat '+'
  }

  if (Parser.getTok().isNot(AsmToken::Real)) {
    Error(S, "Expected floating-point immediate");
    return MatchOperand_ParseFail;
  }

  APFloat RealVal(APFloat::IEEEdouble, Parser.getTok().getString());
  if (Negative) RealVal.changeSign();
  double DblVal = RealVal.convertToDouble();

  Parser.Lex(); // Eat real number
  SMLoc E = Parser.getTok().getLoc();

  Operands.push_back(AArch64Operand::CreateFPImm(DblVal, S, E));
  return MatchOperand_Success;
}


// Automatically generated
static unsigned MatchRegisterName(StringRef Name);

bool
AArch64AsmParser::IdentifyRegister(unsigned &RegNum, SMLoc &RegEndLoc,
                                   StringRef &Layout,
                                   SMLoc &LayoutLoc) const {
  const AsmToken &Tok = Parser.getTok();

  if (Tok.isNot(AsmToken::Identifier))
    return false;

  std::string LowerReg = Tok.getString().lower();
  size_t DotPos = LowerReg.find('.');

  bool IsVec128 = false;
  SMLoc S = Tok.getLoc();
  RegEndLoc = SMLoc::getFromPointer(S.getPointer() + DotPos);

  if (DotPos == std::string::npos) {
    Layout = StringRef();
  } else {
    // Everything afterwards needs to be a literal token, expected to be
    // '.2d','.b' etc for vector registers.

    // This StringSwitch validates the input and (perhaps more importantly)
    // gives us a permanent string to use in the token (a pointer into LowerReg
    // would go out of scope when we return).
    LayoutLoc = SMLoc::getFromPointer(S.getPointer() + DotPos + 1);
    StringRef LayoutText = StringRef(LowerReg).substr(DotPos);

    // See if it's a 128-bit layout first.
    Layout = StringSwitch<const char *>(LayoutText)
      .Case(".q", ".q").Case(".1q", ".1q")
      .Case(".d", ".d").Case(".2d", ".2d")
      .Case(".s", ".s").Case(".4s", ".4s")
      .Case(".h", ".h").Case(".8h", ".8h")
      .Case(".b", ".b").Case(".16b", ".16b")
      .Default("");

    if (Layout.size() != 0)
      IsVec128 = true;
    else {
      Layout = StringSwitch<const char *>(LayoutText)
                   .Case(".1d", ".1d")
                   .Case(".2s", ".2s")
                   .Case(".4h", ".4h")
                   .Case(".8b", ".8b")
                   .Default("");
    }

    if (Layout.size() == 0) {
      // If we've still not pinned it down the register is malformed.
      return false;
    }
  }

  RegNum = MatchRegisterName(LowerReg.substr(0, DotPos));
  if (RegNum == AArch64::NoRegister) {
    RegNum = StringSwitch<unsigned>(LowerReg.substr(0, DotPos))
      .Case("ip0", AArch64::X16)
      .Case("ip1", AArch64::X17)
      .Case("fp", AArch64::X29)
      .Case("lr", AArch64::X30)
      .Case("v0", IsVec128 ? AArch64::Q0 : AArch64::D0)
      .Case("v1", IsVec128 ? AArch64::Q1 : AArch64::D1)
      .Case("v2", IsVec128 ? AArch64::Q2 : AArch64::D2)
      .Case("v3", IsVec128 ? AArch64::Q3 : AArch64::D3)
      .Case("v4", IsVec128 ? AArch64::Q4 : AArch64::D4)
      .Case("v5", IsVec128 ? AArch64::Q5 : AArch64::D5)
      .Case("v6", IsVec128 ? AArch64::Q6 : AArch64::D6)
      .Case("v7", IsVec128 ? AArch64::Q7 : AArch64::D7)
      .Case("v8", IsVec128 ? AArch64::Q8 : AArch64::D8)
      .Case("v9", IsVec128 ? AArch64::Q9 : AArch64::D9)
      .Case("v10", IsVec128 ? AArch64::Q10 : AArch64::D10)
      .Case("v11", IsVec128 ? AArch64::Q11 : AArch64::D11)
      .Case("v12", IsVec128 ? AArch64::Q12 : AArch64::D12)
      .Case("v13", IsVec128 ? AArch64::Q13 : AArch64::D13)
      .Case("v14", IsVec128 ? AArch64::Q14 : AArch64::D14)
      .Case("v15", IsVec128 ? AArch64::Q15 : AArch64::D15)
      .Case("v16", IsVec128 ? AArch64::Q16 : AArch64::D16)
      .Case("v17", IsVec128 ? AArch64::Q17 : AArch64::D17)
      .Case("v18", IsVec128 ? AArch64::Q18 : AArch64::D18)
      .Case("v19", IsVec128 ? AArch64::Q19 : AArch64::D19)
      .Case("v20", IsVec128 ? AArch64::Q20 : AArch64::D20)
      .Case("v21", IsVec128 ? AArch64::Q21 : AArch64::D21)
      .Case("v22", IsVec128 ? AArch64::Q22 : AArch64::D22)
      .Case("v23", IsVec128 ? AArch64::Q23 : AArch64::D23)
      .Case("v24", IsVec128 ? AArch64::Q24 : AArch64::D24)
      .Case("v25", IsVec128 ? AArch64::Q25 : AArch64::D25)
      .Case("v26", IsVec128 ? AArch64::Q26 : AArch64::D26)
      .Case("v27", IsVec128 ? AArch64::Q27 : AArch64::D27)
      .Case("v28", IsVec128 ? AArch64::Q28 : AArch64::D28)
      .Case("v29", IsVec128 ? AArch64::Q29 : AArch64::D29)
      .Case("v30", IsVec128 ? AArch64::Q30 : AArch64::D30)
      .Case("v31", IsVec128 ? AArch64::Q31 : AArch64::D31)
      .Default(AArch64::NoRegister);
  }
  if (RegNum == AArch64::NoRegister)
    return false;

  return true;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseRegister(SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                uint32_t &NumLanes) {
  unsigned RegNum;
  StringRef Layout;
  SMLoc RegEndLoc, LayoutLoc;
  SMLoc S = Parser.getTok().getLoc();

  if (!IdentifyRegister(RegNum, RegEndLoc, Layout, LayoutLoc))
    return MatchOperand_NoMatch;

  Operands.push_back(AArch64Operand::CreateReg(RegNum, S, RegEndLoc));

  if (Layout.size() != 0) {
    unsigned long long TmpLanes = 0;
    llvm::getAsUnsignedInteger(Layout.substr(1), 10, TmpLanes);
    if (TmpLanes != 0) {
      NumLanes = TmpLanes;
    } else {
      // If the number of lanes isn't specified explicitly, a valid instruction
      // will have an element specifier and be capable of acting on the entire
      // vector register.
      switch (Layout.back()) {
      default: llvm_unreachable("Invalid layout specifier");
      case 'b': NumLanes = 16; break;
      case 'h': NumLanes = 8; break;
      case 's': NumLanes = 4; break;
      case 'd': NumLanes = 2; break;
      case 'q': NumLanes = 1; break;
      }
    }

    Operands.push_back(AArch64Operand::CreateToken(Layout, LayoutLoc));
  }

  Parser.Lex();
  return MatchOperand_Success;
}

bool
AArch64AsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                SMLoc &EndLoc) {
  // This callback is used for things like DWARF frame directives in
  // assembly. They don't care about things like NEON layouts or lanes, they
  // just want to be able to produce the DWARF register number.
  StringRef LayoutSpec;
  SMLoc RegEndLoc, LayoutLoc;
  StartLoc = Parser.getTok().getLoc();

  if (!IdentifyRegister(RegNo, RegEndLoc, LayoutSpec, LayoutLoc))
    return true;

  Parser.Lex();
  EndLoc = Parser.getTok().getLoc();

  return false;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseNamedImmOperand(const NamedImmMapper &Mapper,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // Since these operands occur in very limited circumstances, without
  // alternatives, we actually signal an error if there is no match. If relaxing
  // this, beware of unintended consequences: an immediate will be accepted
  // during matching, no matter how it gets into the AArch64Operand.
  const AsmToken &Tok = Parser.getTok();
  SMLoc S = Tok.getLoc();

  if (Tok.is(AsmToken::Identifier)) {
    bool ValidName;
    uint32_t Code = Mapper.fromString(Tok.getString().lower(), ValidName);

    if (!ValidName) {
      Error(S, "operand specifier not recognised");
      return MatchOperand_ParseFail;
    }

    Parser.Lex(); // We're done with the identifier. Eat it

    SMLoc E = Parser.getTok().getLoc();
    const MCExpr *Imm = MCConstantExpr::Create(Code, getContext());
    Operands.push_back(AArch64Operand::CreateImm(Imm, S, E));
    return MatchOperand_Success;
  } else if (Tok.is(AsmToken::Hash)) {
    Parser.Lex();

    const MCExpr *ImmVal;
    if (ParseImmediate(ImmVal) != MatchOperand_Success)
      return MatchOperand_ParseFail;

    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(ImmVal);
    if (!CE || CE->getValue() < 0 || !Mapper.validImm(CE->getValue())) {
      Error(S, "Invalid immediate for instruction");
      return MatchOperand_ParseFail;
    }

    SMLoc E = Parser.getTok().getLoc();
    Operands.push_back(AArch64Operand::CreateImm(ImmVal, S, E));
    return MatchOperand_Success;
  }

  Error(S, "unexpected operand for instruction");
  return MatchOperand_ParseFail;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseSysRegOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  const AsmToken &Tok = Parser.getTok();

  // Any MSR/MRS operand will be an identifier, and we want to store it as some
  // kind of string: SPSel is valid for two different forms of MSR with two
  // different encodings. There's no collision at the moment, but the potential
  // is there.
  if (!Tok.is(AsmToken::Identifier)) {
    return MatchOperand_NoMatch;
  }

  SMLoc S = Tok.getLoc();
  Operands.push_back(AArch64Operand::CreateSysReg(Tok.getString(), S));
  Parser.Lex(); // Eat identifier

  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseLSXAddressOperand(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  SMLoc S = Parser.getTok().getLoc();

  unsigned RegNum;
  SMLoc RegEndLoc, LayoutLoc;
  StringRef Layout;
  if(!IdentifyRegister(RegNum, RegEndLoc, Layout, LayoutLoc)
     || !AArch64MCRegisterClasses[AArch64::GPR64xspRegClassID].contains(RegNum)
     || Layout.size() != 0) {
    // Check Layout.size because we don't want to let "x3.4s" or similar
    // through.
    return MatchOperand_NoMatch;
  }
  Parser.Lex(); // Eat register

  if (Parser.getTok().is(AsmToken::RBrac)) {
    // We're done
    SMLoc E = Parser.getTok().getLoc();
    Operands.push_back(AArch64Operand::CreateWrappedReg(RegNum, S, E));
    return MatchOperand_Success;
  }

  // Otherwise, only ", #0" is valid

  if (Parser.getTok().isNot(AsmToken::Comma)) {
    Error(Parser.getTok().getLoc(), "expected ',' or ']' after register");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat ','

  if (Parser.getTok().isNot(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(), "expected '#0'");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat '#'

  if (Parser.getTok().isNot(AsmToken::Integer)
      || Parser.getTok().getIntVal() != 0 ) {
    Error(Parser.getTok().getLoc(), "expected '#0'");
    return MatchOperand_ParseFail;
  }
  Parser.Lex(); // Eat '0'

  SMLoc E = Parser.getTok().getLoc();
  Operands.push_back(AArch64Operand::CreateWrappedReg(RegNum, S, E));
  return MatchOperand_Success;
}

AArch64AsmParser::OperandMatchResultTy
AArch64AsmParser::ParseShiftExtend(
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  StringRef IDVal = Parser.getTok().getIdentifier();
  std::string LowerID = IDVal.lower();

  A64SE::ShiftExtSpecifiers Spec =
      StringSwitch<A64SE::ShiftExtSpecifiers>(LowerID)
        .Case("lsl", A64SE::LSL)
	.Case("msl", A64SE::MSL)
	.Case("lsr", A64SE::LSR)
	.Case("asr", A64SE::ASR)
	.Case("ror", A64SE::ROR)
	.Case("uxtb", A64SE::UXTB)
	.Case("uxth", A64SE::UXTH)
	.Case("uxtw", A64SE::UXTW)
	.Case("uxtx", A64SE::UXTX)
	.Case("sxtb", A64SE::SXTB)
	.Case("sxth", A64SE::SXTH)
	.Case("sxtw", A64SE::SXTW)
	.Case("sxtx", A64SE::SXTX)
	.Default(A64SE::Invalid);

  if (Spec == A64SE::Invalid)
    return MatchOperand_NoMatch;

  // Eat the shift
  SMLoc S, E;
  S = Parser.getTok().getLoc();
  Parser.Lex();

  if (Spec != A64SE::LSL && Spec != A64SE::LSR && Spec != A64SE::ASR &&
      Spec != A64SE::ROR && Spec != A64SE::MSL) {
    // The shift amount can be omitted for the extending versions, but not real
    // shifts:
    //     add x0, x0, x0, uxtb
    // is valid, and equivalent to
    //     add x0, x0, x0, uxtb #0

    if (Parser.getTok().is(AsmToken::Comma) ||
        Parser.getTok().is(AsmToken::EndOfStatement) ||
        Parser.getTok().is(AsmToken::RBrac)) {
      Operands.push_back(AArch64Operand::CreateShiftExtend(Spec, 0, true,
                                                           S, E));
      return MatchOperand_Success;
    }
  }

  // Eat # at beginning of immediate
  if (!Parser.getTok().is(AsmToken::Hash)) {
    Error(Parser.getTok().getLoc(),
          "expected #imm after shift specifier");
    return MatchOperand_ParseFail;
  }
  Parser.Lex();

  // Make sure we do actually have a number
  if (!Parser.getTok().is(AsmToken::Integer)) {
    Error(Parser.getTok().getLoc(),
          "expected integer shift amount");
    return MatchOperand_ParseFail;
  }
  unsigned Amount = Parser.getTok().getIntVal();
  Parser.Lex();
  E = Parser.getTok().getLoc();

  Operands.push_back(AArch64Operand::CreateShiftExtend(Spec, Amount, false,
                                                       S, E));

  return MatchOperand_Success;
}

/// Try to parse a vector register token, If it is a vector register,
/// the token is eaten and return true. Otherwise return false.
bool AArch64AsmParser::TryParseVector(uint32_t &RegNum, SMLoc &RegEndLoc,
                                      StringRef &Layout, SMLoc &LayoutLoc) {
  bool IsVector = true;

  if (!IdentifyRegister(RegNum, RegEndLoc, Layout, LayoutLoc))
    IsVector = false;
  else if (!AArch64MCRegisterClasses[AArch64::FPR64RegClassID]
                .contains(RegNum) &&
           !AArch64MCRegisterClasses[AArch64::FPR128RegClassID]
                .contains(RegNum))
    IsVector = false;
  else if (Layout.size() == 0)
    IsVector = false;

  if (!IsVector)
    Error(Parser.getTok().getLoc(), "expected vector type register");

  Parser.Lex(); // Eat this token.
  return IsVector;
}


// A vector list contains 1-4 consecutive registers.
// Now there are two kinds of vector list when number of vector > 1:
//   (1) {Vn.layout, Vn+1.layout, ... , Vm.layout}
//   (2) {Vn.layout - Vm.layout}
// If the layout is like .b/.h/.s/.d, also parse the lane.
AArch64AsmParser::OperandMatchResultTy AArch64AsmParser::ParseVectorList(
    SmallVectorImpl<MCParsedAsmOperand *> &Operands) {
  if (Parser.getTok().isNot(AsmToken::LCurly)) {
    Error(Parser.getTok().getLoc(), "'{' expected");
    return MatchOperand_ParseFail;
  }
  SMLoc SLoc = Parser.getTok().getLoc();
  Parser.Lex(); // Eat '{' token.

  unsigned Reg, Count = 1;
  StringRef LayoutStr;
  SMLoc RegEndLoc, LayoutLoc;
  if (!TryParseVector(Reg, RegEndLoc, LayoutStr, LayoutLoc))
    return MatchOperand_ParseFail;

  if (Parser.getTok().is(AsmToken::Minus)) {
    Parser.Lex(); // Eat the minus.

    unsigned Reg2;
    StringRef LayoutStr2;
    SMLoc RegEndLoc2, LayoutLoc2;
    SMLoc RegLoc2 = Parser.getTok().getLoc();

    if (!TryParseVector(Reg2, RegEndLoc2, LayoutStr2, LayoutLoc2))
      return MatchOperand_ParseFail;
    unsigned Space = (Reg < Reg2) ? (Reg2 - Reg) : (Reg2 + 32 - Reg);

    if (LayoutStr != LayoutStr2) {
      Error(LayoutLoc2, "expected the same vector layout");
      return MatchOperand_ParseFail;
    }
    if (Space == 0 || Space > 3) {
      Error(RegLoc2, "invalid number of vectors");
      return MatchOperand_ParseFail;
    }

    Count += Space;
  } else {
    unsigned LastReg = Reg;
    while (Parser.getTok().is(AsmToken::Comma)) {
      Parser.Lex(); // Eat the comma.
      unsigned Reg2;
      StringRef LayoutStr2;
      SMLoc RegEndLoc2, LayoutLoc2;
      SMLoc RegLoc2 = Parser.getTok().getLoc();

      if (!TryParseVector(Reg2, RegEndLoc2, LayoutStr2, LayoutLoc2))
        return MatchOperand_ParseFail;
      unsigned Space = (LastReg < Reg2) ? (Reg2 - LastReg)
                                        : (Reg2 + 32 - LastReg);
      Count++;

      // The space between two vectors should be 1. And they should have the same layout.
      // Total count shouldn't be great than 4
      if (Space != 1) {
        Error(RegLoc2, "invalid space between two vectors");
        return MatchOperand_ParseFail;
      }
      if (LayoutStr != LayoutStr2) {
        Error(LayoutLoc2, "expected the same vector layout");
        return MatchOperand_ParseFail;
      }
      if (Count > 4) {
        Error(RegLoc2, "invalid number of vectors");
        return MatchOperand_ParseFail;
      }

      LastReg = Reg2;
    }
  }

  if (Parser.getTok().isNot(AsmToken::RCurly)) {
    Error(Parser.getTok().getLoc(), "'}' expected");
    return MatchOperand_ParseFail;
  }
  SMLoc ELoc = Parser.getTok().getLoc();
  Parser.Lex(); // Eat '}' token.

  A64Layout::VectorLayout Layout = A64StringToVectorLayout(LayoutStr);
  if (Count > 1) { // If count > 1, create vector list using super register.
    bool IsVec64 = (Layout < A64Layout::VL_16B);
    static unsigned SupRegIDs[3][2] = {
      { AArch64::QPairRegClassID, AArch64::DPairRegClassID },
      { AArch64::QTripleRegClassID, AArch64::DTripleRegClassID },
      { AArch64::QQuadRegClassID, AArch64::DQuadRegClassID }
    };
    unsigned SupRegID = SupRegIDs[Count - 2][static_cast<int>(IsVec64)];
    unsigned Sub0 = IsVec64 ? AArch64::dsub_0 : AArch64::qsub_0;
    const MCRegisterInfo *MRI = getContext().getRegisterInfo();
    Reg = MRI->getMatchingSuperReg(Reg, Sub0,
                                   &AArch64MCRegisterClasses[SupRegID]);
  }
  Operands.push_back(
      AArch64Operand::CreateVectorList(Reg, Count, Layout, SLoc, ELoc));

  if (Parser.getTok().is(AsmToken::LBrac)) {
    uint32_t NumLanes = 0;
    switch(Layout) {
    case A64Layout::VL_B : NumLanes = 16; break;
    case A64Layout::VL_H : NumLanes = 8; break;
    case A64Layout::VL_S : NumLanes = 4; break;
    case A64Layout::VL_D : NumLanes = 2; break;
    default:
      SMLoc Loc = getLexer().getLoc();
      Error(Loc, "expected comma before next operand");
      return MatchOperand_ParseFail;
    }
    return ParseNEONLane(Operands, NumLanes);
  } else {
    return MatchOperand_Success;
  }
}

// FIXME: We would really like to be able to tablegen'erate this.
bool AArch64AsmParser::
validateInstruction(MCInst &Inst,
                    const SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  switch (Inst.getOpcode()) {
  case AArch64::BFIwwii:
  case AArch64::BFIxxii:
  case AArch64::SBFIZwwii:
  case AArch64::SBFIZxxii:
  case AArch64::UBFIZwwii:
  case AArch64::UBFIZxxii:  {
    unsigned ImmOps = Inst.getNumOperands() - 2;
    int64_t ImmR = Inst.getOperand(ImmOps).getImm();
    int64_t ImmS = Inst.getOperand(ImmOps+1).getImm();

    if (ImmR != 0 && ImmS >= ImmR) {
      return Error(Operands[4]->getStartLoc(),
                   "requested insert overflows register");
    }
    return false;
  }
  case AArch64::BFXILwwii:
  case AArch64::BFXILxxii:
  case AArch64::SBFXwwii:
  case AArch64::SBFXxxii:
  case AArch64::UBFXwwii:
  case AArch64::UBFXxxii: {
    unsigned ImmOps = Inst.getNumOperands() - 2;
    int64_t ImmR = Inst.getOperand(ImmOps).getImm();
    int64_t ImmS = Inst.getOperand(ImmOps+1).getImm();
    int64_t RegWidth = 0;
    switch (Inst.getOpcode()) {
    case AArch64::SBFXxxii: case AArch64::UBFXxxii: case AArch64::BFXILxxii:
      RegWidth = 64;
      break;
    case AArch64::SBFXwwii: case AArch64::UBFXwwii: case AArch64::BFXILwwii:
      RegWidth = 32;
      break;
    }

    if (ImmS >= RegWidth || ImmS < ImmR) {
      return Error(Operands[4]->getStartLoc(),
                   "requested extract overflows register");
    }
    return false;
  }
  case AArch64::ICix: {
    int64_t ImmVal = Inst.getOperand(0).getImm();
    A64IC::ICValues ICOp = static_cast<A64IC::ICValues>(ImmVal);
    if (!A64IC::NeedsRegister(ICOp)) {
      return Error(Operands[1]->getStartLoc(),
                   "specified IC op does not use a register");
    }
    return false;
  }
  case AArch64::ICi: {
    int64_t ImmVal = Inst.getOperand(0).getImm();
    A64IC::ICValues ICOp = static_cast<A64IC::ICValues>(ImmVal);
    if (A64IC::NeedsRegister(ICOp)) {
      return Error(Operands[1]->getStartLoc(),
                   "specified IC op requires a register");
    }
    return false;
  }
  case AArch64::TLBIix: {
    int64_t ImmVal = Inst.getOperand(0).getImm();
    A64TLBI::TLBIValues TLBIOp = static_cast<A64TLBI::TLBIValues>(ImmVal);
    if (!A64TLBI::NeedsRegister(TLBIOp)) {
      return Error(Operands[1]->getStartLoc(),
                   "specified TLBI op does not use a register");
    }
    return false;
  }
  case AArch64::TLBIi: {
    int64_t ImmVal = Inst.getOperand(0).getImm();
    A64TLBI::TLBIValues TLBIOp = static_cast<A64TLBI::TLBIValues>(ImmVal);
    if (A64TLBI::NeedsRegister(TLBIOp)) {
      return Error(Operands[1]->getStartLoc(),
                   "specified TLBI op requires a register");
    }
    return false;
  }
  }

  return false;
}


// Parses the instruction *together with* all operands, appending each parsed
// operand to the "Operands" list
bool AArch64AsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                        StringRef Name, SMLoc NameLoc,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  size_t CondCodePos = Name.find('.');

  StringRef Mnemonic = Name.substr(0, CondCodePos);
  Operands.push_back(AArch64Operand::CreateToken(Mnemonic, NameLoc));

  if (CondCodePos != StringRef::npos) {
    // We have a condition code
    SMLoc S = SMLoc::getFromPointer(NameLoc.getPointer() + CondCodePos + 1);
    StringRef CondStr = Name.substr(CondCodePos + 1, StringRef::npos);
    A64CC::CondCodes Code;

    Code = A64StringToCondCode(CondStr);

    if (Code == A64CC::Invalid) {
      Error(S, "invalid condition code");
      Parser.eatToEndOfStatement();
      return true;
    }

    SMLoc DotL = SMLoc::getFromPointer(NameLoc.getPointer() + CondCodePos);

    Operands.push_back(AArch64Operand::CreateToken(".",  DotL));
    SMLoc E = SMLoc::getFromPointer(NameLoc.getPointer() + CondCodePos + 3);
    Operands.push_back(AArch64Operand::CreateCondCode(Code, S, E));
  }

  // Now we parse the operands of this instruction
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Mnemonic)) {
      Parser.eatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, Mnemonic)) {
        Parser.eatToEndOfStatement();
        return true;
      }


      // After successfully parsing some operands there are two special cases to
      // consider (i.e. notional operands not separated by commas). Both are due
      // to memory specifiers:
      //  + An RBrac will end an address for load/store/prefetch
      //  + An '!' will indicate a pre-indexed operation.
      //
      // It's someone else's responsibility to make sure these tokens are sane
      // in the given context!
      if (Parser.getTok().is(AsmToken::RBrac)) {
        SMLoc Loc = Parser.getTok().getLoc();
        Operands.push_back(AArch64Operand::CreateToken("]", Loc));
        Parser.Lex();
      }

      if (Parser.getTok().is(AsmToken::Exclaim)) {
        SMLoc Loc = Parser.getTok().getLoc();
        Operands.push_back(AArch64Operand::CreateToken("!", Loc));
        Parser.Lex();
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "expected comma before next operand");
  }

  // Eat the EndOfStatement
  Parser.Lex();

  return false;
}

bool AArch64AsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".hword")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  else if (IDVal == ".word")
    return ParseDirectiveWord(4, DirectiveID.getLoc());
  else if (IDVal == ".xword")
    return ParseDirectiveWord(8, DirectiveID.getLoc());
  else if (IDVal == ".tlsdesccall")
    return ParseDirectiveTLSDescCall(DirectiveID.getLoc());

  return true;
}

/// parseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool AArch64AsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
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

// parseDirectiveTLSDescCall:
//   ::= .tlsdesccall symbol
bool AArch64AsmParser::ParseDirectiveTLSDescCall(SMLoc L) {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return Error(L, "expected symbol after directive");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);
  const MCSymbolRefExpr *Expr = MCSymbolRefExpr::Create(Sym, getContext());

  MCInst Inst;
  Inst.setOpcode(AArch64::TLSDESCCALL);
  Inst.addOperand(MCOperand::CreateExpr(Expr));

  getParser().getStreamer().EmitInstruction(Inst);
  return false;
}


bool AArch64AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                 SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                                 MCStreamer &Out, unsigned &ErrorInfo,
                                 bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult;
  MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                     MatchingInlineAsm);

  if (ErrorInfo != ~0U && ErrorInfo >= Operands.size())
    return Error(IDLoc, "too few operands for instruction");

  switch (MatchResult) {
  default: break;
  case Match_Success:
    if (validateInstruction(Inst, Operands))
      return true;

    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled");
    return true;
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      ErrorLoc = ((AArch64Operand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");

  case Match_AddSubRegExtendSmall:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
      "expected '[su]xt[bhw]' or 'lsl' with optional integer in range [0, 4]");
  case Match_AddSubRegExtendLarge:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
      "expected 'sxtx' 'uxtx' or 'lsl' with optional integer in range [0, 4]");
  case Match_AddSubRegShift32:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
       "expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 31]");
  case Match_AddSubRegShift64:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
       "expected 'lsl', 'lsr' or 'asr' with optional integer in range [0, 63]");
  case Match_AddSubSecondSource:
      return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
          "expected compatible register, symbol or integer in range [0, 4095]");
  case Match_CVTFixedPos32:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 32]");
  case Match_CVTFixedPos64:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 64]");
  case Match_CondCode:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected AArch64 condition code");
  case Match_FPImm:
    // Any situation which allows a nontrivial floating-point constant also
    // allows a register.
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected compatible register or floating-point constant");
  case Match_FPZero:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected floating-point constant #0.0 or invalid register type");
  case Match_Label:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected label or encodable integer pc offset");
  case Match_Lane1:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected lane specifier '[1]'");
  case Match_LoadStoreExtend32_1:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'uxtw' or 'sxtw' with optional shift of #0");
  case Match_LoadStoreExtend32_2:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'uxtw' or 'sxtw' with optional shift of #0 or #1");
  case Match_LoadStoreExtend32_4:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'uxtw' or 'sxtw' with optional shift of #0 or #2");
  case Match_LoadStoreExtend32_8:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'uxtw' or 'sxtw' with optional shift of #0 or #3");
  case Match_LoadStoreExtend32_16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtw' with optional shift of #0 or #4");
  case Match_LoadStoreExtend64_1:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtx' with optional shift of #0");
  case Match_LoadStoreExtend64_2:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtx' with optional shift of #0 or #1");
  case Match_LoadStoreExtend64_4:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtx' with optional shift of #0 or #2");
  case Match_LoadStoreExtend64_8:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtx' with optional shift of #0 or #3");
  case Match_LoadStoreExtend64_16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'lsl' or 'sxtx' with optional shift of #0 or #4");
  case Match_LoadStoreSImm7_4:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer multiple of 4 in range [-256, 252]");
  case Match_LoadStoreSImm7_8:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer multiple of 8 in range [-512, 508]");
  case Match_LoadStoreSImm7_16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer multiple of 16 in range [-1024, 1016]");
  case Match_LoadStoreSImm9:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [-256, 255]");
  case Match_LoadStoreUImm12_1:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic reference or integer in range [0, 4095]");
  case Match_LoadStoreUImm12_2:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic reference or integer in range [0, 8190]");
  case Match_LoadStoreUImm12_4:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic reference or integer in range [0, 16380]");
  case Match_LoadStoreUImm12_8:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic reference or integer in range [0, 32760]");
  case Match_LoadStoreUImm12_16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic reference or integer in range [0, 65520]");
  case Match_LogicalSecondSource:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected compatible register or logical immediate");
  case Match_MOVWUImm16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected relocated symbol or integer in range [0, 65535]");
  case Match_MRS:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected readable system register");
  case Match_MSR:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected writable system register or pstate");
  case Match_NamedImm_at:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                "expected symbolic 'at' operand: s1e[0-3][rw] or s12e[01][rw]");
  case Match_NamedImm_dbarrier:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
             "expected integer in range [0, 15] or symbolic barrier operand");
  case Match_NamedImm_dc:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected symbolic 'dc' operand");
  case Match_NamedImm_ic:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected 'ic' operand: 'ialluis', 'iallu' or 'ivau'");
  case Match_NamedImm_isb:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 15] or 'sy'");
  case Match_NamedImm_prefetch:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected prefetch hint: p(ld|st|i)l[123](strm|keep)");
  case Match_NamedImm_tlbi:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected translation buffer invalidation operand");
  case Match_UImm16:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 65535]");
  case Match_UImm3:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 7]");
  case Match_UImm4:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 15]");
  case Match_UImm5:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 31]");
  case Match_UImm6:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 63]");
  case Match_UImm7:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 127]");
  case Match_Width32:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [<lsb>, 31]");
  case Match_Width64:
    return Error(((AArch64Operand*)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [<lsb>, 63]");
  case Match_ShrImm8:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 8]");
  case Match_ShrImm16:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 16]");
  case Match_ShrImm32:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 32]");
  case Match_ShrImm64:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [1, 64]");
  case Match_ShlImm8:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 7]");
  case Match_ShlImm16:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 15]");
  case Match_ShlImm32:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 31]");
  case Match_ShlImm64:
    return Error(((AArch64Operand *)Operands[ErrorInfo])->getStartLoc(),
                 "expected integer in range [0, 63]");
  }

  llvm_unreachable("Implement any new match types added!");
  return true;
}

void AArch64Operand::print(raw_ostream &OS) const {
  switch (Kind) {
  case k_CondCode:
    OS << "<CondCode: " << CondCode.Code << ">";
    break;
  case k_FPImmediate:
    OS << "<fpimm: " << FPImm.Val << ">";
    break;
  case k_ImmWithLSL:
    OS << "<immwithlsl: imm=" << ImmWithLSL.Val
       << ", shift=" << ImmWithLSL.ShiftAmount << ">";
    break;
  case k_Immediate:
    getImm()->print(OS);
    break;
  case k_Register:
    OS << "<register " << getReg() << '>';
    break;
  case k_Token:
    OS << '\'' << getToken() << '\'';
    break;
  case k_ShiftExtend:
    OS << "<shift: type=" << ShiftExtend.ShiftType
       << ", amount=" << ShiftExtend.Amount << ">";
    break;
  case k_SysReg: {
    StringRef Name(SysReg.Data, SysReg.Length);
    OS << "<sysreg: " << Name << '>';
    break;
  }
  default:
    llvm_unreachable("No idea how to print this kind of operand");
    break;
  }
}

void AArch64Operand::dump() const {
  print(errs());
}


/// Force static initialization.
extern "C" void LLVMInitializeAArch64AsmParser() {
  RegisterMCAsmParser<AArch64AsmParser> X(TheAArch64Target);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "AArch64GenAsmMatcher.inc"
