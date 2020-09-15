//===- AMDGPUAsmParser.cpp - Parse SI asm to MCInst instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDKernelCodeT.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUTargetStreamer.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <string>

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::amdhsa;

namespace {

class AMDGPUAsmParser;

enum RegisterKind { IS_UNKNOWN, IS_VGPR, IS_SGPR, IS_AGPR, IS_TTMP, IS_SPECIAL };

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

class AMDGPUOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Register,
    Expression
  } Kind;

  SMLoc StartLoc, EndLoc;
  const AMDGPUAsmParser *AsmParser;

public:
  AMDGPUOperand(KindTy Kind_, const AMDGPUAsmParser *AsmParser_)
    : MCParsedAsmOperand(), Kind(Kind_), AsmParser(AsmParser_) {}

  using Ptr = std::unique_ptr<AMDGPUOperand>;

  struct Modifiers {
    bool Abs = false;
    bool Neg = false;
    bool Sext = false;

    bool hasFPModifiers() const { return Abs || Neg; }
    bool hasIntModifiers() const { return Sext; }
    bool hasModifiers() const { return hasFPModifiers() || hasIntModifiers(); }

    int64_t getFPModifiersOperand() const {
      int64_t Operand = 0;
      Operand |= Abs ? SISrcMods::ABS : 0u;
      Operand |= Neg ? SISrcMods::NEG : 0u;
      return Operand;
    }

    int64_t getIntModifiersOperand() const {
      int64_t Operand = 0;
      Operand |= Sext ? SISrcMods::SEXT : 0u;
      return Operand;
    }

    int64_t getModifiersOperand() const {
      assert(!(hasFPModifiers() && hasIntModifiers())
           && "fp and int modifiers should not be used simultaneously");
      if (hasFPModifiers()) {
        return getFPModifiersOperand();
      } else if (hasIntModifiers()) {
        return getIntModifiersOperand();
      } else {
        return 0;
      }
    }

    friend raw_ostream &operator <<(raw_ostream &OS, AMDGPUOperand::Modifiers Mods);
  };

  enum ImmTy {
    ImmTyNone,
    ImmTyGDS,
    ImmTyLDS,
    ImmTyOffen,
    ImmTyIdxen,
    ImmTyAddr64,
    ImmTyOffset,
    ImmTyInstOffset,
    ImmTyOffset0,
    ImmTyOffset1,
    ImmTyDLC,
    ImmTyGLC,
    ImmTySLC,
    ImmTySWZ,
    ImmTyTFE,
    ImmTyD16,
    ImmTyClampSI,
    ImmTyOModSI,
    ImmTyDPP8,
    ImmTyDppCtrl,
    ImmTyDppRowMask,
    ImmTyDppBankMask,
    ImmTyDppBoundCtrl,
    ImmTyDppFi,
    ImmTySdwaDstSel,
    ImmTySdwaSrc0Sel,
    ImmTySdwaSrc1Sel,
    ImmTySdwaDstUnused,
    ImmTyDMask,
    ImmTyDim,
    ImmTyUNorm,
    ImmTyDA,
    ImmTyR128A16,
    ImmTyA16,
    ImmTyLWE,
    ImmTyExpTgt,
    ImmTyExpCompr,
    ImmTyExpVM,
    ImmTyFORMAT,
    ImmTyHwreg,
    ImmTyOff,
    ImmTySendMsg,
    ImmTyInterpSlot,
    ImmTyInterpAttr,
    ImmTyAttrChan,
    ImmTyOpSel,
    ImmTyOpSelHi,
    ImmTyNegLo,
    ImmTyNegHi,
    ImmTySwizzle,
    ImmTyGprIdxMode,
    ImmTyHigh,
    ImmTyBLGP,
    ImmTyCBSZ,
    ImmTyABID,
    ImmTyEndpgm,
  };

private:
  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct ImmOp {
    int64_t Val;
    ImmTy Type;
    bool IsFPImm;
    Modifiers Mods;
  };

  struct RegOp {
    unsigned RegNo;
    Modifiers Mods;
  };

  union {
    TokOp Tok;
    ImmOp Imm;
    RegOp Reg;
    const MCExpr *Expr;
  };

public:
  bool isToken() const override {
    if (Kind == Token)
      return true;

    // When parsing operands, we can't always tell if something was meant to be
    // a token, like 'gds', or an expression that references a global variable.
    // In this case, we assume the string is an expression, and if we need to
    // interpret is a token, then we treat the symbol name as the token.
    return isSymbolRefExpr();
  }

  bool isSymbolRefExpr() const {
    return isExpr() && Expr && isa<MCSymbolRefExpr>(Expr);
  }

  bool isImm() const override {
    return Kind == Immediate;
  }

  bool isInlinableImm(MVT type) const;
  bool isLiteralImm(MVT type) const;

  bool isRegKind() const {
    return Kind == Register;
  }

  bool isReg() const override {
    return isRegKind() && !hasModifiers();
  }

  bool isRegOrImmWithInputMods(unsigned RCID, MVT type) const {
    return isRegClass(RCID) || isInlinableImm(type) || isLiteralImm(type);
  }

  bool isRegOrImmWithInt16InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_32RegClassID, MVT::i16);
  }

  bool isRegOrImmWithInt32InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_32RegClassID, MVT::i32);
  }

  bool isRegOrImmWithInt64InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_64RegClassID, MVT::i64);
  }

  bool isRegOrImmWithFP16InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_32RegClassID, MVT::f16);
  }

  bool isRegOrImmWithFP32InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_32RegClassID, MVT::f32);
  }

  bool isRegOrImmWithFP64InputMods() const {
    return isRegOrImmWithInputMods(AMDGPU::VS_64RegClassID, MVT::f64);
  }

  bool isVReg() const {
    return isRegClass(AMDGPU::VGPR_32RegClassID) ||
           isRegClass(AMDGPU::VReg_64RegClassID) ||
           isRegClass(AMDGPU::VReg_96RegClassID) ||
           isRegClass(AMDGPU::VReg_128RegClassID) ||
           isRegClass(AMDGPU::VReg_160RegClassID) ||
           isRegClass(AMDGPU::VReg_192RegClassID) ||
           isRegClass(AMDGPU::VReg_256RegClassID) ||
           isRegClass(AMDGPU::VReg_512RegClassID) ||
           isRegClass(AMDGPU::VReg_1024RegClassID);
  }

  bool isVReg32() const {
    return isRegClass(AMDGPU::VGPR_32RegClassID);
  }

  bool isVReg32OrOff() const {
    return isOff() || isVReg32();
  }

  bool isNull() const {
    return isRegKind() && getReg() == AMDGPU::SGPR_NULL;
  }

  bool isSDWAOperand(MVT type) const;
  bool isSDWAFP16Operand() const;
  bool isSDWAFP32Operand() const;
  bool isSDWAInt16Operand() const;
  bool isSDWAInt32Operand() const;

  bool isImmTy(ImmTy ImmT) const {
    return isImm() && Imm.Type == ImmT;
  }

  bool isImmModifier() const {
    return isImm() && Imm.Type != ImmTyNone;
  }

  bool isClampSI() const { return isImmTy(ImmTyClampSI); }
  bool isOModSI() const { return isImmTy(ImmTyOModSI); }
  bool isDMask() const { return isImmTy(ImmTyDMask); }
  bool isDim() const { return isImmTy(ImmTyDim); }
  bool isUNorm() const { return isImmTy(ImmTyUNorm); }
  bool isDA() const { return isImmTy(ImmTyDA); }
  bool isR128A16() const { return isImmTy(ImmTyR128A16); }
  bool isGFX10A16() const { return isImmTy(ImmTyA16); }
  bool isLWE() const { return isImmTy(ImmTyLWE); }
  bool isOff() const { return isImmTy(ImmTyOff); }
  bool isExpTgt() const { return isImmTy(ImmTyExpTgt); }
  bool isExpVM() const { return isImmTy(ImmTyExpVM); }
  bool isExpCompr() const { return isImmTy(ImmTyExpCompr); }
  bool isOffen() const { return isImmTy(ImmTyOffen); }
  bool isIdxen() const { return isImmTy(ImmTyIdxen); }
  bool isAddr64() const { return isImmTy(ImmTyAddr64); }
  bool isOffset() const { return isImmTy(ImmTyOffset) && isUInt<16>(getImm()); }
  bool isOffset0() const { return isImmTy(ImmTyOffset0) && isUInt<8>(getImm()); }
  bool isOffset1() const { return isImmTy(ImmTyOffset1) && isUInt<8>(getImm()); }

  bool isFlatOffset() const { return isImmTy(ImmTyOffset) || isImmTy(ImmTyInstOffset); }
  bool isGDS() const { return isImmTy(ImmTyGDS); }
  bool isLDS() const { return isImmTy(ImmTyLDS); }
  bool isDLC() const { return isImmTy(ImmTyDLC); }
  bool isGLC() const { return isImmTy(ImmTyGLC); }
  bool isSLC() const { return isImmTy(ImmTySLC); }
  bool isSWZ() const { return isImmTy(ImmTySWZ); }
  bool isTFE() const { return isImmTy(ImmTyTFE); }
  bool isD16() const { return isImmTy(ImmTyD16); }
  bool isFORMAT() const { return isImmTy(ImmTyFORMAT) && isUInt<7>(getImm()); }
  bool isBankMask() const { return isImmTy(ImmTyDppBankMask); }
  bool isRowMask() const { return isImmTy(ImmTyDppRowMask); }
  bool isBoundCtrl() const { return isImmTy(ImmTyDppBoundCtrl); }
  bool isFI() const { return isImmTy(ImmTyDppFi); }
  bool isSDWADstSel() const { return isImmTy(ImmTySdwaDstSel); }
  bool isSDWASrc0Sel() const { return isImmTy(ImmTySdwaSrc0Sel); }
  bool isSDWASrc1Sel() const { return isImmTy(ImmTySdwaSrc1Sel); }
  bool isSDWADstUnused() const { return isImmTy(ImmTySdwaDstUnused); }
  bool isInterpSlot() const { return isImmTy(ImmTyInterpSlot); }
  bool isInterpAttr() const { return isImmTy(ImmTyInterpAttr); }
  bool isAttrChan() const { return isImmTy(ImmTyAttrChan); }
  bool isOpSel() const { return isImmTy(ImmTyOpSel); }
  bool isOpSelHi() const { return isImmTy(ImmTyOpSelHi); }
  bool isNegLo() const { return isImmTy(ImmTyNegLo); }
  bool isNegHi() const { return isImmTy(ImmTyNegHi); }
  bool isHigh() const { return isImmTy(ImmTyHigh); }

  bool isMod() const {
    return isClampSI() || isOModSI();
  }

  bool isRegOrImm() const {
    return isReg() || isImm();
  }

  bool isRegClass(unsigned RCID) const;

  bool isInlineValue() const;

  bool isRegOrInlineNoMods(unsigned RCID, MVT type) const {
    return (isRegClass(RCID) || isInlinableImm(type)) && !hasModifiers();
  }

  bool isSCSrcB16() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_32RegClassID, MVT::i16);
  }

  bool isSCSrcV2B16() const {
    return isSCSrcB16();
  }

  bool isSCSrcB32() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_32RegClassID, MVT::i32);
  }

  bool isSCSrcB64() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_64RegClassID, MVT::i64);
  }

  bool isBoolReg() const;

  bool isSCSrcF16() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_32RegClassID, MVT::f16);
  }

  bool isSCSrcV2F16() const {
    return isSCSrcF16();
  }

  bool isSCSrcF32() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_32RegClassID, MVT::f32);
  }

  bool isSCSrcF64() const {
    return isRegOrInlineNoMods(AMDGPU::SReg_64RegClassID, MVT::f64);
  }

  bool isSSrcB32() const {
    return isSCSrcB32() || isLiteralImm(MVT::i32) || isExpr();
  }

  bool isSSrcB16() const {
    return isSCSrcB16() || isLiteralImm(MVT::i16);
  }

  bool isSSrcV2B16() const {
    llvm_unreachable("cannot happen");
    return isSSrcB16();
  }

  bool isSSrcB64() const {
    // TODO: Find out how SALU supports extension of 32-bit literals to 64 bits.
    // See isVSrc64().
    return isSCSrcB64() || isLiteralImm(MVT::i64);
  }

  bool isSSrcF32() const {
    return isSCSrcB32() || isLiteralImm(MVT::f32) || isExpr();
  }

  bool isSSrcF64() const {
    return isSCSrcB64() || isLiteralImm(MVT::f64);
  }

  bool isSSrcF16() const {
    return isSCSrcB16() || isLiteralImm(MVT::f16);
  }

  bool isSSrcV2F16() const {
    llvm_unreachable("cannot happen");
    return isSSrcF16();
  }

  bool isSSrcOrLdsB32() const {
    return isRegOrInlineNoMods(AMDGPU::SRegOrLds_32RegClassID, MVT::i32) ||
           isLiteralImm(MVT::i32) || isExpr();
  }

  bool isVCSrcB32() const {
    return isRegOrInlineNoMods(AMDGPU::VS_32RegClassID, MVT::i32);
  }

  bool isVCSrcB64() const {
    return isRegOrInlineNoMods(AMDGPU::VS_64RegClassID, MVT::i64);
  }

  bool isVCSrcB16() const {
    return isRegOrInlineNoMods(AMDGPU::VS_32RegClassID, MVT::i16);
  }

  bool isVCSrcV2B16() const {
    return isVCSrcB16();
  }

  bool isVCSrcF32() const {
    return isRegOrInlineNoMods(AMDGPU::VS_32RegClassID, MVT::f32);
  }

  bool isVCSrcF64() const {
    return isRegOrInlineNoMods(AMDGPU::VS_64RegClassID, MVT::f64);
  }

  bool isVCSrcF16() const {
    return isRegOrInlineNoMods(AMDGPU::VS_32RegClassID, MVT::f16);
  }

  bool isVCSrcV2F16() const {
    return isVCSrcF16();
  }

  bool isVSrcB32() const {
    return isVCSrcF32() || isLiteralImm(MVT::i32) || isExpr();
  }

  bool isVSrcB64() const {
    return isVCSrcF64() || isLiteralImm(MVT::i64);
  }

  bool isVSrcB16() const {
    return isVCSrcB16() || isLiteralImm(MVT::i16);
  }

  bool isVSrcV2B16() const {
    return isVSrcB16() || isLiteralImm(MVT::v2i16);
  }

  bool isVSrcF32() const {
    return isVCSrcF32() || isLiteralImm(MVT::f32) || isExpr();
  }

  bool isVSrcF64() const {
    return isVCSrcF64() || isLiteralImm(MVT::f64);
  }

  bool isVSrcF16() const {
    return isVCSrcF16() || isLiteralImm(MVT::f16);
  }

  bool isVSrcV2F16() const {
    return isVSrcF16() || isLiteralImm(MVT::v2f16);
  }

  bool isVISrcB32() const {
    return isRegOrInlineNoMods(AMDGPU::VGPR_32RegClassID, MVT::i32);
  }

  bool isVISrcB16() const {
    return isRegOrInlineNoMods(AMDGPU::VGPR_32RegClassID, MVT::i16);
  }

  bool isVISrcV2B16() const {
    return isVISrcB16();
  }

  bool isVISrcF32() const {
    return isRegOrInlineNoMods(AMDGPU::VGPR_32RegClassID, MVT::f32);
  }

  bool isVISrcF16() const {
    return isRegOrInlineNoMods(AMDGPU::VGPR_32RegClassID, MVT::f16);
  }

  bool isVISrcV2F16() const {
    return isVISrcF16() || isVISrcB32();
  }

  bool isAISrcB32() const {
    return isRegOrInlineNoMods(AMDGPU::AGPR_32RegClassID, MVT::i32);
  }

  bool isAISrcB16() const {
    return isRegOrInlineNoMods(AMDGPU::AGPR_32RegClassID, MVT::i16);
  }

  bool isAISrcV2B16() const {
    return isAISrcB16();
  }

  bool isAISrcF32() const {
    return isRegOrInlineNoMods(AMDGPU::AGPR_32RegClassID, MVT::f32);
  }

  bool isAISrcF16() const {
    return isRegOrInlineNoMods(AMDGPU::AGPR_32RegClassID, MVT::f16);
  }

  bool isAISrcV2F16() const {
    return isAISrcF16() || isAISrcB32();
  }

  bool isAISrc_128B32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_128RegClassID, MVT::i32);
  }

  bool isAISrc_128B16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_128RegClassID, MVT::i16);
  }

  bool isAISrc_128V2B16() const {
    return isAISrc_128B16();
  }

  bool isAISrc_128F32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_128RegClassID, MVT::f32);
  }

  bool isAISrc_128F16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_128RegClassID, MVT::f16);
  }

  bool isAISrc_128V2F16() const {
    return isAISrc_128F16() || isAISrc_128B32();
  }

  bool isAISrc_512B32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_512RegClassID, MVT::i32);
  }

  bool isAISrc_512B16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_512RegClassID, MVT::i16);
  }

  bool isAISrc_512V2B16() const {
    return isAISrc_512B16();
  }

  bool isAISrc_512F32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_512RegClassID, MVT::f32);
  }

  bool isAISrc_512F16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_512RegClassID, MVT::f16);
  }

  bool isAISrc_512V2F16() const {
    return isAISrc_512F16() || isAISrc_512B32();
  }

  bool isAISrc_1024B32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_1024RegClassID, MVT::i32);
  }

  bool isAISrc_1024B16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_1024RegClassID, MVT::i16);
  }

  bool isAISrc_1024V2B16() const {
    return isAISrc_1024B16();
  }

  bool isAISrc_1024F32() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_1024RegClassID, MVT::f32);
  }

  bool isAISrc_1024F16() const {
    return isRegOrInlineNoMods(AMDGPU::AReg_1024RegClassID, MVT::f16);
  }

  bool isAISrc_1024V2F16() const {
    return isAISrc_1024F16() || isAISrc_1024B32();
  }

  bool isKImmFP32() const {
    return isLiteralImm(MVT::f32);
  }

  bool isKImmFP16() const {
    return isLiteralImm(MVT::f16);
  }

  bool isMem() const override {
    return false;
  }

  bool isExpr() const {
    return Kind == Expression;
  }

  bool isSoppBrTarget() const {
    return isExpr() || isImm();
  }

  bool isSWaitCnt() const;
  bool isHwreg() const;
  bool isSendMsg() const;
  bool isSwizzle() const;
  bool isSMRDOffset8() const;
  bool isSMEMOffset() const;
  bool isSMRDLiteralOffset() const;
  bool isDPP8() const;
  bool isDPPCtrl() const;
  bool isBLGP() const;
  bool isCBSZ() const;
  bool isABID() const;
  bool isGPRIdxMode() const;
  bool isS16Imm() const;
  bool isU16Imm() const;
  bool isEndpgm() const;

  StringRef getExpressionAsToken() const {
    assert(isExpr());
    const MCSymbolRefExpr *S = cast<MCSymbolRefExpr>(Expr);
    return S->getSymbol().getName();
  }

  StringRef getToken() const {
    assert(isToken());

    if (Kind == Expression)
      return getExpressionAsToken();

    return StringRef(Tok.Data, Tok.Length);
  }

  int64_t getImm() const {
    assert(isImm());
    return Imm.Val;
  }

  void setImm(int64_t Val) {
    assert(isImm());
    Imm.Val = Val;
  }

  ImmTy getImmTy() const {
    assert(isImm());
    return Imm.Type;
  }

  unsigned getReg() const override {
    assert(isRegKind());
    return Reg.RegNo;
  }

  SMLoc getStartLoc() const override {
    return StartLoc;
  }

  SMLoc getEndLoc() const override {
    return EndLoc;
  }

  SMRange getLocRange() const {
    return SMRange(StartLoc, EndLoc);
  }

  Modifiers getModifiers() const {
    assert(isRegKind() || isImmTy(ImmTyNone));
    return isRegKind() ? Reg.Mods : Imm.Mods;
  }

  void setModifiers(Modifiers Mods) {
    assert(isRegKind() || isImmTy(ImmTyNone));
    if (isRegKind())
      Reg.Mods = Mods;
    else
      Imm.Mods = Mods;
  }

  bool hasModifiers() const {
    return getModifiers().hasModifiers();
  }

  bool hasFPModifiers() const {
    return getModifiers().hasFPModifiers();
  }

  bool hasIntModifiers() const {
    return getModifiers().hasIntModifiers();
  }

  uint64_t applyInputFPModifiers(uint64_t Val, unsigned Size) const;

  void addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers = true) const;

  void addLiteralImmOperand(MCInst &Inst, int64_t Val, bool ApplyModifiers) const;

  template <unsigned Bitwidth>
  void addKImmFPOperands(MCInst &Inst, unsigned N) const;

  void addKImmFP16Operands(MCInst &Inst, unsigned N) const {
    addKImmFPOperands<16>(Inst, N);
  }

  void addKImmFP32Operands(MCInst &Inst, unsigned N) const {
    addKImmFPOperands<32>(Inst, N);
  }

  void addRegOperands(MCInst &Inst, unsigned N) const;

  void addBoolRegOperands(MCInst &Inst, unsigned N) const {
    addRegOperands(Inst, N);
  }

  void addRegOrImmOperands(MCInst &Inst, unsigned N) const {
    if (isRegKind())
      addRegOperands(Inst, N);
    else if (isExpr())
      Inst.addOperand(MCOperand::createExpr(Expr));
    else
      addImmOperands(Inst, N);
  }

  void addRegOrImmWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Modifiers Mods = getModifiers();
    Inst.addOperand(MCOperand::createImm(Mods.getModifiersOperand()));
    if (isRegKind()) {
      addRegOperands(Inst, N);
    } else {
      addImmOperands(Inst, N, false);
    }
  }

  void addRegOrImmWithFPInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasIntModifiers());
    addRegOrImmWithInputModsOperands(Inst, N);
  }

  void addRegOrImmWithIntInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasFPModifiers());
    addRegOrImmWithInputModsOperands(Inst, N);
  }

  void addRegWithInputModsOperands(MCInst &Inst, unsigned N) const {
    Modifiers Mods = getModifiers();
    Inst.addOperand(MCOperand::createImm(Mods.getModifiersOperand()));
    assert(isRegKind());
    addRegOperands(Inst, N);
  }

  void addRegWithFPInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasIntModifiers());
    addRegWithInputModsOperands(Inst, N);
  }

  void addRegWithIntInputModsOperands(MCInst &Inst, unsigned N) const {
    assert(!hasFPModifiers());
    addRegWithInputModsOperands(Inst, N);
  }

  void addSoppBrTargetOperands(MCInst &Inst, unsigned N) const {
    if (isImm())
      addImmOperands(Inst, N);
    else {
      assert(isExpr());
      Inst.addOperand(MCOperand::createExpr(Expr));
    }
  }

  static void printImmTy(raw_ostream& OS, ImmTy Type) {
    switch (Type) {
    case ImmTyNone: OS << "None"; break;
    case ImmTyGDS: OS << "GDS"; break;
    case ImmTyLDS: OS << "LDS"; break;
    case ImmTyOffen: OS << "Offen"; break;
    case ImmTyIdxen: OS << "Idxen"; break;
    case ImmTyAddr64: OS << "Addr64"; break;
    case ImmTyOffset: OS << "Offset"; break;
    case ImmTyInstOffset: OS << "InstOffset"; break;
    case ImmTyOffset0: OS << "Offset0"; break;
    case ImmTyOffset1: OS << "Offset1"; break;
    case ImmTyDLC: OS << "DLC"; break;
    case ImmTyGLC: OS << "GLC"; break;
    case ImmTySLC: OS << "SLC"; break;
    case ImmTySWZ: OS << "SWZ"; break;
    case ImmTyTFE: OS << "TFE"; break;
    case ImmTyD16: OS << "D16"; break;
    case ImmTyFORMAT: OS << "FORMAT"; break;
    case ImmTyClampSI: OS << "ClampSI"; break;
    case ImmTyOModSI: OS << "OModSI"; break;
    case ImmTyDPP8: OS << "DPP8"; break;
    case ImmTyDppCtrl: OS << "DppCtrl"; break;
    case ImmTyDppRowMask: OS << "DppRowMask"; break;
    case ImmTyDppBankMask: OS << "DppBankMask"; break;
    case ImmTyDppBoundCtrl: OS << "DppBoundCtrl"; break;
    case ImmTyDppFi: OS << "FI"; break;
    case ImmTySdwaDstSel: OS << "SdwaDstSel"; break;
    case ImmTySdwaSrc0Sel: OS << "SdwaSrc0Sel"; break;
    case ImmTySdwaSrc1Sel: OS << "SdwaSrc1Sel"; break;
    case ImmTySdwaDstUnused: OS << "SdwaDstUnused"; break;
    case ImmTyDMask: OS << "DMask"; break;
    case ImmTyDim: OS << "Dim"; break;
    case ImmTyUNorm: OS << "UNorm"; break;
    case ImmTyDA: OS << "DA"; break;
    case ImmTyR128A16: OS << "R128A16"; break;
    case ImmTyA16: OS << "A16"; break;
    case ImmTyLWE: OS << "LWE"; break;
    case ImmTyOff: OS << "Off"; break;
    case ImmTyExpTgt: OS << "ExpTgt"; break;
    case ImmTyExpCompr: OS << "ExpCompr"; break;
    case ImmTyExpVM: OS << "ExpVM"; break;
    case ImmTyHwreg: OS << "Hwreg"; break;
    case ImmTySendMsg: OS << "SendMsg"; break;
    case ImmTyInterpSlot: OS << "InterpSlot"; break;
    case ImmTyInterpAttr: OS << "InterpAttr"; break;
    case ImmTyAttrChan: OS << "AttrChan"; break;
    case ImmTyOpSel: OS << "OpSel"; break;
    case ImmTyOpSelHi: OS << "OpSelHi"; break;
    case ImmTyNegLo: OS << "NegLo"; break;
    case ImmTyNegHi: OS << "NegHi"; break;
    case ImmTySwizzle: OS << "Swizzle"; break;
    case ImmTyGprIdxMode: OS << "GprIdxMode"; break;
    case ImmTyHigh: OS << "High"; break;
    case ImmTyBLGP: OS << "BLGP"; break;
    case ImmTyCBSZ: OS << "CBSZ"; break;
    case ImmTyABID: OS << "ABID"; break;
    case ImmTyEndpgm: OS << "Endpgm"; break;
    }
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case Register:
      OS << "<register " << getReg() << " mods: " << Reg.Mods << '>';
      break;
    case Immediate:
      OS << '<' << getImm();
      if (getImmTy() != ImmTyNone) {
        OS << " type: "; printImmTy(OS, getImmTy());
      }
      OS << " mods: " << Imm.Mods << '>';
      break;
    case Token:
      OS << '\'' << getToken() << '\'';
      break;
    case Expression:
      OS << "<expr " << *Expr << '>';
      break;
    }
  }

  static AMDGPUOperand::Ptr CreateImm(const AMDGPUAsmParser *AsmParser,
                                      int64_t Val, SMLoc Loc,
                                      ImmTy Type = ImmTyNone,
                                      bool IsFPImm = false) {
    auto Op = std::make_unique<AMDGPUOperand>(Immediate, AsmParser);
    Op->Imm.Val = Val;
    Op->Imm.IsFPImm = IsFPImm;
    Op->Imm.Type = Type;
    Op->Imm.Mods = Modifiers();
    Op->StartLoc = Loc;
    Op->EndLoc = Loc;
    return Op;
  }

  static AMDGPUOperand::Ptr CreateToken(const AMDGPUAsmParser *AsmParser,
                                        StringRef Str, SMLoc Loc,
                                        bool HasExplicitEncodingSize = true) {
    auto Res = std::make_unique<AMDGPUOperand>(Token, AsmParser);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    Res->StartLoc = Loc;
    Res->EndLoc = Loc;
    return Res;
  }

  static AMDGPUOperand::Ptr CreateReg(const AMDGPUAsmParser *AsmParser,
                                      unsigned RegNo, SMLoc S,
                                      SMLoc E) {
    auto Op = std::make_unique<AMDGPUOperand>(Register, AsmParser);
    Op->Reg.RegNo = RegNo;
    Op->Reg.Mods = Modifiers();
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static AMDGPUOperand::Ptr CreateExpr(const AMDGPUAsmParser *AsmParser,
                                       const class MCExpr *Expr, SMLoc S) {
    auto Op = std::make_unique<AMDGPUOperand>(Expression, AsmParser);
    Op->Expr = Expr;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }
};

raw_ostream &operator <<(raw_ostream &OS, AMDGPUOperand::Modifiers Mods) {
  OS << "abs:" << Mods.Abs << " neg: " << Mods.Neg << " sext:" << Mods.Sext;
  return OS;
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

// Holds info related to the current kernel, e.g. count of SGPRs used.
// Kernel scope begins at .amdgpu_hsa_kernel directive, ends at next
// .amdgpu_hsa_kernel or at EOF.
class KernelScopeInfo {
  int SgprIndexUnusedMin = -1;
  int VgprIndexUnusedMin = -1;
  MCContext *Ctx = nullptr;

  void usesSgprAt(int i) {
    if (i >= SgprIndexUnusedMin) {
      SgprIndexUnusedMin = ++i;
      if (Ctx) {
        MCSymbol * const Sym = Ctx->getOrCreateSymbol(Twine(".kernel.sgpr_count"));
        Sym->setVariableValue(MCConstantExpr::create(SgprIndexUnusedMin, *Ctx));
      }
    }
  }

  void usesVgprAt(int i) {
    if (i >= VgprIndexUnusedMin) {
      VgprIndexUnusedMin = ++i;
      if (Ctx) {
        MCSymbol * const Sym = Ctx->getOrCreateSymbol(Twine(".kernel.vgpr_count"));
        Sym->setVariableValue(MCConstantExpr::create(VgprIndexUnusedMin, *Ctx));
      }
    }
  }

public:
  KernelScopeInfo() = default;

  void initialize(MCContext &Context) {
    Ctx = &Context;
    usesSgprAt(SgprIndexUnusedMin = -1);
    usesVgprAt(VgprIndexUnusedMin = -1);
  }

  void usesRegister(RegisterKind RegKind, unsigned DwordRegIndex, unsigned RegWidth) {
    switch (RegKind) {
      case IS_SGPR: usesSgprAt(DwordRegIndex + RegWidth - 1); break;
      case IS_AGPR: // fall through
      case IS_VGPR: usesVgprAt(DwordRegIndex + RegWidth - 1); break;
      default: break;
    }
  }
};

class AMDGPUAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  // Number of extra operands parsed after the first optional operand.
  // This may be necessary to skip hardcoded mandatory operands.
  static const unsigned MAX_OPR_LOOKAHEAD = 8;

  unsigned ForcedEncodingSize = 0;
  bool ForcedDPP = false;
  bool ForcedSDWA = false;
  KernelScopeInfo KernelScope;

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "AMDGPUGenAsmMatcher.inc"

  /// }

private:
  bool ParseAsAbsoluteExpression(uint32_t &Ret);
  bool OutOfRangeError(SMRange Range);
  /// Calculate VGPR/SGPR blocks required for given target, reserved
  /// registers, and user-specified NextFreeXGPR values.
  ///
  /// \param Features [in] Target features, used for bug corrections.
  /// \param VCCUsed [in] Whether VCC special SGPR is reserved.
  /// \param FlatScrUsed [in] Whether FLAT_SCRATCH special SGPR is reserved.
  /// \param XNACKUsed [in] Whether XNACK_MASK special SGPR is reserved.
  /// \param EnableWavefrontSize32 [in] Value of ENABLE_WAVEFRONT_SIZE32 kernel
  /// descriptor field, if valid.
  /// \param NextFreeVGPR [in] Max VGPR number referenced, plus one.
  /// \param VGPRRange [in] Token range, used for VGPR diagnostics.
  /// \param NextFreeSGPR [in] Max SGPR number referenced, plus one.
  /// \param SGPRRange [in] Token range, used for SGPR diagnostics.
  /// \param VGPRBlocks [out] Result VGPR block count.
  /// \param SGPRBlocks [out] Result SGPR block count.
  bool calculateGPRBlocks(const FeatureBitset &Features, bool VCCUsed,
                          bool FlatScrUsed, bool XNACKUsed,
                          Optional<bool> EnableWavefrontSize32, unsigned NextFreeVGPR,
                          SMRange VGPRRange, unsigned NextFreeSGPR,
                          SMRange SGPRRange, unsigned &VGPRBlocks,
                          unsigned &SGPRBlocks);
  bool ParseDirectiveAMDGCNTarget();
  bool ParseDirectiveAMDHSAKernel();
  bool ParseDirectiveMajorMinor(uint32_t &Major, uint32_t &Minor);
  bool ParseDirectiveHSACodeObjectVersion();
  bool ParseDirectiveHSACodeObjectISA();
  bool ParseAMDKernelCodeTValue(StringRef ID, amd_kernel_code_t &Header);
  bool ParseDirectiveAMDKernelCodeT();
  bool subtargetHasRegister(const MCRegisterInfo &MRI, unsigned RegNo) const;
  bool ParseDirectiveAMDGPUHsaKernel();

  bool ParseDirectiveISAVersion();
  bool ParseDirectiveHSAMetadata();
  bool ParseDirectivePALMetadataBegin();
  bool ParseDirectivePALMetadata();
  bool ParseDirectiveAMDGPULDS();

  /// Common code to parse out a block of text (typically YAML) between start and
  /// end directives.
  bool ParseToEndDirective(const char *AssemblerDirectiveBegin,
                           const char *AssemblerDirectiveEnd,
                           std::string &CollectString);

  bool AddNextRegisterToList(unsigned& Reg, unsigned& RegWidth,
                             RegisterKind RegKind, unsigned Reg1, SMLoc Loc);
  bool ParseAMDGPURegister(RegisterKind &RegKind, unsigned &Reg,
                           unsigned &RegNum, unsigned &RegWidth,
                           bool RestoreOnFailure = false);
  bool ParseAMDGPURegister(RegisterKind &RegKind, unsigned &Reg,
                           unsigned &RegNum, unsigned &RegWidth,
                           SmallVectorImpl<AsmToken> &Tokens);
  unsigned ParseRegularReg(RegisterKind &RegKind, unsigned &RegNum,
                           unsigned &RegWidth,
                           SmallVectorImpl<AsmToken> &Tokens);
  unsigned ParseSpecialReg(RegisterKind &RegKind, unsigned &RegNum,
                           unsigned &RegWidth,
                           SmallVectorImpl<AsmToken> &Tokens);
  unsigned ParseRegList(RegisterKind &RegKind, unsigned &RegNum,
                        unsigned &RegWidth, SmallVectorImpl<AsmToken> &Tokens);
  bool ParseRegRange(unsigned& Num, unsigned& Width);
  unsigned getRegularReg(RegisterKind RegKind,
                         unsigned RegNum,
                         unsigned RegWidth,
                         SMLoc Loc);

  bool isRegister();
  bool isRegister(const AsmToken &Token, const AsmToken &NextToken) const;
  Optional<StringRef> getGprCountSymbolName(RegisterKind RegKind);
  void initializeGprCountSymbol(RegisterKind RegKind);
  bool updateGprCountSymbols(RegisterKind RegKind, unsigned DwordRegIndex,
                             unsigned RegWidth);
  void cvtMubufImpl(MCInst &Inst, const OperandVector &Operands,
                    bool IsAtomic, bool IsAtomicReturn, bool IsLds = false);
  void cvtDSImpl(MCInst &Inst, const OperandVector &Operands,
                 bool IsGdsHardcoded);

public:
  enum AMDGPUMatchResultTy {
    Match_PreferE32 = FIRST_TARGET_MATCH_RESULT_TY
  };
  enum OperandMode {
    OperandMode_Default,
    OperandMode_NSA,
  };

  using OptionalImmIndexMap = std::map<AMDGPUOperand::ImmTy, unsigned>;

  AMDGPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &_Parser,
               const MCInstrInfo &MII,
               const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), Parser(_Parser) {
    MCAsmParserExtension::Initialize(Parser);

    if (getFeatureBits().none()) {
      // Set default features.
      copySTI().ToggleFeature("southern-islands");
    }

    setAvailableFeatures(ComputeAvailableFeatures(getFeatureBits()));

    {
      // TODO: make those pre-defined variables read-only.
      // Currently there is none suitable machinery in the core llvm-mc for this.
      // MCSymbol::isRedefinable is intended for another purpose, and
      // AsmParser::parseDirectiveSet() cannot be specialized for specific target.
      AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(getSTI().getCPU());
      MCContext &Ctx = getContext();
      if (ISA.Major >= 6 && AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
        MCSymbol *Sym =
            Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_number"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Major, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_minor"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Minor, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".amdgcn.gfx_generation_stepping"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Stepping, Ctx));
      } else {
        MCSymbol *Sym =
            Ctx.getOrCreateSymbol(Twine(".option.machine_version_major"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Major, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".option.machine_version_minor"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Minor, Ctx));
        Sym = Ctx.getOrCreateSymbol(Twine(".option.machine_version_stepping"));
        Sym->setVariableValue(MCConstantExpr::create(ISA.Stepping, Ctx));
      }
      if (ISA.Major >= 6 && AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
        initializeGprCountSymbol(IS_VGPR);
        initializeGprCountSymbol(IS_SGPR);
      } else
        KernelScope.initialize(getContext());
    }
  }

  bool hasXNACK() const {
    return AMDGPU::hasXNACK(getSTI());
  }

  bool hasMIMG_R128() const {
    return AMDGPU::hasMIMG_R128(getSTI());
  }

  bool hasPackedD16() const {
    return AMDGPU::hasPackedD16(getSTI());
  }

  bool hasGFX10A16() const {
    return AMDGPU::hasGFX10A16(getSTI());
  }

  bool isSI() const {
    return AMDGPU::isSI(getSTI());
  }

  bool isCI() const {
    return AMDGPU::isCI(getSTI());
  }

  bool isVI() const {
    return AMDGPU::isVI(getSTI());
  }

  bool isGFX9() const {
    return AMDGPU::isGFX9(getSTI());
  }

  bool isGFX10() const {
    return AMDGPU::isGFX10(getSTI());
  }

  bool isGFX10_BEncoding() const {
    return AMDGPU::isGFX10_BEncoding(getSTI());
  }

  bool hasInv2PiInlineImm() const {
    return getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm];
  }

  bool hasFlatOffsets() const {
    return getFeatureBits()[AMDGPU::FeatureFlatInstOffsets];
  }

  bool hasSGPR102_SGPR103() const {
    return !isVI() && !isGFX9();
  }

  bool hasSGPR104_SGPR105() const {
    return isGFX10();
  }

  bool hasIntClamp() const {
    return getFeatureBits()[AMDGPU::FeatureIntClamp];
  }

  AMDGPUTargetStreamer &getTargetStreamer() {
    MCTargetStreamer &TS = *getParser().getStreamer().getTargetStreamer();
    return static_cast<AMDGPUTargetStreamer &>(TS);
  }

  const MCRegisterInfo *getMRI() const {
    // We need this const_cast because for some reason getContext() is not const
    // in MCAsmParser.
    return const_cast<AMDGPUAsmParser*>(this)->getContext().getRegisterInfo();
  }

  const MCInstrInfo *getMII() const {
    return &MII;
  }

  const FeatureBitset &getFeatureBits() const {
    return getSTI().getFeatureBits();
  }

  void setForcedEncodingSize(unsigned Size) { ForcedEncodingSize = Size; }
  void setForcedDPP(bool ForceDPP_) { ForcedDPP = ForceDPP_; }
  void setForcedSDWA(bool ForceSDWA_) { ForcedSDWA = ForceSDWA_; }

  unsigned getForcedEncodingSize() const { return ForcedEncodingSize; }
  bool isForcedVOP3() const { return ForcedEncodingSize == 64; }
  bool isForcedDPP() const { return ForcedDPP; }
  bool isForcedSDWA() const { return ForcedSDWA; }
  ArrayRef<unsigned> getMatchedVariants() const;

  std::unique_ptr<AMDGPUOperand> parseRegister(bool RestoreOnFailure = false);
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc,
                     bool RestoreOnFailure);
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  OperandMatchResultTy tryParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                        SMLoc &EndLoc) override;
  unsigned checkTargetMatchPredicate(MCInst &Inst) override;
  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseDirective(AsmToken DirectiveID) override;
  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Mnemonic,
                                    OperandMode Mode = OperandMode_Default);
  StringRef parseMnemonicSuffix(StringRef Name);
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  //bool ProcessInstruction(MCInst &Inst);

  OperandMatchResultTy parseIntWithPrefix(const char *Prefix, int64_t &Int);

  OperandMatchResultTy
  parseIntWithPrefix(const char *Prefix, OperandVector &Operands,
                     AMDGPUOperand::ImmTy ImmTy = AMDGPUOperand::ImmTyNone,
                     bool (*ConvertResult)(int64_t &) = nullptr);

  OperandMatchResultTy
  parseOperandArrayWithPrefix(const char *Prefix,
                              OperandVector &Operands,
                              AMDGPUOperand::ImmTy ImmTy = AMDGPUOperand::ImmTyNone,
                              bool (*ConvertResult)(int64_t&) = nullptr);

  OperandMatchResultTy
  parseNamedBit(const char *Name, OperandVector &Operands,
                AMDGPUOperand::ImmTy ImmTy = AMDGPUOperand::ImmTyNone);
  OperandMatchResultTy parseStringWithPrefix(StringRef Prefix,
                                             StringRef &Value);

  bool isModifier();
  bool isOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isRegOrOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isNamedOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const;
  bool isOpcodeModifierWithVal(const AsmToken &Token, const AsmToken &NextToken) const;
  bool parseSP3NegModifier();
  OperandMatchResultTy parseImm(OperandVector &Operands, bool HasSP3AbsModifier = false);
  OperandMatchResultTy parseReg(OperandVector &Operands);
  OperandMatchResultTy parseRegOrImm(OperandVector &Operands, bool HasSP3AbsMod = false);
  OperandMatchResultTy parseRegOrImmWithFPInputMods(OperandVector &Operands, bool AllowImm = true);
  OperandMatchResultTy parseRegOrImmWithIntInputMods(OperandVector &Operands, bool AllowImm = true);
  OperandMatchResultTy parseRegWithFPInputMods(OperandVector &Operands);
  OperandMatchResultTy parseRegWithIntInputMods(OperandVector &Operands);
  OperandMatchResultTy parseVReg32OrOff(OperandVector &Operands);
  OperandMatchResultTy parseDfmtNfmt(int64_t &Format);
  OperandMatchResultTy parseUfmt(int64_t &Format);
  OperandMatchResultTy parseSymbolicSplitFormat(StringRef FormatStr, SMLoc Loc, int64_t &Format);
  OperandMatchResultTy parseSymbolicUnifiedFormat(StringRef FormatStr, SMLoc Loc, int64_t &Format);
  OperandMatchResultTy parseFORMAT(OperandVector &Operands);
  OperandMatchResultTy parseSymbolicOrNumericFormat(int64_t &Format);
  OperandMatchResultTy parseNumericFormat(int64_t &Format);
  bool tryParseFmt(const char *Pref, int64_t MaxVal, int64_t &Val);
  bool matchDfmtNfmt(int64_t &Dfmt, int64_t &Nfmt, StringRef FormatStr, SMLoc Loc);

  void cvtDSOffset01(MCInst &Inst, const OperandVector &Operands);
  void cvtDS(MCInst &Inst, const OperandVector &Operands) { cvtDSImpl(Inst, Operands, false); }
  void cvtDSGds(MCInst &Inst, const OperandVector &Operands) { cvtDSImpl(Inst, Operands, true); }
  void cvtExp(MCInst &Inst, const OperandVector &Operands);

  bool parseCnt(int64_t &IntVal);
  OperandMatchResultTy parseSWaitCntOps(OperandVector &Operands);
  OperandMatchResultTy parseHwreg(OperandVector &Operands);

private:
  struct OperandInfoTy {
    int64_t Id;
    bool IsSymbolic = false;
    bool IsDefined = false;

    OperandInfoTy(int64_t Id_) : Id(Id_) {}
  };

  bool parseSendMsgBody(OperandInfoTy &Msg, OperandInfoTy &Op, OperandInfoTy &Stream);
  bool validateSendMsg(const OperandInfoTy &Msg,
                       const OperandInfoTy &Op,
                       const OperandInfoTy &Stream,
                       const SMLoc Loc);

  bool parseHwregBody(OperandInfoTy &HwReg, int64_t &Offset, int64_t &Width);
  bool validateHwreg(const OperandInfoTy &HwReg,
                     const int64_t Offset,
                     const int64_t Width,
                     const SMLoc Loc);

  OperandMatchResultTy parseExpTgtImpl(StringRef Str, uint8_t &Val);
  SMLoc getFlatOffsetLoc(const OperandVector &Operands) const;
  SMLoc getSMEMOffsetLoc(const OperandVector &Operands) const;

  bool validateInstruction(const MCInst &Inst, const SMLoc &IDLoc, const OperandVector &Operands);
  bool validateFlatOffset(const MCInst &Inst, const OperandVector &Operands);
  bool validateSMEMOffset(const MCInst &Inst, const OperandVector &Operands);
  bool validateSOPLiteral(const MCInst &Inst) const;
  bool validateConstantBusLimitations(const MCInst &Inst);
  bool validateEarlyClobberLimitations(const MCInst &Inst);
  bool validateIntClampSupported(const MCInst &Inst);
  bool validateMIMGAtomicDMask(const MCInst &Inst);
  bool validateMIMGGatherDMask(const MCInst &Inst);
  bool validateMovrels(const MCInst &Inst);
  bool validateMIMGDataSize(const MCInst &Inst);
  bool validateMIMGAddrSize(const MCInst &Inst);
  bool validateMIMGD16(const MCInst &Inst);
  bool validateMIMGDim(const MCInst &Inst);
  bool validateLdsDirect(const MCInst &Inst);
  bool validateOpSel(const MCInst &Inst);
  bool validateVccOperand(unsigned Reg) const;
  bool validateVOP3Literal(const MCInst &Inst) const;
  bool validateMAIAccWrite(const MCInst &Inst);
  unsigned getConstantBusLimit(unsigned Opcode) const;
  bool usesConstantBus(const MCInst &Inst, unsigned OpIdx);
  bool isInlineConstant(const MCInst &Inst, unsigned OpIdx) const;
  unsigned findImplicitSGPRReadInVOP(const MCInst &Inst) const;

  bool isId(const StringRef Id) const;
  bool isId(const AsmToken &Token, const StringRef Id) const;
  bool isToken(const AsmToken::TokenKind Kind) const;
  bool trySkipId(const StringRef Id);
  bool trySkipId(const StringRef Id, const AsmToken::TokenKind Kind);
  bool trySkipToken(const AsmToken::TokenKind Kind);
  bool skipToken(const AsmToken::TokenKind Kind, const StringRef ErrMsg);
  bool parseString(StringRef &Val, const StringRef ErrMsg = "expected a string");
  bool parseId(StringRef &Val, const StringRef ErrMsg);

  void peekTokens(MutableArrayRef<AsmToken> Tokens);
  AsmToken::TokenKind getTokenKind() const;
  bool parseExpr(int64_t &Imm);
  bool parseExpr(OperandVector &Operands);
  StringRef getTokenStr() const;
  AsmToken peekToken();
  AsmToken getToken() const;
  SMLoc getLoc() const;
  void lex();

public:
  OperandMatchResultTy parseOptionalOperand(OperandVector &Operands);
  OperandMatchResultTy parseOptionalOpr(OperandVector &Operands);

  OperandMatchResultTy parseExpTgt(OperandVector &Operands);
  OperandMatchResultTy parseSendMsgOp(OperandVector &Operands);
  OperandMatchResultTy parseInterpSlot(OperandVector &Operands);
  OperandMatchResultTy parseInterpAttr(OperandVector &Operands);
  OperandMatchResultTy parseSOppBrTarget(OperandVector &Operands);
  OperandMatchResultTy parseBoolReg(OperandVector &Operands);

  bool parseSwizzleOperands(const unsigned OpNum, int64_t* Op,
                            const unsigned MinVal,
                            const unsigned MaxVal,
                            const StringRef ErrMsg);
  OperandMatchResultTy parseSwizzleOp(OperandVector &Operands);
  bool parseSwizzleOffset(int64_t &Imm);
  bool parseSwizzleMacro(int64_t &Imm);
  bool parseSwizzleQuadPerm(int64_t &Imm);
  bool parseSwizzleBitmaskPerm(int64_t &Imm);
  bool parseSwizzleBroadcast(int64_t &Imm);
  bool parseSwizzleSwap(int64_t &Imm);
  bool parseSwizzleReverse(int64_t &Imm);

  OperandMatchResultTy parseGPRIdxMode(OperandVector &Operands);
  int64_t parseGPRIdxMacro();

  void cvtMubuf(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, false, false); }
  void cvtMubufAtomic(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, true, false); }
  void cvtMubufAtomicReturn(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, true, true); }
  void cvtMubufLds(MCInst &Inst, const OperandVector &Operands) { cvtMubufImpl(Inst, Operands, false, false, true); }
  void cvtMtbuf(MCInst &Inst, const OperandVector &Operands);

  AMDGPUOperand::Ptr defaultDLC() const;
  AMDGPUOperand::Ptr defaultGLC() const;
  AMDGPUOperand::Ptr defaultSLC() const;

  AMDGPUOperand::Ptr defaultSMRDOffset8() const;
  AMDGPUOperand::Ptr defaultSMEMOffset() const;
  AMDGPUOperand::Ptr defaultSMRDLiteralOffset() const;
  AMDGPUOperand::Ptr defaultFlatOffset() const;

  OperandMatchResultTy parseOModOperand(OperandVector &Operands);

  void cvtVOP3(MCInst &Inst, const OperandVector &Operands,
               OptionalImmIndexMap &OptionalIdx);
  void cvtVOP3OpSel(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3(MCInst &Inst, const OperandVector &Operands);
  void cvtVOP3P(MCInst &Inst, const OperandVector &Operands);

  void cvtVOP3Interp(MCInst &Inst, const OperandVector &Operands);

  void cvtMIMG(MCInst &Inst, const OperandVector &Operands,
               bool IsAtomic = false);
  void cvtMIMGAtomic(MCInst &Inst, const OperandVector &Operands);

  OperandMatchResultTy parseDim(OperandVector &Operands);
  OperandMatchResultTy parseDPP8(OperandVector &Operands);
  OperandMatchResultTy parseDPPCtrl(OperandVector &Operands);
  AMDGPUOperand::Ptr defaultRowMask() const;
  AMDGPUOperand::Ptr defaultBankMask() const;
  AMDGPUOperand::Ptr defaultBoundCtrl() const;
  AMDGPUOperand::Ptr defaultFI() const;
  void cvtDPP(MCInst &Inst, const OperandVector &Operands, bool IsDPP8 = false);
  void cvtDPP8(MCInst &Inst, const OperandVector &Operands) { cvtDPP(Inst, Operands, true); }

  OperandMatchResultTy parseSDWASel(OperandVector &Operands, StringRef Prefix,
                                    AMDGPUOperand::ImmTy Type);
  OperandMatchResultTy parseSDWADstUnused(OperandVector &Operands);
  void cvtSdwaVOP1(MCInst &Inst, const OperandVector &Operands);
  void cvtSdwaVOP2(MCInst &Inst, const OperandVector &Operands);
  void cvtSdwaVOP2b(MCInst &Inst, const OperandVector &Operands);
  void cvtSdwaVOP2e(MCInst &Inst, const OperandVector &Operands);
  void cvtSdwaVOPC(MCInst &Inst, const OperandVector &Operands);
  void cvtSDWA(MCInst &Inst, const OperandVector &Operands,
               uint64_t BasicInstType,
               bool SkipDstVcc = false,
               bool SkipSrcVcc = false);

  AMDGPUOperand::Ptr defaultBLGP() const;
  AMDGPUOperand::Ptr defaultCBSZ() const;
  AMDGPUOperand::Ptr defaultABID() const;

  OperandMatchResultTy parseEndpgmOp(OperandVector &Operands);
  AMDGPUOperand::Ptr defaultEndpgmImmOperands() const;
};

struct OptionalOperand {
  const char *Name;
  AMDGPUOperand::ImmTy Type;
  bool IsBit;
  bool (*ConvertResult)(int64_t&);
};

} // end anonymous namespace

// May be called with integer type with equivalent bitwidth.
static const fltSemantics *getFltSemantics(unsigned Size) {
  switch (Size) {
  case 4:
    return &APFloat::IEEEsingle();
  case 8:
    return &APFloat::IEEEdouble();
  case 2:
    return &APFloat::IEEEhalf();
  default:
    llvm_unreachable("unsupported fp type");
  }
}

static const fltSemantics *getFltSemantics(MVT VT) {
  return getFltSemantics(VT.getSizeInBits() / 8);
}

static const fltSemantics *getOpFltSemantics(uint8_t OperandType) {
  switch (OperandType) {
  case AMDGPU::OPERAND_REG_IMM_INT32:
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
    return &APFloat::IEEEsingle();
  case AMDGPU::OPERAND_REG_IMM_INT64:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_INT64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    return &APFloat::IEEEdouble();
  case AMDGPU::OPERAND_REG_IMM_INT16:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16:
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
    return &APFloat::IEEEhalf();
  default:
    llvm_unreachable("unsupported fp type");
  }
}

//===----------------------------------------------------------------------===//
// Operand
//===----------------------------------------------------------------------===//

static bool canLosslesslyConvertToFPType(APFloat &FPLiteral, MVT VT) {
  bool Lost;

  // Convert literal to single precision
  APFloat::opStatus Status = FPLiteral.convert(*getFltSemantics(VT),
                                               APFloat::rmNearestTiesToEven,
                                               &Lost);
  // We allow precision lost but not overflow or underflow
  if (Status != APFloat::opOK &&
      Lost &&
      ((Status & APFloat::opOverflow)  != 0 ||
       (Status & APFloat::opUnderflow) != 0)) {
    return false;
  }

  return true;
}

static bool isSafeTruncation(int64_t Val, unsigned Size) {
  return isUIntN(Size, Val) || isIntN(Size, Val);
}

static bool isInlineableLiteralOp16(int64_t Val, MVT VT, bool HasInv2Pi) {
  if (VT.getScalarType() == MVT::i16) {
    // FP immediate values are broken.
    return isInlinableIntLiteral(Val);
  }

  // f16/v2f16 operands work correctly for all values.
  return AMDGPU::isInlinableLiteral16(Val, HasInv2Pi);
}

bool AMDGPUOperand::isInlinableImm(MVT type) const {

  // This is a hack to enable named inline values like
  // shared_base with both 32-bit and 64-bit operands.
  // Note that these values are defined as
  // 32-bit operands only.
  if (isInlineValue()) {
    return true;
  }

  if (!isImmTy(ImmTyNone)) {
    // Only plain immediates are inlinable (e.g. "clamp" attribute is not)
    return false;
  }
  // TODO: We should avoid using host float here. It would be better to
  // check the float bit values which is what a few other places do.
  // We've had bot failures before due to weird NaN support on mips hosts.

  APInt Literal(64, Imm.Val);

  if (Imm.IsFPImm) { // We got fp literal token
    if (type == MVT::f64 || type == MVT::i64) { // Expected 64-bit operand
      return AMDGPU::isInlinableLiteral64(Imm.Val,
                                          AsmParser->hasInv2PiInlineImm());
    }

    APFloat FPLiteral(APFloat::IEEEdouble(), APInt(64, Imm.Val));
    if (!canLosslesslyConvertToFPType(FPLiteral, type))
      return false;

    if (type.getScalarSizeInBits() == 16) {
      return isInlineableLiteralOp16(
        static_cast<int16_t>(FPLiteral.bitcastToAPInt().getZExtValue()),
        type, AsmParser->hasInv2PiInlineImm());
    }

    // Check if single precision literal is inlinable
    return AMDGPU::isInlinableLiteral32(
      static_cast<int32_t>(FPLiteral.bitcastToAPInt().getZExtValue()),
      AsmParser->hasInv2PiInlineImm());
  }

  // We got int literal token.
  if (type == MVT::f64 || type == MVT::i64) { // Expected 64-bit operand
    return AMDGPU::isInlinableLiteral64(Imm.Val,
                                        AsmParser->hasInv2PiInlineImm());
  }

  if (!isSafeTruncation(Imm.Val, type.getScalarSizeInBits())) {
    return false;
  }

  if (type.getScalarSizeInBits() == 16) {
    return isInlineableLiteralOp16(
      static_cast<int16_t>(Literal.getLoBits(16).getSExtValue()),
      type, AsmParser->hasInv2PiInlineImm());
  }

  return AMDGPU::isInlinableLiteral32(
    static_cast<int32_t>(Literal.getLoBits(32).getZExtValue()),
    AsmParser->hasInv2PiInlineImm());
}

bool AMDGPUOperand::isLiteralImm(MVT type) const {
  // Check that this immediate can be added as literal
  if (!isImmTy(ImmTyNone)) {
    return false;
  }

  if (!Imm.IsFPImm) {
    // We got int literal token.

    if (type == MVT::f64 && hasFPModifiers()) {
      // Cannot apply fp modifiers to int literals preserving the same semantics
      // for VOP1/2/C and VOP3 because of integer truncation. To avoid ambiguity,
      // disable these cases.
      return false;
    }

    unsigned Size = type.getSizeInBits();
    if (Size == 64)
      Size = 32;

    // FIXME: 64-bit operands can zero extend, sign extend, or pad zeroes for FP
    // types.
    return isSafeTruncation(Imm.Val, Size);
  }

  // We got fp literal token
  if (type == MVT::f64) { // Expected 64-bit fp operand
    // We would set low 64-bits of literal to zeroes but we accept this literals
    return true;
  }

  if (type == MVT::i64) { // Expected 64-bit int operand
    // We don't allow fp literals in 64-bit integer instructions. It is
    // unclear how we should encode them.
    return false;
  }

  // We allow fp literals with f16x2 operands assuming that the specified
  // literal goes into the lower half and the upper half is zero. We also
  // require that the literal may be losslesly converted to f16.
  MVT ExpectedType = (type == MVT::v2f16)? MVT::f16 :
                     (type == MVT::v2i16)? MVT::i16 : type;

  APFloat FPLiteral(APFloat::IEEEdouble(), APInt(64, Imm.Val));
  return canLosslesslyConvertToFPType(FPLiteral, ExpectedType);
}

bool AMDGPUOperand::isRegClass(unsigned RCID) const {
  return isRegKind() && AsmParser->getMRI()->getRegClass(RCID).contains(getReg());
}

bool AMDGPUOperand::isSDWAOperand(MVT type) const {
  if (AsmParser->isVI())
    return isVReg32();
  else if (AsmParser->isGFX9() || AsmParser->isGFX10())
    return isRegClass(AMDGPU::VS_32RegClassID) || isInlinableImm(type);
  else
    return false;
}

bool AMDGPUOperand::isSDWAFP16Operand() const {
  return isSDWAOperand(MVT::f16);
}

bool AMDGPUOperand::isSDWAFP32Operand() const {
  return isSDWAOperand(MVT::f32);
}

bool AMDGPUOperand::isSDWAInt16Operand() const {
  return isSDWAOperand(MVT::i16);
}

bool AMDGPUOperand::isSDWAInt32Operand() const {
  return isSDWAOperand(MVT::i32);
}

bool AMDGPUOperand::isBoolReg() const {
  return (AsmParser->getFeatureBits()[AMDGPU::FeatureWavefrontSize64] && isSCSrcB64()) ||
         (AsmParser->getFeatureBits()[AMDGPU::FeatureWavefrontSize32] && isSCSrcB32());
}

uint64_t AMDGPUOperand::applyInputFPModifiers(uint64_t Val, unsigned Size) const
{
  assert(isImmTy(ImmTyNone) && Imm.Mods.hasFPModifiers());
  assert(Size == 2 || Size == 4 || Size == 8);

  const uint64_t FpSignMask = (1ULL << (Size * 8 - 1));

  if (Imm.Mods.Abs) {
    Val &= ~FpSignMask;
  }
  if (Imm.Mods.Neg) {
    Val ^= FpSignMask;
  }

  return Val;
}

void AMDGPUOperand::addImmOperands(MCInst &Inst, unsigned N, bool ApplyModifiers) const {
  if (AMDGPU::isSISrcOperand(AsmParser->getMII()->get(Inst.getOpcode()),
                             Inst.getNumOperands())) {
    addLiteralImmOperand(Inst, Imm.Val,
                         ApplyModifiers &
                         isImmTy(ImmTyNone) && Imm.Mods.hasFPModifiers());
  } else {
    assert(!isImmTy(ImmTyNone) || !hasModifiers());
    Inst.addOperand(MCOperand::createImm(Imm.Val));
  }
}

void AMDGPUOperand::addLiteralImmOperand(MCInst &Inst, int64_t Val, bool ApplyModifiers) const {
  const auto& InstDesc = AsmParser->getMII()->get(Inst.getOpcode());
  auto OpNum = Inst.getNumOperands();
  // Check that this operand accepts literals
  assert(AMDGPU::isSISrcOperand(InstDesc, OpNum));

  if (ApplyModifiers) {
    assert(AMDGPU::isSISrcFPOperand(InstDesc, OpNum));
    const unsigned Size = Imm.IsFPImm ? sizeof(double) : getOperandSize(InstDesc, OpNum);
    Val = applyInputFPModifiers(Val, Size);
  }

  APInt Literal(64, Val);
  uint8_t OpTy = InstDesc.OpInfo[OpNum].OperandType;

  if (Imm.IsFPImm) { // We got fp literal token
    switch (OpTy) {
    case AMDGPU::OPERAND_REG_IMM_INT64:
    case AMDGPU::OPERAND_REG_IMM_FP64:
    case AMDGPU::OPERAND_REG_INLINE_C_INT64:
    case AMDGPU::OPERAND_REG_INLINE_C_FP64:
      if (AMDGPU::isInlinableLiteral64(Literal.getZExtValue(),
                                       AsmParser->hasInv2PiInlineImm())) {
        Inst.addOperand(MCOperand::createImm(Literal.getZExtValue()));
        return;
      }

      // Non-inlineable
      if (AMDGPU::isSISrcFPOperand(InstDesc, OpNum)) { // Expected 64-bit fp operand
        // For fp operands we check if low 32 bits are zeros
        if (Literal.getLoBits(32) != 0) {
          const_cast<AMDGPUAsmParser *>(AsmParser)->Warning(Inst.getLoc(),
          "Can't encode literal as exact 64-bit floating-point operand. "
          "Low 32-bits will be set to zero");
        }

        Inst.addOperand(MCOperand::createImm(Literal.lshr(32).getZExtValue()));
        return;
      }

      // We don't allow fp literals in 64-bit integer instructions. It is
      // unclear how we should encode them. This case should be checked earlier
      // in predicate methods (isLiteralImm())
      llvm_unreachable("fp literal in 64-bit integer instruction.");

    case AMDGPU::OPERAND_REG_IMM_INT32:
    case AMDGPU::OPERAND_REG_IMM_FP32:
    case AMDGPU::OPERAND_REG_INLINE_C_INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
    case AMDGPU::OPERAND_REG_IMM_INT16:
    case AMDGPU::OPERAND_REG_IMM_FP16:
    case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
    case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
    case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16:
    case AMDGPU::OPERAND_REG_IMM_V2INT16:
    case AMDGPU::OPERAND_REG_IMM_V2FP16: {
      bool lost;
      APFloat FPLiteral(APFloat::IEEEdouble(), Literal);
      // Convert literal to single precision
      FPLiteral.convert(*getOpFltSemantics(OpTy),
                        APFloat::rmNearestTiesToEven, &lost);
      // We allow precision lost but not overflow or underflow. This should be
      // checked earlier in isLiteralImm()

      uint64_t ImmVal = FPLiteral.bitcastToAPInt().getZExtValue();
      Inst.addOperand(MCOperand::createImm(ImmVal));
      return;
    }
    default:
      llvm_unreachable("invalid operand size");
    }

    return;
  }

  // We got int literal token.
  // Only sign extend inline immediates.
  switch (OpTy) {
  case AMDGPU::OPERAND_REG_IMM_INT32:
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
    if (isSafeTruncation(Val, 32) &&
        AMDGPU::isInlinableLiteral32(static_cast<int32_t>(Val),
                                     AsmParser->hasInv2PiInlineImm())) {
      Inst.addOperand(MCOperand::createImm(Val));
      return;
    }

    Inst.addOperand(MCOperand::createImm(Val & 0xffffffff));
    return;

  case AMDGPU::OPERAND_REG_IMM_INT64:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_INT64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    if (AMDGPU::isInlinableLiteral64(Val, AsmParser->hasInv2PiInlineImm())) {
      Inst.addOperand(MCOperand::createImm(Val));
      return;
    }

    Inst.addOperand(MCOperand::createImm(Lo_32(Val)));
    return;

  case AMDGPU::OPERAND_REG_IMM_INT16:
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
    if (isSafeTruncation(Val, 16) &&
        AMDGPU::isInlinableLiteral16(static_cast<int16_t>(Val),
                                     AsmParser->hasInv2PiInlineImm())) {
      Inst.addOperand(MCOperand::createImm(Val));
      return;
    }

    Inst.addOperand(MCOperand::createImm(Val & 0xffff));
    return;

  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16: {
    assert(isSafeTruncation(Val, 16));
    assert(AMDGPU::isInlinableLiteral16(static_cast<int16_t>(Val),
                                        AsmParser->hasInv2PiInlineImm()));

    Inst.addOperand(MCOperand::createImm(Val));
    return;
  }
  default:
    llvm_unreachable("invalid operand size");
  }
}

template <unsigned Bitwidth>
void AMDGPUOperand::addKImmFPOperands(MCInst &Inst, unsigned N) const {
  APInt Literal(64, Imm.Val);

  if (!Imm.IsFPImm) {
    // We got int literal token.
    Inst.addOperand(MCOperand::createImm(Literal.getLoBits(Bitwidth).getZExtValue()));
    return;
  }

  bool Lost;
  APFloat FPLiteral(APFloat::IEEEdouble(), Literal);
  FPLiteral.convert(*getFltSemantics(Bitwidth / 8),
                    APFloat::rmNearestTiesToEven, &Lost);
  Inst.addOperand(MCOperand::createImm(FPLiteral.bitcastToAPInt().getZExtValue()));
}

void AMDGPUOperand::addRegOperands(MCInst &Inst, unsigned N) const {
  Inst.addOperand(MCOperand::createReg(AMDGPU::getMCReg(getReg(), AsmParser->getSTI())));
}

static bool isInlineValue(unsigned Reg) {
  switch (Reg) {
  case AMDGPU::SRC_SHARED_BASE:
  case AMDGPU::SRC_SHARED_LIMIT:
  case AMDGPU::SRC_PRIVATE_BASE:
  case AMDGPU::SRC_PRIVATE_LIMIT:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
    return true;
  case AMDGPU::SRC_VCCZ:
  case AMDGPU::SRC_EXECZ:
  case AMDGPU::SRC_SCC:
    return true;
  case AMDGPU::SGPR_NULL:
    return true;
  default:
    return false;
  }
}

bool AMDGPUOperand::isInlineValue() const {
  return isRegKind() && ::isInlineValue(getReg());
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

static int getRegClass(RegisterKind Is, unsigned RegWidth) {
  if (Is == IS_VGPR) {
    switch (RegWidth) {
      default: return -1;
      case 1: return AMDGPU::VGPR_32RegClassID;
      case 2: return AMDGPU::VReg_64RegClassID;
      case 3: return AMDGPU::VReg_96RegClassID;
      case 4: return AMDGPU::VReg_128RegClassID;
      case 5: return AMDGPU::VReg_160RegClassID;
      case 6: return AMDGPU::VReg_192RegClassID;
      case 8: return AMDGPU::VReg_256RegClassID;
      case 16: return AMDGPU::VReg_512RegClassID;
      case 32: return AMDGPU::VReg_1024RegClassID;
    }
  } else if (Is == IS_TTMP) {
    switch (RegWidth) {
      default: return -1;
      case 1: return AMDGPU::TTMP_32RegClassID;
      case 2: return AMDGPU::TTMP_64RegClassID;
      case 4: return AMDGPU::TTMP_128RegClassID;
      case 8: return AMDGPU::TTMP_256RegClassID;
      case 16: return AMDGPU::TTMP_512RegClassID;
    }
  } else if (Is == IS_SGPR) {
    switch (RegWidth) {
      default: return -1;
      case 1: return AMDGPU::SGPR_32RegClassID;
      case 2: return AMDGPU::SGPR_64RegClassID;
      case 3: return AMDGPU::SGPR_96RegClassID;
      case 4: return AMDGPU::SGPR_128RegClassID;
      case 5: return AMDGPU::SGPR_160RegClassID;
      case 6: return AMDGPU::SGPR_192RegClassID;
      case 8: return AMDGPU::SGPR_256RegClassID;
      case 16: return AMDGPU::SGPR_512RegClassID;
    }
  } else if (Is == IS_AGPR) {
    switch (RegWidth) {
      default: return -1;
      case 1: return AMDGPU::AGPR_32RegClassID;
      case 2: return AMDGPU::AReg_64RegClassID;
      case 3: return AMDGPU::AReg_96RegClassID;
      case 4: return AMDGPU::AReg_128RegClassID;
      case 5: return AMDGPU::AReg_160RegClassID;
      case 6: return AMDGPU::AReg_192RegClassID;
      case 8: return AMDGPU::AReg_256RegClassID;
      case 16: return AMDGPU::AReg_512RegClassID;
      case 32: return AMDGPU::AReg_1024RegClassID;
    }
  }
  return -1;
}

static unsigned getSpecialRegForName(StringRef RegName) {
  return StringSwitch<unsigned>(RegName)
    .Case("exec", AMDGPU::EXEC)
    .Case("vcc", AMDGPU::VCC)
    .Case("flat_scratch", AMDGPU::FLAT_SCR)
    .Case("xnack_mask", AMDGPU::XNACK_MASK)
    .Case("shared_base", AMDGPU::SRC_SHARED_BASE)
    .Case("src_shared_base", AMDGPU::SRC_SHARED_BASE)
    .Case("shared_limit", AMDGPU::SRC_SHARED_LIMIT)
    .Case("src_shared_limit", AMDGPU::SRC_SHARED_LIMIT)
    .Case("private_base", AMDGPU::SRC_PRIVATE_BASE)
    .Case("src_private_base", AMDGPU::SRC_PRIVATE_BASE)
    .Case("private_limit", AMDGPU::SRC_PRIVATE_LIMIT)
    .Case("src_private_limit", AMDGPU::SRC_PRIVATE_LIMIT)
    .Case("pops_exiting_wave_id", AMDGPU::SRC_POPS_EXITING_WAVE_ID)
    .Case("src_pops_exiting_wave_id", AMDGPU::SRC_POPS_EXITING_WAVE_ID)
    .Case("lds_direct", AMDGPU::LDS_DIRECT)
    .Case("src_lds_direct", AMDGPU::LDS_DIRECT)
    .Case("m0", AMDGPU::M0)
    .Case("vccz", AMDGPU::SRC_VCCZ)
    .Case("src_vccz", AMDGPU::SRC_VCCZ)
    .Case("execz", AMDGPU::SRC_EXECZ)
    .Case("src_execz", AMDGPU::SRC_EXECZ)
    .Case("scc", AMDGPU::SRC_SCC)
    .Case("src_scc", AMDGPU::SRC_SCC)
    .Case("tba", AMDGPU::TBA)
    .Case("tma", AMDGPU::TMA)
    .Case("flat_scratch_lo", AMDGPU::FLAT_SCR_LO)
    .Case("flat_scratch_hi", AMDGPU::FLAT_SCR_HI)
    .Case("xnack_mask_lo", AMDGPU::XNACK_MASK_LO)
    .Case("xnack_mask_hi", AMDGPU::XNACK_MASK_HI)
    .Case("vcc_lo", AMDGPU::VCC_LO)
    .Case("vcc_hi", AMDGPU::VCC_HI)
    .Case("exec_lo", AMDGPU::EXEC_LO)
    .Case("exec_hi", AMDGPU::EXEC_HI)
    .Case("tma_lo", AMDGPU::TMA_LO)
    .Case("tma_hi", AMDGPU::TMA_HI)
    .Case("tba_lo", AMDGPU::TBA_LO)
    .Case("tba_hi", AMDGPU::TBA_HI)
    .Case("pc", AMDGPU::PC_REG)
    .Case("null", AMDGPU::SGPR_NULL)
    .Default(AMDGPU::NoRegister);
}

bool AMDGPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                    SMLoc &EndLoc, bool RestoreOnFailure) {
  auto R = parseRegister();
  if (!R) return true;
  assert(R->isReg());
  RegNo = R->getReg();
  StartLoc = R->getStartLoc();
  EndLoc = R->getEndLoc();
  return false;
}

bool AMDGPUAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                    SMLoc &EndLoc) {
  return ParseRegister(RegNo, StartLoc, EndLoc, /*RestoreOnFailure=*/false);
}

OperandMatchResultTy AMDGPUAsmParser::tryParseRegister(unsigned &RegNo,
                                                       SMLoc &StartLoc,
                                                       SMLoc &EndLoc) {
  bool Result =
      ParseRegister(RegNo, StartLoc, EndLoc, /*RestoreOnFailure=*/true);
  bool PendingErrors = getParser().hasPendingError();
  getParser().clearPendingErrors();
  if (PendingErrors)
    return MatchOperand_ParseFail;
  if (Result)
    return MatchOperand_NoMatch;
  return MatchOperand_Success;
}

bool AMDGPUAsmParser::AddNextRegisterToList(unsigned &Reg, unsigned &RegWidth,
                                            RegisterKind RegKind, unsigned Reg1,
                                            SMLoc Loc) {
  switch (RegKind) {
  case IS_SPECIAL:
    if (Reg == AMDGPU::EXEC_LO && Reg1 == AMDGPU::EXEC_HI) {
      Reg = AMDGPU::EXEC;
      RegWidth = 2;
      return true;
    }
    if (Reg == AMDGPU::FLAT_SCR_LO && Reg1 == AMDGPU::FLAT_SCR_HI) {
      Reg = AMDGPU::FLAT_SCR;
      RegWidth = 2;
      return true;
    }
    if (Reg == AMDGPU::XNACK_MASK_LO && Reg1 == AMDGPU::XNACK_MASK_HI) {
      Reg = AMDGPU::XNACK_MASK;
      RegWidth = 2;
      return true;
    }
    if (Reg == AMDGPU::VCC_LO && Reg1 == AMDGPU::VCC_HI) {
      Reg = AMDGPU::VCC;
      RegWidth = 2;
      return true;
    }
    if (Reg == AMDGPU::TBA_LO && Reg1 == AMDGPU::TBA_HI) {
      Reg = AMDGPU::TBA;
      RegWidth = 2;
      return true;
    }
    if (Reg == AMDGPU::TMA_LO && Reg1 == AMDGPU::TMA_HI) {
      Reg = AMDGPU::TMA;
      RegWidth = 2;
      return true;
    }
    Error(Loc, "register does not fit in the list");
    return false;
  case IS_VGPR:
  case IS_SGPR:
  case IS_AGPR:
  case IS_TTMP:
    if (Reg1 != Reg + RegWidth) {
      Error(Loc, "registers in a list must have consecutive indices");
      return false;
    }
    RegWidth++;
    return true;
  default:
    llvm_unreachable("unexpected register kind");
  }
}

struct RegInfo {
  StringLiteral Name;
  RegisterKind Kind;
};

static constexpr RegInfo RegularRegisters[] = {
  {{"v"},    IS_VGPR},
  {{"s"},    IS_SGPR},
  {{"ttmp"}, IS_TTMP},
  {{"acc"},  IS_AGPR},
  {{"a"},    IS_AGPR},
};

static bool isRegularReg(RegisterKind Kind) {
  return Kind == IS_VGPR ||
         Kind == IS_SGPR ||
         Kind == IS_TTMP ||
         Kind == IS_AGPR;
}

static const RegInfo* getRegularRegInfo(StringRef Str) {
  for (const RegInfo &Reg : RegularRegisters)
    if (Str.startswith(Reg.Name))
      return &Reg;
  return nullptr;
}

static bool getRegNum(StringRef Str, unsigned& Num) {
  return !Str.getAsInteger(10, Num);
}

bool
AMDGPUAsmParser::isRegister(const AsmToken &Token,
                            const AsmToken &NextToken) const {

  // A list of consecutive registers: [s0,s1,s2,s3]
  if (Token.is(AsmToken::LBrac))
    return true;

  if (!Token.is(AsmToken::Identifier))
    return false;

  // A single register like s0 or a range of registers like s[0:1]

  StringRef Str = Token.getString();
  const RegInfo *Reg = getRegularRegInfo(Str);
  if (Reg) {
    StringRef RegName = Reg->Name;
    StringRef RegSuffix = Str.substr(RegName.size());
    if (!RegSuffix.empty()) {
      unsigned Num;
      // A single register with an index: rXX
      if (getRegNum(RegSuffix, Num))
        return true;
    } else {
      // A range of registers: r[XX:YY].
      if (NextToken.is(AsmToken::LBrac))
        return true;
    }
  }

  return getSpecialRegForName(Str) != AMDGPU::NoRegister;
}

bool
AMDGPUAsmParser::isRegister()
{
  return isRegister(getToken(), peekToken());
}

unsigned
AMDGPUAsmParser::getRegularReg(RegisterKind RegKind,
                               unsigned RegNum,
                               unsigned RegWidth,
                               SMLoc Loc) {

  assert(isRegularReg(RegKind));

  unsigned AlignSize = 1;
  if (RegKind == IS_SGPR || RegKind == IS_TTMP) {
    // SGPR and TTMP registers must be aligned.
    // Max required alignment is 4 dwords.
    AlignSize = std::min(RegWidth, 4u);
  }

  if (RegNum % AlignSize != 0) {
    Error(Loc, "invalid register alignment");
    return AMDGPU::NoRegister;
  }

  unsigned RegIdx = RegNum / AlignSize;
  int RCID = getRegClass(RegKind, RegWidth);
  if (RCID == -1) {
    Error(Loc, "invalid or unsupported register size");
    return AMDGPU::NoRegister;
  }

  const MCRegisterInfo *TRI = getContext().getRegisterInfo();
  const MCRegisterClass RC = TRI->getRegClass(RCID);
  if (RegIdx >= RC.getNumRegs()) {
    Error(Loc, "register index is out of range");
    return AMDGPU::NoRegister;
  }

  return RC.getRegister(RegIdx);
}

bool
AMDGPUAsmParser::ParseRegRange(unsigned& Num, unsigned& Width) {
  int64_t RegLo, RegHi;
  if (!skipToken(AsmToken::LBrac, "missing register index"))
    return false;

  SMLoc FirstIdxLoc = getLoc();
  SMLoc SecondIdxLoc;

  if (!parseExpr(RegLo))
    return false;

  if (trySkipToken(AsmToken::Colon)) {
    SecondIdxLoc = getLoc();
    if (!parseExpr(RegHi))
      return false;
  } else {
    RegHi = RegLo;
  }

  if (!skipToken(AsmToken::RBrac, "expected a closing square bracket"))
    return false;

  if (!isUInt<32>(RegLo)) {
    Error(FirstIdxLoc, "invalid register index");
    return false;
  }

  if (!isUInt<32>(RegHi)) {
    Error(SecondIdxLoc, "invalid register index");
    return false;
  }

  if (RegLo > RegHi) {
    Error(FirstIdxLoc, "first register index should not exceed second index");
    return false;
  }

  Num = static_cast<unsigned>(RegLo);
  Width = (RegHi - RegLo) + 1;
  return true;
}

unsigned AMDGPUAsmParser::ParseSpecialReg(RegisterKind &RegKind,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  assert(isToken(AsmToken::Identifier));
  unsigned Reg = getSpecialRegForName(getTokenStr());
  if (Reg) {
    RegNum = 0;
    RegWidth = 1;
    RegKind = IS_SPECIAL;
    Tokens.push_back(getToken());
    lex(); // skip register name
  }
  return Reg;
}

unsigned AMDGPUAsmParser::ParseRegularReg(RegisterKind &RegKind,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  assert(isToken(AsmToken::Identifier));
  StringRef RegName = getTokenStr();
  auto Loc = getLoc();

  const RegInfo *RI = getRegularRegInfo(RegName);
  if (!RI) {
    Error(Loc, "invalid register name");
    return AMDGPU::NoRegister;
  }

  Tokens.push_back(getToken());
  lex(); // skip register name

  RegKind = RI->Kind;
  StringRef RegSuffix = RegName.substr(RI->Name.size());
  if (!RegSuffix.empty()) {
    // Single 32-bit register: vXX.
    if (!getRegNum(RegSuffix, RegNum)) {
      Error(Loc, "invalid register index");
      return AMDGPU::NoRegister;
    }
    RegWidth = 1;
  } else {
    // Range of registers: v[XX:YY]. ":YY" is optional.
    if (!ParseRegRange(RegNum, RegWidth))
      return AMDGPU::NoRegister;
  }

  return getRegularReg(RegKind, RegNum, RegWidth, Loc);
}

unsigned AMDGPUAsmParser::ParseRegList(RegisterKind &RegKind, unsigned &RegNum,
                                       unsigned &RegWidth,
                                       SmallVectorImpl<AsmToken> &Tokens) {
  unsigned Reg = AMDGPU::NoRegister;
  auto ListLoc = getLoc();

  if (!skipToken(AsmToken::LBrac,
                 "expected a register or a list of registers")) {
    return AMDGPU::NoRegister;
  }

  // List of consecutive registers, e.g.: [s0,s1,s2,s3]

  auto Loc = getLoc();
  if (!ParseAMDGPURegister(RegKind, Reg, RegNum, RegWidth))
    return AMDGPU::NoRegister;
  if (RegWidth != 1) {
    Error(Loc, "expected a single 32-bit register");
    return AMDGPU::NoRegister;
  }

  for (; trySkipToken(AsmToken::Comma); ) {
    RegisterKind NextRegKind;
    unsigned NextReg, NextRegNum, NextRegWidth;
    Loc = getLoc();

    if (!ParseAMDGPURegister(NextRegKind, NextReg,
                             NextRegNum, NextRegWidth,
                             Tokens)) {
      return AMDGPU::NoRegister;
    }
    if (NextRegWidth != 1) {
      Error(Loc, "expected a single 32-bit register");
      return AMDGPU::NoRegister;
    }
    if (NextRegKind != RegKind) {
      Error(Loc, "registers in a list must be of the same kind");
      return AMDGPU::NoRegister;
    }
    if (!AddNextRegisterToList(Reg, RegWidth, RegKind, NextReg, Loc))
      return AMDGPU::NoRegister;
  }

  if (!skipToken(AsmToken::RBrac,
                 "expected a comma or a closing square bracket")) {
    return AMDGPU::NoRegister;
  }

  if (isRegularReg(RegKind))
    Reg = getRegularReg(RegKind, RegNum, RegWidth, ListLoc);

  return Reg;
}

bool AMDGPUAsmParser::ParseAMDGPURegister(RegisterKind &RegKind, unsigned &Reg,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          SmallVectorImpl<AsmToken> &Tokens) {
  auto Loc = getLoc();
  Reg = AMDGPU::NoRegister;

  if (isToken(AsmToken::Identifier)) {
    Reg = ParseSpecialReg(RegKind, RegNum, RegWidth, Tokens);
    if (Reg == AMDGPU::NoRegister)
      Reg = ParseRegularReg(RegKind, RegNum, RegWidth, Tokens);
  } else {
    Reg = ParseRegList(RegKind, RegNum, RegWidth, Tokens);
  }

  const MCRegisterInfo *TRI = getContext().getRegisterInfo();
  if (Reg == AMDGPU::NoRegister) {
    assert(Parser.hasPendingError());
    return false;
  }

  if (!subtargetHasRegister(*TRI, Reg)) {
    if (Reg == AMDGPU::SGPR_NULL) {
      Error(Loc, "'null' operand is not supported on this GPU");
    } else {
      Error(Loc, "register not available on this GPU");
    }
    return false;
  }

  return true;
}

bool AMDGPUAsmParser::ParseAMDGPURegister(RegisterKind &RegKind, unsigned &Reg,
                                          unsigned &RegNum, unsigned &RegWidth,
                                          bool RestoreOnFailure /*=false*/) {
  Reg = AMDGPU::NoRegister;

  SmallVector<AsmToken, 1> Tokens;
  if (ParseAMDGPURegister(RegKind, Reg, RegNum, RegWidth, Tokens)) {
    if (RestoreOnFailure) {
      while (!Tokens.empty()) {
        getLexer().UnLex(Tokens.pop_back_val());
      }
    }
    return true;
  }
  return false;
}

Optional<StringRef>
AMDGPUAsmParser::getGprCountSymbolName(RegisterKind RegKind) {
  switch (RegKind) {
  case IS_VGPR:
    return StringRef(".amdgcn.next_free_vgpr");
  case IS_SGPR:
    return StringRef(".amdgcn.next_free_sgpr");
  default:
    return None;
  }
}

void AMDGPUAsmParser::initializeGprCountSymbol(RegisterKind RegKind) {
  auto SymbolName = getGprCountSymbolName(RegKind);
  assert(SymbolName && "initializing invalid register kind");
  MCSymbol *Sym = getContext().getOrCreateSymbol(*SymbolName);
  Sym->setVariableValue(MCConstantExpr::create(0, getContext()));
}

bool AMDGPUAsmParser::updateGprCountSymbols(RegisterKind RegKind,
                                            unsigned DwordRegIndex,
                                            unsigned RegWidth) {
  // Symbols are only defined for GCN targets
  if (AMDGPU::getIsaVersion(getSTI().getCPU()).Major < 6)
    return true;

  auto SymbolName = getGprCountSymbolName(RegKind);
  if (!SymbolName)
    return true;
  MCSymbol *Sym = getContext().getOrCreateSymbol(*SymbolName);

  int64_t NewMax = DwordRegIndex + RegWidth - 1;
  int64_t OldCount;

  if (!Sym->isVariable())
    return !Error(getParser().getTok().getLoc(),
                  ".amdgcn.next_free_{v,s}gpr symbols must be variable");
  if (!Sym->getVariableValue(false)->evaluateAsAbsolute(OldCount))
    return !Error(
        getParser().getTok().getLoc(),
        ".amdgcn.next_free_{v,s}gpr symbols must be absolute expressions");

  if (OldCount <= NewMax)
    Sym->setVariableValue(MCConstantExpr::create(NewMax + 1, getContext()));

  return true;
}

std::unique_ptr<AMDGPUOperand>
AMDGPUAsmParser::parseRegister(bool RestoreOnFailure) {
  const auto &Tok = Parser.getTok();
  SMLoc StartLoc = Tok.getLoc();
  SMLoc EndLoc = Tok.getEndLoc();
  RegisterKind RegKind;
  unsigned Reg, RegNum, RegWidth;

  if (!ParseAMDGPURegister(RegKind, Reg, RegNum, RegWidth)) {
    return nullptr;
  }
  if (AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
    if (!updateGprCountSymbols(RegKind, RegNum, RegWidth))
      return nullptr;
  } else
    KernelScope.usesRegister(RegKind, RegNum, RegWidth);
  return AMDGPUOperand::CreateReg(this, Reg, StartLoc, EndLoc);
}

OperandMatchResultTy
AMDGPUAsmParser::parseImm(OperandVector &Operands, bool HasSP3AbsModifier) {
  // TODO: add syntactic sugar for 1/(2*PI)

  assert(!isRegister());
  assert(!isModifier());

  const auto& Tok = getToken();
  const auto& NextTok = peekToken();
  bool IsReal = Tok.is(AsmToken::Real);
  SMLoc S = getLoc();
  bool Negate = false;

  if (!IsReal && Tok.is(AsmToken::Minus) && NextTok.is(AsmToken::Real)) {
    lex();
    IsReal = true;
    Negate = true;
  }

  if (IsReal) {
    // Floating-point expressions are not supported.
    // Can only allow floating-point literals with an
    // optional sign.

    StringRef Num = getTokenStr();
    lex();

    APFloat RealVal(APFloat::IEEEdouble());
    auto roundMode = APFloat::rmNearestTiesToEven;
    if (errorToBool(RealVal.convertFromString(Num, roundMode).takeError())) {
      return MatchOperand_ParseFail;
    }
    if (Negate)
      RealVal.changeSign();

    Operands.push_back(
      AMDGPUOperand::CreateImm(this, RealVal.bitcastToAPInt().getZExtValue(), S,
                               AMDGPUOperand::ImmTyNone, true));

    return MatchOperand_Success;

  } else {
    int64_t IntVal;
    const MCExpr *Expr;
    SMLoc S = getLoc();

    if (HasSP3AbsModifier) {
      // This is a workaround for handling expressions
      // as arguments of SP3 'abs' modifier, for example:
      //     |1.0|
      //     |-1|
      //     |1+x|
      // This syntax is not compatible with syntax of standard
      // MC expressions (due to the trailing '|').
      SMLoc EndLoc;
      if (getParser().parsePrimaryExpr(Expr, EndLoc, nullptr))
        return MatchOperand_ParseFail;
    } else {
      if (Parser.parseExpression(Expr))
        return MatchOperand_ParseFail;
    }

    if (Expr->evaluateAsAbsolute(IntVal)) {
      Operands.push_back(AMDGPUOperand::CreateImm(this, IntVal, S));
    } else {
      Operands.push_back(AMDGPUOperand::CreateExpr(this, Expr, S));
    }

    return MatchOperand_Success;
  }

  return MatchOperand_NoMatch;
}

OperandMatchResultTy
AMDGPUAsmParser::parseReg(OperandVector &Operands) {
  if (!isRegister())
    return MatchOperand_NoMatch;

  if (auto R = parseRegister()) {
    assert(R->isReg());
    Operands.push_back(std::move(R));
    return MatchOperand_Success;
  }
  return MatchOperand_ParseFail;
}

OperandMatchResultTy
AMDGPUAsmParser::parseRegOrImm(OperandVector &Operands, bool HasSP3AbsMod) {
  auto res = parseReg(Operands);
  if (res != MatchOperand_NoMatch) {
    return res;
  } else if (isModifier()) {
    return MatchOperand_NoMatch;
  } else {
    return parseImm(Operands, HasSP3AbsMod);
  }
}

bool
AMDGPUAsmParser::isNamedOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const {
  if (Token.is(AsmToken::Identifier) && NextToken.is(AsmToken::LParen)) {
    const auto &str = Token.getString();
    return str == "abs" || str == "neg" || str == "sext";
  }
  return false;
}

bool
AMDGPUAsmParser::isOpcodeModifierWithVal(const AsmToken &Token, const AsmToken &NextToken) const {
  return Token.is(AsmToken::Identifier) && NextToken.is(AsmToken::Colon);
}

bool
AMDGPUAsmParser::isOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const {
  return isNamedOperandModifier(Token, NextToken) || Token.is(AsmToken::Pipe);
}

bool
AMDGPUAsmParser::isRegOrOperandModifier(const AsmToken &Token, const AsmToken &NextToken) const {
  return isRegister(Token, NextToken) || isOperandModifier(Token, NextToken);
}

// Check if this is an operand modifier or an opcode modifier
// which may look like an expression but it is not. We should
// avoid parsing these modifiers as expressions. Currently
// recognized sequences are:
//   |...|
//   abs(...)
//   neg(...)
//   sext(...)
//   -reg
//   -|...|
//   -abs(...)
//   name:...
// Note that simple opcode modifiers like 'gds' may be parsed as
// expressions; this is a special case. See getExpressionAsToken.
//
bool
AMDGPUAsmParser::isModifier() {

  AsmToken Tok = getToken();
  AsmToken NextToken[2];
  peekTokens(NextToken);

  return isOperandModifier(Tok, NextToken[0]) ||
         (Tok.is(AsmToken::Minus) && isRegOrOperandModifier(NextToken[0], NextToken[1])) ||
         isOpcodeModifierWithVal(Tok, NextToken[0]);
}

// Check if the current token is an SP3 'neg' modifier.
// Currently this modifier is allowed in the following context:
//
// 1. Before a register, e.g. "-v0", "-v[...]" or "-[v0,v1]".
// 2. Before an 'abs' modifier: -abs(...)
// 3. Before an SP3 'abs' modifier: -|...|
//
// In all other cases "-" is handled as a part
// of an expression that follows the sign.
//
// Note: When "-" is followed by an integer literal,
// this is interpreted as integer negation rather
// than a floating-point NEG modifier applied to N.
// Beside being contr-intuitive, such use of floating-point
// NEG modifier would have resulted in different meaning
// of integer literals used with VOP1/2/C and VOP3,
// for example:
//    v_exp_f32_e32 v5, -1 // VOP1: src0 = 0xFFFFFFFF
//    v_exp_f32_e64 v5, -1 // VOP3: src0 = 0x80000001
// Negative fp literals with preceding "-" are
// handled likewise for unifomtity
//
bool
AMDGPUAsmParser::parseSP3NegModifier() {

  AsmToken NextToken[2];
  peekTokens(NextToken);

  if (isToken(AsmToken::Minus) &&
      (isRegister(NextToken[0], NextToken[1]) ||
       NextToken[0].is(AsmToken::Pipe) ||
       isId(NextToken[0], "abs"))) {
    lex();
    return true;
  }

  return false;
}

OperandMatchResultTy
AMDGPUAsmParser::parseRegOrImmWithFPInputMods(OperandVector &Operands,
                                              bool AllowImm) {
  bool Neg, SP3Neg;
  bool Abs, SP3Abs;
  SMLoc Loc;

  // Disable ambiguous constructs like '--1' etc. Should use neg(-1) instead.
  if (isToken(AsmToken::Minus) && peekToken().is(AsmToken::Minus)) {
    Error(getLoc(), "invalid syntax, expected 'neg' modifier");
    return MatchOperand_ParseFail;
  }

  SP3Neg = parseSP3NegModifier();

  Loc = getLoc();
  Neg = trySkipId("neg");
  if (Neg && SP3Neg) {
    Error(Loc, "expected register or immediate");
    return MatchOperand_ParseFail;
  }
  if (Neg && !skipToken(AsmToken::LParen, "expected left paren after neg"))
    return MatchOperand_ParseFail;

  Abs = trySkipId("abs");
  if (Abs && !skipToken(AsmToken::LParen, "expected left paren after abs"))
    return MatchOperand_ParseFail;

  Loc = getLoc();
  SP3Abs = trySkipToken(AsmToken::Pipe);
  if (Abs && SP3Abs) {
    Error(Loc, "expected register or immediate");
    return MatchOperand_ParseFail;
  }

  OperandMatchResultTy Res;
  if (AllowImm) {
    Res = parseRegOrImm(Operands, SP3Abs);
  } else {
    Res = parseReg(Operands);
  }
  if (Res != MatchOperand_Success) {
    return (SP3Neg || Neg || SP3Abs || Abs)? MatchOperand_ParseFail : Res;
  }

  if (SP3Abs && !skipToken(AsmToken::Pipe, "expected vertical bar"))
    return MatchOperand_ParseFail;
  if (Abs && !skipToken(AsmToken::RParen, "expected closing parentheses"))
    return MatchOperand_ParseFail;
  if (Neg && !skipToken(AsmToken::RParen, "expected closing parentheses"))
    return MatchOperand_ParseFail;

  AMDGPUOperand::Modifiers Mods;
  Mods.Abs = Abs || SP3Abs;
  Mods.Neg = Neg || SP3Neg;

  if (Mods.hasFPModifiers()) {
    AMDGPUOperand &Op = static_cast<AMDGPUOperand &>(*Operands.back());
    if (Op.isExpr()) {
      Error(Op.getStartLoc(), "expected an absolute expression");
      return MatchOperand_ParseFail;
    }
    Op.setModifiers(Mods);
  }
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseRegOrImmWithIntInputMods(OperandVector &Operands,
                                               bool AllowImm) {
  bool Sext = trySkipId("sext");
  if (Sext && !skipToken(AsmToken::LParen, "expected left paren after sext"))
    return MatchOperand_ParseFail;

  OperandMatchResultTy Res;
  if (AllowImm) {
    Res = parseRegOrImm(Operands);
  } else {
    Res = parseReg(Operands);
  }
  if (Res != MatchOperand_Success) {
    return Sext? MatchOperand_ParseFail : Res;
  }

  if (Sext && !skipToken(AsmToken::RParen, "expected closing parentheses"))
    return MatchOperand_ParseFail;

  AMDGPUOperand::Modifiers Mods;
  Mods.Sext = Sext;

  if (Mods.hasIntModifiers()) {
    AMDGPUOperand &Op = static_cast<AMDGPUOperand &>(*Operands.back());
    if (Op.isExpr()) {
      Error(Op.getStartLoc(), "expected an absolute expression");
      return MatchOperand_ParseFail;
    }
    Op.setModifiers(Mods);
  }

  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseRegWithFPInputMods(OperandVector &Operands) {
  return parseRegOrImmWithFPInputMods(Operands, false);
}

OperandMatchResultTy
AMDGPUAsmParser::parseRegWithIntInputMods(OperandVector &Operands) {
  return parseRegOrImmWithIntInputMods(Operands, false);
}

OperandMatchResultTy AMDGPUAsmParser::parseVReg32OrOff(OperandVector &Operands) {
  auto Loc = getLoc();
  if (trySkipId("off")) {
    Operands.push_back(AMDGPUOperand::CreateImm(this, 0, Loc,
                                                AMDGPUOperand::ImmTyOff, false));
    return MatchOperand_Success;
  }

  if (!isRegister())
    return MatchOperand_NoMatch;

  std::unique_ptr<AMDGPUOperand> Reg = parseRegister();
  if (Reg) {
    Operands.push_back(std::move(Reg));
    return MatchOperand_Success;
  }

  return MatchOperand_ParseFail;

}

unsigned AMDGPUAsmParser::checkTargetMatchPredicate(MCInst &Inst) {
  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;

  if ((getForcedEncodingSize() == 32 && (TSFlags & SIInstrFlags::VOP3)) ||
      (getForcedEncodingSize() == 64 && !(TSFlags & SIInstrFlags::VOP3)) ||
      (isForcedDPP() && !(TSFlags & SIInstrFlags::DPP)) ||
      (isForcedSDWA() && !(TSFlags & SIInstrFlags::SDWA)) )
    return Match_InvalidOperand;

  if ((TSFlags & SIInstrFlags::VOP3) &&
      (TSFlags & SIInstrFlags::VOPAsmPrefer32Bit) &&
      getForcedEncodingSize() != 64)
    return Match_PreferE32;

  if (Inst.getOpcode() == AMDGPU::V_MAC_F32_sdwa_vi ||
      Inst.getOpcode() == AMDGPU::V_MAC_F16_sdwa_vi) {
    // v_mac_f32/16 allow only dst_sel == DWORD;
    auto OpNum =
        AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::dst_sel);
    const auto &Op = Inst.getOperand(OpNum);
    if (!Op.isImm() || Op.getImm() != AMDGPU::SDWA::SdwaSel::DWORD) {
      return Match_InvalidOperand;
    }
  }

  return Match_Success;
}

// What asm variants we should check
ArrayRef<unsigned> AMDGPUAsmParser::getMatchedVariants() const {
  if (getForcedEncodingSize() == 32) {
    static const unsigned Variants[] = {AMDGPUAsmVariants::DEFAULT};
    return makeArrayRef(Variants);
  }

  if (isForcedVOP3()) {
    static const unsigned Variants[] = {AMDGPUAsmVariants::VOP3};
    return makeArrayRef(Variants);
  }

  if (isForcedSDWA()) {
    static const unsigned Variants[] = {AMDGPUAsmVariants::SDWA,
                                        AMDGPUAsmVariants::SDWA9};
    return makeArrayRef(Variants);
  }

  if (isForcedDPP()) {
    static const unsigned Variants[] = {AMDGPUAsmVariants::DPP};
    return makeArrayRef(Variants);
  }

  static const unsigned Variants[] = {
    AMDGPUAsmVariants::DEFAULT, AMDGPUAsmVariants::VOP3,
    AMDGPUAsmVariants::SDWA, AMDGPUAsmVariants::SDWA9, AMDGPUAsmVariants::DPP
  };

  return makeArrayRef(Variants);
}

unsigned AMDGPUAsmParser::findImplicitSGPRReadInVOP(const MCInst &Inst) const {
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  const unsigned Num = Desc.getNumImplicitUses();
  for (unsigned i = 0; i < Num; ++i) {
    unsigned Reg = Desc.ImplicitUses[i];
    switch (Reg) {
    case AMDGPU::FLAT_SCR:
    case AMDGPU::VCC:
    case AMDGPU::VCC_LO:
    case AMDGPU::VCC_HI:
    case AMDGPU::M0:
      return Reg;
    default:
      break;
    }
  }
  return AMDGPU::NoRegister;
}

// NB: This code is correct only when used to check constant
// bus limitations because GFX7 support no f16 inline constants.
// Note that there are no cases when a GFX7 opcode violates
// constant bus limitations due to the use of an f16 constant.
bool AMDGPUAsmParser::isInlineConstant(const MCInst &Inst,
                                       unsigned OpIdx) const {
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());

  if (!AMDGPU::isSISrcOperand(Desc, OpIdx)) {
    return false;
  }

  const MCOperand &MO = Inst.getOperand(OpIdx);

  int64_t Val = MO.getImm();
  auto OpSize = AMDGPU::getOperandSize(Desc, OpIdx);

  switch (OpSize) { // expected operand size
  case 8:
    return AMDGPU::isInlinableLiteral64(Val, hasInv2PiInlineImm());
  case 4:
    return AMDGPU::isInlinableLiteral32(Val, hasInv2PiInlineImm());
  case 2: {
    const unsigned OperandType = Desc.OpInfo[OpIdx].OperandType;
    if (OperandType == AMDGPU::OPERAND_REG_IMM_INT16 ||
        OperandType == AMDGPU::OPERAND_REG_INLINE_C_INT16 ||
        OperandType == AMDGPU::OPERAND_REG_INLINE_AC_INT16)
      return AMDGPU::isInlinableIntLiteral(Val);

    if (OperandType == AMDGPU::OPERAND_REG_INLINE_C_V2INT16 ||
        OperandType == AMDGPU::OPERAND_REG_INLINE_AC_V2INT16 ||
        OperandType == AMDGPU::OPERAND_REG_IMM_V2INT16)
      return AMDGPU::isInlinableIntLiteralV216(Val);

    if (OperandType == AMDGPU::OPERAND_REG_INLINE_C_V2FP16 ||
        OperandType == AMDGPU::OPERAND_REG_INLINE_AC_V2FP16 ||
        OperandType == AMDGPU::OPERAND_REG_IMM_V2FP16)
      return AMDGPU::isInlinableLiteralV216(Val, hasInv2PiInlineImm());

    return AMDGPU::isInlinableLiteral16(Val, hasInv2PiInlineImm());
  }
  default:
    llvm_unreachable("invalid operand size");
  }
}

unsigned AMDGPUAsmParser::getConstantBusLimit(unsigned Opcode) const {
  if (!isGFX10())
    return 1;

  switch (Opcode) {
  // 64-bit shift instructions can use only one scalar value input
  case AMDGPU::V_LSHLREV_B64:
  case AMDGPU::V_LSHLREV_B64_gfx10:
  case AMDGPU::V_LSHL_B64:
  case AMDGPU::V_LSHRREV_B64:
  case AMDGPU::V_LSHRREV_B64_gfx10:
  case AMDGPU::V_LSHR_B64:
  case AMDGPU::V_ASHRREV_I64:
  case AMDGPU::V_ASHRREV_I64_gfx10:
  case AMDGPU::V_ASHR_I64:
    return 1;
  default:
    return 2;
  }
}

bool AMDGPUAsmParser::usesConstantBus(const MCInst &Inst, unsigned OpIdx) {
  const MCOperand &MO = Inst.getOperand(OpIdx);
  if (MO.isImm()) {
    return !isInlineConstant(Inst, OpIdx);
  } else if (MO.isReg()) {
    auto Reg = MO.getReg();
    const MCRegisterInfo *TRI = getContext().getRegisterInfo();
    return isSGPR(mc2PseudoReg(Reg), TRI) && Reg != SGPR_NULL;
  } else {
    return true;
  }
}

bool AMDGPUAsmParser::validateConstantBusLimitations(const MCInst &Inst) {
  const unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opcode);
  unsigned ConstantBusUseCount = 0;
  unsigned NumLiterals = 0;
  unsigned LiteralSize;

  if (Desc.TSFlags &
      (SIInstrFlags::VOPC |
       SIInstrFlags::VOP1 | SIInstrFlags::VOP2 |
       SIInstrFlags::VOP3 | SIInstrFlags::VOP3P |
       SIInstrFlags::SDWA)) {
    // Check special imm operands (used by madmk, etc)
    if (AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::imm) != -1) {
      ++ConstantBusUseCount;
    }

    SmallDenseSet<unsigned> SGPRsUsed;
    unsigned SGPRUsed = findImplicitSGPRReadInVOP(Inst);
    if (SGPRUsed != AMDGPU::NoRegister) {
      SGPRsUsed.insert(SGPRUsed);
      ++ConstantBusUseCount;
    }

    const int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
    const int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
    const int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

    const int OpIndices[] = { Src0Idx, Src1Idx, Src2Idx };

    for (int OpIdx : OpIndices) {
      if (OpIdx == -1) break;

      const MCOperand &MO = Inst.getOperand(OpIdx);
      if (usesConstantBus(Inst, OpIdx)) {
        if (MO.isReg()) {
          const unsigned Reg = mc2PseudoReg(MO.getReg());
          // Pairs of registers with a partial intersections like these
          //   s0, s[0:1]
          //   flat_scratch_lo, flat_scratch
          //   flat_scratch_lo, flat_scratch_hi
          // are theoretically valid but they are disabled anyway.
          // Note that this code mimics SIInstrInfo::verifyInstruction
          if (!SGPRsUsed.count(Reg)) {
            SGPRsUsed.insert(Reg);
            ++ConstantBusUseCount;
          }
        } else { // Expression or a literal

          if (Desc.OpInfo[OpIdx].OperandType == MCOI::OPERAND_IMMEDIATE)
            continue; // special operand like VINTERP attr_chan

          // An instruction may use only one literal.
          // This has been validated on the previous step.
          // See validateVOP3Literal.
          // This literal may be used as more than one operand.
          // If all these operands are of the same size,
          // this literal counts as one scalar value.
          // Otherwise it counts as 2 scalar values.
          // See "GFX10 Shader Programming", section 3.6.2.3.

          unsigned Size = AMDGPU::getOperandSize(Desc, OpIdx);
          if (Size < 4) Size = 4;

          if (NumLiterals == 0) {
            NumLiterals = 1;
            LiteralSize = Size;
          } else if (LiteralSize != Size) {
            NumLiterals = 2;
          }
        }
      }
    }
  }
  ConstantBusUseCount += NumLiterals;

  return ConstantBusUseCount <= getConstantBusLimit(Opcode);
}

bool AMDGPUAsmParser::validateEarlyClobberLimitations(const MCInst &Inst) {
  const unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opcode);

  const int DstIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::vdst);
  if (DstIdx == -1 ||
      Desc.getOperandConstraint(DstIdx, MCOI::EARLY_CLOBBER) == -1) {
    return true;
  }

  const MCRegisterInfo *TRI = getContext().getRegisterInfo();

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  const int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  const int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  assert(DstIdx != -1);
  const MCOperand &Dst = Inst.getOperand(DstIdx);
  assert(Dst.isReg());
  const unsigned DstReg = mc2PseudoReg(Dst.getReg());

  const int SrcIndices[] = { Src0Idx, Src1Idx, Src2Idx };

  for (int SrcIdx : SrcIndices) {
    if (SrcIdx == -1) break;
    const MCOperand &Src = Inst.getOperand(SrcIdx);
    if (Src.isReg()) {
      const unsigned SrcReg = mc2PseudoReg(Src.getReg());
      if (isRegIntersect(DstReg, SrcReg, TRI)) {
        return false;
      }
    }
  }

  return true;
}

bool AMDGPUAsmParser::validateIntClampSupported(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::IntClamp) != 0 && !hasIntClamp()) {
    int ClampIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::clamp);
    assert(ClampIdx != -1);
    return Inst.getOperand(ClampIdx).getImm() == 0;
  }

  return true;
}

bool AMDGPUAsmParser::validateMIMGDataSize(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::MIMG) == 0)
    return true;

  int VDataIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdata);
  int DMaskIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dmask);
  int TFEIdx   = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::tfe);

  assert(VDataIdx != -1);
  assert(DMaskIdx != -1);
  assert(TFEIdx != -1);

  unsigned VDataSize = AMDGPU::getRegOperandSize(getMRI(), Desc, VDataIdx);
  unsigned TFESize = Inst.getOperand(TFEIdx).getImm()? 1 : 0;
  unsigned DMask = Inst.getOperand(DMaskIdx).getImm() & 0xf;
  if (DMask == 0)
    DMask = 1;

  unsigned DataSize =
    (Desc.TSFlags & SIInstrFlags::Gather4) ? 4 : countPopulation(DMask);
  if (hasPackedD16()) {
    int D16Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::d16);
    if (D16Idx >= 0 && Inst.getOperand(D16Idx).getImm())
      DataSize = (DataSize + 1) / 2;
  }

  return (VDataSize / 4) == DataSize + TFESize;
}

bool AMDGPUAsmParser::validateMIMGAddrSize(const MCInst &Inst) {
  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::MIMG) == 0 || !isGFX10())
    return true;

  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Opc);
  const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
  int VAddr0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr0);
  int SrsrcIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::srsrc);
  int DimIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dim);

  assert(VAddr0Idx != -1);
  assert(SrsrcIdx != -1);
  assert(DimIdx != -1);
  assert(SrsrcIdx > VAddr0Idx);

  unsigned Dim = Inst.getOperand(DimIdx).getImm();
  const AMDGPU::MIMGDimInfo *DimInfo = AMDGPU::getMIMGDimInfoByEncoding(Dim);
  bool IsNSA = SrsrcIdx - VAddr0Idx > 1;
  unsigned VAddrSize =
      IsNSA ? SrsrcIdx - VAddr0Idx
            : AMDGPU::getRegOperandSize(getMRI(), Desc, VAddr0Idx) / 4;

  unsigned AddrSize = BaseOpcode->NumExtraArgs +
                      (BaseOpcode->Gradients ? DimInfo->NumGradients : 0) +
                      (BaseOpcode->Coordinates ? DimInfo->NumCoords : 0) +
                      (BaseOpcode->LodOrClampOrMip ? 1 : 0);
  if (!IsNSA) {
    if (AddrSize > 8)
      AddrSize = 16;
    else if (AddrSize > 4)
      AddrSize = 8;
  }

  return VAddrSize == AddrSize;
}

bool AMDGPUAsmParser::validateMIMGAtomicDMask(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::MIMG) == 0)
    return true;
  if (!Desc.mayLoad() || !Desc.mayStore())
    return true; // Not atomic

  int DMaskIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dmask);
  unsigned DMask = Inst.getOperand(DMaskIdx).getImm() & 0xf;

  // This is an incomplete check because image_atomic_cmpswap
  // may only use 0x3 and 0xf while other atomic operations
  // may use 0x1 and 0x3. However these limitations are
  // verified when we check that dmask matches dst size.
  return DMask == 0x1 || DMask == 0x3 || DMask == 0xf;
}

bool AMDGPUAsmParser::validateMIMGGatherDMask(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::Gather4) == 0)
    return true;

  int DMaskIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dmask);
  unsigned DMask = Inst.getOperand(DMaskIdx).getImm() & 0xf;

  // GATHER4 instructions use dmask in a different fashion compared to
  // other MIMG instructions. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  return DMask == 0x1 || DMask == 0x2 || DMask == 0x4 || DMask == 0x8;
}

static bool IsMovrelsSDWAOpcode(const unsigned Opcode)
{
  switch (Opcode) {
  case AMDGPU::V_MOVRELS_B32_sdwa_gfx10:
  case AMDGPU::V_MOVRELSD_B32_sdwa_gfx10:
  case AMDGPU::V_MOVRELSD_2_B32_sdwa_gfx10:
    return true;
  default:
    return false;
  }
}

// movrels* opcodes should only allow VGPRS as src0.
// This is specified in .td description for vop1/vop3,
// but sdwa is handled differently. See isSDWAOperand.
bool AMDGPUAsmParser::validateMovrels(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::SDWA) == 0 || !IsMovrelsSDWAOpcode(Opc))
    return true;

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  assert(Src0Idx != -1);

  const MCOperand &Src0 = Inst.getOperand(Src0Idx);
  if (!Src0.isReg())
    return false;

  auto Reg = Src0.getReg();
  const MCRegisterInfo *TRI = getContext().getRegisterInfo();
  return !isSGPR(mc2PseudoReg(Reg), TRI);
}

bool AMDGPUAsmParser::validateMAIAccWrite(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();

  if (Opc != AMDGPU::V_ACCVGPR_WRITE_B32_vi)
    return true;

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  assert(Src0Idx != -1);

  const MCOperand &Src0 = Inst.getOperand(Src0Idx);
  if (!Src0.isReg())
    return true;

  auto Reg = Src0.getReg();
  const MCRegisterInfo *TRI = getContext().getRegisterInfo();
  if (isSGPR(mc2PseudoReg(Reg), TRI)) {
    Error(getLoc(), "source operand must be either a VGPR or an inline constant");
    return false;
  }

  return true;
}

bool AMDGPUAsmParser::validateMIMGD16(const MCInst &Inst) {

  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::MIMG) == 0)
    return true;

  int D16Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::d16);
  if (D16Idx >= 0 && Inst.getOperand(D16Idx).getImm()) {
    if (isCI() || isSI())
      return false;
  }

  return true;
}

bool AMDGPUAsmParser::validateMIMGDim(const MCInst &Inst) {
  const unsigned Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  if ((Desc.TSFlags & SIInstrFlags::MIMG) == 0)
    return true;

  int DimIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::dim);
  if (DimIdx < 0)
    return true;

  long Imm = Inst.getOperand(DimIdx).getImm();
  if (Imm < 0 || Imm >= 8)
    return false;

  return true;
}

static bool IsRevOpcode(const unsigned Opcode)
{
  switch (Opcode) {
  case AMDGPU::V_SUBREV_F32_e32:
  case AMDGPU::V_SUBREV_F32_e64:
  case AMDGPU::V_SUBREV_F32_e32_gfx10:
  case AMDGPU::V_SUBREV_F32_e32_gfx6_gfx7:
  case AMDGPU::V_SUBREV_F32_e32_vi:
  case AMDGPU::V_SUBREV_F32_e64_gfx10:
  case AMDGPU::V_SUBREV_F32_e64_gfx6_gfx7:
  case AMDGPU::V_SUBREV_F32_e64_vi:

  case AMDGPU::V_SUBREV_CO_U32_e32:
  case AMDGPU::V_SUBREV_CO_U32_e64:
  case AMDGPU::V_SUBREV_I32_e32_gfx6_gfx7:
  case AMDGPU::V_SUBREV_I32_e64_gfx6_gfx7:

  case AMDGPU::V_SUBBREV_U32_e32:
  case AMDGPU::V_SUBBREV_U32_e64:
  case AMDGPU::V_SUBBREV_U32_e32_gfx6_gfx7:
  case AMDGPU::V_SUBBREV_U32_e32_vi:
  case AMDGPU::V_SUBBREV_U32_e64_gfx6_gfx7:
  case AMDGPU::V_SUBBREV_U32_e64_vi:

  case AMDGPU::V_SUBREV_U32_e32:
  case AMDGPU::V_SUBREV_U32_e64:
  case AMDGPU::V_SUBREV_U32_e32_gfx9:
  case AMDGPU::V_SUBREV_U32_e32_vi:
  case AMDGPU::V_SUBREV_U32_e64_gfx9:
  case AMDGPU::V_SUBREV_U32_e64_vi:

  case AMDGPU::V_SUBREV_F16_e32:
  case AMDGPU::V_SUBREV_F16_e64:
  case AMDGPU::V_SUBREV_F16_e32_gfx10:
  case AMDGPU::V_SUBREV_F16_e32_vi:
  case AMDGPU::V_SUBREV_F16_e64_gfx10:
  case AMDGPU::V_SUBREV_F16_e64_vi:

  case AMDGPU::V_SUBREV_U16_e32:
  case AMDGPU::V_SUBREV_U16_e64:
  case AMDGPU::V_SUBREV_U16_e32_vi:
  case AMDGPU::V_SUBREV_U16_e64_vi:

  case AMDGPU::V_SUBREV_CO_U32_e32_gfx9:
  case AMDGPU::V_SUBREV_CO_U32_e64_gfx10:
  case AMDGPU::V_SUBREV_CO_U32_e64_gfx9:

  case AMDGPU::V_SUBBREV_CO_U32_e32_gfx9:
  case AMDGPU::V_SUBBREV_CO_U32_e64_gfx9:

  case AMDGPU::V_SUBREV_NC_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_NC_U32_e64_gfx10:

  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_e64_gfx10:

  case AMDGPU::V_LSHRREV_B32_e32:
  case AMDGPU::V_LSHRREV_B32_e64:
  case AMDGPU::V_LSHRREV_B32_e32_gfx6_gfx7:
  case AMDGPU::V_LSHRREV_B32_e64_gfx6_gfx7:
  case AMDGPU::V_LSHRREV_B32_e32_vi:
  case AMDGPU::V_LSHRREV_B32_e64_vi:
  case AMDGPU::V_LSHRREV_B32_e32_gfx10:
  case AMDGPU::V_LSHRREV_B32_e64_gfx10:

  case AMDGPU::V_ASHRREV_I32_e32:
  case AMDGPU::V_ASHRREV_I32_e64:
  case AMDGPU::V_ASHRREV_I32_e32_gfx10:
  case AMDGPU::V_ASHRREV_I32_e32_gfx6_gfx7:
  case AMDGPU::V_ASHRREV_I32_e32_vi:
  case AMDGPU::V_ASHRREV_I32_e64_gfx10:
  case AMDGPU::V_ASHRREV_I32_e64_gfx6_gfx7:
  case AMDGPU::V_ASHRREV_I32_e64_vi:

  case AMDGPU::V_LSHLREV_B32_e32:
  case AMDGPU::V_LSHLREV_B32_e64:
  case AMDGPU::V_LSHLREV_B32_e32_gfx10:
  case AMDGPU::V_LSHLREV_B32_e32_gfx6_gfx7:
  case AMDGPU::V_LSHLREV_B32_e32_vi:
  case AMDGPU::V_LSHLREV_B32_e64_gfx10:
  case AMDGPU::V_LSHLREV_B32_e64_gfx6_gfx7:
  case AMDGPU::V_LSHLREV_B32_e64_vi:

  case AMDGPU::V_LSHLREV_B16_e32:
  case AMDGPU::V_LSHLREV_B16_e64:
  case AMDGPU::V_LSHLREV_B16_e32_vi:
  case AMDGPU::V_LSHLREV_B16_e64_vi:
  case AMDGPU::V_LSHLREV_B16_gfx10:

  case AMDGPU::V_LSHRREV_B16_e32:
  case AMDGPU::V_LSHRREV_B16_e64:
  case AMDGPU::V_LSHRREV_B16_e32_vi:
  case AMDGPU::V_LSHRREV_B16_e64_vi:
  case AMDGPU::V_LSHRREV_B16_gfx10:

  case AMDGPU::V_ASHRREV_I16_e32:
  case AMDGPU::V_ASHRREV_I16_e64:
  case AMDGPU::V_ASHRREV_I16_e32_vi:
  case AMDGPU::V_ASHRREV_I16_e64_vi:
  case AMDGPU::V_ASHRREV_I16_gfx10:

  case AMDGPU::V_LSHLREV_B64:
  case AMDGPU::V_LSHLREV_B64_gfx10:
  case AMDGPU::V_LSHLREV_B64_vi:

  case AMDGPU::V_LSHRREV_B64:
  case AMDGPU::V_LSHRREV_B64_gfx10:
  case AMDGPU::V_LSHRREV_B64_vi:

  case AMDGPU::V_ASHRREV_I64:
  case AMDGPU::V_ASHRREV_I64_gfx10:
  case AMDGPU::V_ASHRREV_I64_vi:

  case AMDGPU::V_PK_LSHLREV_B16:
  case AMDGPU::V_PK_LSHLREV_B16_gfx10:
  case AMDGPU::V_PK_LSHLREV_B16_vi:

  case AMDGPU::V_PK_LSHRREV_B16:
  case AMDGPU::V_PK_LSHRREV_B16_gfx10:
  case AMDGPU::V_PK_LSHRREV_B16_vi:
  case AMDGPU::V_PK_ASHRREV_I16:
  case AMDGPU::V_PK_ASHRREV_I16_gfx10:
  case AMDGPU::V_PK_ASHRREV_I16_vi:
    return true;
  default:
    return false;
  }
}

bool AMDGPUAsmParser::validateLdsDirect(const MCInst &Inst) {

  using namespace SIInstrFlags;
  const unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opcode);

  // lds_direct register is defined so that it can be used
  // with 9-bit operands only. Ignore encodings which do not accept these.
  if ((Desc.TSFlags & (VOP1 | VOP2 | VOP3 | VOPC | VOP3P | SIInstrFlags::SDWA)) == 0)
    return true;

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  const int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  const int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  const int SrcIndices[] = { Src1Idx, Src2Idx };

  // lds_direct cannot be specified as either src1 or src2.
  for (int SrcIdx : SrcIndices) {
    if (SrcIdx == -1) break;
    const MCOperand &Src = Inst.getOperand(SrcIdx);
    if (Src.isReg() && Src.getReg() == LDS_DIRECT) {
      return false;
    }
  }

  if (Src0Idx == -1)
    return true;

  const MCOperand &Src = Inst.getOperand(Src0Idx);
  if (!Src.isReg() || Src.getReg() != LDS_DIRECT)
    return true;

  // lds_direct is specified as src0. Check additional limitations.
  return (Desc.TSFlags & SIInstrFlags::SDWA) == 0 && !IsRevOpcode(Opcode);
}

SMLoc AMDGPUAsmParser::getFlatOffsetLoc(const OperandVector &Operands) const {
  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);
    if (Op.isFlatOffset())
      return Op.getStartLoc();
  }
  return getLoc();
}

bool AMDGPUAsmParser::validateFlatOffset(const MCInst &Inst,
                                         const OperandVector &Operands) {
  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;
  if ((TSFlags & SIInstrFlags::FLAT) == 0)
    return true;

  auto Opcode = Inst.getOpcode();
  auto OpNum = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::offset);
  assert(OpNum != -1);

  const auto &Op = Inst.getOperand(OpNum);
  if (!hasFlatOffsets() && Op.getImm() != 0) {
    Error(getFlatOffsetLoc(Operands),
          "flat offset modifier is not supported on this GPU");
    return false;
  }

  // Address offset is 12-bit signed for GFX10, 13-bit for GFX9.
  // For FLAT segment the offset must be positive;
  // MSB is ignored and forced to zero.
  unsigned OffsetSize = isGFX9() ? 13 : 12;
  if (TSFlags & SIInstrFlags::IsNonFlatSeg) {
    if (!isIntN(OffsetSize, Op.getImm())) {
      Error(getFlatOffsetLoc(Operands),
            isGFX9() ? "expected a 13-bit signed offset" :
                       "expected a 12-bit signed offset");
      return false;
    }
  } else {
    if (!isUIntN(OffsetSize - 1, Op.getImm())) {
      Error(getFlatOffsetLoc(Operands),
            isGFX9() ? "expected a 12-bit unsigned offset" :
                       "expected an 11-bit unsigned offset");
      return false;
    }
  }

  return true;
}

SMLoc AMDGPUAsmParser::getSMEMOffsetLoc(const OperandVector &Operands) const {
  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);
    if (Op.isSMEMOffset())
      return Op.getStartLoc();
  }
  return getLoc();
}

bool AMDGPUAsmParser::validateSMEMOffset(const MCInst &Inst,
                                         const OperandVector &Operands) {
  if (isCI() || isSI())
    return true;

  uint64_t TSFlags = MII.get(Inst.getOpcode()).TSFlags;
  if ((TSFlags & SIInstrFlags::SMRD) == 0)
    return true;

  auto Opcode = Inst.getOpcode();
  auto OpNum = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::offset);
  if (OpNum == -1)
    return true;

  const auto &Op = Inst.getOperand(OpNum);
  if (!Op.isImm())
    return true;

  uint64_t Offset = Op.getImm();
  bool IsBuffer = AMDGPU::getSMEMIsBuffer(Opcode);
  if (AMDGPU::isLegalSMRDEncodedUnsignedOffset(getSTI(), Offset) ||
      AMDGPU::isLegalSMRDEncodedSignedOffset(getSTI(), Offset, IsBuffer))
    return true;

  Error(getSMEMOffsetLoc(Operands),
        (isVI() || IsBuffer) ? "expected a 20-bit unsigned offset" :
                               "expected a 21-bit signed offset");

  return false;
}

bool AMDGPUAsmParser::validateSOPLiteral(const MCInst &Inst) const {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opcode);
  if (!(Desc.TSFlags & (SIInstrFlags::SOP2 | SIInstrFlags::SOPC)))
    return true;

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  const int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);

  const int OpIndices[] = { Src0Idx, Src1Idx };

  unsigned NumExprs = 0;
  unsigned NumLiterals = 0;
  uint32_t LiteralValue;

  for (int OpIdx : OpIndices) {
    if (OpIdx == -1) break;

    const MCOperand &MO = Inst.getOperand(OpIdx);
    // Exclude special imm operands (like that used by s_set_gpr_idx_on)
    if (AMDGPU::isSISrcOperand(Desc, OpIdx)) {
      if (MO.isImm() && !isInlineConstant(Inst, OpIdx)) {
        uint32_t Value = static_cast<uint32_t>(MO.getImm());
        if (NumLiterals == 0 || LiteralValue != Value) {
          LiteralValue = Value;
          ++NumLiterals;
        }
      } else if (MO.isExpr()) {
        ++NumExprs;
      }
    }
  }

  return NumLiterals + NumExprs <= 1;
}

bool AMDGPUAsmParser::validateOpSel(const MCInst &Inst) {
  const unsigned Opc = Inst.getOpcode();
  if (Opc == AMDGPU::V_PERMLANE16_B32_gfx10 ||
      Opc == AMDGPU::V_PERMLANEX16_B32_gfx10) {
    int OpSelIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel);
    unsigned OpSel = Inst.getOperand(OpSelIdx).getImm();

    if (OpSel & ~3)
      return false;
  }
  return true;
}

// Check if VCC register matches wavefront size
bool AMDGPUAsmParser::validateVccOperand(unsigned Reg) const {
  auto FB = getFeatureBits();
  return (FB[AMDGPU::FeatureWavefrontSize64] && Reg == AMDGPU::VCC) ||
    (FB[AMDGPU::FeatureWavefrontSize32] && Reg == AMDGPU::VCC_LO);
}

// VOP3 literal is only allowed in GFX10+ and only one can be used
bool AMDGPUAsmParser::validateVOP3Literal(const MCInst &Inst) const {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opcode);
  if (!(Desc.TSFlags & (SIInstrFlags::VOP3 | SIInstrFlags::VOP3P)))
    return true;

  const int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  const int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  const int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  const int OpIndices[] = { Src0Idx, Src1Idx, Src2Idx };

  unsigned NumExprs = 0;
  unsigned NumLiterals = 0;
  uint32_t LiteralValue;

  for (int OpIdx : OpIndices) {
    if (OpIdx == -1) break;

    const MCOperand &MO = Inst.getOperand(OpIdx);
    if (!MO.isImm() && !MO.isExpr())
      continue;
    if (!AMDGPU::isSISrcOperand(Desc, OpIdx))
      continue;

    if (OpIdx == Src2Idx && (Desc.TSFlags & SIInstrFlags::IsMAI) &&
        getFeatureBits()[AMDGPU::FeatureMFMAInlineLiteralBug])
      return false;

    if (MO.isImm() && !isInlineConstant(Inst, OpIdx)) {
      uint32_t Value = static_cast<uint32_t>(MO.getImm());
      if (NumLiterals == 0 || LiteralValue != Value) {
        LiteralValue = Value;
        ++NumLiterals;
      }
    } else if (MO.isExpr()) {
      ++NumExprs;
    }
  }
  NumLiterals += NumExprs;

  return !NumLiterals ||
         (NumLiterals == 1 && getFeatureBits()[AMDGPU::FeatureVOP3Literal]);
}

bool AMDGPUAsmParser::validateInstruction(const MCInst &Inst,
                                          const SMLoc &IDLoc,
                                          const OperandVector &Operands) {
  if (!validateLdsDirect(Inst)) {
    Error(IDLoc,
      "invalid use of lds_direct");
    return false;
  }
  if (!validateSOPLiteral(Inst)) {
    Error(IDLoc,
      "only one literal operand is allowed");
    return false;
  }
  if (!validateVOP3Literal(Inst)) {
    Error(IDLoc,
      "invalid literal operand");
    return false;
  }
  if (!validateConstantBusLimitations(Inst)) {
    Error(IDLoc,
      "invalid operand (violates constant bus restrictions)");
    return false;
  }
  if (!validateEarlyClobberLimitations(Inst)) {
    Error(IDLoc,
      "destination must be different than all sources");
    return false;
  }
  if (!validateIntClampSupported(Inst)) {
    Error(IDLoc,
      "integer clamping is not supported on this GPU");
    return false;
  }
  if (!validateOpSel(Inst)) {
    Error(IDLoc,
      "invalid op_sel operand");
    return false;
  }
  // For MUBUF/MTBUF d16 is a part of opcode, so there is nothing to validate.
  if (!validateMIMGD16(Inst)) {
    Error(IDLoc,
      "d16 modifier is not supported on this GPU");
    return false;
  }
  if (!validateMIMGDim(Inst)) {
    Error(IDLoc, "dim modifier is required on this GPU");
    return false;
  }
  if (!validateMIMGDataSize(Inst)) {
    Error(IDLoc,
      "image data size does not match dmask and tfe");
    return false;
  }
  if (!validateMIMGAddrSize(Inst)) {
    Error(IDLoc,
      "image address size does not match dim and a16");
    return false;
  }
  if (!validateMIMGAtomicDMask(Inst)) {
    Error(IDLoc,
      "invalid atomic image dmask");
    return false;
  }
  if (!validateMIMGGatherDMask(Inst)) {
    Error(IDLoc,
      "invalid image_gather dmask: only one bit must be set");
    return false;
  }
  if (!validateMovrels(Inst)) {
    Error(IDLoc, "source operand must be a VGPR");
    return false;
  }
  if (!validateFlatOffset(Inst, Operands)) {
    return false;
  }
  if (!validateSMEMOffset(Inst, Operands)) {
    return false;
  }
  if (!validateMAIAccWrite(Inst)) {
    return false;
  }

  return true;
}

static std::string AMDGPUMnemonicSpellCheck(StringRef S,
                                            const FeatureBitset &FBS,
                                            unsigned VariantID = 0);

bool AMDGPUAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned Result = Match_Success;
  for (auto Variant : getMatchedVariants()) {
    uint64_t EI;
    auto R = MatchInstructionImpl(Operands, Inst, EI, MatchingInlineAsm,
                                  Variant);
    // We order match statuses from least to most specific. We use most specific
    // status as resulting
    // Match_MnemonicFail < Match_InvalidOperand < Match_MissingFeature < Match_PreferE32
    if ((R == Match_Success) ||
        (R == Match_PreferE32) ||
        (R == Match_MissingFeature && Result != Match_PreferE32) ||
        (R == Match_InvalidOperand && Result != Match_MissingFeature
                                   && Result != Match_PreferE32) ||
        (R == Match_MnemonicFail   && Result != Match_InvalidOperand
                                   && Result != Match_MissingFeature
                                   && Result != Match_PreferE32)) {
      Result = R;
      ErrorInfo = EI;
    }
    if (R == Match_Success)
      break;
  }

  switch (Result) {
  default: break;
  case Match_Success:
    if (!validateInstruction(Inst, IDLoc, Operands)) {
      return true;
    }
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;

  case Match_MissingFeature:
    return Error(IDLoc, "instruction not supported on this GPU");

  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion = AMDGPUMnemonicSpellCheck(
        ((AMDGPUOperand &)*Operands[0]).getToken(), FBS);
    return Error(IDLoc, "invalid instruction" + Suggestion,
                 ((AMDGPUOperand &)*Operands[0]).getLocRange());
  }

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size()) {
        return Error(IDLoc, "too few operands for instruction");
      }
      ErrorLoc = ((AMDGPUOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }

  case Match_PreferE32:
    return Error(IDLoc, "internal error: instruction without _e64 suffix "
                        "should be encoded as e32");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool AMDGPUAsmParser::ParseAsAbsoluteExpression(uint32_t &Ret) {
  int64_t Tmp = -1;
  if (getLexer().isNot(AsmToken::Integer) && getLexer().isNot(AsmToken::Identifier)) {
    return true;
  }
  if (getParser().parseAbsoluteExpression(Tmp)) {
    return true;
  }
  Ret = static_cast<uint32_t>(Tmp);
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveMajorMinor(uint32_t &Major,
                                               uint32_t &Minor) {
  if (ParseAsAbsoluteExpression(Major))
    return TokError("invalid major version");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("minor version number required, comma expected");
  Lex();

  if (ParseAsAbsoluteExpression(Minor))
    return TokError("invalid minor version");

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDGCNTarget() {
  if (getSTI().getTargetTriple().getArch() != Triple::amdgcn)
    return TokError("directive only supported for amdgcn architecture");

  std::string Target;

  SMLoc TargetStart = getTok().getLoc();
  if (getParser().parseEscapedString(Target))
    return true;
  SMRange TargetRange = SMRange(TargetStart, getTok().getLoc());

  std::string ExpectedTarget;
  raw_string_ostream ExpectedTargetOS(ExpectedTarget);
  IsaInfo::streamIsaVersion(&getSTI(), ExpectedTargetOS);

  if (Target != ExpectedTargetOS.str())
    return getParser().Error(TargetRange.Start, "target must match options",
                             TargetRange);

  getTargetStreamer().EmitDirectiveAMDGCNTarget(Target);
  return false;
}

bool AMDGPUAsmParser::OutOfRangeError(SMRange Range) {
  return getParser().Error(Range.Start, "value out of range", Range);
}

bool AMDGPUAsmParser::calculateGPRBlocks(
    const FeatureBitset &Features, bool VCCUsed, bool FlatScrUsed,
    bool XNACKUsed, Optional<bool> EnableWavefrontSize32, unsigned NextFreeVGPR,
    SMRange VGPRRange, unsigned NextFreeSGPR, SMRange SGPRRange,
    unsigned &VGPRBlocks, unsigned &SGPRBlocks) {
  // TODO(scott.linder): These calculations are duplicated from
  // AMDGPUAsmPrinter::getSIProgramInfo and could be unified.
  IsaVersion Version = getIsaVersion(getSTI().getCPU());

  unsigned NumVGPRs = NextFreeVGPR;
  unsigned NumSGPRs = NextFreeSGPR;

  if (Version.Major >= 10)
    NumSGPRs = 0;
  else {
    unsigned MaxAddressableNumSGPRs =
        IsaInfo::getAddressableNumSGPRs(&getSTI());

    if (Version.Major >= 8 && !Features.test(FeatureSGPRInitBug) &&
        NumSGPRs > MaxAddressableNumSGPRs)
      return OutOfRangeError(SGPRRange);

    NumSGPRs +=
        IsaInfo::getNumExtraSGPRs(&getSTI(), VCCUsed, FlatScrUsed, XNACKUsed);

    if ((Version.Major <= 7 || Features.test(FeatureSGPRInitBug)) &&
        NumSGPRs > MaxAddressableNumSGPRs)
      return OutOfRangeError(SGPRRange);

    if (Features.test(FeatureSGPRInitBug))
      NumSGPRs = IsaInfo::FIXED_NUM_SGPRS_FOR_INIT_BUG;
  }

  VGPRBlocks =
      IsaInfo::getNumVGPRBlocks(&getSTI(), NumVGPRs, EnableWavefrontSize32);
  SGPRBlocks = IsaInfo::getNumSGPRBlocks(&getSTI(), NumSGPRs);

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDHSAKernel() {
  if (getSTI().getTargetTriple().getArch() != Triple::amdgcn)
    return TokError("directive only supported for amdgcn architecture");

  if (getSTI().getTargetTriple().getOS() != Triple::AMDHSA)
    return TokError("directive only supported for amdhsa OS");

  StringRef KernelName;
  if (getParser().parseIdentifier(KernelName))
    return true;

  kernel_descriptor_t KD = getDefaultAmdhsaKernelDescriptor(&getSTI());

  StringSet<> Seen;

  IsaVersion IVersion = getIsaVersion(getSTI().getCPU());

  SMRange VGPRRange;
  uint64_t NextFreeVGPR = 0;
  SMRange SGPRRange;
  uint64_t NextFreeSGPR = 0;
  unsigned UserSGPRCount = 0;
  bool ReserveVCC = true;
  bool ReserveFlatScr = true;
  bool ReserveXNACK = hasXNACK();
  Optional<bool> EnableWavefrontSize32;

  while (true) {
    while (getLexer().is(AsmToken::EndOfStatement))
      Lex();

    if (getLexer().isNot(AsmToken::Identifier))
      return TokError("expected .amdhsa_ directive or .end_amdhsa_kernel");

    StringRef ID = getTok().getIdentifier();
    SMRange IDRange = getTok().getLocRange();
    Lex();

    if (ID == ".end_amdhsa_kernel")
      break;

    if (Seen.find(ID) != Seen.end())
      return TokError(".amdhsa_ directives cannot be repeated");
    Seen.insert(ID);

    SMLoc ValStart = getTok().getLoc();
    int64_t IVal;
    if (getParser().parseAbsoluteExpression(IVal))
      return true;
    SMLoc ValEnd = getTok().getLoc();
    SMRange ValRange = SMRange(ValStart, ValEnd);

    if (IVal < 0)
      return OutOfRangeError(ValRange);

    uint64_t Val = IVal;

#define PARSE_BITS_ENTRY(FIELD, ENTRY, VALUE, RANGE)                           \
  if (!isUInt<ENTRY##_WIDTH>(VALUE))                                           \
    return OutOfRangeError(RANGE);                                             \
  AMDHSA_BITS_SET(FIELD, ENTRY, VALUE);

    if (ID == ".amdhsa_group_segment_fixed_size") {
      if (!isUInt<sizeof(KD.group_segment_fixed_size) * CHAR_BIT>(Val))
        return OutOfRangeError(ValRange);
      KD.group_segment_fixed_size = Val;
    } else if (ID == ".amdhsa_private_segment_fixed_size") {
      if (!isUInt<sizeof(KD.private_segment_fixed_size) * CHAR_BIT>(Val))
        return OutOfRangeError(ValRange);
      KD.private_segment_fixed_size = Val;
    } else if (ID == ".amdhsa_user_sgpr_private_segment_buffer") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
                       Val, ValRange);
      if (Val)
        UserSGPRCount += 4;
    } else if (ID == ".amdhsa_user_sgpr_dispatch_ptr") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, Val,
                       ValRange);
      if (Val)
        UserSGPRCount += 2;
    } else if (ID == ".amdhsa_user_sgpr_queue_ptr") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, Val,
                       ValRange);
      if (Val)
        UserSGPRCount += 2;
    } else if (ID == ".amdhsa_user_sgpr_kernarg_segment_ptr") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                       Val, ValRange);
      if (Val)
        UserSGPRCount += 2;
    } else if (ID == ".amdhsa_user_sgpr_dispatch_id") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, Val,
                       ValRange);
      if (Val)
        UserSGPRCount += 2;
    } else if (ID == ".amdhsa_user_sgpr_flat_scratch_init") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, Val,
                       ValRange);
      if (Val)
        UserSGPRCount += 2;
    } else if (ID == ".amdhsa_user_sgpr_private_segment_size") {
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE,
                       Val, ValRange);
      if (Val)
        UserSGPRCount += 1;
    } else if (ID == ".amdhsa_wavefront_size32") {
      if (IVersion.Major < 10)
        return getParser().Error(IDRange.Start, "directive requires gfx10+",
                                 IDRange);
      EnableWavefrontSize32 = Val;
      PARSE_BITS_ENTRY(KD.kernel_code_properties,
                       KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
                       Val, ValRange);
    } else if (ID == ".amdhsa_system_sgpr_private_segment_wavefront_offset") {
      PARSE_BITS_ENTRY(
          KD.compute_pgm_rsrc2,
          COMPUTE_PGM_RSRC2_ENABLE_SGPR_PRIVATE_SEGMENT_WAVEFRONT_OFFSET, Val,
          ValRange);
    } else if (ID == ".amdhsa_system_sgpr_workgroup_id_x") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, Val,
                       ValRange);
    } else if (ID == ".amdhsa_system_sgpr_workgroup_id_y") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, Val,
                       ValRange);
    } else if (ID == ".amdhsa_system_sgpr_workgroup_id_z") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, Val,
                       ValRange);
    } else if (ID == ".amdhsa_system_sgpr_workgroup_info") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, Val,
                       ValRange);
    } else if (ID == ".amdhsa_system_vgpr_workitem_id") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID, Val,
                       ValRange);
    } else if (ID == ".amdhsa_next_free_vgpr") {
      VGPRRange = ValRange;
      NextFreeVGPR = Val;
    } else if (ID == ".amdhsa_next_free_sgpr") {
      SGPRRange = ValRange;
      NextFreeSGPR = Val;
    } else if (ID == ".amdhsa_reserve_vcc") {
      if (!isUInt<1>(Val))
        return OutOfRangeError(ValRange);
      ReserveVCC = Val;
    } else if (ID == ".amdhsa_reserve_flat_scratch") {
      if (IVersion.Major < 7)
        return getParser().Error(IDRange.Start, "directive requires gfx7+",
                                 IDRange);
      if (!isUInt<1>(Val))
        return OutOfRangeError(ValRange);
      ReserveFlatScr = Val;
    } else if (ID == ".amdhsa_reserve_xnack_mask") {
      if (IVersion.Major < 8)
        return getParser().Error(IDRange.Start, "directive requires gfx8+",
                                 IDRange);
      if (!isUInt<1>(Val))
        return OutOfRangeError(ValRange);
      ReserveXNACK = Val;
    } else if (ID == ".amdhsa_float_round_mode_32") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32, Val, ValRange);
    } else if (ID == ".amdhsa_float_round_mode_16_64") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64, Val, ValRange);
    } else if (ID == ".amdhsa_float_denorm_mode_32") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32, Val, ValRange);
    } else if (ID == ".amdhsa_float_denorm_mode_16_64") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64, Val,
                       ValRange);
    } else if (ID == ".amdhsa_dx10_clamp") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, Val, ValRange);
    } else if (ID == ".amdhsa_ieee_mode") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE,
                       Val, ValRange);
    } else if (ID == ".amdhsa_fp16_overflow") {
      if (IVersion.Major < 9)
        return getParser().Error(IDRange.Start, "directive requires gfx9+",
                                 IDRange);
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_FP16_OVFL, Val,
                       ValRange);
    } else if (ID == ".amdhsa_workgroup_processor_mode") {
      if (IVersion.Major < 10)
        return getParser().Error(IDRange.Start, "directive requires gfx10+",
                                 IDRange);
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_WGP_MODE, Val,
                       ValRange);
    } else if (ID == ".amdhsa_memory_ordered") {
      if (IVersion.Major < 10)
        return getParser().Error(IDRange.Start, "directive requires gfx10+",
                                 IDRange);
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_MEM_ORDERED, Val,
                       ValRange);
    } else if (ID == ".amdhsa_forward_progress") {
      if (IVersion.Major < 10)
        return getParser().Error(IDRange.Start, "directive requires gfx10+",
                                 IDRange);
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_FWD_PROGRESS, Val,
                       ValRange);
    } else if (ID == ".amdhsa_exception_fp_ieee_invalid_op") {
      PARSE_BITS_ENTRY(
          KD.compute_pgm_rsrc2,
          COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION, Val,
          ValRange);
    } else if (ID == ".amdhsa_exception_fp_denorm_src") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE,
                       Val, ValRange);
    } else if (ID == ".amdhsa_exception_fp_ieee_div_zero") {
      PARSE_BITS_ENTRY(
          KD.compute_pgm_rsrc2,
          COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO, Val,
          ValRange);
    } else if (ID == ".amdhsa_exception_fp_ieee_overflow") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW,
                       Val, ValRange);
    } else if (ID == ".amdhsa_exception_fp_ieee_underflow") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW,
                       Val, ValRange);
    } else if (ID == ".amdhsa_exception_fp_ieee_inexact") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT,
                       Val, ValRange);
    } else if (ID == ".amdhsa_exception_int_div_zero") {
      PARSE_BITS_ENTRY(KD.compute_pgm_rsrc2,
                       COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO,
                       Val, ValRange);
    } else {
      return getParser().Error(IDRange.Start,
                               "unknown .amdhsa_kernel directive", IDRange);
    }

#undef PARSE_BITS_ENTRY
  }

  if (Seen.find(".amdhsa_next_free_vgpr") == Seen.end())
    return TokError(".amdhsa_next_free_vgpr directive is required");

  if (Seen.find(".amdhsa_next_free_sgpr") == Seen.end())
    return TokError(".amdhsa_next_free_sgpr directive is required");

  unsigned VGPRBlocks;
  unsigned SGPRBlocks;
  if (calculateGPRBlocks(getFeatureBits(), ReserveVCC, ReserveFlatScr,
                         ReserveXNACK, EnableWavefrontSize32, NextFreeVGPR,
                         VGPRRange, NextFreeSGPR, SGPRRange, VGPRBlocks,
                         SGPRBlocks))
    return true;

  if (!isUInt<COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH>(
          VGPRBlocks))
    return OutOfRangeError(VGPRRange);
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                  COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, VGPRBlocks);

  if (!isUInt<COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH>(
          SGPRBlocks))
    return OutOfRangeError(SGPRRange);
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc1,
                  COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  SGPRBlocks);

  if (!isUInt<COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH>(UserSGPRCount))
    return TokError("too many user SGPRs enabled");
  AMDHSA_BITS_SET(KD.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                  UserSGPRCount);

  getTargetStreamer().EmitAmdhsaKernelDescriptor(
      getSTI(), KernelName, KD, NextFreeVGPR, NextFreeSGPR, ReserveVCC,
      ReserveFlatScr, ReserveXNACK);
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveHSACodeObjectVersion() {
  uint32_t Major;
  uint32_t Minor;

  if (ParseDirectiveMajorMinor(Major, Minor))
    return true;

  getTargetStreamer().EmitDirectiveHSACodeObjectVersion(Major, Minor);
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveHSACodeObjectISA() {
  uint32_t Major;
  uint32_t Minor;
  uint32_t Stepping;
  StringRef VendorName;
  StringRef ArchName;

  // If this directive has no arguments, then use the ISA version for the
  // targeted GPU.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(getSTI().getCPU());
    getTargetStreamer().EmitDirectiveHSACodeObjectISA(ISA.Major, ISA.Minor,
                                                      ISA.Stepping,
                                                      "AMD", "AMDGPU");
    return false;
  }

  if (ParseDirectiveMajorMinor(Major, Minor))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("stepping version number required, comma expected");
  Lex();

  if (ParseAsAbsoluteExpression(Stepping))
    return TokError("invalid stepping version");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("vendor name required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::String))
    return TokError("invalid vendor name");

  VendorName = getLexer().getTok().getStringContents();
  Lex();

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("arch name required, comma expected");
  Lex();

  if (getLexer().isNot(AsmToken::String))
    return TokError("invalid arch name");

  ArchName = getLexer().getTok().getStringContents();
  Lex();

  getTargetStreamer().EmitDirectiveHSACodeObjectISA(Major, Minor, Stepping,
                                                    VendorName, ArchName);
  return false;
}

bool AMDGPUAsmParser::ParseAMDKernelCodeTValue(StringRef ID,
                                               amd_kernel_code_t &Header) {
  // max_scratch_backing_memory_byte_size is deprecated. Ignore it while parsing
  // assembly for backwards compatibility.
  if (ID == "max_scratch_backing_memory_byte_size") {
    Parser.eatToEndOfStatement();
    return false;
  }

  SmallString<40> ErrStr;
  raw_svector_ostream Err(ErrStr);
  if (!parseAmdKernelCodeField(ID, getParser(), Header, Err)) {
    return TokError(Err.str());
  }
  Lex();

  if (ID == "enable_wavefront_size32") {
    if (Header.code_properties & AMD_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) {
      if (!isGFX10())
        return TokError("enable_wavefront_size32=1 is only allowed on GFX10+");
      if (!getFeatureBits()[AMDGPU::FeatureWavefrontSize32])
        return TokError("enable_wavefront_size32=1 requires +WavefrontSize32");
    } else {
      if (!getFeatureBits()[AMDGPU::FeatureWavefrontSize64])
        return TokError("enable_wavefront_size32=0 requires +WavefrontSize64");
    }
  }

  if (ID == "wavefront_size") {
    if (Header.wavefront_size == 5) {
      if (!isGFX10())
        return TokError("wavefront_size=5 is only allowed on GFX10+");
      if (!getFeatureBits()[AMDGPU::FeatureWavefrontSize32])
        return TokError("wavefront_size=5 requires +WavefrontSize32");
    } else if (Header.wavefront_size == 6) {
      if (!getFeatureBits()[AMDGPU::FeatureWavefrontSize64])
        return TokError("wavefront_size=6 requires +WavefrontSize64");
    }
  }

  if (ID == "enable_wgp_mode") {
    if (G_00B848_WGP_MODE(Header.compute_pgm_resource_registers) && !isGFX10())
      return TokError("enable_wgp_mode=1 is only allowed on GFX10+");
  }

  if (ID == "enable_mem_ordered") {
    if (G_00B848_MEM_ORDERED(Header.compute_pgm_resource_registers) && !isGFX10())
      return TokError("enable_mem_ordered=1 is only allowed on GFX10+");
  }

  if (ID == "enable_fwd_progress") {
    if (G_00B848_FWD_PROGRESS(Header.compute_pgm_resource_registers) && !isGFX10())
      return TokError("enable_fwd_progress=1 is only allowed on GFX10+");
  }

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDKernelCodeT() {
  amd_kernel_code_t Header;
  AMDGPU::initDefaultAMDKernelCodeT(Header, &getSTI());

  while (true) {
    // Lex EndOfStatement.  This is in a while loop, because lexing a comment
    // will set the current token to EndOfStatement.
    while(getLexer().is(AsmToken::EndOfStatement))
      Lex();

    if (getLexer().isNot(AsmToken::Identifier))
      return TokError("expected value identifier or .end_amd_kernel_code_t");

    StringRef ID = getLexer().getTok().getIdentifier();
    Lex();

    if (ID == ".end_amd_kernel_code_t")
      break;

    if (ParseAMDKernelCodeTValue(ID, Header))
      return true;
  }

  getTargetStreamer().EmitAMDKernelCodeT(Header);

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveAMDGPUHsaKernel() {
  if (getLexer().isNot(AsmToken::Identifier))
    return TokError("expected symbol name");

  StringRef KernelName = Parser.getTok().getString();

  getTargetStreamer().EmitAMDGPUSymbolType(KernelName,
                                           ELF::STT_AMDGPU_HSA_KERNEL);
  Lex();
  if (!AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI()))
    KernelScope.initialize(getContext());
  return false;
}

bool AMDGPUAsmParser::ParseDirectiveISAVersion() {
  if (getSTI().getTargetTriple().getArch() != Triple::amdgcn) {
    return Error(getParser().getTok().getLoc(),
                 ".amd_amdgpu_isa directive is not available on non-amdgcn "
                 "architectures");
  }

  auto ISAVersionStringFromASM = getLexer().getTok().getStringContents();

  std::string ISAVersionStringFromSTI;
  raw_string_ostream ISAVersionStreamFromSTI(ISAVersionStringFromSTI);
  IsaInfo::streamIsaVersion(&getSTI(), ISAVersionStreamFromSTI);

  if (ISAVersionStringFromASM != ISAVersionStreamFromSTI.str()) {
    return Error(getParser().getTok().getLoc(),
                 ".amd_amdgpu_isa directive does not match triple and/or mcpu "
                 "arguments specified through the command line");
  }

  getTargetStreamer().EmitISAVersion(ISAVersionStreamFromSTI.str());
  Lex();

  return false;
}

bool AMDGPUAsmParser::ParseDirectiveHSAMetadata() {
  const char *AssemblerDirectiveBegin;
  const char *AssemblerDirectiveEnd;
  std::tie(AssemblerDirectiveBegin, AssemblerDirectiveEnd) =
      AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI())
          ? std::make_tuple(HSAMD::V3::AssemblerDirectiveBegin,
                            HSAMD::V3::AssemblerDirectiveEnd)
          : std::make_tuple(HSAMD::AssemblerDirectiveBegin,
                            HSAMD::AssemblerDirectiveEnd);

  if (getSTI().getTargetTriple().getOS() != Triple::AMDHSA) {
    return Error(getParser().getTok().getLoc(),
                 (Twine(AssemblerDirectiveBegin) + Twine(" directive is "
                 "not available on non-amdhsa OSes")).str());
  }

  std::string HSAMetadataString;
  if (ParseToEndDirective(AssemblerDirectiveBegin, AssemblerDirectiveEnd,
                          HSAMetadataString))
    return true;

  if (IsaInfo::hasCodeObjectV3(&getSTI())) {
    if (!getTargetStreamer().EmitHSAMetadataV3(HSAMetadataString))
      return Error(getParser().getTok().getLoc(), "invalid HSA metadata");
  } else {
    if (!getTargetStreamer().EmitHSAMetadataV2(HSAMetadataString))
      return Error(getParser().getTok().getLoc(), "invalid HSA metadata");
  }

  return false;
}

/// Common code to parse out a block of text (typically YAML) between start and
/// end directives.
bool AMDGPUAsmParser::ParseToEndDirective(const char *AssemblerDirectiveBegin,
                                          const char *AssemblerDirectiveEnd,
                                          std::string &CollectString) {

  raw_string_ostream CollectStream(CollectString);

  getLexer().setSkipSpace(false);

  bool FoundEnd = false;
  while (!getLexer().is(AsmToken::Eof)) {
    while (getLexer().is(AsmToken::Space)) {
      CollectStream << getLexer().getTok().getString();
      Lex();
    }

    if (getLexer().is(AsmToken::Identifier)) {
      StringRef ID = getLexer().getTok().getIdentifier();
      if (ID == AssemblerDirectiveEnd) {
        Lex();
        FoundEnd = true;
        break;
      }
    }

    CollectStream << Parser.parseStringToEndOfStatement()
                  << getContext().getAsmInfo()->getSeparatorString();

    Parser.eatToEndOfStatement();
  }

  getLexer().setSkipSpace(true);

  if (getLexer().is(AsmToken::Eof) && !FoundEnd) {
    return TokError(Twine("expected directive ") +
                    Twine(AssemblerDirectiveEnd) + Twine(" not found"));
  }

  CollectStream.flush();
  return false;
}

/// Parse the assembler directive for new MsgPack-format PAL metadata.
bool AMDGPUAsmParser::ParseDirectivePALMetadataBegin() {
  std::string String;
  if (ParseToEndDirective(AMDGPU::PALMD::AssemblerDirectiveBegin,
                          AMDGPU::PALMD::AssemblerDirectiveEnd, String))
    return true;

  auto PALMetadata = getTargetStreamer().getPALMetadata();
  if (!PALMetadata->setFromString(String))
    return Error(getParser().getTok().getLoc(), "invalid PAL metadata");
  return false;
}

/// Parse the assembler directive for old linear-format PAL metadata.
bool AMDGPUAsmParser::ParseDirectivePALMetadata() {
  if (getSTI().getTargetTriple().getOS() != Triple::AMDPAL) {
    return Error(getParser().getTok().getLoc(),
                 (Twine(PALMD::AssemblerDirective) + Twine(" directive is "
                 "not available on non-amdpal OSes")).str());
  }

  auto PALMetadata = getTargetStreamer().getPALMetadata();
  PALMetadata->setLegacy();
  for (;;) {
    uint32_t Key, Value;
    if (ParseAsAbsoluteExpression(Key)) {
      return TokError(Twine("invalid value in ") +
                      Twine(PALMD::AssemblerDirective));
    }
    if (getLexer().isNot(AsmToken::Comma)) {
      return TokError(Twine("expected an even number of values in ") +
                      Twine(PALMD::AssemblerDirective));
    }
    Lex();
    if (ParseAsAbsoluteExpression(Value)) {
      return TokError(Twine("invalid value in ") +
                      Twine(PALMD::AssemblerDirective));
    }
    PALMetadata->setRegister(Key, Value);
    if (getLexer().isNot(AsmToken::Comma))
      break;
    Lex();
  }
  return false;
}

/// ParseDirectiveAMDGPULDS
///  ::= .amdgpu_lds identifier ',' size_expression [',' align_expression]
bool AMDGPUAsmParser::ParseDirectiveAMDGPULDS() {
  if (getParser().checkForValidSection())
    return true;

  StringRef Name;
  SMLoc NameLoc = getLexer().getLoc();
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");

  MCSymbol *Symbol = getContext().getOrCreateSymbol(Name);
  if (parseToken(AsmToken::Comma, "expected ','"))
    return true;

  unsigned LocalMemorySize = AMDGPU::IsaInfo::getLocalMemorySize(&getSTI());

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().parseAbsoluteExpression(Size))
    return true;
  if (Size < 0)
    return Error(SizeLoc, "size must be non-negative");
  if (Size > LocalMemorySize)
    return Error(SizeLoc, "size is too large");

  int64_t Alignment = 4;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    SMLoc AlignLoc = getLexer().getLoc();
    if (getParser().parseAbsoluteExpression(Alignment))
      return true;
    if (Alignment < 0 || !isPowerOf2_64(Alignment))
      return Error(AlignLoc, "alignment must be a power of two");

    // Alignment larger than the size of LDS is possible in theory, as long
    // as the linker manages to place to symbol at address 0, but we do want
    // to make sure the alignment fits nicely into a 32-bit integer.
    if (Alignment >= 1u << 31)
      return Error(AlignLoc, "alignment is too large");
  }

  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.amdgpu_lds' directive"))
    return true;

  Symbol->redefineIfPossible();
  if (!Symbol->isUndefined())
    return Error(NameLoc, "invalid symbol redefinition");

  getTargetStreamer().emitAMDGPULDS(Symbol, Size, Align(Alignment));
  return false;
}

bool AMDGPUAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getString();

  if (AMDGPU::IsaInfo::hasCodeObjectV3(&getSTI())) {
    if (IDVal == ".amdgcn_target")
      return ParseDirectiveAMDGCNTarget();

    if (IDVal == ".amdhsa_kernel")
      return ParseDirectiveAMDHSAKernel();

    // TODO: Restructure/combine with PAL metadata directive.
    if (IDVal == AMDGPU::HSAMD::V3::AssemblerDirectiveBegin)
      return ParseDirectiveHSAMetadata();
  } else {
    if (IDVal == ".hsa_code_object_version")
      return ParseDirectiveHSACodeObjectVersion();

    if (IDVal == ".hsa_code_object_isa")
      return ParseDirectiveHSACodeObjectISA();

    if (IDVal == ".amd_kernel_code_t")
      return ParseDirectiveAMDKernelCodeT();

    if (IDVal == ".amdgpu_hsa_kernel")
      return ParseDirectiveAMDGPUHsaKernel();

    if (IDVal == ".amd_amdgpu_isa")
      return ParseDirectiveISAVersion();

    if (IDVal == AMDGPU::HSAMD::AssemblerDirectiveBegin)
      return ParseDirectiveHSAMetadata();
  }

  if (IDVal == ".amdgpu_lds")
    return ParseDirectiveAMDGPULDS();

  if (IDVal == PALMD::AssemblerDirectiveBegin)
    return ParseDirectivePALMetadataBegin();

  if (IDVal == PALMD::AssemblerDirective)
    return ParseDirectivePALMetadata();

  return true;
}

bool AMDGPUAsmParser::subtargetHasRegister(const MCRegisterInfo &MRI,
                                           unsigned RegNo) const {

  for (MCRegAliasIterator R(AMDGPU::TTMP12_TTMP13_TTMP14_TTMP15, &MRI, true);
       R.isValid(); ++R) {
    if (*R == RegNo)
      return isGFX9() || isGFX10();
  }

  // GFX10 has 2 more SGPRs 104 and 105.
  for (MCRegAliasIterator R(AMDGPU::SGPR104_SGPR105, &MRI, true);
       R.isValid(); ++R) {
    if (*R == RegNo)
      return hasSGPR104_SGPR105();
  }

  switch (RegNo) {
  case AMDGPU::SRC_SHARED_BASE:
  case AMDGPU::SRC_SHARED_LIMIT:
  case AMDGPU::SRC_PRIVATE_BASE:
  case AMDGPU::SRC_PRIVATE_LIMIT:
  case AMDGPU::SRC_POPS_EXITING_WAVE_ID:
    return !isCI() && !isSI() && !isVI();
  case AMDGPU::TBA:
  case AMDGPU::TBA_LO:
  case AMDGPU::TBA_HI:
  case AMDGPU::TMA:
  case AMDGPU::TMA_LO:
  case AMDGPU::TMA_HI:
    return !isGFX9() && !isGFX10();
  case AMDGPU::XNACK_MASK:
  case AMDGPU::XNACK_MASK_LO:
  case AMDGPU::XNACK_MASK_HI:
    return !isCI() && !isSI() && !isGFX10() && hasXNACK();
  case AMDGPU::SGPR_NULL:
    return isGFX10();
  default:
    break;
  }

  if (isCI())
    return true;

  if (isSI() || isGFX10()) {
    // No flat_scr on SI.
    // On GFX10 flat scratch is not a valid register operand and can only be
    // accessed with s_setreg/s_getreg.
    switch (RegNo) {
    case AMDGPU::FLAT_SCR:
    case AMDGPU::FLAT_SCR_LO:
    case AMDGPU::FLAT_SCR_HI:
      return false;
    default:
      return true;
    }
  }

  // VI only has 102 SGPRs, so make sure we aren't trying to use the 2 more that
  // SI/CI have.
  for (MCRegAliasIterator R(AMDGPU::SGPR102_SGPR103, &MRI, true);
       R.isValid(); ++R) {
    if (*R == RegNo)
      return hasSGPR102_SGPR103();
  }

  return true;
}

OperandMatchResultTy
AMDGPUAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic,
                              OperandMode Mode) {
  // Try to parse with a custom parser
  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If we successfully parsed the operand or if there as an error parsing,
  // we are done.
  //
  // If we are parsing after we reach EndOfStatement then this means we
  // are appending default values to the Operands list.  This is only done
  // by custom parser, so we shouldn't continue on to the generic parsing.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail ||
      getLexer().is(AsmToken::EndOfStatement))
    return ResTy;

  if (Mode == OperandMode_NSA && getLexer().is(AsmToken::LBrac)) {
    unsigned Prefix = Operands.size();
    SMLoc LBraceLoc = getTok().getLoc();
    Parser.Lex(); // eat the '['

    for (;;) {
      ResTy = parseReg(Operands);
      if (ResTy != MatchOperand_Success)
        return ResTy;

      if (getLexer().is(AsmToken::RBrac))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return MatchOperand_ParseFail;
      Parser.Lex();
    }

    if (Operands.size() - Prefix > 1) {
      Operands.insert(Operands.begin() + Prefix,
                      AMDGPUOperand::CreateToken(this, "[", LBraceLoc));
      Operands.push_back(AMDGPUOperand::CreateToken(this, "]",
                                                    getTok().getLoc()));
    }

    Parser.Lex(); // eat the ']'
    return MatchOperand_Success;
  }

  return parseRegOrImm(Operands);
}

StringRef AMDGPUAsmParser::parseMnemonicSuffix(StringRef Name) {
  // Clear any forced encodings from the previous instruction.
  setForcedEncodingSize(0);
  setForcedDPP(false);
  setForcedSDWA(false);

  if (Name.endswith("_e64")) {
    setForcedEncodingSize(64);
    return Name.substr(0, Name.size() - 4);
  } else if (Name.endswith("_e32")) {
    setForcedEncodingSize(32);
    return Name.substr(0, Name.size() - 4);
  } else if (Name.endswith("_dpp")) {
    setForcedDPP(true);
    return Name.substr(0, Name.size() - 4);
  } else if (Name.endswith("_sdwa")) {
    setForcedSDWA(true);
    return Name.substr(0, Name.size() - 5);
  }
  return Name;
}

bool AMDGPUAsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name,
                                       SMLoc NameLoc, OperandVector &Operands) {
  // Add the instruction mnemonic
  Name = parseMnemonicSuffix(Name);
  Operands.push_back(AMDGPUOperand::CreateToken(this, Name, NameLoc));

  bool IsMIMG = Name.startswith("image_");

  while (!getLexer().is(AsmToken::EndOfStatement)) {
    OperandMode Mode = OperandMode_Default;
    if (IsMIMG && isGFX10() && Operands.size() == 2)
      Mode = OperandMode_NSA;
    OperandMatchResultTy Res = parseOperand(Operands, Name, Mode);

    // Eat the comma or space if there is one.
    if (getLexer().is(AsmToken::Comma))
      Parser.Lex();

    if (Res != MatchOperand_Success) {
      if (!Parser.hasPendingError()) {
        // FIXME: use real operand location rather than the current location.
        StringRef Msg =
          (Res == MatchOperand_ParseFail) ? "failed parsing operand." :
                                            "not a valid operand.";
        Error(getLexer().getLoc(), Msg);
      }
      while (!getLexer().is(AsmToken::EndOfStatement)) {
        Parser.Lex();
      }
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, int64_t &IntVal) {

  if (!trySkipId(Prefix, AsmToken::Colon))
    return MatchOperand_NoMatch;

  return parseExpr(IntVal) ? MatchOperand_Success : MatchOperand_ParseFail;
}

OperandMatchResultTy
AMDGPUAsmParser::parseIntWithPrefix(const char *Prefix, OperandVector &Operands,
                                    AMDGPUOperand::ImmTy ImmTy,
                                    bool (*ConvertResult)(int64_t&)) {
  SMLoc S = getLoc();
  int64_t Value = 0;

  OperandMatchResultTy Res = parseIntWithPrefix(Prefix, Value);
  if (Res != MatchOperand_Success)
    return Res;

  if (ConvertResult && !ConvertResult(Value)) {
    Error(S, "invalid " + StringRef(Prefix) + " value.");
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Value, S, ImmTy));
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseOperandArrayWithPrefix(const char *Prefix,
                                             OperandVector &Operands,
                                             AMDGPUOperand::ImmTy ImmTy,
                                             bool (*ConvertResult)(int64_t&)) {
  SMLoc S = getLoc();
  if (!trySkipId(Prefix, AsmToken::Colon))
    return MatchOperand_NoMatch;

  if (!skipToken(AsmToken::LBrac, "expected a left square bracket"))
    return MatchOperand_ParseFail;

  unsigned Val = 0;
  const unsigned MaxSize = 4;

  // FIXME: How to verify the number of elements matches the number of src
  // operands?
  for (int I = 0; ; ++I) {
    int64_t Op;
    SMLoc Loc = getLoc();
    if (!parseExpr(Op))
      return MatchOperand_ParseFail;

    if (Op != 0 && Op != 1) {
      Error(Loc, "invalid " + StringRef(Prefix) + " value.");
      return MatchOperand_ParseFail;
    }

    Val |= (Op << I);

    if (trySkipToken(AsmToken::RBrac))
      break;

    if (I + 1 == MaxSize) {
      Error(getLoc(), "expected a closing square bracket");
      return MatchOperand_ParseFail;
    }

    if (!skipToken(AsmToken::Comma, "expected a comma"))
      return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Val, S, ImmTy));
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseNamedBit(const char *Name, OperandVector &Operands,
                               AMDGPUOperand::ImmTy ImmTy) {
  int64_t Bit = 0;
  SMLoc S = Parser.getTok().getLoc();

  // We are at the end of the statement, and this is a default argument, so
  // use a default value.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    switch(getLexer().getKind()) {
      case AsmToken::Identifier: {
        StringRef Tok = Parser.getTok().getString();
        if (Tok == Name) {
          if (Tok == "r128" && !hasMIMG_R128())
            Error(S, "r128 modifier is not supported on this GPU");
          if (Tok == "a16" && !isGFX9() && !hasGFX10A16())
            Error(S, "a16 modifier is not supported on this GPU");
          Bit = 1;
          Parser.Lex();
        } else if (Tok.startswith("no") && Tok.endswith(Name)) {
          Bit = 0;
          Parser.Lex();
        } else {
          return MatchOperand_NoMatch;
        }
        break;
      }
      default:
        return MatchOperand_NoMatch;
    }
  }

  if (!isGFX10() && ImmTy == AMDGPUOperand::ImmTyDLC)
    return MatchOperand_ParseFail;

  if (isGFX9() && ImmTy == AMDGPUOperand::ImmTyA16)
    ImmTy = AMDGPUOperand::ImmTyR128A16;

  Operands.push_back(AMDGPUOperand::CreateImm(this, Bit, S, ImmTy));
  return MatchOperand_Success;
}

static void addOptionalImmOperand(
  MCInst& Inst, const OperandVector& Operands,
  AMDGPUAsmParser::OptionalImmIndexMap& OptionalIdx,
  AMDGPUOperand::ImmTy ImmT,
  int64_t Default = 0) {
  auto i = OptionalIdx.find(ImmT);
  if (i != OptionalIdx.end()) {
    unsigned Idx = i->second;
    ((AMDGPUOperand &)*Operands[Idx]).addImmOperands(Inst, 1);
  } else {
    Inst.addOperand(MCOperand::createImm(Default));
  }
}

OperandMatchResultTy
AMDGPUAsmParser::parseStringWithPrefix(StringRef Prefix, StringRef &Value) {
  if (getLexer().isNot(AsmToken::Identifier)) {
    return MatchOperand_NoMatch;
  }
  StringRef Tok = Parser.getTok().getString();
  if (Tok != Prefix) {
    return MatchOperand_NoMatch;
  }

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Colon)) {
    return MatchOperand_ParseFail;
  }

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Identifier)) {
    return MatchOperand_ParseFail;
  }

  Value = Parser.getTok().getString();
  return MatchOperand_Success;
}

//===----------------------------------------------------------------------===//
// MTBUF format
//===----------------------------------------------------------------------===//

bool AMDGPUAsmParser::tryParseFmt(const char *Pref,
                                  int64_t MaxVal,
                                  int64_t &Fmt) {
  int64_t Val;
  SMLoc Loc = getLoc();

  auto Res = parseIntWithPrefix(Pref, Val);
  if (Res == MatchOperand_ParseFail)
    return false;
  if (Res == MatchOperand_NoMatch)
    return true;

  if (Val < 0 || Val > MaxVal) {
    Error(Loc, Twine("out of range ", StringRef(Pref)));
    return false;
  }

  Fmt = Val;
  return true;
}

// dfmt and nfmt (in a tbuffer instruction) are parsed as one to allow their
// values to live in a joint format operand in the MCInst encoding.
OperandMatchResultTy
AMDGPUAsmParser::parseDfmtNfmt(int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int64_t Dfmt = DFMT_UNDEF;
  int64_t Nfmt = NFMT_UNDEF;

  // dfmt and nfmt can appear in either order, and each is optional.
  for (int I = 0; I < 2; ++I) {
    if (Dfmt == DFMT_UNDEF && !tryParseFmt("dfmt", DFMT_MAX, Dfmt))
      return MatchOperand_ParseFail;

    if (Nfmt == NFMT_UNDEF && !tryParseFmt("nfmt", NFMT_MAX, Nfmt)) {
      return MatchOperand_ParseFail;
    }
    // Skip optional comma between dfmt/nfmt
    // but guard against 2 commas following each other.
    if ((Dfmt == DFMT_UNDEF) != (Nfmt == NFMT_UNDEF) &&
        !peekToken().is(AsmToken::Comma)) {
      trySkipToken(AsmToken::Comma);
    }
  }

  if (Dfmt == DFMT_UNDEF && Nfmt == NFMT_UNDEF)
    return MatchOperand_NoMatch;

  Dfmt = (Dfmt == DFMT_UNDEF) ? DFMT_DEFAULT : Dfmt;
  Nfmt = (Nfmt == NFMT_UNDEF) ? NFMT_DEFAULT : Nfmt;

  Format = encodeDfmtNfmt(Dfmt, Nfmt);
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseUfmt(int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int64_t Fmt = UFMT_UNDEF;

  if (!tryParseFmt("format", UFMT_MAX, Fmt))
    return MatchOperand_ParseFail;

  if (Fmt == UFMT_UNDEF)
    return MatchOperand_NoMatch;

  Format = Fmt;
  return MatchOperand_Success;
}

bool AMDGPUAsmParser::matchDfmtNfmt(int64_t &Dfmt,
                                    int64_t &Nfmt,
                                    StringRef FormatStr,
                                    SMLoc Loc) {
  using namespace llvm::AMDGPU::MTBUFFormat;
  int64_t Format;

  Format = getDfmt(FormatStr);
  if (Format != DFMT_UNDEF) {
    Dfmt = Format;
    return true;
  }

  Format = getNfmt(FormatStr, getSTI());
  if (Format != NFMT_UNDEF) {
    Nfmt = Format;
    return true;
  }

  Error(Loc, "unsupported format");
  return false;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSymbolicSplitFormat(StringRef FormatStr,
                                          SMLoc FormatLoc,
                                          int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int64_t Dfmt = DFMT_UNDEF;
  int64_t Nfmt = NFMT_UNDEF;
  if (!matchDfmtNfmt(Dfmt, Nfmt, FormatStr, FormatLoc))
    return MatchOperand_ParseFail;

  if (trySkipToken(AsmToken::Comma)) {
    StringRef Str;
    SMLoc Loc = getLoc();
    if (!parseId(Str, "expected a format string") ||
        !matchDfmtNfmt(Dfmt, Nfmt, Str, Loc)) {
      return MatchOperand_ParseFail;
    }
    if (Dfmt == DFMT_UNDEF) {
      Error(Loc, "duplicate numeric format");
      return MatchOperand_ParseFail;
    } else if (Nfmt == NFMT_UNDEF) {
      Error(Loc, "duplicate data format");
      return MatchOperand_ParseFail;
    }
  }

  Dfmt = (Dfmt == DFMT_UNDEF) ? DFMT_DEFAULT : Dfmt;
  Nfmt = (Nfmt == NFMT_UNDEF) ? NFMT_DEFAULT : Nfmt;

  if (isGFX10()) {
    auto Ufmt = convertDfmtNfmt2Ufmt(Dfmt, Nfmt);
    if (Ufmt == UFMT_UNDEF) {
      Error(FormatLoc, "unsupported format");
      return MatchOperand_ParseFail;
    }
    Format = Ufmt;
  } else {
    Format = encodeDfmtNfmt(Dfmt, Nfmt);
  }

  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSymbolicUnifiedFormat(StringRef FormatStr,
                                            SMLoc Loc,
                                            int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  auto Id = getUnifiedFormat(FormatStr);
  if (Id == UFMT_UNDEF)
    return MatchOperand_NoMatch;

  if (!isGFX10()) {
    Error(Loc, "unified format is not supported on this GPU");
    return MatchOperand_ParseFail;
  }

  Format = Id;
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseNumericFormat(int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;
  SMLoc Loc = getLoc();

  if (!parseExpr(Format))
    return MatchOperand_ParseFail;
  if (!isValidFormatEncoding(Format, getSTI())) {
    Error(Loc, "out of range format");
    return MatchOperand_ParseFail;
  }

  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSymbolicOrNumericFormat(int64_t &Format) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  if (!trySkipId("format", AsmToken::Colon))
    return MatchOperand_NoMatch;

  if (trySkipToken(AsmToken::LBrac)) {
    StringRef FormatStr;
    SMLoc Loc = getLoc();
    if (!parseId(FormatStr, "expected a format string"))
      return MatchOperand_ParseFail;

    auto Res = parseSymbolicUnifiedFormat(FormatStr, Loc, Format);
    if (Res == MatchOperand_NoMatch)
      Res = parseSymbolicSplitFormat(FormatStr, Loc, Format);
    if (Res != MatchOperand_Success)
      return Res;

    if (!skipToken(AsmToken::RBrac, "expected a closing square bracket"))
      return MatchOperand_ParseFail;

    return MatchOperand_Success;
  }

  return parseNumericFormat(Format);
}

OperandMatchResultTy
AMDGPUAsmParser::parseFORMAT(OperandVector &Operands) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int64_t Format = getDefaultFormatEncoding(getSTI());
  OperandMatchResultTy Res;
  SMLoc Loc = getLoc();

  // Parse legacy format syntax.
  Res = isGFX10() ? parseUfmt(Format) : parseDfmtNfmt(Format);
  if (Res == MatchOperand_ParseFail)
    return Res;

  bool FormatFound = (Res == MatchOperand_Success);

  Operands.push_back(
    AMDGPUOperand::CreateImm(this, Format, Loc, AMDGPUOperand::ImmTyFORMAT));

  if (FormatFound)
    trySkipToken(AsmToken::Comma);

  if (isToken(AsmToken::EndOfStatement)) {
    // We are expecting an soffset operand,
    // but let matcher handle the error.
    return MatchOperand_Success;
  }

  // Parse soffset.
  Res = parseRegOrImm(Operands);
  if (Res != MatchOperand_Success)
    return Res;

  trySkipToken(AsmToken::Comma);

  if (!FormatFound) {
    Res = parseSymbolicOrNumericFormat(Format);
    if (Res == MatchOperand_ParseFail)
      return Res;
    if (Res == MatchOperand_Success) {
      auto Size = Operands.size();
      AMDGPUOperand &Op = static_cast<AMDGPUOperand &>(*Operands[Size - 2]);
      assert(Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyFORMAT);
      Op.setImm(Format);
    }
    return MatchOperand_Success;
  }

  if (isId("format") && peekToken().is(AsmToken::Colon)) {
    Error(getLoc(), "duplicate format");
    return MatchOperand_ParseFail;
  }
  return MatchOperand_Success;
}

//===----------------------------------------------------------------------===//
// ds
//===----------------------------------------------------------------------===//

void AMDGPUAsmParser::cvtDSOffset01(MCInst &Inst,
                                    const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOffset0);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOffset1);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGDS);

  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}

void AMDGPUAsmParser::cvtDSImpl(MCInst &Inst, const OperandVector &Operands,
                                bool IsGdsHardcoded) {
  OptionalImmIndexMap OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    if (Op.isToken() && Op.getToken() == "gds") {
      IsGdsHardcoded = true;
      continue;
    }

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  AMDGPUOperand::ImmTy OffsetType =
    (Inst.getOpcode() == AMDGPU::DS_SWIZZLE_B32_gfx10 ||
     Inst.getOpcode() == AMDGPU::DS_SWIZZLE_B32_gfx6_gfx7 ||
     Inst.getOpcode() == AMDGPU::DS_SWIZZLE_B32_vi) ? AMDGPUOperand::ImmTySwizzle :
                                                      AMDGPUOperand::ImmTyOffset;

  addOptionalImmOperand(Inst, Operands, OptionalIdx, OffsetType);

  if (!IsGdsHardcoded) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGDS);
  }
  Inst.addOperand(MCOperand::createReg(AMDGPU::M0)); // m0
}

void AMDGPUAsmParser::cvtExp(MCInst &Inst, const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;

  unsigned OperandIdx[4];
  unsigned EnMask = 0;
  int SrcIdx = 0;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      assert(SrcIdx < 4);
      OperandIdx[SrcIdx] = Inst.size();
      Op.addRegOperands(Inst, 1);
      ++SrcIdx;
      continue;
    }

    if (Op.isOff()) {
      assert(SrcIdx < 4);
      OperandIdx[SrcIdx] = Inst.size();
      Inst.addOperand(MCOperand::createReg(AMDGPU::NoRegister));
      ++SrcIdx;
      continue;
    }

    if (Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyExpTgt) {
      Op.addImmOperands(Inst, 1);
      continue;
    }

    if (Op.isToken() && Op.getToken() == "done")
      continue;

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  assert(SrcIdx == 4);

  bool Compr = false;
  if (OptionalIdx.find(AMDGPUOperand::ImmTyExpCompr) != OptionalIdx.end()) {
    Compr = true;
    Inst.getOperand(OperandIdx[1]) = Inst.getOperand(OperandIdx[2]);
    Inst.getOperand(OperandIdx[2]).setReg(AMDGPU::NoRegister);
    Inst.getOperand(OperandIdx[3]).setReg(AMDGPU::NoRegister);
  }

  for (auto i = 0; i < SrcIdx; ++i) {
    if (Inst.getOperand(OperandIdx[i]).getReg() != AMDGPU::NoRegister) {
      EnMask |= Compr? (0x3 << i * 2) : (0x1 << i);
    }
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyExpVM);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyExpCompr);

  Inst.addOperand(MCOperand::createImm(EnMask));
}

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

static bool
encodeCnt(
  const AMDGPU::IsaVersion ISA,
  int64_t &IntVal,
  int64_t CntVal,
  bool Saturate,
  unsigned (*encode)(const IsaVersion &Version, unsigned, unsigned),
  unsigned (*decode)(const IsaVersion &Version, unsigned))
{
  bool Failed = false;

  IntVal = encode(ISA, IntVal, CntVal);
  if (CntVal != decode(ISA, IntVal)) {
    if (Saturate) {
      IntVal = encode(ISA, IntVal, -1);
    } else {
      Failed = true;
    }
  }
  return Failed;
}

bool AMDGPUAsmParser::parseCnt(int64_t &IntVal) {

  SMLoc CntLoc = getLoc();
  StringRef CntName = getTokenStr();

  if (!skipToken(AsmToken::Identifier, "expected a counter name") ||
      !skipToken(AsmToken::LParen, "expected a left parenthesis"))
    return false;

  int64_t CntVal;
  SMLoc ValLoc = getLoc();
  if (!parseExpr(CntVal))
    return false;

  AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(getSTI().getCPU());

  bool Failed = true;
  bool Sat = CntName.endswith("_sat");

  if (CntName == "vmcnt" || CntName == "vmcnt_sat") {
    Failed = encodeCnt(ISA, IntVal, CntVal, Sat, encodeVmcnt, decodeVmcnt);
  } else if (CntName == "expcnt" || CntName == "expcnt_sat") {
    Failed = encodeCnt(ISA, IntVal, CntVal, Sat, encodeExpcnt, decodeExpcnt);
  } else if (CntName == "lgkmcnt" || CntName == "lgkmcnt_sat") {
    Failed = encodeCnt(ISA, IntVal, CntVal, Sat, encodeLgkmcnt, decodeLgkmcnt);
  } else {
    Error(CntLoc, "invalid counter name " + CntName);
    return false;
  }

  if (Failed) {
    Error(ValLoc, "too large value for " + CntName);
    return false;
  }

  if (!skipToken(AsmToken::RParen, "expected a closing parenthesis"))
    return false;

  if (trySkipToken(AsmToken::Amp) || trySkipToken(AsmToken::Comma)) {
    if (isToken(AsmToken::EndOfStatement)) {
      Error(getLoc(), "expected a counter name");
      return false;
    }
  }

  return true;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSWaitCntOps(OperandVector &Operands) {
  AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(getSTI().getCPU());
  int64_t Waitcnt = getWaitcntBitMask(ISA);
  SMLoc S = getLoc();

  if (isToken(AsmToken::Identifier) && peekToken().is(AsmToken::LParen)) {
    while (!isToken(AsmToken::EndOfStatement)) {
      if (!parseCnt(Waitcnt))
        return MatchOperand_ParseFail;
    }
  } else {
    if (!parseExpr(Waitcnt))
      return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Waitcnt, S));
  return MatchOperand_Success;
}

bool
AMDGPUOperand::isSWaitCnt() const {
  return isImm();
}

//===----------------------------------------------------------------------===//
// hwreg
//===----------------------------------------------------------------------===//

bool
AMDGPUAsmParser::parseHwregBody(OperandInfoTy &HwReg,
                                int64_t &Offset,
                                int64_t &Width) {
  using namespace llvm::AMDGPU::Hwreg;

  // The register may be specified by name or using a numeric code
  if (isToken(AsmToken::Identifier) &&
      (HwReg.Id = getHwregId(getTokenStr())) >= 0) {
    HwReg.IsSymbolic = true;
    lex(); // skip message name
  } else if (!parseExpr(HwReg.Id)) {
    return false;
  }

  if (trySkipToken(AsmToken::RParen))
    return true;

  // parse optional params
  return
    skipToken(AsmToken::Comma, "expected a comma or a closing parenthesis") &&
    parseExpr(Offset) &&
    skipToken(AsmToken::Comma, "expected a comma") &&
    parseExpr(Width) &&
    skipToken(AsmToken::RParen, "expected a closing parenthesis");
}

bool
AMDGPUAsmParser::validateHwreg(const OperandInfoTy &HwReg,
                               const int64_t Offset,
                               const int64_t Width,
                               const SMLoc Loc) {

  using namespace llvm::AMDGPU::Hwreg;

  if (HwReg.IsSymbolic && !isValidHwreg(HwReg.Id, getSTI())) {
    Error(Loc, "specified hardware register is not supported on this GPU");
    return false;
  } else if (!isValidHwreg(HwReg.Id)) {
    Error(Loc, "invalid code of hardware register: only 6-bit values are legal");
    return false;
  } else if (!isValidHwregOffset(Offset)) {
    Error(Loc, "invalid bit offset: only 5-bit values are legal");
    return false;
  } else if (!isValidHwregWidth(Width)) {
    Error(Loc, "invalid bitfield width: only values from 1 to 32 are legal");
    return false;
  }
  return true;
}

OperandMatchResultTy
AMDGPUAsmParser::parseHwreg(OperandVector &Operands) {
  using namespace llvm::AMDGPU::Hwreg;

  int64_t ImmVal = 0;
  SMLoc Loc = getLoc();

  if (trySkipId("hwreg", AsmToken::LParen)) {
    OperandInfoTy HwReg(ID_UNKNOWN_);
    int64_t Offset = OFFSET_DEFAULT_;
    int64_t Width = WIDTH_DEFAULT_;
    if (parseHwregBody(HwReg, Offset, Width) &&
        validateHwreg(HwReg, Offset, Width, Loc)) {
      ImmVal = encodeHwreg(HwReg.Id, Offset, Width);
    } else {
      return MatchOperand_ParseFail;
    }
  } else if (parseExpr(ImmVal)) {
    if (ImmVal < 0 || !isUInt<16>(ImmVal)) {
      Error(Loc, "invalid immediate: only 16-bit values are legal");
      return MatchOperand_ParseFail;
    }
  } else {
    return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, ImmVal, Loc, AMDGPUOperand::ImmTyHwreg));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isHwreg() const {
  return isImmTy(ImmTyHwreg);
}

//===----------------------------------------------------------------------===//
// sendmsg
//===----------------------------------------------------------------------===//

bool
AMDGPUAsmParser::parseSendMsgBody(OperandInfoTy &Msg,
                                  OperandInfoTy &Op,
                                  OperandInfoTy &Stream) {
  using namespace llvm::AMDGPU::SendMsg;

  if (isToken(AsmToken::Identifier) && (Msg.Id = getMsgId(getTokenStr())) >= 0) {
    Msg.IsSymbolic = true;
    lex(); // skip message name
  } else if (!parseExpr(Msg.Id)) {
    return false;
  }

  if (trySkipToken(AsmToken::Comma)) {
    Op.IsDefined = true;
    if (isToken(AsmToken::Identifier) &&
        (Op.Id = getMsgOpId(Msg.Id, getTokenStr())) >= 0) {
      lex(); // skip operation name
    } else if (!parseExpr(Op.Id)) {
      return false;
    }

    if (trySkipToken(AsmToken::Comma)) {
      Stream.IsDefined = true;
      if (!parseExpr(Stream.Id))
        return false;
    }
  }

  return skipToken(AsmToken::RParen, "expected a closing parenthesis");
}

bool
AMDGPUAsmParser::validateSendMsg(const OperandInfoTy &Msg,
                                 const OperandInfoTy &Op,
                                 const OperandInfoTy &Stream,
                                 const SMLoc S) {
  using namespace llvm::AMDGPU::SendMsg;

  // Validation strictness depends on whether message is specified
  // in a symbolc or in a numeric form. In the latter case
  // only encoding possibility is checked.
  bool Strict = Msg.IsSymbolic;

  if (!isValidMsgId(Msg.Id, getSTI(), Strict)) {
    Error(S, "invalid message id");
    return false;
  } else if (Strict && (msgRequiresOp(Msg.Id) != Op.IsDefined)) {
    Error(S, Op.IsDefined ?
             "message does not support operations" :
             "missing message operation");
    return false;
  } else if (!isValidMsgOp(Msg.Id, Op.Id, Strict)) {
    Error(S, "invalid operation id");
    return false;
  } else if (Strict && !msgSupportsStream(Msg.Id, Op.Id) && Stream.IsDefined) {
    Error(S, "message operation does not support streams");
    return false;
  } else if (!isValidMsgStream(Msg.Id, Op.Id, Stream.Id, Strict)) {
    Error(S, "invalid message stream id");
    return false;
  }
  return true;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSendMsgOp(OperandVector &Operands) {
  using namespace llvm::AMDGPU::SendMsg;

  int64_t ImmVal = 0;
  SMLoc Loc = getLoc();

  if (trySkipId("sendmsg", AsmToken::LParen)) {
    OperandInfoTy Msg(ID_UNKNOWN_);
    OperandInfoTy Op(OP_NONE_);
    OperandInfoTy Stream(STREAM_ID_NONE_);
    if (parseSendMsgBody(Msg, Op, Stream) &&
        validateSendMsg(Msg, Op, Stream, Loc)) {
      ImmVal = encodeMsg(Msg.Id, Op.Id, Stream.Id);
    } else {
      return MatchOperand_ParseFail;
    }
  } else if (parseExpr(ImmVal)) {
    if (ImmVal < 0 || !isUInt<16>(ImmVal)) {
      Error(Loc, "invalid immediate: only 16-bit values are legal");
      return MatchOperand_ParseFail;
    }
  } else {
    return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, ImmVal, Loc, AMDGPUOperand::ImmTySendMsg));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isSendMsg() const {
  return isImmTy(ImmTySendMsg);
}

//===----------------------------------------------------------------------===//
// v_interp
//===----------------------------------------------------------------------===//

OperandMatchResultTy AMDGPUAsmParser::parseInterpSlot(OperandVector &Operands) {
  if (getLexer().getKind() != AsmToken::Identifier)
    return MatchOperand_NoMatch;

  StringRef Str = Parser.getTok().getString();
  int Slot = StringSwitch<int>(Str)
    .Case("p10", 0)
    .Case("p20", 1)
    .Case("p0", 2)
    .Default(-1);

  SMLoc S = Parser.getTok().getLoc();
  if (Slot == -1)
    return MatchOperand_ParseFail;

  Parser.Lex();
  Operands.push_back(AMDGPUOperand::CreateImm(this, Slot, S,
                                              AMDGPUOperand::ImmTyInterpSlot));
  return MatchOperand_Success;
}

OperandMatchResultTy AMDGPUAsmParser::parseInterpAttr(OperandVector &Operands) {
  if (getLexer().getKind() != AsmToken::Identifier)
    return MatchOperand_NoMatch;

  StringRef Str = Parser.getTok().getString();
  if (!Str.startswith("attr"))
    return MatchOperand_NoMatch;

  StringRef Chan = Str.take_back(2);
  int AttrChan = StringSwitch<int>(Chan)
    .Case(".x", 0)
    .Case(".y", 1)
    .Case(".z", 2)
    .Case(".w", 3)
    .Default(-1);
  if (AttrChan == -1)
    return MatchOperand_ParseFail;

  Str = Str.drop_back(2).drop_front(4);

  uint8_t Attr;
  if (Str.getAsInteger(10, Attr))
    return MatchOperand_ParseFail;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex();
  if (Attr > 63) {
    Error(S, "out of bounds attr");
    return MatchOperand_ParseFail;
  }

  SMLoc SChan = SMLoc::getFromPointer(Chan.data());

  Operands.push_back(AMDGPUOperand::CreateImm(this, Attr, S,
                                              AMDGPUOperand::ImmTyInterpAttr));
  Operands.push_back(AMDGPUOperand::CreateImm(this, AttrChan, SChan,
                                              AMDGPUOperand::ImmTyAttrChan));
  return MatchOperand_Success;
}

//===----------------------------------------------------------------------===//
// exp
//===----------------------------------------------------------------------===//

OperandMatchResultTy AMDGPUAsmParser::parseExpTgtImpl(StringRef Str,
                                                      uint8_t &Val) {
  if (Str == "null") {
    Val = 9;
    return MatchOperand_Success;
  }

  if (Str.startswith("mrt")) {
    Str = Str.drop_front(3);
    if (Str == "z") { // == mrtz
      Val = 8;
      return MatchOperand_Success;
    }

    if (Str.getAsInteger(10, Val))
      return MatchOperand_ParseFail;

    if (Val > 7) {
      Error(getLoc(), "invalid exp target");
      return MatchOperand_ParseFail;
    }

    return MatchOperand_Success;
  }

  if (Str.startswith("pos")) {
    Str = Str.drop_front(3);
    if (Str.getAsInteger(10, Val))
      return MatchOperand_ParseFail;

    if (Val > 4 || (Val == 4 && !isGFX10())) {
      Error(getLoc(), "invalid exp target");
      return MatchOperand_ParseFail;
    }

    Val += 12;
    return MatchOperand_Success;
  }

  if (isGFX10() && Str == "prim") {
    Val = 20;
    return MatchOperand_Success;
  }

  if (Str.startswith("param")) {
    Str = Str.drop_front(5);
    if (Str.getAsInteger(10, Val))
      return MatchOperand_ParseFail;

    if (Val >= 32) {
      Error(getLoc(), "invalid exp target");
      return MatchOperand_ParseFail;
    }

    Val += 32;
    return MatchOperand_Success;
  }

  if (Str.startswith("invalid_target_")) {
    Str = Str.drop_front(15);
    if (Str.getAsInteger(10, Val))
      return MatchOperand_ParseFail;

    Error(getLoc(), "invalid exp target");
    return MatchOperand_ParseFail;
  }

  return MatchOperand_NoMatch;
}

OperandMatchResultTy AMDGPUAsmParser::parseExpTgt(OperandVector &Operands) {
  uint8_t Val;
  StringRef Str = Parser.getTok().getString();

  auto Res = parseExpTgtImpl(Str, Val);
  if (Res != MatchOperand_Success)
    return Res;

  SMLoc S = Parser.getTok().getLoc();
  Parser.Lex();

  Operands.push_back(AMDGPUOperand::CreateImm(this, Val, S,
                                              AMDGPUOperand::ImmTyExpTgt));
  return MatchOperand_Success;
}

//===----------------------------------------------------------------------===//
// parser helpers
//===----------------------------------------------------------------------===//

bool
AMDGPUAsmParser::isId(const AsmToken &Token, const StringRef Id) const {
  return Token.is(AsmToken::Identifier) && Token.getString() == Id;
}

bool
AMDGPUAsmParser::isId(const StringRef Id) const {
  return isId(getToken(), Id);
}

bool
AMDGPUAsmParser::isToken(const AsmToken::TokenKind Kind) const {
  return getTokenKind() == Kind;
}

bool
AMDGPUAsmParser::trySkipId(const StringRef Id) {
  if (isId(Id)) {
    lex();
    return true;
  }
  return false;
}

bool
AMDGPUAsmParser::trySkipId(const StringRef Id, const AsmToken::TokenKind Kind) {
  if (isId(Id) && peekToken().is(Kind)) {
    lex();
    lex();
    return true;
  }
  return false;
}

bool
AMDGPUAsmParser::trySkipToken(const AsmToken::TokenKind Kind) {
  if (isToken(Kind)) {
    lex();
    return true;
  }
  return false;
}

bool
AMDGPUAsmParser::skipToken(const AsmToken::TokenKind Kind,
                           const StringRef ErrMsg) {
  if (!trySkipToken(Kind)) {
    Error(getLoc(), ErrMsg);
    return false;
  }
  return true;
}

bool
AMDGPUAsmParser::parseExpr(int64_t &Imm) {
  return !getParser().parseAbsoluteExpression(Imm);
}

bool
AMDGPUAsmParser::parseExpr(OperandVector &Operands) {
  SMLoc S = getLoc();

  const MCExpr *Expr;
  if (Parser.parseExpression(Expr))
    return false;

  int64_t IntVal;
  if (Expr->evaluateAsAbsolute(IntVal)) {
    Operands.push_back(AMDGPUOperand::CreateImm(this, IntVal, S));
  } else {
    Operands.push_back(AMDGPUOperand::CreateExpr(this, Expr, S));
  }
  return true;
}

bool
AMDGPUAsmParser::parseString(StringRef &Val, const StringRef ErrMsg) {
  if (isToken(AsmToken::String)) {
    Val = getToken().getStringContents();
    lex();
    return true;
  } else {
    Error(getLoc(), ErrMsg);
    return false;
  }
}

bool
AMDGPUAsmParser::parseId(StringRef &Val, const StringRef ErrMsg) {
  if (isToken(AsmToken::Identifier)) {
    Val = getTokenStr();
    lex();
    return true;
  } else {
    Error(getLoc(), ErrMsg);
    return false;
  }
}

AsmToken
AMDGPUAsmParser::getToken() const {
  return Parser.getTok();
}

AsmToken
AMDGPUAsmParser::peekToken() {
  return isToken(AsmToken::EndOfStatement) ? getToken() : getLexer().peekTok();
}

void
AMDGPUAsmParser::peekTokens(MutableArrayRef<AsmToken> Tokens) {
  auto TokCount = getLexer().peekTokens(Tokens);

  for (auto Idx = TokCount; Idx < Tokens.size(); ++Idx)
    Tokens[Idx] = AsmToken(AsmToken::Error, "");
}

AsmToken::TokenKind
AMDGPUAsmParser::getTokenKind() const {
  return getLexer().getKind();
}

SMLoc
AMDGPUAsmParser::getLoc() const {
  return getToken().getLoc();
}

StringRef
AMDGPUAsmParser::getTokenStr() const {
  return getToken().getString();
}

void
AMDGPUAsmParser::lex() {
  Parser.Lex();
}

//===----------------------------------------------------------------------===//
// swizzle
//===----------------------------------------------------------------------===//

LLVM_READNONE
static unsigned
encodeBitmaskPerm(const unsigned AndMask,
                  const unsigned OrMask,
                  const unsigned XorMask) {
  using namespace llvm::AMDGPU::Swizzle;

  return BITMASK_PERM_ENC |
         (AndMask << BITMASK_AND_SHIFT) |
         (OrMask  << BITMASK_OR_SHIFT)  |
         (XorMask << BITMASK_XOR_SHIFT);
}

bool
AMDGPUAsmParser::parseSwizzleOperands(const unsigned OpNum, int64_t* Op,
                                      const unsigned MinVal,
                                      const unsigned MaxVal,
                                      const StringRef ErrMsg) {
  for (unsigned i = 0; i < OpNum; ++i) {
    if (!skipToken(AsmToken::Comma, "expected a comma")){
      return false;
    }
    SMLoc ExprLoc = Parser.getTok().getLoc();
    if (!parseExpr(Op[i])) {
      return false;
    }
    if (Op[i] < MinVal || Op[i] > MaxVal) {
      Error(ExprLoc, ErrMsg);
      return false;
    }
  }

  return true;
}

bool
AMDGPUAsmParser::parseSwizzleQuadPerm(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  int64_t Lane[LANE_NUM];
  if (parseSwizzleOperands(LANE_NUM, Lane, 0, LANE_MAX,
                           "expected a 2-bit lane id")) {
    Imm = QUAD_PERM_ENC;
    for (unsigned I = 0; I < LANE_NUM; ++I) {
      Imm |= Lane[I] << (LANE_SHIFT * I);
    }
    return true;
  }
  return false;
}

bool
AMDGPUAsmParser::parseSwizzleBroadcast(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  SMLoc S = Parser.getTok().getLoc();
  int64_t GroupSize;
  int64_t LaneIdx;

  if (!parseSwizzleOperands(1, &GroupSize,
                            2, 32,
                            "group size must be in the interval [2,32]")) {
    return false;
  }
  if (!isPowerOf2_64(GroupSize)) {
    Error(S, "group size must be a power of two");
    return false;
  }
  if (parseSwizzleOperands(1, &LaneIdx,
                           0, GroupSize - 1,
                           "lane id must be in the interval [0,group size - 1]")) {
    Imm = encodeBitmaskPerm(BITMASK_MAX - GroupSize + 1, LaneIdx, 0);
    return true;
  }
  return false;
}

bool
AMDGPUAsmParser::parseSwizzleReverse(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  SMLoc S = Parser.getTok().getLoc();
  int64_t GroupSize;

  if (!parseSwizzleOperands(1, &GroupSize,
      2, 32, "group size must be in the interval [2,32]")) {
    return false;
  }
  if (!isPowerOf2_64(GroupSize)) {
    Error(S, "group size must be a power of two");
    return false;
  }

  Imm = encodeBitmaskPerm(BITMASK_MAX, 0, GroupSize - 1);
  return true;
}

bool
AMDGPUAsmParser::parseSwizzleSwap(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  SMLoc S = Parser.getTok().getLoc();
  int64_t GroupSize;

  if (!parseSwizzleOperands(1, &GroupSize,
      1, 16, "group size must be in the interval [1,16]")) {
    return false;
  }
  if (!isPowerOf2_64(GroupSize)) {
    Error(S, "group size must be a power of two");
    return false;
  }

  Imm = encodeBitmaskPerm(BITMASK_MAX, 0, GroupSize);
  return true;
}

bool
AMDGPUAsmParser::parseSwizzleBitmaskPerm(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  if (!skipToken(AsmToken::Comma, "expected a comma")) {
    return false;
  }

  StringRef Ctl;
  SMLoc StrLoc = Parser.getTok().getLoc();
  if (!parseString(Ctl)) {
    return false;
  }
  if (Ctl.size() != BITMASK_WIDTH) {
    Error(StrLoc, "expected a 5-character mask");
    return false;
  }

  unsigned AndMask = 0;
  unsigned OrMask = 0;
  unsigned XorMask = 0;

  for (size_t i = 0; i < Ctl.size(); ++i) {
    unsigned Mask = 1 << (BITMASK_WIDTH - 1 - i);
    switch(Ctl[i]) {
    default:
      Error(StrLoc, "invalid mask");
      return false;
    case '0':
      break;
    case '1':
      OrMask |= Mask;
      break;
    case 'p':
      AndMask |= Mask;
      break;
    case 'i':
      AndMask |= Mask;
      XorMask |= Mask;
      break;
    }
  }

  Imm = encodeBitmaskPerm(AndMask, OrMask, XorMask);
  return true;
}

bool
AMDGPUAsmParser::parseSwizzleOffset(int64_t &Imm) {

  SMLoc OffsetLoc = Parser.getTok().getLoc();

  if (!parseExpr(Imm)) {
    return false;
  }
  if (!isUInt<16>(Imm)) {
    Error(OffsetLoc, "expected a 16-bit offset");
    return false;
  }
  return true;
}

bool
AMDGPUAsmParser::parseSwizzleMacro(int64_t &Imm) {
  using namespace llvm::AMDGPU::Swizzle;

  if (skipToken(AsmToken::LParen, "expected a left parentheses")) {

    SMLoc ModeLoc = Parser.getTok().getLoc();
    bool Ok = false;

    if (trySkipId(IdSymbolic[ID_QUAD_PERM])) {
      Ok = parseSwizzleQuadPerm(Imm);
    } else if (trySkipId(IdSymbolic[ID_BITMASK_PERM])) {
      Ok = parseSwizzleBitmaskPerm(Imm);
    } else if (trySkipId(IdSymbolic[ID_BROADCAST])) {
      Ok = parseSwizzleBroadcast(Imm);
    } else if (trySkipId(IdSymbolic[ID_SWAP])) {
      Ok = parseSwizzleSwap(Imm);
    } else if (trySkipId(IdSymbolic[ID_REVERSE])) {
      Ok = parseSwizzleReverse(Imm);
    } else {
      Error(ModeLoc, "expected a swizzle mode");
    }

    return Ok && skipToken(AsmToken::RParen, "expected a closing parentheses");
  }

  return false;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSwizzleOp(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  int64_t Imm = 0;

  if (trySkipId("offset")) {

    bool Ok = false;
    if (skipToken(AsmToken::Colon, "expected a colon")) {
      if (trySkipId("swizzle")) {
        Ok = parseSwizzleMacro(Imm);
      } else {
        Ok = parseSwizzleOffset(Imm);
      }
    }

    Operands.push_back(AMDGPUOperand::CreateImm(this, Imm, S, AMDGPUOperand::ImmTySwizzle));

    return Ok? MatchOperand_Success : MatchOperand_ParseFail;
  } else {
    // Swizzle "offset" operand is optional.
    // If it is omitted, try parsing other optional operands.
    return parseOptionalOpr(Operands);
  }
}

bool
AMDGPUOperand::isSwizzle() const {
  return isImmTy(ImmTySwizzle);
}

//===----------------------------------------------------------------------===//
// VGPR Index Mode
//===----------------------------------------------------------------------===//

int64_t AMDGPUAsmParser::parseGPRIdxMacro() {

  using namespace llvm::AMDGPU::VGPRIndexMode;

  if (trySkipToken(AsmToken::RParen)) {
    return OFF;
  }

  int64_t Imm = 0;

  while (true) {
    unsigned Mode = 0;
    SMLoc S = Parser.getTok().getLoc();

    for (unsigned ModeId = ID_MIN; ModeId <= ID_MAX; ++ModeId) {
      if (trySkipId(IdSymbolic[ModeId])) {
        Mode = 1 << ModeId;
        break;
      }
    }

    if (Mode == 0) {
      Error(S, (Imm == 0)?
               "expected a VGPR index mode or a closing parenthesis" :
               "expected a VGPR index mode");
      return UNDEF;
    }

    if (Imm & Mode) {
      Error(S, "duplicate VGPR index mode");
      return UNDEF;
    }
    Imm |= Mode;

    if (trySkipToken(AsmToken::RParen))
      break;
    if (!skipToken(AsmToken::Comma,
                   "expected a comma or a closing parenthesis"))
      return UNDEF;
  }

  return Imm;
}

OperandMatchResultTy
AMDGPUAsmParser::parseGPRIdxMode(OperandVector &Operands) {

  using namespace llvm::AMDGPU::VGPRIndexMode;

  int64_t Imm = 0;
  SMLoc S = Parser.getTok().getLoc();

  if (getLexer().getKind() == AsmToken::Identifier &&
      Parser.getTok().getString() == "gpr_idx" &&
      getLexer().peekTok().is(AsmToken::LParen)) {

    Parser.Lex();
    Parser.Lex();

    Imm = parseGPRIdxMacro();
    if (Imm == UNDEF)
      return MatchOperand_ParseFail;

  } else {
    if (getParser().parseAbsoluteExpression(Imm))
      return MatchOperand_ParseFail;
    if (Imm < 0 || !isUInt<4>(Imm)) {
      Error(S, "invalid immediate: only 4-bit values are legal");
      return MatchOperand_ParseFail;
    }
  }

  Operands.push_back(
      AMDGPUOperand::CreateImm(this, Imm, S, AMDGPUOperand::ImmTyGprIdxMode));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isGPRIdxMode() const {
  return isImmTy(ImmTyGprIdxMode);
}

//===----------------------------------------------------------------------===//
// sopp branch targets
//===----------------------------------------------------------------------===//

OperandMatchResultTy
AMDGPUAsmParser::parseSOppBrTarget(OperandVector &Operands) {

  // Make sure we are not parsing something
  // that looks like a label or an expression but is not.
  // This will improve error messages.
  if (isRegister() || isModifier())
    return MatchOperand_NoMatch;

  if (!parseExpr(Operands))
    return MatchOperand_ParseFail;

  AMDGPUOperand &Opr = ((AMDGPUOperand &)*Operands[Operands.size() - 1]);
  assert(Opr.isImm() || Opr.isExpr());
  SMLoc Loc = Opr.getStartLoc();

  // Currently we do not support arbitrary expressions as branch targets.
  // Only labels and absolute expressions are accepted.
  if (Opr.isExpr() && !Opr.isSymbolRefExpr()) {
    Error(Loc, "expected an absolute expression or a label");
  } else if (Opr.isImm() && !Opr.isS16Imm()) {
    Error(Loc, "expected a 16-bit signed jump offset");
  }

  return MatchOperand_Success;
}

//===----------------------------------------------------------------------===//
// Boolean holding registers
//===----------------------------------------------------------------------===//

OperandMatchResultTy
AMDGPUAsmParser::parseBoolReg(OperandVector &Operands) {
  return parseReg(Operands);
}

//===----------------------------------------------------------------------===//
// mubuf
//===----------------------------------------------------------------------===//

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultDLC() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyDLC);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultGLC() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyGLC);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultSLC() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTySLC);
}

void AMDGPUAsmParser::cvtMubufImpl(MCInst &Inst,
                               const OperandVector &Operands,
                               bool IsAtomic,
                               bool IsAtomicReturn,
                               bool IsLds) {
  bool IsLdsOpcode = IsLds;
  bool HasLdsModifier = false;
  OptionalImmIndexMap OptionalIdx;
  assert(IsAtomicReturn ? IsAtomic : true);
  unsigned FirstOperandIdx = 1;

  for (unsigned i = FirstOperandIdx, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      // Insert a tied src for atomic return dst.
      // This cannot be postponed as subsequent calls to
      // addImmOperands rely on correct number of MC operands.
      if (IsAtomicReturn && i == FirstOperandIdx)
        Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle the case where soffset is an immediate
    if (Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyNone) {
      Op.addImmOperands(Inst, 1);
      continue;
    }

    HasLdsModifier |= Op.isLDS();

    // Handle tokens like 'offen' which are sometimes hard-coded into the
    // asm string.  There are no MCInst operands for these.
    if (Op.isToken()) {
      continue;
    }
    assert(Op.isImm());

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  // This is a workaround for an llvm quirk which may result in an
  // incorrect instruction selection. Lds and non-lds versions of
  // MUBUF instructions are identical except that lds versions
  // have mandatory 'lds' modifier. However this modifier follows
  // optional modifiers and llvm asm matcher regards this 'lds'
  // modifier as an optional one. As a result, an lds version
  // of opcode may be selected even if it has no 'lds' modifier.
  if (IsLdsOpcode && !HasLdsModifier) {
    int NoLdsOpcode = AMDGPU::getMUBUFNoLdsInst(Inst.getOpcode());
    if (NoLdsOpcode != -1) { // Got lds version - correct it.
      Inst.setOpcode(NoLdsOpcode);
      IsLdsOpcode = false;
    }
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOffset);
  if (!IsAtomic) { // glc is hard-coded.
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  }
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);

  if (!IsLdsOpcode) { // tfe is not legal with lds opcodes
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
  }

  if (isGFX10())
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDLC);
}

void AMDGPUAsmParser::cvtMtbuf(MCInst &Inst, const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;

  for (unsigned i = 1, e = Operands.size(); i != e; ++i) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[i]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
      continue;
    }

    // Handle the case where soffset is an immediate
    if (Op.isImm() && Op.getImmTy() == AMDGPUOperand::ImmTyNone) {
      Op.addImmOperands(Inst, 1);
      continue;
    }

    // Handle tokens like 'offen' which are sometimes hard-coded into the
    // asm string.  There are no MCInst operands for these.
    if (Op.isToken()) {
      continue;
    }
    assert(Op.isImm());

    // Handle optional arguments
    OptionalIdx[Op.getImmTy()] = i;
  }

  addOptionalImmOperand(Inst, Operands, OptionalIdx,
                        AMDGPUOperand::ImmTyOffset);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyFORMAT);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);

  if (isGFX10())
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDLC);
}

//===----------------------------------------------------------------------===//
// mimg
//===----------------------------------------------------------------------===//

void AMDGPUAsmParser::cvtMIMG(MCInst &Inst, const OperandVector &Operands,
                              bool IsAtomic) {
  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  if (IsAtomic) {
    // Add src, same as dst
    assert(Desc.getNumDefs() == 1);
    ((AMDGPUOperand &)*Operands[I - 1]).addRegOperands(Inst, 1);
  }

  OptionalImmIndexMap OptionalIdx;

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);

    // Add the register arguments
    if (Op.isReg()) {
      Op.addRegOperands(Inst, 1);
    } else if (Op.isImmModifier()) {
      OptionalIdx[Op.getImmTy()] = I;
    } else if (!Op.isToken()) {
      llvm_unreachable("unexpected operand type");
    }
  }

  bool IsGFX10 = isGFX10();

  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDMask);
  if (IsGFX10)
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDim, -1);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyUNorm);
  if (IsGFX10)
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyGLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySLC);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyR128A16);
  if (IsGFX10)
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyA16);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyTFE);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyLWE);
  if (!IsGFX10)
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDA);
  addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyD16);
}

void AMDGPUAsmParser::cvtMIMGAtomic(MCInst &Inst, const OperandVector &Operands) {
  cvtMIMG(Inst, Operands, true);
}

//===----------------------------------------------------------------------===//
// smrd
//===----------------------------------------------------------------------===//

bool AMDGPUOperand::isSMRDOffset8() const {
  return isImm() && isUInt<8>(getImm());
}

bool AMDGPUOperand::isSMEMOffset() const {
  return isImm(); // Offset range is checked later by validator.
}

bool AMDGPUOperand::isSMRDLiteralOffset() const {
  // 32-bit literals are only supported on CI and we only want to use them
  // when the offset is > 8-bits.
  return isImm() && !isUInt<8>(getImm()) && isUInt<32>(getImm());
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultSMRDOffset8() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyOffset);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultSMEMOffset() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyOffset);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultSMRDLiteralOffset() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyOffset);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultFlatOffset() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyOffset);
}

//===----------------------------------------------------------------------===//
// vop3
//===----------------------------------------------------------------------===//

static bool ConvertOmodMul(int64_t &Mul) {
  if (Mul != 1 && Mul != 2 && Mul != 4)
    return false;

  Mul >>= 1;
  return true;
}

static bool ConvertOmodDiv(int64_t &Div) {
  if (Div == 1) {
    Div = 0;
    return true;
  }

  if (Div == 2) {
    Div = 3;
    return true;
  }

  return false;
}

static bool ConvertBoundCtrl(int64_t &BoundCtrl) {
  if (BoundCtrl == 0) {
    BoundCtrl = 1;
    return true;
  }

  if (BoundCtrl == -1) {
    BoundCtrl = 0;
    return true;
  }

  return false;
}

// Note: the order in this table matches the order of operands in AsmString.
static const OptionalOperand AMDGPUOptionalOperandTable[] = {
  {"offen",   AMDGPUOperand::ImmTyOffen, true, nullptr},
  {"idxen",   AMDGPUOperand::ImmTyIdxen, true, nullptr},
  {"addr64",  AMDGPUOperand::ImmTyAddr64, true, nullptr},
  {"offset0", AMDGPUOperand::ImmTyOffset0, false, nullptr},
  {"offset1", AMDGPUOperand::ImmTyOffset1, false, nullptr},
  {"gds",     AMDGPUOperand::ImmTyGDS, true, nullptr},
  {"lds",     AMDGPUOperand::ImmTyLDS, true, nullptr},
  {"offset",  AMDGPUOperand::ImmTyOffset, false, nullptr},
  {"inst_offset", AMDGPUOperand::ImmTyInstOffset, false, nullptr},
  {"dlc",     AMDGPUOperand::ImmTyDLC, true, nullptr},
  {"glc",     AMDGPUOperand::ImmTyGLC, true, nullptr},
  {"slc",     AMDGPUOperand::ImmTySLC, true, nullptr},
  {"swz",     AMDGPUOperand::ImmTySWZ, true, nullptr},
  {"tfe",     AMDGPUOperand::ImmTyTFE, true, nullptr},
  {"d16",     AMDGPUOperand::ImmTyD16, true, nullptr},
  {"high",    AMDGPUOperand::ImmTyHigh, true, nullptr},
  {"clamp",   AMDGPUOperand::ImmTyClampSI, true, nullptr},
  {"omod",    AMDGPUOperand::ImmTyOModSI, false, ConvertOmodMul},
  {"unorm",   AMDGPUOperand::ImmTyUNorm, true, nullptr},
  {"da",      AMDGPUOperand::ImmTyDA,    true, nullptr},
  {"r128",    AMDGPUOperand::ImmTyR128A16,  true, nullptr},
  {"a16",     AMDGPUOperand::ImmTyA16,  true, nullptr},
  {"lwe",     AMDGPUOperand::ImmTyLWE,   true, nullptr},
  {"d16",     AMDGPUOperand::ImmTyD16,   true, nullptr},
  {"dmask",   AMDGPUOperand::ImmTyDMask, false, nullptr},
  {"dim",     AMDGPUOperand::ImmTyDim,   false, nullptr},
  {"row_mask",   AMDGPUOperand::ImmTyDppRowMask, false, nullptr},
  {"bank_mask",  AMDGPUOperand::ImmTyDppBankMask, false, nullptr},
  {"bound_ctrl", AMDGPUOperand::ImmTyDppBoundCtrl, false, ConvertBoundCtrl},
  {"fi",         AMDGPUOperand::ImmTyDppFi, false, nullptr},
  {"dst_sel",    AMDGPUOperand::ImmTySdwaDstSel, false, nullptr},
  {"src0_sel",   AMDGPUOperand::ImmTySdwaSrc0Sel, false, nullptr},
  {"src1_sel",   AMDGPUOperand::ImmTySdwaSrc1Sel, false, nullptr},
  {"dst_unused", AMDGPUOperand::ImmTySdwaDstUnused, false, nullptr},
  {"compr", AMDGPUOperand::ImmTyExpCompr, true, nullptr },
  {"vm", AMDGPUOperand::ImmTyExpVM, true, nullptr},
  {"op_sel", AMDGPUOperand::ImmTyOpSel, false, nullptr},
  {"op_sel_hi", AMDGPUOperand::ImmTyOpSelHi, false, nullptr},
  {"neg_lo", AMDGPUOperand::ImmTyNegLo, false, nullptr},
  {"neg_hi", AMDGPUOperand::ImmTyNegHi, false, nullptr},
  {"blgp", AMDGPUOperand::ImmTyBLGP, false, nullptr},
  {"cbsz", AMDGPUOperand::ImmTyCBSZ, false, nullptr},
  {"abid", AMDGPUOperand::ImmTyABID, false, nullptr}
};

OperandMatchResultTy AMDGPUAsmParser::parseOptionalOperand(OperandVector &Operands) {

  OperandMatchResultTy res = parseOptionalOpr(Operands);

  // This is a hack to enable hardcoded mandatory operands which follow
  // optional operands.
  //
  // Current design assumes that all operands after the first optional operand
  // are also optional. However implementation of some instructions violates
  // this rule (see e.g. flat/global atomic which have hardcoded 'glc' operands).
  //
  // To alleviate this problem, we have to (implicitly) parse extra operands
  // to make sure autogenerated parser of custom operands never hit hardcoded
  // mandatory operands.

  for (unsigned i = 0; i < MAX_OPR_LOOKAHEAD; ++i) {
    if (res != MatchOperand_Success ||
        isToken(AsmToken::EndOfStatement))
      break;

    trySkipToken(AsmToken::Comma);
    res = parseOptionalOpr(Operands);
  }

  return res;
}

OperandMatchResultTy AMDGPUAsmParser::parseOptionalOpr(OperandVector &Operands) {
  OperandMatchResultTy res;
  for (const OptionalOperand &Op : AMDGPUOptionalOperandTable) {
    // try to parse any optional operand here
    if (Op.IsBit) {
      res = parseNamedBit(Op.Name, Operands, Op.Type);
    } else if (Op.Type == AMDGPUOperand::ImmTyOModSI) {
      res = parseOModOperand(Operands);
    } else if (Op.Type == AMDGPUOperand::ImmTySdwaDstSel ||
               Op.Type == AMDGPUOperand::ImmTySdwaSrc0Sel ||
               Op.Type == AMDGPUOperand::ImmTySdwaSrc1Sel) {
      res = parseSDWASel(Operands, Op.Name, Op.Type);
    } else if (Op.Type == AMDGPUOperand::ImmTySdwaDstUnused) {
      res = parseSDWADstUnused(Operands);
    } else if (Op.Type == AMDGPUOperand::ImmTyOpSel ||
               Op.Type == AMDGPUOperand::ImmTyOpSelHi ||
               Op.Type == AMDGPUOperand::ImmTyNegLo ||
               Op.Type == AMDGPUOperand::ImmTyNegHi) {
      res = parseOperandArrayWithPrefix(Op.Name, Operands, Op.Type,
                                        Op.ConvertResult);
    } else if (Op.Type == AMDGPUOperand::ImmTyDim) {
      res = parseDim(Operands);
    } else {
      res = parseIntWithPrefix(Op.Name, Operands, Op.Type, Op.ConvertResult);
    }
    if (res != MatchOperand_NoMatch) {
      return res;
    }
  }
  return MatchOperand_NoMatch;
}

OperandMatchResultTy AMDGPUAsmParser::parseOModOperand(OperandVector &Operands) {
  StringRef Name = Parser.getTok().getString();
  if (Name == "mul") {
    return parseIntWithPrefix("mul", Operands,
                              AMDGPUOperand::ImmTyOModSI, ConvertOmodMul);
  }

  if (Name == "div") {
    return parseIntWithPrefix("div", Operands,
                              AMDGPUOperand::ImmTyOModSI, ConvertOmodDiv);
  }

  return MatchOperand_NoMatch;
}

void AMDGPUAsmParser::cvtVOP3OpSel(MCInst &Inst, const OperandVector &Operands) {
  cvtVOP3P(Inst, Operands);

  int Opc = Inst.getOpcode();

  int SrcNum;
  const int Ops[] = { AMDGPU::OpName::src0,
                      AMDGPU::OpName::src1,
                      AMDGPU::OpName::src2 };
  for (SrcNum = 0;
       SrcNum < 3 && AMDGPU::getNamedOperandIdx(Opc, Ops[SrcNum]) != -1;
       ++SrcNum);
  assert(SrcNum > 0);

  int OpSelIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel);
  unsigned OpSel = Inst.getOperand(OpSelIdx).getImm();

  if ((OpSel & (1 << SrcNum)) != 0) {
    int ModIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
    uint32_t ModVal = Inst.getOperand(ModIdx).getImm();
    Inst.getOperand(ModIdx).setImm(ModVal | SISrcMods::DST_OP_SEL);
  }
}

static bool isRegOrImmWithInputMods(const MCInstrDesc &Desc, unsigned OpNum) {
      // 1. This operand is input modifiers
  return Desc.OpInfo[OpNum].OperandType == AMDGPU::OPERAND_INPUT_MODS
      // 2. This is not last operand
      && Desc.NumOperands > (OpNum + 1)
      // 3. Next operand is register class
      && Desc.OpInfo[OpNum + 1].RegClass != -1
      // 4. Next register is not tied to any other operand
      && Desc.getOperandConstraint(OpNum + 1, MCOI::OperandConstraint::TIED_TO) == -1;
}

void AMDGPUAsmParser::cvtVOP3Interp(MCInst &Inst, const OperandVector &Operands)
{
  OptionalImmIndexMap OptionalIdx;
  unsigned Opc = Inst.getOpcode();

  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
    if (isRegOrImmWithInputMods(Desc, Inst.getNumOperands())) {
      Op.addRegOrImmWithFPInputModsOperands(Inst, 2);
    } else if (Op.isInterpSlot() ||
               Op.isInterpAttr() ||
               Op.isAttrChan()) {
      Inst.addOperand(MCOperand::createImm(Op.getImm()));
    } else if (Op.isImmModifier()) {
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      llvm_unreachable("unhandled operand type");
    }
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::high) != -1) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyHigh);
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::clamp) != -1) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClampSI);
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::omod) != -1) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOModSI);
  }
}

void AMDGPUAsmParser::cvtVOP3(MCInst &Inst, const OperandVector &Operands,
                              OptionalImmIndexMap &OptionalIdx) {
  unsigned Opc = Inst.getOpcode();

  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers) != -1) {
    // This instruction has src modifiers
    for (unsigned E = Operands.size(); I != E; ++I) {
      AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
      if (isRegOrImmWithInputMods(Desc, Inst.getNumOperands())) {
        Op.addRegOrImmWithFPInputModsOperands(Inst, 2);
      } else if (Op.isImmModifier()) {
        OptionalIdx[Op.getImmTy()] = I;
      } else if (Op.isRegOrImm()) {
        Op.addRegOrImmOperands(Inst, 1);
      } else {
        llvm_unreachable("unhandled operand type");
      }
    }
  } else {
    // No src modifiers
    for (unsigned E = Operands.size(); I != E; ++I) {
      AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
      if (Op.isMod()) {
        OptionalIdx[Op.getImmTy()] = I;
      } else {
        Op.addRegOrImmOperands(Inst, 1);
      }
    }
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::clamp) != -1) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClampSI);
  }

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::omod) != -1) {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOModSI);
  }

  // Special case v_mac_{f16, f32} and v_fmac_{f16, f32} (gfx906/gfx10+):
  // it has src2 register operand that is tied to dst operand
  // we don't allow modifiers for this operand in assembler so src2_modifiers
  // should be 0.
  if (Opc == AMDGPU::V_MAC_F32_e64_gfx6_gfx7 ||
      Opc == AMDGPU::V_MAC_F32_e64_gfx10 ||
      Opc == AMDGPU::V_MAC_F32_e64_vi ||
      Opc == AMDGPU::V_MAC_F16_e64_vi ||
      Opc == AMDGPU::V_FMAC_F32_e64_gfx10 ||
      Opc == AMDGPU::V_FMAC_F32_e64_vi ||
      Opc == AMDGPU::V_FMAC_F16_e64_gfx10) {
    auto it = Inst.begin();
    std::advance(it, AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2_modifiers));
    it = Inst.insert(it, MCOperand::createImm(0)); // no modifiers for src2
    ++it;
    Inst.insert(it, Inst.getOperand(0)); // src2 = dst
  }
}

void AMDGPUAsmParser::cvtVOP3(MCInst &Inst, const OperandVector &Operands) {
  OptionalImmIndexMap OptionalIdx;
  cvtVOP3(Inst, Operands, OptionalIdx);
}

void AMDGPUAsmParser::cvtVOP3P(MCInst &Inst,
                               const OperandVector &Operands) {
  OptionalImmIndexMap OptIdx;
  const int Opc = Inst.getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);

  const bool IsPacked = (Desc.TSFlags & SIInstrFlags::IsPacked) != 0;

  cvtVOP3(Inst, Operands, OptIdx);

  if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst_in) != -1) {
    assert(!IsPacked);
    Inst.addOperand(Inst.getOperand(0));
  }

  // FIXME: This is messy. Parse the modifiers as if it was a normal VOP3
  // instruction, and then figure out where to actually put the modifiers

  addOptionalImmOperand(Inst, Operands, OptIdx, AMDGPUOperand::ImmTyOpSel);

  int OpSelHiIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel_hi);
  if (OpSelHiIdx != -1) {
    int DefaultVal = IsPacked ? -1 : 0;
    addOptionalImmOperand(Inst, Operands, OptIdx, AMDGPUOperand::ImmTyOpSelHi,
                          DefaultVal);
  }

  int NegLoIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::neg_lo);
  if (NegLoIdx != -1) {
    assert(IsPacked);
    addOptionalImmOperand(Inst, Operands, OptIdx, AMDGPUOperand::ImmTyNegLo);
    addOptionalImmOperand(Inst, Operands, OptIdx, AMDGPUOperand::ImmTyNegHi);
  }

  const int Ops[] = { AMDGPU::OpName::src0,
                      AMDGPU::OpName::src1,
                      AMDGPU::OpName::src2 };
  const int ModOps[] = { AMDGPU::OpName::src0_modifiers,
                         AMDGPU::OpName::src1_modifiers,
                         AMDGPU::OpName::src2_modifiers };

  int OpSelIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel);

  unsigned OpSel = Inst.getOperand(OpSelIdx).getImm();
  unsigned OpSelHi = 0;
  unsigned NegLo = 0;
  unsigned NegHi = 0;

  if (OpSelHiIdx != -1) {
    OpSelHi = Inst.getOperand(OpSelHiIdx).getImm();
  }

  if (NegLoIdx != -1) {
    int NegHiIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::neg_hi);
    NegLo = Inst.getOperand(NegLoIdx).getImm();
    NegHi = Inst.getOperand(NegHiIdx).getImm();
  }

  for (int J = 0; J < 3; ++J) {
    int OpIdx = AMDGPU::getNamedOperandIdx(Opc, Ops[J]);
    if (OpIdx == -1)
      break;

    uint32_t ModVal = 0;

    if ((OpSel & (1 << J)) != 0)
      ModVal |= SISrcMods::OP_SEL_0;

    if ((OpSelHi & (1 << J)) != 0)
      ModVal |= SISrcMods::OP_SEL_1;

    if ((NegLo & (1 << J)) != 0)
      ModVal |= SISrcMods::NEG;

    if ((NegHi & (1 << J)) != 0)
      ModVal |= SISrcMods::NEG_HI;

    int ModIdx = AMDGPU::getNamedOperandIdx(Opc, ModOps[J]);

    Inst.getOperand(ModIdx).setImm(Inst.getOperand(ModIdx).getImm() | ModVal);
  }
}

//===----------------------------------------------------------------------===//
// dpp
//===----------------------------------------------------------------------===//

bool AMDGPUOperand::isDPP8() const {
  return isImmTy(ImmTyDPP8);
}

bool AMDGPUOperand::isDPPCtrl() const {
  using namespace AMDGPU::DPP;

  bool result = isImm() && getImmTy() == ImmTyDppCtrl && isUInt<9>(getImm());
  if (result) {
    int64_t Imm = getImm();
    return (Imm >= DppCtrl::QUAD_PERM_FIRST && Imm <= DppCtrl::QUAD_PERM_LAST) ||
           (Imm >= DppCtrl::ROW_SHL_FIRST && Imm <= DppCtrl::ROW_SHL_LAST) ||
           (Imm >= DppCtrl::ROW_SHR_FIRST && Imm <= DppCtrl::ROW_SHR_LAST) ||
           (Imm >= DppCtrl::ROW_ROR_FIRST && Imm <= DppCtrl::ROW_ROR_LAST) ||
           (Imm == DppCtrl::WAVE_SHL1) ||
           (Imm == DppCtrl::WAVE_ROL1) ||
           (Imm == DppCtrl::WAVE_SHR1) ||
           (Imm == DppCtrl::WAVE_ROR1) ||
           (Imm == DppCtrl::ROW_MIRROR) ||
           (Imm == DppCtrl::ROW_HALF_MIRROR) ||
           (Imm == DppCtrl::BCAST15) ||
           (Imm == DppCtrl::BCAST31) ||
           (Imm >= DppCtrl::ROW_SHARE_FIRST && Imm <= DppCtrl::ROW_SHARE_LAST) ||
           (Imm >= DppCtrl::ROW_XMASK_FIRST && Imm <= DppCtrl::ROW_XMASK_LAST);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// mAI
//===----------------------------------------------------------------------===//

bool AMDGPUOperand::isBLGP() const {
  return isImm() && getImmTy() == ImmTyBLGP && isUInt<3>(getImm());
}

bool AMDGPUOperand::isCBSZ() const {
  return isImm() && getImmTy() == ImmTyCBSZ && isUInt<3>(getImm());
}

bool AMDGPUOperand::isABID() const {
  return isImm() && getImmTy() == ImmTyABID && isUInt<4>(getImm());
}

bool AMDGPUOperand::isS16Imm() const {
  return isImm() && (isInt<16>(getImm()) || isUInt<16>(getImm()));
}

bool AMDGPUOperand::isU16Imm() const {
  return isImm() && isUInt<16>(getImm());
}

OperandMatchResultTy AMDGPUAsmParser::parseDim(OperandVector &Operands) {
  if (!isGFX10())
    return MatchOperand_NoMatch;

  SMLoc S = Parser.getTok().getLoc();

  if (getLexer().isNot(AsmToken::Identifier))
    return MatchOperand_NoMatch;
  if (getLexer().getTok().getString() != "dim")
    return MatchOperand_NoMatch;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Colon))
    return MatchOperand_ParseFail;

  Parser.Lex();

  // We want to allow "dim:1D" etc., but the initial 1 is tokenized as an
  // integer.
  std::string Token;
  if (getLexer().is(AsmToken::Integer)) {
    SMLoc Loc = getLexer().getTok().getEndLoc();
    Token = std::string(getLexer().getTok().getString());
    Parser.Lex();
    if (getLexer().getTok().getLoc() != Loc)
      return MatchOperand_ParseFail;
  }
  if (getLexer().isNot(AsmToken::Identifier))
    return MatchOperand_ParseFail;
  Token += getLexer().getTok().getString();

  StringRef DimId = Token;
  if (DimId.startswith("SQ_RSRC_IMG_"))
    DimId = DimId.substr(12);

  const AMDGPU::MIMGDimInfo *DimInfo = AMDGPU::getMIMGDimInfoByAsmSuffix(DimId);
  if (!DimInfo)
    return MatchOperand_ParseFail;

  Parser.Lex();

  Operands.push_back(AMDGPUOperand::CreateImm(this, DimInfo->Encoding, S,
                                              AMDGPUOperand::ImmTyDim));
  return MatchOperand_Success;
}

OperandMatchResultTy AMDGPUAsmParser::parseDPP8(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  StringRef Prefix;

  if (getLexer().getKind() == AsmToken::Identifier) {
    Prefix = Parser.getTok().getString();
  } else {
    return MatchOperand_NoMatch;
  }

  if (Prefix != "dpp8")
    return parseDPPCtrl(Operands);
  if (!isGFX10())
    return MatchOperand_NoMatch;

  // dpp8:[%d,%d,%d,%d,%d,%d,%d,%d]

  int64_t Sels[8];

  Parser.Lex();
  if (getLexer().isNot(AsmToken::Colon))
    return MatchOperand_ParseFail;

  Parser.Lex();
  if (getLexer().isNot(AsmToken::LBrac))
    return MatchOperand_ParseFail;

  Parser.Lex();
  if (getParser().parseAbsoluteExpression(Sels[0]))
    return MatchOperand_ParseFail;
  if (0 > Sels[0] || 7 < Sels[0])
    return MatchOperand_ParseFail;

  for (size_t i = 1; i < 8; ++i) {
    if (getLexer().isNot(AsmToken::Comma))
      return MatchOperand_ParseFail;

    Parser.Lex();
    if (getParser().parseAbsoluteExpression(Sels[i]))
      return MatchOperand_ParseFail;
    if (0 > Sels[i] || 7 < Sels[i])
      return MatchOperand_ParseFail;
  }

  if (getLexer().isNot(AsmToken::RBrac))
    return MatchOperand_ParseFail;
  Parser.Lex();

  unsigned DPP8 = 0;
  for (size_t i = 0; i < 8; ++i)
    DPP8 |= (Sels[i] << (i * 3));

  Operands.push_back(AMDGPUOperand::CreateImm(this, DPP8, S, AMDGPUOperand::ImmTyDPP8));
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseDPPCtrl(OperandVector &Operands) {
  using namespace AMDGPU::DPP;

  SMLoc S = Parser.getTok().getLoc();
  StringRef Prefix;
  int64_t Int;

  if (getLexer().getKind() == AsmToken::Identifier) {
    Prefix = Parser.getTok().getString();
  } else {
    return MatchOperand_NoMatch;
  }

  if (Prefix == "row_mirror") {
    Int = DppCtrl::ROW_MIRROR;
    Parser.Lex();
  } else if (Prefix == "row_half_mirror") {
    Int = DppCtrl::ROW_HALF_MIRROR;
    Parser.Lex();
  } else {
    // Check to prevent parseDPPCtrlOps from eating invalid tokens
    if (Prefix != "quad_perm"
        && Prefix != "row_shl"
        && Prefix != "row_shr"
        && Prefix != "row_ror"
        && Prefix != "wave_shl"
        && Prefix != "wave_rol"
        && Prefix != "wave_shr"
        && Prefix != "wave_ror"
        && Prefix != "row_bcast"
        && Prefix != "row_share"
        && Prefix != "row_xmask") {
      return MatchOperand_NoMatch;
    }

    if (!isGFX10() && (Prefix == "row_share" || Prefix == "row_xmask"))
      return MatchOperand_NoMatch;

    if (!isVI() && !isGFX9() &&
        (Prefix == "wave_shl" || Prefix == "wave_shr" ||
         Prefix == "wave_rol" || Prefix == "wave_ror" ||
         Prefix == "row_bcast"))
      return MatchOperand_NoMatch;

    Parser.Lex();
    if (getLexer().isNot(AsmToken::Colon))
      return MatchOperand_ParseFail;

    if (Prefix == "quad_perm") {
      // quad_perm:[%d,%d,%d,%d]
      Parser.Lex();
      if (getLexer().isNot(AsmToken::LBrac))
        return MatchOperand_ParseFail;
      Parser.Lex();

      if (getParser().parseAbsoluteExpression(Int) || !(0 <= Int && Int <=3))
        return MatchOperand_ParseFail;

      for (int i = 0; i < 3; ++i) {
        if (getLexer().isNot(AsmToken::Comma))
          return MatchOperand_ParseFail;
        Parser.Lex();

        int64_t Temp;
        if (getParser().parseAbsoluteExpression(Temp) || !(0 <= Temp && Temp <=3))
          return MatchOperand_ParseFail;
        const int shift = i*2 + 2;
        Int += (Temp << shift);
      }

      if (getLexer().isNot(AsmToken::RBrac))
        return MatchOperand_ParseFail;
      Parser.Lex();
    } else {
      // sel:%d
      Parser.Lex();
      if (getParser().parseAbsoluteExpression(Int))
        return MatchOperand_ParseFail;

      if (Prefix == "row_shl" && 1 <= Int && Int <= 15) {
        Int |= DppCtrl::ROW_SHL0;
      } else if (Prefix == "row_shr" && 1 <= Int && Int <= 15) {
        Int |= DppCtrl::ROW_SHR0;
      } else if (Prefix == "row_ror" && 1 <= Int && Int <= 15) {
        Int |= DppCtrl::ROW_ROR0;
      } else if (Prefix == "wave_shl" && 1 == Int) {
        Int = DppCtrl::WAVE_SHL1;
      } else if (Prefix == "wave_rol" && 1 == Int) {
        Int = DppCtrl::WAVE_ROL1;
      } else if (Prefix == "wave_shr" && 1 == Int) {
        Int = DppCtrl::WAVE_SHR1;
      } else if (Prefix == "wave_ror" && 1 == Int) {
        Int = DppCtrl::WAVE_ROR1;
      } else if (Prefix == "row_bcast") {
        if (Int == 15) {
          Int = DppCtrl::BCAST15;
        } else if (Int == 31) {
          Int = DppCtrl::BCAST31;
        } else {
          return MatchOperand_ParseFail;
        }
      } else if (Prefix == "row_share" && 0 <= Int && Int <= 15) {
        Int |= DppCtrl::ROW_SHARE_FIRST;
      } else if (Prefix == "row_xmask" && 0 <= Int && Int <= 15) {
        Int |= DppCtrl::ROW_XMASK_FIRST;
      } else {
        return MatchOperand_ParseFail;
      }
    }
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Int, S, AMDGPUOperand::ImmTyDppCtrl));
  return MatchOperand_Success;
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultRowMask() const {
  return AMDGPUOperand::CreateImm(this, 0xf, SMLoc(), AMDGPUOperand::ImmTyDppRowMask);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultEndpgmImmOperands() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyEndpgm);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultBankMask() const {
  return AMDGPUOperand::CreateImm(this, 0xf, SMLoc(), AMDGPUOperand::ImmTyDppBankMask);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultBoundCtrl() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyDppBoundCtrl);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultFI() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyDppFi);
}

void AMDGPUAsmParser::cvtDPP(MCInst &Inst, const OperandVector &Operands, bool IsDPP8) {
  OptionalImmIndexMap OptionalIdx;

  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  int Fi = 0;
  for (unsigned E = Operands.size(); I != E; ++I) {
    auto TiedTo = Desc.getOperandConstraint(Inst.getNumOperands(),
                                            MCOI::TIED_TO);
    if (TiedTo != -1) {
      assert((unsigned)TiedTo < Inst.getNumOperands());
      // handle tied old or src2 for MAC instructions
      Inst.addOperand(Inst.getOperand(TiedTo));
    }
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
    // Add the register arguments
    if (Op.isReg() && validateVccOperand(Op.getReg())) {
      // VOP2b (v_add_u32, v_sub_u32 ...) dpp use "vcc" token.
      // Skip it.
      continue;
    }

    if (IsDPP8) {
      if (Op.isDPP8()) {
        Op.addImmOperands(Inst, 1);
      } else if (isRegOrImmWithInputMods(Desc, Inst.getNumOperands())) {
        Op.addRegWithFPInputModsOperands(Inst, 2);
      } else if (Op.isFI()) {
        Fi = Op.getImm();
      } else if (Op.isReg()) {
        Op.addRegOperands(Inst, 1);
      } else {
        llvm_unreachable("Invalid operand type");
      }
    } else {
      if (isRegOrImmWithInputMods(Desc, Inst.getNumOperands())) {
        Op.addRegWithFPInputModsOperands(Inst, 2);
      } else if (Op.isDPPCtrl()) {
        Op.addImmOperands(Inst, 1);
      } else if (Op.isImm()) {
        // Handle optional arguments
        OptionalIdx[Op.getImmTy()] = I;
      } else {
        llvm_unreachable("Invalid operand type");
      }
    }
  }

  if (IsDPP8) {
    using namespace llvm::AMDGPU::DPP;
    Inst.addOperand(MCOperand::createImm(Fi? DPP8_FI_1 : DPP8_FI_0));
  } else {
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppRowMask, 0xf);
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppBankMask, 0xf);
    addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppBoundCtrl);
    if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::fi) != -1) {
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyDppFi);
    }
  }
}

//===----------------------------------------------------------------------===//
// sdwa
//===----------------------------------------------------------------------===//

OperandMatchResultTy
AMDGPUAsmParser::parseSDWASel(OperandVector &Operands, StringRef Prefix,
                              AMDGPUOperand::ImmTy Type) {
  using namespace llvm::AMDGPU::SDWA;

  SMLoc S = Parser.getTok().getLoc();
  StringRef Value;
  OperandMatchResultTy res;

  res = parseStringWithPrefix(Prefix, Value);
  if (res != MatchOperand_Success) {
    return res;
  }

  int64_t Int;
  Int = StringSwitch<int64_t>(Value)
        .Case("BYTE_0", SdwaSel::BYTE_0)
        .Case("BYTE_1", SdwaSel::BYTE_1)
        .Case("BYTE_2", SdwaSel::BYTE_2)
        .Case("BYTE_3", SdwaSel::BYTE_3)
        .Case("WORD_0", SdwaSel::WORD_0)
        .Case("WORD_1", SdwaSel::WORD_1)
        .Case("DWORD", SdwaSel::DWORD)
        .Default(0xffffffff);
  Parser.Lex(); // eat last token

  if (Int == 0xffffffff) {
    return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Int, S, Type));
  return MatchOperand_Success;
}

OperandMatchResultTy
AMDGPUAsmParser::parseSDWADstUnused(OperandVector &Operands) {
  using namespace llvm::AMDGPU::SDWA;

  SMLoc S = Parser.getTok().getLoc();
  StringRef Value;
  OperandMatchResultTy res;

  res = parseStringWithPrefix("dst_unused", Value);
  if (res != MatchOperand_Success) {
    return res;
  }

  int64_t Int;
  Int = StringSwitch<int64_t>(Value)
        .Case("UNUSED_PAD", DstUnused::UNUSED_PAD)
        .Case("UNUSED_SEXT", DstUnused::UNUSED_SEXT)
        .Case("UNUSED_PRESERVE", DstUnused::UNUSED_PRESERVE)
        .Default(0xffffffff);
  Parser.Lex(); // eat last token

  if (Int == 0xffffffff) {
    return MatchOperand_ParseFail;
  }

  Operands.push_back(AMDGPUOperand::CreateImm(this, Int, S, AMDGPUOperand::ImmTySdwaDstUnused));
  return MatchOperand_Success;
}

void AMDGPUAsmParser::cvtSdwaVOP1(MCInst &Inst, const OperandVector &Operands) {
  cvtSDWA(Inst, Operands, SIInstrFlags::VOP1);
}

void AMDGPUAsmParser::cvtSdwaVOP2(MCInst &Inst, const OperandVector &Operands) {
  cvtSDWA(Inst, Operands, SIInstrFlags::VOP2);
}

void AMDGPUAsmParser::cvtSdwaVOP2b(MCInst &Inst, const OperandVector &Operands) {
  cvtSDWA(Inst, Operands, SIInstrFlags::VOP2, true, true);
}

void AMDGPUAsmParser::cvtSdwaVOP2e(MCInst &Inst, const OperandVector &Operands) {
  cvtSDWA(Inst, Operands, SIInstrFlags::VOP2, false, true);
}

void AMDGPUAsmParser::cvtSdwaVOPC(MCInst &Inst, const OperandVector &Operands) {
  cvtSDWA(Inst, Operands, SIInstrFlags::VOPC, isVI());
}

void AMDGPUAsmParser::cvtSDWA(MCInst &Inst, const OperandVector &Operands,
                              uint64_t BasicInstType,
                              bool SkipDstVcc,
                              bool SkipSrcVcc) {
  using namespace llvm::AMDGPU::SDWA;

  OptionalImmIndexMap OptionalIdx;
  bool SkipVcc = SkipDstVcc || SkipSrcVcc;
  bool SkippedVcc = false;

  unsigned I = 1;
  const MCInstrDesc &Desc = MII.get(Inst.getOpcode());
  for (unsigned J = 0; J < Desc.getNumDefs(); ++J) {
    ((AMDGPUOperand &)*Operands[I++]).addRegOperands(Inst, 1);
  }

  for (unsigned E = Operands.size(); I != E; ++I) {
    AMDGPUOperand &Op = ((AMDGPUOperand &)*Operands[I]);
    if (SkipVcc && !SkippedVcc && Op.isReg() &&
        (Op.getReg() == AMDGPU::VCC || Op.getReg() == AMDGPU::VCC_LO)) {
      // VOP2b (v_add_u32, v_sub_u32 ...) sdwa use "vcc" token as dst.
      // Skip it if it's 2nd (e.g. v_add_i32_sdwa v1, vcc, v2, v3)
      // or 4th (v_addc_u32_sdwa v1, vcc, v2, v3, vcc) operand.
      // Skip VCC only if we didn't skip it on previous iteration.
      // Note that src0 and src1 occupy 2 slots each because of modifiers.
      if (BasicInstType == SIInstrFlags::VOP2 &&
          ((SkipDstVcc && Inst.getNumOperands() == 1) ||
           (SkipSrcVcc && Inst.getNumOperands() == 5))) {
        SkippedVcc = true;
        continue;
      } else if (BasicInstType == SIInstrFlags::VOPC &&
                 Inst.getNumOperands() == 0) {
        SkippedVcc = true;
        continue;
      }
    }
    if (isRegOrImmWithInputMods(Desc, Inst.getNumOperands())) {
      Op.addRegOrImmWithInputModsOperands(Inst, 2);
    } else if (Op.isImm()) {
      // Handle optional arguments
      OptionalIdx[Op.getImmTy()] = I;
    } else {
      llvm_unreachable("Invalid operand type");
    }
    SkippedVcc = false;
  }

  if (Inst.getOpcode() != AMDGPU::V_NOP_sdwa_gfx10 &&
      Inst.getOpcode() != AMDGPU::V_NOP_sdwa_gfx9 &&
      Inst.getOpcode() != AMDGPU::V_NOP_sdwa_vi) {
    // v_nop_sdwa_sdwa_vi/gfx9 has no optional sdwa arguments
    switch (BasicInstType) {
    case SIInstrFlags::VOP1:
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClampSI, 0);
      if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::omod) != -1) {
        addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOModSI, 0);
      }
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaDstSel, SdwaSel::DWORD);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaDstUnused, DstUnused::UNUSED_PRESERVE);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaSrc0Sel, SdwaSel::DWORD);
      break;

    case SIInstrFlags::VOP2:
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClampSI, 0);
      if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::omod) != -1) {
        addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyOModSI, 0);
      }
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaDstSel, SdwaSel::DWORD);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaDstUnused, DstUnused::UNUSED_PRESERVE);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaSrc0Sel, SdwaSel::DWORD);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaSrc1Sel, SdwaSel::DWORD);
      break;

    case SIInstrFlags::VOPC:
      if (AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::clamp) != -1)
        addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTyClampSI, 0);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaSrc0Sel, SdwaSel::DWORD);
      addOptionalImmOperand(Inst, Operands, OptionalIdx, AMDGPUOperand::ImmTySdwaSrc1Sel, SdwaSel::DWORD);
      break;

    default:
      llvm_unreachable("Invalid instruction type. Only VOP1, VOP2 and VOPC allowed");
    }
  }

  // special case v_mac_{f16, f32}:
  // it has src2 register operand that is tied to dst operand
  if (Inst.getOpcode() == AMDGPU::V_MAC_F32_sdwa_vi ||
      Inst.getOpcode() == AMDGPU::V_MAC_F16_sdwa_vi)  {
    auto it = Inst.begin();
    std::advance(
      it, AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::src2));
    Inst.insert(it, Inst.getOperand(0)); // src2 = dst
  }
}

//===----------------------------------------------------------------------===//
// mAI
//===----------------------------------------------------------------------===//

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultBLGP() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyBLGP);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultCBSZ() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyCBSZ);
}

AMDGPUOperand::Ptr AMDGPUAsmParser::defaultABID() const {
  return AMDGPUOperand::CreateImm(this, 0, SMLoc(), AMDGPUOperand::ImmTyABID);
}

/// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAMDGPUAsmParser() {
  RegisterMCAsmParser<AMDGPUAsmParser> A(getTheAMDGPUTarget());
  RegisterMCAsmParser<AMDGPUAsmParser> B(getTheGCNTarget());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "AMDGPUGenAsmMatcher.inc"

// This fuction should be defined after auto-generated include so that we have
// MatchClassKind enum defined
unsigned AMDGPUAsmParser::validateTargetOperandClass(MCParsedAsmOperand &Op,
                                                     unsigned Kind) {
  // Tokens like "glc" would be parsed as immediate operands in ParseOperand().
  // But MatchInstructionImpl() expects to meet token and fails to validate
  // operand. This method checks if we are given immediate operand but expect to
  // get corresponding token.
  AMDGPUOperand &Operand = (AMDGPUOperand&)Op;
  switch (Kind) {
  case MCK_addr64:
    return Operand.isAddr64() ? Match_Success : Match_InvalidOperand;
  case MCK_gds:
    return Operand.isGDS() ? Match_Success : Match_InvalidOperand;
  case MCK_lds:
    return Operand.isLDS() ? Match_Success : Match_InvalidOperand;
  case MCK_glc:
    return Operand.isGLC() ? Match_Success : Match_InvalidOperand;
  case MCK_idxen:
    return Operand.isIdxen() ? Match_Success : Match_InvalidOperand;
  case MCK_offen:
    return Operand.isOffen() ? Match_Success : Match_InvalidOperand;
  case MCK_SSrcB32:
    // When operands have expression values, they will return true for isToken,
    // because it is not possible to distinguish between a token and an
    // expression at parse time. MatchInstructionImpl() will always try to
    // match an operand as a token, when isToken returns true, and when the
    // name of the expression is not a valid token, the match will fail,
    // so we need to handle it here.
    return Operand.isSSrcB32() ? Match_Success : Match_InvalidOperand;
  case MCK_SSrcF32:
    return Operand.isSSrcF32() ? Match_Success : Match_InvalidOperand;
  case MCK_SoppBrTarget:
    return Operand.isSoppBrTarget() ? Match_Success : Match_InvalidOperand;
  case MCK_VReg32OrOff:
    return Operand.isVReg32OrOff() ? Match_Success : Match_InvalidOperand;
  case MCK_InterpSlot:
    return Operand.isInterpSlot() ? Match_Success : Match_InvalidOperand;
  case MCK_Attr:
    return Operand.isInterpAttr() ? Match_Success : Match_InvalidOperand;
  case MCK_AttrChan:
    return Operand.isAttrChan() ? Match_Success : Match_InvalidOperand;
  case MCK_ImmSMEMOffset:
    return Operand.isSMEMOffset() ? Match_Success : Match_InvalidOperand;
  case MCK_SReg_64:
  case MCK_SReg_64_XEXEC:
    // Null is defined as a 32-bit register but
    // it should also be enabled with 64-bit operands.
    // The following code enables it for SReg_64 operands
    // used as source and destination. Remaining source
    // operands are handled in isInlinableImm.
    return Operand.isNull() ? Match_Success : Match_InvalidOperand;
  default:
    return Match_InvalidOperand;
  }
}

//===----------------------------------------------------------------------===//
// endpgm
//===----------------------------------------------------------------------===//

OperandMatchResultTy AMDGPUAsmParser::parseEndpgmOp(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  int64_t Imm = 0;

  if (!parseExpr(Imm)) {
    // The operand is optional, if not present default to 0
    Imm = 0;
  }

  if (!isUInt<16>(Imm)) {
    Error(S, "expected a 16-bit value");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(
      AMDGPUOperand::CreateImm(this, Imm, S, AMDGPUOperand::ImmTyEndpgm));
  return MatchOperand_Success;
}

bool AMDGPUOperand::isEndpgm() const { return isImmTy(ImmTyEndpgm); }
