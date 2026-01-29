//===- WebAssemblyLegalizerInfo.h --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for
/// WebAssembly
//===----------------------------------------------------------------------===//

#include "WebAssemblyLegalizerInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "wasm-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

WebAssemblyLegalizerInfo::WebAssemblyLegalizerInfo(
    const WebAssemblySubtarget &ST) {
  using namespace TargetOpcode;
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT s64 = LLT::scalar(64);

  const LLT p0 = LLT::pointer(0, ST.hasAddr64() ? 64 : 32);
  const LLT p0s = LLT::scalar(ST.hasAddr64() ? 64 : 32);

  getActionDefinitionsBuilder(G_GLOBAL_VALUE).legalFor({p0});

  getActionDefinitionsBuilder(G_PHI)
      .legalFor({p0, s32, s64})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);
  getActionDefinitionsBuilder(G_BR).alwaysLegal();
  getActionDefinitionsBuilder(G_BRCOND).legalFor({s32}).clampScalar(0, s32,
                                                                    s32);
  getActionDefinitionsBuilder(G_BRJT).legalFor({{p0, p0s}});

  getActionDefinitionsBuilder(G_SELECT)
      .legalFor({{s32, s32}, {s64, s32}, {p0, s32}})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s32);

  getActionDefinitionsBuilder(G_JUMP_TABLE).legalFor({p0});

  getActionDefinitionsBuilder(G_ICMP)
      .legalFor({{s32, s32}, {s32, s64}, {s32, p0}})
      //.scalarize(0)
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64)
      .clampScalar(0, s32, s32);

  getActionDefinitionsBuilder(G_FCMP)
      .customFor({{s32, s32}, {s32, s64}})
      //.scalarize(0)
      .clampScalar(0, s32, s32)
      .libcall();

  getActionDefinitionsBuilder(G_FRAME_INDEX).legalFor({p0});

  getActionDefinitionsBuilder(G_CONSTANT)
      .legalFor({s32, s64, p0})
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(G_FCONSTANT)
      .legalFor({s32, s64})
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalFor({s32, s64, p0})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder(
      {G_ADD, G_SUB, G_MUL, G_UDIV, G_SDIV, G_UREM, G_SREM})
      .legalFor({s32, s64})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_ASHR, G_LSHR, G_SHL, G_CTLZ, G_CTLZ_ZERO_UNDEF,
                               G_CTTZ, G_CTTZ_ZERO_UNDEF, G_CTPOP})
      .legalFor({{s32, s32}, {s64, s64}})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64)
      .minScalarSameAs(1, 0)
      .maxScalarSameAs(1, 0);

  getActionDefinitionsBuilder({G_FSHL, G_FSHR}).lower();

  getActionDefinitionsBuilder({G_ROTL, G_ROTR})
      .legalFor({{s32, s32}, {s64, s64}})
      //.scalarize(0)
      .lower();

  getActionDefinitionsBuilder({G_SCMP, G_UCMP}).lower();

  getActionDefinitionsBuilder({G_AND, G_OR, G_XOR})
      .legalFor({s32, s64})
      //.scalarize(0)
      .widenScalarToNextPow2(0)
      .clampScalar(0, s32, s64);

  getActionDefinitionsBuilder({G_UMIN, G_UMAX, G_SMIN, G_SMAX}).lower();
  getActionDefinitionsBuilder({G_USUBSAT, G_USUBO}).lower();

  getActionDefinitionsBuilder({G_FADD, G_FSUB, G_FDIV, G_FMUL, G_FNEG, G_FABS,
                               G_FCEIL, G_FFLOOR, G_FSQRT, G_INTRINSIC_TRUNC,
                               G_FNEARBYINT, G_FRINT, G_INTRINSIC_ROUNDEVEN,
                               G_FMINIMUM, G_FMAXIMUM, G_STRICT_FMUL})
      .legalFor({s32, s64})
      //.scalarize(0)
      .minScalar(0, s32);

  getActionDefinitionsBuilder({G_FMINNUM, G_FMAXNUM})
      .customFor({s32, s64})
      //.scalarize(0)
      .minScalar(0, s32);

  getActionDefinitionsBuilder(
      {G_VECREDUCE_OR, G_VECREDUCE_AND}); //.scalarize(1);

  getActionDefinitionsBuilder(G_BITCAST)
      .customIf([=](const LegalityQuery &Query) {
        // Handle casts from i1 vectors to scalars.
        LLT DstTy = Query.Types[0];
        LLT SrcTy = Query.Types[1];
        return DstTy.isScalar() && SrcTy.isVector() &&
               SrcTy.getScalarSizeInBits() == 1;
      })
      .lowerIf([=](const LegalityQuery &Query) {
        return Query.Types[0].isVector() != Query.Types[1].isVector();
      });
  //.scalarize(0);

  getActionDefinitionsBuilder(G_MERGE_VALUES)
      .lowerFor({{s64, s32}, {s64, s16}, {s64, s8}, {s32, s16}, {s32, s8}});

  getActionDefinitionsBuilder(G_FCANONICALIZE)
      .customFor({s32, s64})
      //.scalarize(0)
      .minScalar(0, s32);

  getActionDefinitionsBuilder({G_FMA, G_FREM})
      .libcallFor({s32, s64})
      .minScalar(0, s32);

  getActionDefinitionsBuilder(G_LROUND).libcallForCartesianProduct({s32},
                                                                   {s32, s64});

  getActionDefinitionsBuilder(G_LLROUND).libcallForCartesianProduct({s64},
                                                                    {s32, s64});

  getActionDefinitionsBuilder(G_FCOPYSIGN)
      .legalFor({s32, s64})
      //.scalarize(0)
      .minScalar(0, s32)
      .minScalarSameAs(1, 0)
      .maxScalarSameAs(1, 0);

  if (ST.hasNontrappingFPToInt()) {
    getActionDefinitionsBuilder(
        {G_FPTOUI, G_FPTOUI_SAT, G_FPTOSI, G_FPTOSI_SAT})
        .legalForCartesianProduct({s32, s64}, {s32, s64})
        .minScalar(1, s32)
        .widenScalarToNextPow2(0)
        .clampScalar(0, s32, s64);
  }

  getActionDefinitionsBuilder({G_UITOFP, G_SITOFP})
      .legalForCartesianProduct({s32, s64}, {s32, s64})
      //.scalarize(0)
      .minScalar(1, s32)
      .widenScalarToNextPow2(1)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder(G_PTRTOINT)
      .legalFor({p0s, p0})
      .customForCartesianProduct({s32, s64}, {p0});
  getActionDefinitionsBuilder(G_INTTOPTR)
      .legalFor({p0, p0s})
      .customForCartesianProduct({p0}, {s32, s64});

  getActionDefinitionsBuilder(G_PTR_ADD).legalFor({{p0, p0s}});
  getActionDefinitionsBuilder(G_PTRMASK).legalFor({{p0, p0s}});

  getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc(
          {{s32, p0, s32, 1}, {s64, p0, s64, 1}, {p0, p0, p0, 1}})
      .customIf(
          LegalityPredicates::typePairAndMemDescInSet(0, 1, /*MMOIdx*/ 0,
                                                      {{s32, p0, s8, 1},
                                                       {s32, p0, s16, 1},

                                                       {s64, p0, s8, 1},
                                                       {s64, p0, s16, 1},
                                                       {s64, p0, s32, 1}}))
      .clampScalar(0, s32, s64)
      .lowerIfMemSizeNotByteSizePow2();
  //.scalarize(0);

  getActionDefinitionsBuilder(G_STORE)
      .legalForTypesWithMemDesc(
          {{s32, p0, s32, 1}, {s64, p0, s64, 1}, {p0, p0, p0, 1}})
      .legalForTypesWithMemDesc({{s32, p0, s8, 1},
                                 {s32, p0, s16, 1},

                                 {s64, p0, s8, 1},
                                 {s64, p0, s16, 1},
                                 {s64, p0, s32, 1}})
      .clampScalar(0, s32, s64)
      .lowerIf([=](const LegalityQuery &Query) {
        return Query.Types[0].isScalar() &&
               Query.Types[0] != Query.MMODescrs[0].MemoryTy;
      })
      .bitcastIf(
          [=](const LegalityQuery &Query) {
            // Handle stores of i1 vectors.
            LLT Ty = Query.Types[0];
            return Ty.isVector() && Ty.getScalarSizeInBits() == 1;
          },
          [=](const LegalityQuery &Query) {
            const LLT VecTy = Query.Types[0];
            return std::pair(0, LLT::scalar(VecTy.getSizeInBits()));
          });
  //.scalarize(0);

  getActionDefinitionsBuilder(
      {G_SHUFFLE_VECTOR, G_EXTRACT_VECTOR_ELT, G_INSERT_VECTOR_ELT})
      .lower();

  getActionDefinitionsBuilder({G_ZEXTLOAD, G_SEXTLOAD})
      .legalForTypesWithMemDesc({{s32, p0, s8, 1},
                                 {s32, p0, s16, 1},

                                 {s64, p0, s8, 1},
                                 {s64, p0, s16, 1},
                                 {s64, p0, s32, 1}})
      .clampScalar(0, s32, s64)
      .lowerIfMemSizeNotByteSizePow2();

  if (ST.hasBulkMemoryOpt()) {
    getActionDefinitionsBuilder(G_BZERO).unsupported();

    getActionDefinitionsBuilder(G_MEMSET)
        .customForCartesianProduct({p0}, {s8, s32}, {p0s})
        .immIdx(0);

    getActionDefinitionsBuilder({G_MEMCPY, G_MEMMOVE})
        .customForCartesianProduct({p0}, {p0}, {p0s})
        .immIdx(0);

    getActionDefinitionsBuilder(G_MEMCPY_INLINE)
        .customForCartesianProduct({p0}, {p0}, {p0s});
  } else {
    getActionDefinitionsBuilder({G_BZERO, G_MEMCPY, G_MEMMOVE, G_MEMSET})
        .libcall();
  }

  // TODO: figure out how to combine G_ANYEXT of G_ASSERT_{S|Z}EXT (or
  // appropriate G_AND and G_SEXT_IN_REG?) to a G_{S|Z}EXT + G_ASSERT_{S|Z}EXT
  // for better optimization (since G_ANYEXT will lower to a ZEXT or SEXT
  // instruction anyway).

  getActionDefinitionsBuilder(G_ANYEXT)
      .legalFor({{s64, s32}})
      //.scalarize(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64);

  getActionDefinitionsBuilder({G_SEXT, G_ZEXT})
      .legalFor({{s64, s32}})
      //.scalarize(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64)
      .lower();

  if (ST.hasSignExt()) {
    getActionDefinitionsBuilder(G_SEXT_INREG)
        .clampScalar(0, s32, s64)
        .customFor({s32, s64});
  } else {
    getActionDefinitionsBuilder(G_SEXT_INREG).lower();
  }

  getActionDefinitionsBuilder(G_TRUNC)
      .legalFor({{s32, s64}})
      //.scalarize(0)
      .clampScalar(0, s32, s64)
      .clampScalar(1, s32, s64)
      .lower();

  getActionDefinitionsBuilder(G_FPEXT).legalFor({{s64, s32}}); //.scalarize(0);

  getActionDefinitionsBuilder(G_FPTRUNC).legalFor(
      {{s32, s64}}); //.scalarize(0);

  getActionDefinitionsBuilder(G_VASTART).legalFor({p0});
  getActionDefinitionsBuilder(G_VAARG)
      .customIf(([=](const LegalityQuery &Query) {
        return Query.Types[1] == p0 && (Query.Types[0].isScalar());
      })) // TODO: replace this with typical lowerForCartesianProduct once widen
          // and narrow scalar works
      .clampScalar(
          0, s32,
          s64) // Note: currently not functional (not implemented in core)
      .widenScalarToNextPow2(0, /*Min*/ 8); // Note: currently not functional
                                            // (not implemented in core)

  getActionDefinitionsBuilder(G_DYN_STACKALLOC).lowerFor({{p0, p0s}});

  getActionDefinitionsBuilder({G_STACKSAVE, G_STACKRESTORE}).lower();

  getLegacyLegalizerInfo().computeTables();
}

static MachineInstrBuilder buildEqz(MachineIRBuilder &MIBuilder,
                                    const DstOp &Dst, const SrcOp &Src0) {
  auto Zero = MIBuilder.buildConstant(Dst.getLLTTy(*MIBuilder.getMRI()), 0);
  return MIBuilder.buildICmp(llvm::CmpInst::ICMP_EQ, Dst, Src0, Zero);
}

bool WebAssemblyLegalizerInfo::legalizeCustom(
    LegalizerHelper &Helper, MachineInstr &MI,
    LostDebugLocObserver &LocObserver) const {
  auto &MRI = *Helper.MIRBuilder.getMRI();
  auto &MIRBuilder = Helper.MIRBuilder;

  const LLT s1 = LLT::scalar(1);

  switch (MI.getOpcode()) {
  case TargetOpcode::G_LOAD: {
    // If we are here we are looking at a valid load with a MemTy smaller than the ValueTy
    // just turn it into a G_ZEXTLOAD to aid optimizations

    MI.setDesc(MIRBuilder.getTII().get(TargetOpcode::G_ZEXTLOAD));
    return true;
  }
  case TargetOpcode::G_FCANONICALIZE: {
    auto One =
        MIRBuilder.buildFConstant(MRI.getType(MI.getOperand(0).getReg()), 1.0)
            .getReg(0);

    MIRBuilder.buildInstr(TargetOpcode::G_STRICT_FMUL)
        .addDef(MI.getOperand(0).getReg())
        .addUse(MI.getOperand(1).getReg())
        .addUse(One)
        .setMIFlags(MI.getFlags())
        .setMIFlag(MachineInstr::MIFlag::NoFPExcept);

    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_FMINNUM: {
    if (!MI.getFlag(MachineInstr::MIFlag::FmNoNans))
      return false;

    if (MI.getFlag(MachineInstr::MIFlag::FmNsz)) {
      MIRBuilder.buildInstr(TargetOpcode::G_FMINIMUM)
          .addDef(MI.getOperand(0).getReg())
          .addUse(MI.getOperand(1).getReg())
          .addUse(MI.getOperand(2).getReg())
          .setMIFlags(MI.getFlags());
    } else {
      auto Cond = MIRBuilder
                      .buildFCmp(CmpInst::Predicate::FCMP_OLT, s1,
                                 MI.getOperand(1), MI.getOperand(2))
                      .getReg(0);

      MIRBuilder.buildSelect(MI.getOperand(0), Cond, MI.getOperand(1),
                             MI.getOperand(2));
    }
    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_FMAXNUM: {
    if (!MI.getFlag(MachineInstr::MIFlag::FmNoNans))
      return false;
    if (MI.getFlag(MachineInstr::MIFlag::FmNsz)) {
      MIRBuilder.buildInstr(TargetOpcode::G_FMAXIMUM)
          .addDef(MI.getOperand(0).getReg())
          .addUse(MI.getOperand(1).getReg())
          .addUse(MI.getOperand(2).getReg())
          .setMIFlags(MI.getFlags());
    } else {
      auto Cond = MIRBuilder
                      .buildFCmp(CmpInst::Predicate::FCMP_OGT, s1,
                                 MI.getOperand(1), MI.getOperand(2))
                      .getReg(0);

      MIRBuilder.buildSelect(MI.getOperand(0), Cond, MI.getOperand(1),
                             MI.getOperand(2));
    }
    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_PTRTOINT: {
    auto TmpReg = MRI.createGenericVirtualRegister(
        LLT::scalar(MIRBuilder.getDataLayout().getPointerSizeInBits(0)));

    MIRBuilder.buildPtrToInt(TmpReg, MI.getOperand(1));
    MIRBuilder.buildAnyExtOrTrunc(MI.getOperand(0), TmpReg);
    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_INTTOPTR: {
    auto TmpReg = MRI.createGenericVirtualRegister(
        LLT::scalar(MIRBuilder.getDataLayout().getPointerSizeInBits(0)));

    MIRBuilder.buildAnyExtOrTrunc(TmpReg, MI.getOperand(1));
    MIRBuilder.buildIntToPtr(MI.getOperand(0), TmpReg);
    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_BITCAST: {
    if (MIRBuilder.getMF().getSubtarget<WebAssemblySubtarget>().hasSIMD128()) {
      return false;
    }

    auto [DstReg, DstTy, SrcReg, SrcTy] = MI.getFirst2RegLLTs();

    if (!DstTy.isScalar() || !SrcTy.isVector() || SrcTy.getElementType() != s1)
      return false;

    Register ResultReg = MRI.createGenericVirtualRegister(DstTy);
    MIRBuilder.buildConstant(ResultReg, 0);

    for (unsigned i = 0; i < SrcTy.getNumElements(); i++) {
      auto Elm = MRI.createGenericVirtualRegister(s1);
      auto ExtElm = MRI.createGenericVirtualRegister(DstTy);
      auto ShiftedElm = MRI.createGenericVirtualRegister(DstTy);
      auto Idx = MRI.createGenericVirtualRegister(LLT::scalar(8));
      auto NewResultReg = MRI.createGenericVirtualRegister(DstTy);

      MIRBuilder.buildConstant(Idx, i);
      MIRBuilder.buildExtractVectorElement(Elm, SrcReg, Idx);
      MIRBuilder.buildZExt(ExtElm, Elm, false);
      MIRBuilder.buildShl(ShiftedElm, ExtElm, Idx);
      MIRBuilder.buildOr(NewResultReg, ResultReg, ShiftedElm);

      ResultReg = NewResultReg;
    }

    MIRBuilder.buildCopy(DstReg, ResultReg);

    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_FCMP: {
    Register LHS = MI.getOperand(2).getReg();
    Register RHS = MI.getOperand(3).getReg();
    CmpInst::Predicate Cond =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());

    auto CmpWidth = MRI.getType(LHS).getSizeInBits();
    assert(CmpWidth == MRI.getType(RHS).getSizeInBits() &&
           "LHS and RHS for FCMP are diffrent lengths???");

    switch (Cond) {
    case CmpInst::FCMP_FALSE:
      return false;
    case CmpInst::FCMP_OEQ:
      return true;
    case CmpInst::FCMP_OGT:
      return true;
    case CmpInst::FCMP_OGE:
      return true;
    case CmpInst::FCMP_OLT:
      return true;
    case CmpInst::FCMP_OLE:
      return true;
    case CmpInst::FCMP_ONE: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          MIRBuilder.buildOr(
              s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLT, s1, LHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGT, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_ORD: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          MIRBuilder.buildAnd(
              s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OEQ, s1, RHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_OEQ, s1, LHS, LHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_UNO: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          MIRBuilder.buildOr(
              s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_UNE, s1, RHS, RHS).getReg(0),
              MIRBuilder.buildFCmp(CmpInst::FCMP_UNE, s1, LHS, LHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_UEQ: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          buildEqz(
              MIRBuilder, s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_ONE, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_UGT: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          buildEqz(
              MIRBuilder, s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLE, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_UGE: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          buildEqz(
              MIRBuilder, s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OLT, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_ULT: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          buildEqz(
              MIRBuilder, s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGE, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_ULE: {
      MIRBuilder.buildAnyExt(
          MI.getOperand(0).getReg(),
          buildEqz(
              MIRBuilder, s1,
              MIRBuilder.buildFCmp(CmpInst::FCMP_OGT, s1, LHS, RHS).getReg(0)));
      break;
    }
    case CmpInst::FCMP_UNE:
      return true;
    case CmpInst::FCMP_TRUE:
      return false;
    default:
      llvm_unreachable("Unknown FCMP predicate");
    }

    MI.eraseFromParent();

    return true;
  }
  case TargetOpcode::G_SEXT_INREG: {
    assert(MI.getOperand(2).isImm() && "Expected immediate");

    // Mark only 8/16/32-bit SEXT_INREG as legal
    auto [DstReg, SrcReg] = MI.getFirst2Regs();
    auto DstType = MRI.getType(DstReg);
    auto ExtFromWidth = MI.getOperand(2).getImm();

    if (ExtFromWidth == 8 || ExtFromWidth == 16 ||
        (DstType.getScalarSizeInBits() == 64 && ExtFromWidth == 32)) {
      return true;
    }

    Register TmpRes = MRI.createGenericVirtualRegister(DstType);

    auto MIBSz = MIRBuilder.buildConstant(
        DstType, DstType.getScalarSizeInBits() - ExtFromWidth);
    MIRBuilder.buildShl(TmpRes, SrcReg, MIBSz->getOperand(0));
    MIRBuilder.buildAShr(DstReg, TmpRes, MIBSz->getOperand(0));
    MI.eraseFromParent();

    return true;
  }
  case TargetOpcode::G_MEMSET: {
    // Anyext the value being set to 32 bit (only the bottom 8 bits are read by
    // the instruction).
    auto &Value = MI.getOperand(1);

    if (MRI.getType(Value.getReg()).getSizeInBits() == 8) {
      Register ExtValueReg =
          Helper.MIRBuilder.buildAnyExt(LLT::scalar(32), Value).getReg(0);
      Value.setReg(ExtValueReg);
      return true;
    }

    MachineIRBuilder B(MI);
    assert(MI.getNumMemOperands() == 1);

    auto *DstMem = MI.memoperands()[0];
    auto &Dst = MI.getOperand(0);
    auto &Len = MI.getOperand(1);
    // auto TailCall = I.getOperand(3);

    auto PointerWidth = MI.getMF()->getDataLayout().getPointerSizeInBits();
    auto PtrIsI64 = PointerWidth == 64;

    B.buildInstr(PtrIsI64 ? WebAssembly::MEMSET_A64 : WebAssembly::MEMSET_A32)
        .addImm(0) // DstMem idx
        .add(Dst)
        .add(Value)
        .add(Len)
        .addMemOperand(DstMem);

    MI.eraseFromParent();

    return true;
  }
  case TargetOpcode::G_MEMCPY:
  case TargetOpcode::G_MEMMOVE: {
    MachineIRBuilder B(MI);
    assert(MI.getNumMemOperands() == 2);

    auto *DstMem = MI.memoperands()[0];
    auto *SrcMem = MI.memoperands()[1];
    auto &Dst = MI.getOperand(0);
    auto &Src = MI.getOperand(1);
    auto &Len = MI.getOperand(2);
    // auto TailCall = I.getOperand(3);

    auto PointerWidth = MI.getMF()->getDataLayout().getPointerSizeInBits();
    auto PtrIsI64 = PointerWidth == 64;

    B.buildInstr(PtrIsI64 ? WebAssembly::MEMCPY_A64 : WebAssembly::MEMCPY_A32)
        .addImm(0) // DstMem idx
        .addImm(0) // SrcMem idx
        .add(Dst)
        .add(Src)
        .add(Len)
        .addMemOperand(DstMem)
        .addMemOperand(SrcMem);

    MI.eraseFromParent();
    return true;
  }
  case TargetOpcode::G_VAARG: {
    MachineFunction &MF = MIRBuilder.getMF();
    Align Alignment(MI.getOperand(2).getImm());
    Register Dst = MI.getOperand(0).getReg();
    Register ListPtr = MI.getOperand(1).getReg();

    LLT PtrTy = MRI.getType(ListPtr);
    LLT IntPtrTy = LLT::scalar(PtrTy.getSizeInBits());

    const unsigned PtrSize = PtrTy.getSizeInBits() / 8;
    const Align PtrAlign = Align(PtrSize);
    auto List = MIRBuilder.buildLoad(
        PtrTy, ListPtr,
        *MF.getMachineMemOperand(MachinePointerInfo(),
                                 MachineMemOperand::MOLoad, PtrTy, PtrAlign));

    MachineInstrBuilder DstPtr;
    if (Alignment > PtrAlign) {
      // Realign the list to the actual required alignment.
      auto AlignMinus1 =
          MIRBuilder.buildConstant(IntPtrTy, Alignment.value() - 1);
      auto ListTmp = MIRBuilder.buildPtrAdd(PtrTy, List, AlignMinus1.getReg(0));
      DstPtr = MIRBuilder.buildMaskLowPtrBits(PtrTy, ListTmp, Log2(Alignment));
    } else
      DstPtr = List;

    LLT ValTy = MRI.getType(Dst);
    uint64_t ValSize = ValTy.getSizeInBits() / 8;
    MIRBuilder.buildLoad(Dst, DstPtr,
                         *MF.getMachineMemOperand(
                             MachinePointerInfo(), MachineMemOperand::MOLoad,
                             ValTy, std::max(Alignment, PtrAlign)));

    auto Size = MIRBuilder.buildConstant(IntPtrTy, alignTo(ValSize, PtrAlign));

    auto NewList = MIRBuilder.buildPtrAdd(PtrTy, DstPtr, Size.getReg(0));

    MIRBuilder.buildStore(NewList, ListPtr,
                          *MF.getMachineMemOperand(MachinePointerInfo(),
                                                   MachineMemOperand::MOStore,
                                                   PtrTy, PtrAlign));

    MI.eraseFromParent();

    return true;
  }
  default:
    break;
  }
  return false;
}

bool WebAssemblyLegalizerInfo::legalizeIntrinsic(LegalizerHelper &Helper,
                                                 MachineInstr &MI) const {
  auto &MRI = *Helper.MIRBuilder.getMRI();
  auto &MIRBuilder = Helper.MIRBuilder;

  switch (cast<GIntrinsic>(MI).getIntrinsicID()) {
  default:
    return true;
  case Intrinsic::vacopy: {
    auto PointerWidth = MI.getMF()->getDataLayout().getPointerSize();

    MachineFunction &MF = *MI.getMF();
    auto Val = MF.getRegInfo().createGenericVirtualRegister(
        LLT::scalar(PointerWidth * 8));

    MIRBuilder.buildLoad(Val, MI.getOperand(2),
                         *MF.getMachineMemOperand(
                             MachinePointerInfo(), MachineMemOperand::MOLoad,
                             PointerWidth, Align(PointerWidth)));
    MIRBuilder.buildStore(Val, MI.getOperand(1),
                          *MF.getMachineMemOperand(
                              MachinePointerInfo(), MachineMemOperand::MOStore,
                              PointerWidth, Align(PointerWidth)));
    MI.eraseFromParent();
    return true;
  }
  case Intrinsic::clear_cache:
    reportFatalUsageError("llvm.clear_cache is not supported on wasm");
  }
}
