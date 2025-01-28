//===- llvm/CodeGen/GlobalISel/InferTypeInfoPass.cpp - StripTypeInfoPass ---*-
// C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the InferTypeInfoPass class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InferTypeInfoPass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/LoadStoreOpt.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mir-infer-type-info"

using namespace llvm;

char InferTypeInfo::ID = 0;

INITIALIZE_PASS_BEGIN(InferTypeInfo, DEBUG_TYPE, "TODO", false, false)
INITIALIZE_PASS_END(InferTypeInfo, DEBUG_TYPE, "TODO", false, false)

void InferTypeInfo::init(MachineFunction &MF) {
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  Builder.setMF(MF);
}

void InferTypeInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static LLT updateType(LLT Ty, bool FP) {
  LLT InferredScalarTy =
    FP ? LLT::floatingPoint(Ty.getScalarSizeInBits(), LLT::FPInfo::IEEE_FLOAT)
         : LLT::integer(Ty.getScalarSizeInBits());
  LLT InferredTy =
      Ty.isVector() ? Ty.changeElementType(InferredScalarTy) : InferredScalarTy;

  return InferredTy;
}

void InferTypeInfo::updateDef(Register Reg) {
  LLT Ty = MRI->getType(Reg);
  LLT InferredTy = updateType(Ty, true);

  if (Ty == InferredTy)
    return;

  MRI->setType(Reg, InferredTy);
}

void InferTypeInfo::updateUse(MachineOperand &Op, bool FP) {
  assert(Op.isReg());
  Register Reg = Op.getReg();
  LLT Ty = MRI->getType(Reg);
  LLT InferredTy = updateType(Ty, FP);

  if (Ty == InferredTy)
    return;

  Register NewReg = MRI->cloneVirtualRegister(Reg);
  MRI->setType(NewReg, InferredTy);
  
  MachineOperand *Def = MRI->getOneDef(Reg);
  MachineInstr *MI = Op.getParent();
  MachineBasicBlock *MBB = MI->getParent();

  Builder.setInsertPt(*MBB, MI);
  Builder.buildBitcast(NewReg, Def->getReg());
  Op.setReg(NewReg);
}

constexpr unsigned MaxFPRSearchDepth = 5;

bool InferTypeInfo::shouldBeFP(MachineOperand &Op, unsigned Depth = 0) const {
  if (Depth > MaxFPRSearchDepth)
    return false;

  if (!Op.isReg())
    return false;

  MachineInstr &MI = *Op.getParent();

  auto Pred = [&](MachineOperand &O) { return shouldBeFP(O, Depth + 1); };

  // TODO: cache FP registers

  switch (MI.getOpcode()) {
  // def and use fp instructions
  case TargetOpcode::G_FABS:
  case TargetOpcode::G_FADD:
  case TargetOpcode::G_FCANONICALIZE:
  case TargetOpcode::G_FCEIL:
  case TargetOpcode::G_FCONSTANT:
  case TargetOpcode::G_FCOPYSIGN:
  case TargetOpcode::G_FCOS:
  case TargetOpcode::G_FDIV:
  case TargetOpcode::G_FEXP2:
  case TargetOpcode::G_FEXP:
  case TargetOpcode::G_FFLOOR:
  case TargetOpcode::G_FLOG10:
  case TargetOpcode::G_FLOG2:
  case TargetOpcode::G_FLOG:
  case TargetOpcode::G_FMA:
  case TargetOpcode::G_FMAD:
  case TargetOpcode::G_FMAXIMUM:
  case TargetOpcode::G_FMAXNUM:
  case TargetOpcode::G_FMAXNUM_IEEE:
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMINNUM_IEEE:
  case TargetOpcode::G_FMUL:
  case TargetOpcode::G_FNEARBYINT:
  case TargetOpcode::G_FNEG:
  case TargetOpcode::G_FPEXT:
  case TargetOpcode::G_FPOW:
  case TargetOpcode::G_FPTRUNC:
  case TargetOpcode::G_FREM:
  case TargetOpcode::G_FRINT:
  case TargetOpcode::G_FSIN:
  case TargetOpcode::G_FTAN:
  case TargetOpcode::G_FACOS:
  case TargetOpcode::G_FASIN:
  case TargetOpcode::G_FATAN:
  case TargetOpcode::G_FATAN2:
  case TargetOpcode::G_FCOSH:
  case TargetOpcode::G_FSINH:
  case TargetOpcode::G_FTANH:
  case TargetOpcode::G_FSQRT:
  case TargetOpcode::G_FSUB:
  case TargetOpcode::G_STRICT_FSUB:
  case TargetOpcode::G_STRICT_FADD:
  case TargetOpcode::G_STRICT_FDIV:
  case TargetOpcode::G_STRICT_FLDEXP:
  case TargetOpcode::G_STRICT_FMA:
  case TargetOpcode::G_STRICT_FMUL:
  case TargetOpcode::G_STRICT_FREM:
  case TargetOpcode::G_STRICT_FSQRT:
  case TargetOpcode::G_INTRINSIC_ROUND:
  case TargetOpcode::G_INTRINSIC_ROUNDEVEN:
  case TargetOpcode::G_INTRINSIC_TRUNC:
  case TargetOpcode::G_VECREDUCE_FADD:
  case TargetOpcode::G_VECREDUCE_FMUL:
  case TargetOpcode::G_VECREDUCE_FMAX:
  case TargetOpcode::G_VECREDUCE_FMIN:
  case TargetOpcode::G_VECREDUCE_FMAXIMUM:
  case TargetOpcode::G_VECREDUCE_FMINIMUM:
  case TargetOpcode::G_VECREDUCE_SEQ_FADD:
  case TargetOpcode::G_VECREDUCE_SEQ_FMUL:
    return true;
  case TargetOpcode::G_FPOWI: {
    return Op.isDef() || Op.getReg() == MI.getOperand(1).getReg();
  }
  // use only fp instructions
  case TargetOpcode::G_SITOFP:
  case TargetOpcode::G_UITOFP:
    return Op.isDef();
  // def only fp instructions
  case TargetOpcode::G_FPTOSI:
  case TargetOpcode::G_FPTOUI:
  case TargetOpcode::G_FPTOSI_SAT:
  case TargetOpcode::G_FPTOUI_SAT:
  case TargetOpcode::G_FCMP:
  case TargetOpcode::G_LROUND:
  case TargetOpcode::G_LLROUND:
    return Op.isUse();
  case TargetOpcode::G_FREEZE:
  case TargetOpcode::G_IMPLICIT_DEF:
  case TargetOpcode::G_PHI:
  case TargetOpcode::G_SELECT:
  case TargetOpcode::G_BUILD_VECTOR:
  case TargetOpcode::G_CONCAT_VECTORS:
  case TargetOpcode::G_INSERT_SUBVECTOR:
  case TargetOpcode::G_EXTRACT_SUBVECTOR:
  case TargetOpcode::G_SHUFFLE_VECTOR:
  case TargetOpcode::G_SPLAT_VECTOR:
  case TargetOpcode::G_STEP_VECTOR:
  case TargetOpcode::G_VECTOR_COMPRESS: {
    return all_of(MI.all_defs(),
                  [&](MachineOperand &O) {
                    return all_of(MRI->use_operands(O.getReg()), Pred);
                  }) &&
           all_of(MI.all_uses(), [&](MachineOperand &O) {
             return all_of(MRI->def_operands(O.getReg()), Pred);
           });
  }
  case TargetOpcode::G_INSERT_VECTOR_ELT:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT: {
    MachineOperand &Dst = MI.getOperand(0);
    MachineOperand &LHS = MI.getOperand(1);
    MachineOperand &RHS = MI.getOperand(2);

    return all_of(MRI->use_operands(Dst.getReg()), Pred) &&
           (!LHS.isReg() || all_of(MRI->def_operands(LHS.getReg()), Pred)) &&
           (!RHS.isReg() || all_of(MRI->def_operands(RHS.getReg()), Pred));
  }
  case TargetOpcode::G_STORE:
  case TargetOpcode::G_INDEXED_STORE: {
    MachineOperand &Val = MI.getOperand(0);
    return Op.getReg() == Val.getReg() && all_of(MRI->def_operands(Op.getReg()), Pred);
  } 
  case TargetOpcode::G_INDEXED_LOAD:
  case TargetOpcode::G_LOAD: {
    MachineOperand &Dst = MI.getOperand(0);
    return Op.getReg() == Dst.getReg() && all_of(MRI->use_operands(Dst.getReg()), Pred);
  }
  case TargetOpcode::G_ATOMICRMW_FADD:
  case TargetOpcode::G_ATOMICRMW_FSUB:
  case TargetOpcode::G_ATOMICRMW_FMAX:
  case TargetOpcode::G_ATOMICRMW_FMIN: {
    MachineOperand &WriteBack = MI.getOperand(0);
    MachineOperand &FPOp = MI.getOperand(2);
    return Op.getReg() == WriteBack.getReg() || Op.getReg() == FPOp.getReg();
  }
  case TargetOpcode::G_INTRINSIC_CONVERGENT:
  case TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS:
  case TargetOpcode::G_INTRINSIC: {
      GIntrinsic *Intrinsic = dyn_cast<GIntrinsic>(&MI);
      if (!Intrinsic)
        return false;

      unsigned Idx = Op.getOperandNo() - (Op.getOperandNo() > Intrinsic->getNumExplicitDefs());
      switch (Intrinsic->getIntrinsicID()) {
        case Intrinsic::amdgcn_rcp:
        case Intrinsic::amdgcn_rcp_legacy:
        case Intrinsic::amdgcn_rsq:
        case Intrinsic::amdgcn_rsq_clamp:
        case Intrinsic::amdgcn_rsq_legacy:
        case Intrinsic::amdgcn_sqrt:
        case Intrinsic::amdgcn_log:
        case Intrinsic::amdgcn_log_clamp:
        case Intrinsic::amdgcn_sin:
        case Intrinsic::amdgcn_exp:
        case Intrinsic::amdgcn_cos:
        case Intrinsic::amdgcn_exp2:
        case Intrinsic::amdgcn_fdiv_fast:
        case Intrinsic::amdgcn_fdot2:
        case Intrinsic::amdgcn_fdot2_f16_f16:
        case Intrinsic::amdgcn_fma_legacy:
        case Intrinsic::amdgcn_fmad_ftz:
        case Intrinsic::amdgcn_fmed3:
        case Intrinsic::amdgcn_fmul_legacy:
        case Intrinsic::amdgcn_fract:
        case Intrinsic::amdgcn_frexp_exp:
        case Intrinsic::amdgcn_div_fixup:
        case Intrinsic::amdgcn_div_scale:
        case Intrinsic::amdgcn_cvt_pkrtz:
        case Intrinsic::amdgcn_fdot2_bf16_bf16:
        case Intrinsic::amdgcn_fdot2_f32_bf16:
        case Intrinsic::amdgcn_fdot2c_f32_bf16: 
        case Intrinsic::amdgcn_dot4_f32_bf8_fp8:
        case Intrinsic::amdgcn_dot4_f32_fp8_bf8:
        case Intrinsic::amdgcn_dot4_f32_bf8_bf8:
        case Intrinsic::amdgcn_dot4_f32_fp8_fp8:
          return true;
        case Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8:
        case Intrinsic::amdgcn_mfma_f32_16x16x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_16x16x8bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8:
        case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:
        case Intrinsic::amdgcn_mfma_f32_4x4x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8:
        case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
        case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
        case Intrinsic::amdgcn_mfma_f32_16x16x8_xf32:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_f16:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8:
        case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
        case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
        case Intrinsic::amdgcn_mfma_f32_32x32x4_xf32:
        case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
        case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
        case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
        case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
        case Intrinsic::amdgcn_mfma_f64_16x16x4f64:
        case Intrinsic::amdgcn_mfma_f64_4x4x4f64:
        case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
        case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
        case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8:
        case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8:
        case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x4bf16:
        case Intrinsic::amdgcn_mfma_f32_32x32x4bf16_1k:
        case Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k:
        case Intrinsic::amdgcn_smfmac_f32_16x16x128_bf8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x128_bf8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x128_fp8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x128_fp8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x32_bf16:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_bf16:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_bf8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_bf8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_fp8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x16_bf16:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_bf8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_bf8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_fp8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x64_bf8_bf8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x64_bf8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x64_fp8_bf8:
        case Intrinsic::amdgcn_swmmac_bf16_16x16x32_bf16:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_bf16:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_bf8_bf8:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_bf8_fp8:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_fp8_bf8:
        case Intrinsic::amdgcn_swmmac_f16_16x16x32_f16:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_f16:
        case Intrinsic::amdgcn_swmmac_f32_16x16x32_fp8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_bf16:
        case Intrinsic::amdgcn_smfmac_f32_16x16x32_f16:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_f16:
        case Intrinsic::amdgcn_smfmac_f32_16x16x64_fp8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x16_f16:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_f16:
        case Intrinsic::amdgcn_smfmac_f32_32x32x32_fp8_fp8:
        case Intrinsic::amdgcn_smfmac_f32_32x32x64_fp8_fp8:
        case Intrinsic::amdgcn_wmma_bf16_16x16x16_bf16:
        case Intrinsic::amdgcn_wmma_bf16_16x16x16_bf16_tied:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_bf16:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_bf8_bf8:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_bf8_fp8:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_bf8:
        case Intrinsic::amdgcn_wmma_f16_16x16x16_f16:
        case Intrinsic::amdgcn_wmma_f16_16x16x16_f16_tied:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_f16:
        case Intrinsic::amdgcn_wmma_f32_16x16x16_fp8_fp8:
          return Idx == 0 || Idx == 1 || Idx == 2 || Idx == 3;
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_1d:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_1darray:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_2d:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_2darray:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_2darraymsaa:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_2dmsaa:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_3d:
        case Intrinsic::amdgcn_image_atomic_pk_add_bf16_cube:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_1d:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_1darray:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_2d:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_2darray:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_2darraymsaa:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_2dmsaa:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_3d:
        case Intrinsic::amdgcn_image_atomic_pk_add_f16_cube:
          return Idx == 0 || Idx == 1;
        case Intrinsic::amdgcn_flat_atomic_fmax_num:
        case Intrinsic::amdgcn_flat_atomic_fmin_num:
        case Intrinsic::amdgcn_global_atomic_fmax_num:
        case Intrinsic::amdgcn_global_atomic_fmin_num:
          return Idx == 0 || Idx == 2;
        case Intrinsic::amdgcn_raw_buffer_atomic_fadd:
        case Intrinsic::amdgcn_raw_buffer_atomic_fmax:
        case Intrinsic::amdgcn_raw_buffer_atomic_fmin:
        case Intrinsic::amdgcn_raw_ptr_buffer_atomic_fadd:
        case Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmax:
        case Intrinsic::amdgcn_raw_ptr_buffer_atomic_fmin:      
        case Intrinsic::amdgcn_struct_buffer_atomic_fadd:
        case Intrinsic::amdgcn_struct_buffer_atomic_fmax:
        case Intrinsic::amdgcn_struct_buffer_atomic_fmin:
        case Intrinsic::amdgcn_struct_ptr_buffer_atomic_fadd:
        case Intrinsic::amdgcn_struct_ptr_buffer_atomic_fmax:
        case Intrinsic::amdgcn_struct_ptr_buffer_atomic_fmin:
          return Idx == 0 || Idx == 1;
        case Intrinsic::amdgcn_interp_p1:
        case Intrinsic::amdgcn_interp_p1_f16:
          return Idx == 0 || Idx == 1;
        case Intrinsic::amdgcn_interp_p2:
        case Intrinsic::amdgcn_interp_p2_f16:
        case Intrinsic::amdgcn_interp_p2_rtz_f16:
          return Idx == 0 || Idx == 1 || Idx == 2;
        case Intrinsic::amdgcn_interp_inreg_p2:
        case Intrinsic::amdgcn_interp_inreg_p2_f16:
        case Intrinsic::amdgcn_interp_p10_rtz_f16:
        case Intrinsic::amdgcn_interp_inreg_p10:
        case Intrinsic::amdgcn_interp_inreg_p10_f16:
          return Idx == 0 || Idx == 1 || Idx == 2 || Idx == 3;
        case Intrinsic::amdgcn_fcmp: 
          return Idx == 1 || Idx == 2;
        case Intrinsic::amdgcn_class:
          return Idx == 1;
        case Intrinsic::amdgcn_cvt_pknorm_i16:
        case Intrinsic::amdgcn_cvt_pknorm_u16:
          return Idx == 1 || Idx == 2;
        case Intrinsic::amdgcn_div_fmas:
          return Idx >= 0 && Idx <= 3;
        default: {
          dbgs() << "unhandled intrinsic in" << MF->getName() << " " << MI;
        }
      }
      return false;
  }
  default:
    break;
  }

  return false;
}

bool InferTypeInfo::inferTypeInfo(MachineFunction &MF) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.instrs()) {

      for (auto &Def : MI.all_defs()) {
        Register Reg = Def.getReg();
        if (!Reg.isVirtual())
          continue;

        if (shouldBeFP(Def)) {
          updateDef(Reg);
          Changed |= true;
        }
      }

      for (auto &Use : MI.all_uses()) {
        Register Reg = Use.getReg();
        if (!Reg.isVirtual())
          continue;

        bool IsFPDef = MRI->getVRegDef(Reg) &&
            all_of(MRI->def_operands(Reg),
                   [&](MachineOperand &Op) { return shouldBeFP(Op); });
        bool IsFPUse = shouldBeFP(Use);

        if (IsFPUse && !IsFPDef) {
          updateUse(Use, true);
          Changed |= true;
        } else if (!IsFPUse && IsFPDef) {
          updateUse(Use, false);
          Changed |= true;
        }
      }

      for (auto &MemOp: MI.memoperands()) {
        bool IsFP = any_of(MI.all_defs(), [&](MachineOperand &O){ return shouldBeFP(O); }) ||
          any_of(MI.all_uses(), [&](MachineOperand &O){ return shouldBeFP(O); });

          if (!IsFP)
            continue;

          LLT Ty = MemOp->getType();
          LLT NewTy = updateType(Ty, true);
          MemOp->setType(NewTy);
      }
    }
  }

  return Changed;
}

bool InferTypeInfo::runOnMachineFunction(MachineFunction &MF) {
  init(MF);
  bool Changed = false;
  Changed |= inferTypeInfo(MF);
  return Changed;
}