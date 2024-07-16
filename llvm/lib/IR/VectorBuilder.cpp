//===- VectorBuilder.cpp - Builder for VP Intrinsics ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VectorBuilder class, which is used as a convenient
// way to create VP intrinsics as if they were LLVM instructions with a
// consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/VectorBuilder.h>

namespace llvm {

void VectorBuilder::handleError(const char *ErrorMsg) const {
  if (ErrorHandling == Behavior::SilentlyReturnNone)
    return;
  report_fatal_error(ErrorMsg);
}

Module &VectorBuilder::getModule() const {
  return *Builder.GetInsertBlock()->getModule();
}

Value *VectorBuilder::getAllTrueMask() {
  return Builder.getAllOnesMask(StaticVectorLength);
}

Value &VectorBuilder::requestMask() {
  if (Mask)
    return *Mask;

  return *getAllTrueMask();
}

Value &VectorBuilder::requestEVL() {
  if (ExplicitVectorLength)
    return *ExplicitVectorLength;

  assert(!StaticVectorLength.isScalable() && "TODO vscale lowering");
  auto *IntTy = Builder.getInt32Ty();
  return *ConstantInt::get(IntTy, StaticVectorLength.getFixedValue());
}

Value *VectorBuilder::createVectorInstruction(unsigned Opcode, Type *ReturnTy,
                                              ArrayRef<Value *> InstOpArray,
                                              const Twine &Name) {
  auto VPID = VPIntrinsic::getForOpcode(Opcode);
  if (VPID == Intrinsic::not_intrinsic)
    return returnWithError<Value *>("No VPIntrinsic for this opcode");
  return createVectorInstructionImpl(VPID, ReturnTy, InstOpArray, Name);
}

Value *VectorBuilder::createSimpleTargetReduction(RecurKind Kind, Type *ValTy,
                                                  ArrayRef<Value *> InstOpArray,
                                                  const Twine &Name) {
  Intrinsic::ID VPID;
  switch (Kind) {
  case RecurKind::Add:
    VPID = Intrinsic::vp_reduce_add;
    break;
  case RecurKind::Mul:
    VPID = Intrinsic::vp_reduce_mul;
    break;
  case RecurKind::And:
    VPID = Intrinsic::vp_reduce_and;
    break;
  case RecurKind::Or:
    VPID = Intrinsic::vp_reduce_or;
    break;
  case RecurKind::Xor:
    VPID = Intrinsic::vp_reduce_xor;
    break;
  case RecurKind::FMulAdd:
  case RecurKind::FAdd:
    VPID = Intrinsic::vp_reduce_fadd;
    break;
  case RecurKind::FMul:
    VPID = Intrinsic::vp_reduce_fmul;
    break;
  case RecurKind::SMax:
    VPID = Intrinsic::vp_reduce_smax;
    break;
  case RecurKind::SMin:
    VPID = Intrinsic::vp_reduce_smin;
    break;
  case RecurKind::UMax:
    VPID = Intrinsic::vp_reduce_umax;
    break;
  case RecurKind::UMin:
    VPID = Intrinsic::vp_reduce_umin;
    break;
  case RecurKind::FMax:
    VPID = Intrinsic::vp_reduce_fmax;
    break;
  case RecurKind::FMin:
    VPID = Intrinsic::vp_reduce_fmin;
    break;
  case RecurKind::FMaximum:
    VPID = Intrinsic::vp_reduce_fmaximum;
    break;
  case RecurKind::FMinimum:
    VPID = Intrinsic::vp_reduce_fminimum;
    break;
  default:
    llvm_unreachable("No VPIntrinsic for this reduction");
  }
  return createVectorInstructionImpl(VPID, ValTy, InstOpArray, Name);
}

Value *VectorBuilder::createVectorInstructionImpl(Intrinsic::ID VPID,
                                                  Type *ReturnTy,
                                                  ArrayRef<Value *> InstOpArray,
                                                  const Twine &Name) {
  auto MaskPosOpt = VPIntrinsic::getMaskParamPos(VPID);
  auto VLenPosOpt = VPIntrinsic::getVectorLengthParamPos(VPID);
  size_t NumInstParams = InstOpArray.size();
  size_t NumVPParams =
      NumInstParams + MaskPosOpt.has_value() + VLenPosOpt.has_value();

  SmallVector<Value *, 6> IntrinParams;

  // Whether the mask and vlen parameter are at the end of the parameter list.
  bool TrailingMaskAndVLen =
      std::min<size_t>(MaskPosOpt.value_or(NumInstParams),
                       VLenPosOpt.value_or(NumInstParams)) >= NumInstParams;

  if (TrailingMaskAndVLen) {
    // Fast path for trailing mask, vector length.
    IntrinParams.append(InstOpArray.begin(), InstOpArray.end());
    IntrinParams.resize(NumVPParams);
  } else {
    IntrinParams.resize(NumVPParams);
    // Insert mask and evl operands in between the instruction operands.
    for (size_t VPParamIdx = 0, ParamIdx = 0; VPParamIdx < NumVPParams;
         ++VPParamIdx) {
      if ((MaskPosOpt && MaskPosOpt.value_or(NumVPParams) == VPParamIdx) ||
          (VLenPosOpt && VLenPosOpt.value_or(NumVPParams) == VPParamIdx))
        continue;
      assert(ParamIdx < NumInstParams);
      IntrinParams[VPParamIdx] = InstOpArray[ParamIdx++];
    }
  }

  if (MaskPosOpt)
    IntrinParams[*MaskPosOpt] = &requestMask();
  if (VLenPosOpt)
    IntrinParams[*VLenPosOpt] = &requestEVL();

  auto *VPDecl = VPIntrinsic::getDeclarationForParams(&getModule(), VPID,
                                                      ReturnTy, IntrinParams);
  return Builder.CreateCall(VPDecl, IntrinParams, Name);
}

} // namespace llvm
