//===-- AMDGPUISelLowering.cpp - AMDGPU Common DAG lowering functions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This is the parent TargetLowering class for hardware code gen
/// targets.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUISelLowering.h"
#include "AMDGPU.h"
#include "AMDGPUFrameLowering.h"
#include "AMDGPUIntrinsicInfo.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "R600MachineFunctionInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "SIInstrInfo.h"
using namespace llvm;

static bool allocateKernArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                            CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State) {
  MachineFunction &MF = State.getMachineFunction();
  AMDGPUMachineFunction *MFI = MF.getInfo<AMDGPUMachineFunction>();

  uint64_t Offset = MFI->allocateKernArg(ValVT.getStoreSize(),
                                         ArgFlags.getOrigAlign());
  State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  return true;
}

#include "AMDGPUGenCallingConv.inc"

// Find a larger type to do a load / store of a vector with.
EVT AMDGPUTargetLowering::getEquivalentMemType(LLVMContext &Ctx, EVT VT) {
  unsigned StoreSize = VT.getStoreSizeInBits();
  if (StoreSize <= 32)
    return EVT::getIntegerVT(Ctx, StoreSize);

  assert(StoreSize % 32 == 0 && "Store size not a multiple of 32");
  return EVT::getVectorVT(Ctx, MVT::i32, StoreSize / 32);
}

AMDGPUTargetLowering::AMDGPUTargetLowering(const TargetMachine &TM,
                                           const AMDGPUSubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  // Lower floating point store/load to integer store/load to reduce the number
  // of patterns in tablegen.
  setOperationAction(ISD::LOAD, MVT::f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f32, MVT::i32);

  setOperationAction(ISD::LOAD, MVT::v2f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v4f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::LOAD, MVT::v8f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::LOAD, MVT::v16f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::LOAD, MVT::i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::i64, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v2i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2i64, MVT::v4i32);

  setOperationAction(ISD::LOAD, MVT::f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f64, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v2f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f64, MVT::v4i32);

  // There are no 64-bit extloads. These should be done as a 32-bit extload and
  // an extension to 64-bit.
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, MVT::i64, VT, Expand);
    setLoadExtAction(ISD::SEXTLOAD, MVT::i64, VT, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, MVT::i64, VT, Expand);
  }

  for (MVT VT : MVT::integer_valuetypes()) {
    if (VT == MVT::i64)
      continue;

    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i32, Expand);
  }

  for (MVT VT : MVT::integer_vector_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v4i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v4i16, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v4i16, Expand);
  }

  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f32, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f32, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f32, MVT::v8f16, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8f32, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8f16, Expand);

  setOperationAction(ISD::STORE, MVT::f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::f32, MVT::i32);

  setOperationAction(ISD::STORE, MVT::v2f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v4f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::STORE, MVT::v8f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::STORE, MVT::v16f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::STORE, MVT::i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::i64, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v2i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2i64, MVT::v4i32);

  setOperationAction(ISD::STORE, MVT::f64, Promote);
  AddPromotedToType(ISD::STORE, MVT::f64, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v2f64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f64, MVT::v4i32);

  setTruncStoreAction(MVT::v2i32, MVT::v2i8, Custom);
  setTruncStoreAction(MVT::v2i32, MVT::v2i16, Custom);

  setTruncStoreAction(MVT::v4i32, MVT::v4i8, Custom);
  setTruncStoreAction(MVT::v4i32, MVT::v4i16, Expand);

  setTruncStoreAction(MVT::v8i32, MVT::v8i16, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i8, Expand);
  setTruncStoreAction(MVT::v16i32, MVT::v16i16, Expand);

  setTruncStoreAction(MVT::i64, MVT::i1, Expand);
  setTruncStoreAction(MVT::i64, MVT::i8, Expand);
  setTruncStoreAction(MVT::i64, MVT::i16, Expand);
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);

  setTruncStoreAction(MVT::v2i64, MVT::v2i1, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i16, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i32, Expand);

  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::v2f32, MVT::v2f16, Expand);
  setTruncStoreAction(MVT::v4f32, MVT::v4f16, Expand);
  setTruncStoreAction(MVT::v8f32, MVT::v8f16, Expand);

  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  setTruncStoreAction(MVT::v2f64, MVT::v2f32, Expand);
  setTruncStoreAction(MVT::v2f64, MVT::v2f16, Expand);

  setTruncStoreAction(MVT::v4f64, MVT::v4f32, Expand);
  setTruncStoreAction(MVT::v4f64, MVT::v4f16, Expand);

  setTruncStoreAction(MVT::v8f64, MVT::v8f32, Expand);
  setTruncStoreAction(MVT::v8f64, MVT::v8f16, Expand);


  setOperationAction(ISD::Constant, MVT::i32, Legal);
  setOperationAction(ISD::Constant, MVT::i64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  // This is totally unsupported, just custom lower to produce an error.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Custom);

  // We need to custom lower some of the intrinsics
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  // Library functions.  These default to Expand, but we have instructions
  // for them.
  setOperationAction(ISD::FCEIL,  MVT::f32, Legal);
  setOperationAction(ISD::FEXP2,  MVT::f32, Legal);
  setOperationAction(ISD::FPOW,   MVT::f32, Legal);
  setOperationAction(ISD::FLOG2,  MVT::f32, Legal);
  setOperationAction(ISD::FABS,   MVT::f32, Legal);
  setOperationAction(ISD::FFLOOR, MVT::f32, Legal);
  setOperationAction(ISD::FRINT,  MVT::f32, Legal);
  setOperationAction(ISD::FTRUNC, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);

  setOperationAction(ISD::FROUND, MVT::f32, Custom);
  setOperationAction(ISD::FROUND, MVT::f64, Custom);

  setOperationAction(ISD::FNEARBYINT, MVT::f32, Custom);
  setOperationAction(ISD::FNEARBYINT, MVT::f64, Custom);

  setOperationAction(ISD::FREM, MVT::f32, Custom);
  setOperationAction(ISD::FREM, MVT::f64, Custom);

  // v_mad_f32 does not support denormals according to some sources.
  if (!Subtarget->hasFP32Denormals())
    setOperationAction(ISD::FMAD, MVT::f32, Legal);

  // Expand to fneg + fadd.
  setOperationAction(ISD::FSUB, MVT::f64, Expand);

  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4f32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8i32, Custom);

  if (Subtarget->getGeneration() < AMDGPUSubtarget::SEA_ISLANDS) {
    setOperationAction(ISD::FCEIL, MVT::f64, Custom);
    setOperationAction(ISD::FTRUNC, MVT::f64, Custom);
    setOperationAction(ISD::FRINT, MVT::f64, Custom);
    setOperationAction(ISD::FFLOOR, MVT::f64, Custom);
  }

  if (!Subtarget->hasBFI()) {
    // fcopysign can be done in a single instruction with BFI.
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  }

  setOperationAction(ISD::FP16_TO_FP, MVT::f64, Expand);

  const MVT ScalarIntVTs[] = { MVT::i32, MVT::i64 };
  for (MVT VT : ScalarIntVTs) {
    // These should use [SU]DIVREM, so set them to expand
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);

    // GPU does not have divrem function for signed or unsigned.
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Custom);

    // GPU does not have [S|U]MUL_LOHI functions as a single instruction.
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);

    setOperationAction(ISD::BSWAP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
  }

  if (!Subtarget->hasBCNT(32))
    setOperationAction(ISD::CTPOP, MVT::i32, Expand);

  if (!Subtarget->hasBCNT(64))
    setOperationAction(ISD::CTPOP, MVT::i64, Expand);

  // The hardware supports 32-bit ROTR, but not ROTL.
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i64, Expand);
  setOperationAction(ISD::ROTR, MVT::i64, Expand);

  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::MULHU, MVT::i64, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);

  setOperationAction(ISD::SMIN, MVT::i32, Legal);
  setOperationAction(ISD::UMIN, MVT::i32, Legal);
  setOperationAction(ISD::SMAX, MVT::i32, Legal);
  setOperationAction(ISD::UMAX, MVT::i32, Legal);

  if (Subtarget->hasFFBH())
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Custom);

  if (Subtarget->hasFFBL())
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Legal);

  setOperationAction(ISD::CTLZ, MVT::i64, Custom);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i64, Custom);

  // We only really have 32-bit BFE instructions (and 16-bit on VI).
  //
  // On SI+ there are 64-bit BFEs, but they are scalar only and there isn't any
  // effort to match them now. We want this to be false for i64 cases when the
  // extraction isn't restricted to the upper or lower half. Ideally we would
  // have some pass reduce 64-bit extracts to 32-bit if possible. Extracts that
  // span the midpoint are probably relatively rare, so don't worry about them
  // for now.
  if (Subtarget->hasBFE())
    setHasExtractBitsInsn(true);

  static const MVT::SimpleValueType VectorIntTypes[] = {
    MVT::v2i32, MVT::v4i32
  };

  for (MVT VT : VectorIntTypes) {
    // Expand the following operations for the current type by default.
    setOperationAction(ISD::ADD,  VT, Expand);
    setOperationAction(ISD::AND,  VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, VT, Expand);
    setOperationAction(ISD::MUL,  VT, Expand);
    setOperationAction(ISD::OR,   VT, Expand);
    setOperationAction(ISD::SHL,  VT, Expand);
    setOperationAction(ISD::SRA,  VT, Expand);
    setOperationAction(ISD::SRL,  VT, Expand);
    setOperationAction(ISD::ROTL, VT, Expand);
    setOperationAction(ISD::ROTR, VT, Expand);
    setOperationAction(ISD::SUB,  VT, Expand);
    setOperationAction(ISD::SINT_TO_FP, VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, VT, Expand);
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Expand);
    setOperationAction(ISD::ADDC, VT, Expand);
    setOperationAction(ISD::SUBC, VT, Expand);
    setOperationAction(ISD::ADDE, VT, Expand);
    setOperationAction(ISD::SUBE, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::XOR,  VT, Expand);
    setOperationAction(ISD::BSWAP, VT, Expand);
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
  }

  static const MVT::SimpleValueType FloatVectorTypes[] = {
    MVT::v2f32, MVT::v4f32
  };

  for (MVT VT : FloatVectorTypes) {
    setOperationAction(ISD::FABS, VT, Expand);
    setOperationAction(ISD::FMINNUM, VT, Expand);
    setOperationAction(ISD::FMAXNUM, VT, Expand);
    setOperationAction(ISD::FADD, VT, Expand);
    setOperationAction(ISD::FCEIL, VT, Expand);
    setOperationAction(ISD::FCOS, VT, Expand);
    setOperationAction(ISD::FDIV, VT, Expand);
    setOperationAction(ISD::FEXP2, VT, Expand);
    setOperationAction(ISD::FLOG2, VT, Expand);
    setOperationAction(ISD::FREM, VT, Expand);
    setOperationAction(ISD::FPOW, VT, Expand);
    setOperationAction(ISD::FFLOOR, VT, Expand);
    setOperationAction(ISD::FTRUNC, VT, Expand);
    setOperationAction(ISD::FMUL, VT, Expand);
    setOperationAction(ISD::FMA, VT, Expand);
    setOperationAction(ISD::FRINT, VT, Expand);
    setOperationAction(ISD::FNEARBYINT, VT, Expand);
    setOperationAction(ISD::FSQRT, VT, Expand);
    setOperationAction(ISD::FSIN, VT, Expand);
    setOperationAction(ISD::FSUB, VT, Expand);
    setOperationAction(ISD::FNEG, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
  }

  // This causes using an unrolled select operation rather than expansion with
  // bit operations. This is in general better, but the alternative using BFI
  // instructions may be better if the select sources are SGPRs.
  setOperationAction(ISD::SELECT, MVT::v2f32, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::SELECT, MVT::v4f32, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v4f32, MVT::v4i32);

  setBooleanContents(ZeroOrNegativeOneBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  setSchedulingPreference(Sched::RegPressure);
  setJumpIsExpensive(true);

  // SI at least has hardware support for floating point exceptions, but no way
  // of using or handling them is implemented. They are also optional in OpenCL
  // (Section 7.3)
  setHasFloatingPointExceptions(Subtarget->hasFPExceptions());

  setSelectIsExpensive(false);
  PredictableSelectIsExpensive = false;

  // We want to find all load dependencies for long chains of stores to enable
  // merging into very wide vectors. The problem is with vectors with > 4
  // elements. MergeConsecutiveStores will attempt to merge these because x8/x16
  // vectors are a legal type, even though we have to split the loads
  // usually. When we can more precisely specify load legality per address
  // space, we should be able to make FindBetterChain/MergeConsecutiveStores
  // smarter so that they can figure out what to do in 2 iterations without all
  // N > 4 stores on the same chain.
  GatherAllAliasesMaxDepth = 16;

  // FIXME: Need to really handle these.
  MaxStoresPerMemcpy  = 4096;
  MaxStoresPerMemmove = 4096;
  MaxStoresPerMemset  = 4096;

  setTargetDAGCombine(ISD::BITCAST);
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::SHL);
  setTargetDAGCombine(ISD::SRA);
  setTargetDAGCombine(ISD::SRL);
  setTargetDAGCombine(ISD::MUL);
  setTargetDAGCombine(ISD::MULHU);
  setTargetDAGCombine(ISD::MULHS);
  setTargetDAGCombine(ISD::SELECT);
  setTargetDAGCombine(ISD::SELECT_CC);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::FADD);
  setTargetDAGCombine(ISD::FSUB);
}

//===----------------------------------------------------------------------===//
// Target Information
//===----------------------------------------------------------------------===//

MVT AMDGPUTargetLowering::getVectorIdxTy(const DataLayout &) const {
  return MVT::i32;
}

bool AMDGPUTargetLowering::isSelectSupported(SelectSupportKind SelType) const {
  return true;
}

// The backend supports 32 and 64 bit floating point immediates.
// FIXME: Why are we reporting vectors of FP immediates as legal?
bool AMDGPUTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  EVT ScalarVT = VT.getScalarType();
  return (ScalarVT == MVT::f32 || ScalarVT == MVT::f64);
}

// We don't want to shrink f64 / f32 constants.
bool AMDGPUTargetLowering::ShouldShrinkFPConstant(EVT VT) const {
  EVT ScalarVT = VT.getScalarType();
  return (ScalarVT != MVT::f32 && ScalarVT != MVT::f64);
}

bool AMDGPUTargetLowering::shouldReduceLoadWidth(SDNode *N,
                                                 ISD::LoadExtType,
                                                 EVT NewVT) const {

  unsigned NewSize = NewVT.getStoreSizeInBits();

  // If we are reducing to a 32-bit load, this is always better.
  if (NewSize == 32)
    return true;

  EVT OldVT = N->getValueType(0);
  unsigned OldSize = OldVT.getStoreSizeInBits();

  // Don't produce extloads from sub 32-bit types. SI doesn't have scalar
  // extloads, so doing one requires using a buffer_load. In cases where we
  // still couldn't use a scalar load, using the wider load shouldn't really
  // hurt anything.

  // If the old size already had to be an extload, there's no harm in continuing
  // to reduce the width.
  return (OldSize < 32);
}

bool AMDGPUTargetLowering::isLoadBitCastBeneficial(EVT LoadTy,
                                                   EVT CastTy) const {

  assert(LoadTy.getSizeInBits() == CastTy.getSizeInBits());

  if (LoadTy.getScalarType() == MVT::i32)
    return false;

  unsigned LScalarSize = LoadTy.getScalarSizeInBits();
  unsigned CastScalarSize = CastTy.getScalarSizeInBits();

  return (LScalarSize < CastScalarSize) ||
         (CastScalarSize >= 32);
}

// SI+ has instructions for cttz / ctlz for 32-bit values. This is probably also
// profitable with the expansion for 64-bit since it's generally good to
// speculate things.
// FIXME: These should really have the size as a parameter.
bool AMDGPUTargetLowering::isCheapToSpeculateCttz() const {
  return true;
}

bool AMDGPUTargetLowering::isCheapToSpeculateCtlz() const {
  return true;
}

//===---------------------------------------------------------------------===//
// Target Properties
//===---------------------------------------------------------------------===//

bool AMDGPUTargetLowering::isFAbsFree(EVT VT) const {
  assert(VT.isFloatingPoint());
  return VT == MVT::f32 || VT == MVT::f64;
}

bool AMDGPUTargetLowering::isFNegFree(EVT VT) const {
  assert(VT.isFloatingPoint());
  return VT == MVT::f32 || VT == MVT::f64;
}

bool AMDGPUTargetLowering:: storeOfVectorConstantIsCheap(EVT MemVT,
                                                         unsigned NumElem,
                                                         unsigned AS) const {
  return true;
}

bool AMDGPUTargetLowering::aggressivelyPreferBuildVectorSources(EVT VecVT) const {
  // There are few operations which truly have vector input operands. Any vector
  // operation is going to involve operations on each component, and a
  // build_vector will be a copy per element, so it always makes sense to use a
  // build_vector input in place of the extracted element to avoid a copy into a
  // super register.
  //
  // We should probably only do this if all users are extracts only, but this
  // should be the common case.
  return true;
}

bool AMDGPUTargetLowering::isTruncateFree(EVT Source, EVT Dest) const {
  // Truncate is just accessing a subregister.
  return Dest.bitsLT(Source) && (Dest.getSizeInBits() % 32 == 0);
}

bool AMDGPUTargetLowering::isTruncateFree(Type *Source, Type *Dest) const {
  // Truncate is just accessing a subregister.
  return Dest->getPrimitiveSizeInBits() < Source->getPrimitiveSizeInBits() &&
         (Dest->getPrimitiveSizeInBits() % 32 == 0);
}

bool AMDGPUTargetLowering::isZExtFree(Type *Src, Type *Dest) const {
  unsigned SrcSize = Src->getScalarSizeInBits();
  unsigned DestSize = Dest->getScalarSizeInBits();

  return SrcSize == 32 && DestSize == 64;
}

bool AMDGPUTargetLowering::isZExtFree(EVT Src, EVT Dest) const {
  // Any register load of a 64-bit value really requires 2 32-bit moves. For all
  // practical purposes, the extra mov 0 to load a 64-bit is free.  As used,
  // this will enable reducing 64-bit operations the 32-bit, which is always
  // good.
  return Src == MVT::i32 && Dest == MVT::i64;
}

bool AMDGPUTargetLowering::isZExtFree(SDValue Val, EVT VT2) const {
  return isZExtFree(Val.getValueType(), VT2);
}

bool AMDGPUTargetLowering::isNarrowingProfitable(EVT SrcVT, EVT DestVT) const {
  // There aren't really 64-bit registers, but pairs of 32-bit ones and only a
  // limited number of native 64-bit operations. Shrinking an operation to fit
  // in a single 32-bit register should always be helpful. As currently used,
  // this is much less general than the name suggests, and is only used in
  // places trying to reduce the sizes of loads. Shrinking loads to < 32-bits is
  // not profitable, and may actually be harmful.
  return SrcVT.getSizeInBits() > 32 && DestVT.getSizeInBits() == 32;
}

//===---------------------------------------------------------------------===//
// TargetLowering Callbacks
//===---------------------------------------------------------------------===//

void AMDGPUTargetLowering::AnalyzeFormalArguments(CCState &State,
                             const SmallVectorImpl<ISD::InputArg> &Ins) const {

  State.AnalyzeFormalArguments(Ins, CC_AMDGPU);
}

void AMDGPUTargetLowering::AnalyzeReturn(CCState &State,
                           const SmallVectorImpl<ISD::OutputArg> &Outs) const {

  State.AnalyzeReturn(Outs, RetCC_SI);
}

SDValue
AMDGPUTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool isVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  return DAG.getNode(AMDGPUISD::ENDPGM, DL, MVT::Other, Chain);
}

//===---------------------------------------------------------------------===//
// Target specific lowering
//===---------------------------------------------------------------------===//

SDValue AMDGPUTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                        SmallVectorImpl<SDValue> &InVals) const {
  SDValue Callee = CLI.Callee;
  SelectionDAG &DAG = CLI.DAG;

  const Function &Fn = *DAG.getMachineFunction().getFunction();

  StringRef FuncName("<unknown>");

  if (const ExternalSymbolSDNode *G = dyn_cast<ExternalSymbolSDNode>(Callee))
    FuncName = G->getSymbol();
  else if (const GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    FuncName = G->getGlobal()->getName();

  DiagnosticInfoUnsupported NoCalls(
      Fn, "unsupported call to function " + FuncName, CLI.DL.getDebugLoc());
  DAG.getContext()->diagnose(NoCalls);

  for (unsigned I = 0, E = CLI.Ins.size(); I != E; ++I)
    InVals.push_back(DAG.getUNDEF(CLI.Ins[I].VT));

  return DAG.getEntryNode();
}

SDValue AMDGPUTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  const Function &Fn = *DAG.getMachineFunction().getFunction();

  DiagnosticInfoUnsupported NoDynamicAlloca(Fn, "unsupported dynamic alloca",
                                            SDLoc(Op).getDebugLoc());
  DAG.getContext()->diagnose(NoDynamicAlloca);
  auto Ops = {DAG.getConstant(0, SDLoc(), Op.getValueType()), Op.getOperand(0)};
  return DAG.getMergeValues(Ops, SDLoc());
}

SDValue AMDGPUTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    Op->dump(&DAG);
    llvm_unreachable("Custom lowering code for this"
                     "instruction is not implemented yet!");
    break;
  case ISD::SIGN_EXTEND_INREG: return LowerSIGN_EXTEND_INREG(Op, DAG);
  case ISD::CONCAT_VECTORS: return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR: return LowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::UDIVREM: return LowerUDIVREM(Op, DAG);
  case ISD::SDIVREM: return LowerSDIVREM(Op, DAG);
  case ISD::FREM: return LowerFREM(Op, DAG);
  case ISD::FCEIL: return LowerFCEIL(Op, DAG);
  case ISD::FTRUNC: return LowerFTRUNC(Op, DAG);
  case ISD::FRINT: return LowerFRINT(Op, DAG);
  case ISD::FNEARBYINT: return LowerFNEARBYINT(Op, DAG);
  case ISD::FROUND: return LowerFROUND(Op, DAG);
  case ISD::FFLOOR: return LowerFFLOOR(Op, DAG);
  case ISD::SINT_TO_FP: return LowerSINT_TO_FP(Op, DAG);
  case ISD::UINT_TO_FP: return LowerUINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT: return LowerFP_TO_SINT(Op, DAG);
  case ISD::FP_TO_UINT: return LowerFP_TO_UINT(Op, DAG);
  case ISD::CTLZ:
  case ISD::CTLZ_ZERO_UNDEF:
    return LowerCTLZ(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  }
  return Op;
}

void AMDGPUTargetLowering::ReplaceNodeResults(SDNode *N,
                                              SmallVectorImpl<SDValue> &Results,
                                              SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::SIGN_EXTEND_INREG:
    // Different parts of legalization seem to interpret which type of
    // sign_extend_inreg is the one to check for custom lowering. The extended
    // from type is what really matters, but some places check for custom
    // lowering of the result type. This results in trying to use
    // ReplaceNodeResults to sext_in_reg to an illegal type, so we'll just do
    // nothing here and let the illegal result integer be handled normally.
    return;
  default:
    return;
  }
}

static bool hasDefinedInitializer(const GlobalValue *GV) {
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar || !GVar->hasInitializer())
    return false;

  return !isa<UndefValue>(GVar->getInitializer());
}

SDValue AMDGPUTargetLowering::LowerGlobalAddress(AMDGPUMachineFunction* MFI,
                                                 SDValue Op,
                                                 SelectionDAG &DAG) const {

  const DataLayout &DL = DAG.getDataLayout();
  GlobalAddressSDNode *G = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = G->getGlobal();

  switch (G->getAddressSpace()) {
  case AMDGPUAS::LOCAL_ADDRESS: {
    // XXX: What does the value of G->getOffset() mean?
    assert(G->getOffset() == 0 &&
         "Do not know what to do with an non-zero offset");

    // TODO: We could emit code to handle the initialization somewhere.
    if (hasDefinedInitializer(GV))
      break;

    unsigned Offset = MFI->allocateLDSGlobal(DL, *GV);
    return DAG.getConstant(Offset, SDLoc(Op), Op.getValueType());
  }
  }

  const Function &Fn = *DAG.getMachineFunction().getFunction();
  DiagnosticInfoUnsupported BadInit(
      Fn, "unsupported initializer for address space", SDLoc(Op).getDebugLoc());
  DAG.getContext()->diagnose(BadInit);
  return SDValue();
}

SDValue AMDGPUTargetLowering::LowerCONCAT_VECTORS(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SmallVector<SDValue, 8> Args;

  for (const SDUse &U : Op->ops())
    DAG.ExtractVectorElements(U.get(), Args);

  return DAG.getBuildVector(Op.getValueType(), SDLoc(Op), Args);
}

SDValue AMDGPUTargetLowering::LowerEXTRACT_SUBVECTOR(SDValue Op,
                                                     SelectionDAG &DAG) const {

  SmallVector<SDValue, 8> Args;
  unsigned Start = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  EVT VT = Op.getValueType();
  DAG.ExtractVectorElements(Op.getOperand(0), Args, Start,
                            VT.getVectorNumElements());

  return DAG.getBuildVector(Op.getValueType(), SDLoc(Op), Args);
}

SDValue AMDGPUTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
    SelectionDAG &DAG) const {
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  switch (IntrinsicID) {
    default: return Op;
    case AMDGPUIntrinsic::AMDGPU_clamp: // Legacy name.
      return DAG.getNode(AMDGPUISD::CLAMP, DL, VT,
                         Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_bfe_i32:
      return DAG.getNode(AMDGPUISD::BFE_I32, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));

    case AMDGPUIntrinsic::AMDGPU_bfe_u32:
      return DAG.getNode(AMDGPUISD::BFE_U32, DL, VT,
                         Op.getOperand(1),
                         Op.getOperand(2),
                         Op.getOperand(3));
  }
}

/// \brief Generate Min/Max node
SDValue AMDGPUTargetLowering::CombineFMinMaxLegacy(const SDLoc &DL, EVT VT,
                                                   SDValue LHS, SDValue RHS,
                                                   SDValue True, SDValue False,
                                                   SDValue CC,
                                                   DAGCombinerInfo &DCI) const {
  if (Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return SDValue();

  if (!(LHS == True && RHS == False) && !(LHS == False && RHS == True))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  ISD::CondCode CCOpcode = cast<CondCodeSDNode>(CC)->get();
  switch (CCOpcode) {
  case ISD::SETOEQ:
  case ISD::SETONE:
  case ISD::SETUNE:
  case ISD::SETNE:
  case ISD::SETUEQ:
  case ISD::SETEQ:
  case ISD::SETFALSE:
  case ISD::SETFALSE2:
  case ISD::SETTRUE:
  case ISD::SETTRUE2:
  case ISD::SETUO:
  case ISD::SETO:
    break;
  case ISD::SETULE:
  case ISD::SETULT: {
    if (LHS == True)
      return DAG.getNode(AMDGPUISD::FMIN_LEGACY, DL, VT, RHS, LHS);
    return DAG.getNode(AMDGPUISD::FMAX_LEGACY, DL, VT, LHS, RHS);
  }
  case ISD::SETOLE:
  case ISD::SETOLT:
  case ISD::SETLE:
  case ISD::SETLT: {
    // Ordered. Assume ordered for undefined.

    // Only do this after legalization to avoid interfering with other combines
    // which might occur.
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG &&
        !DCI.isCalledByLegalizer())
      return SDValue();

    // We need to permute the operands to get the correct NaN behavior. The
    // selected operand is the second one based on the failing compare with NaN,
    // so permute it based on the compare type the hardware uses.
    if (LHS == True)
      return DAG.getNode(AMDGPUISD::FMIN_LEGACY, DL, VT, LHS, RHS);
    return DAG.getNode(AMDGPUISD::FMAX_LEGACY, DL, VT, RHS, LHS);
  }
  case ISD::SETUGE:
  case ISD::SETUGT: {
    if (LHS == True)
      return DAG.getNode(AMDGPUISD::FMAX_LEGACY, DL, VT, RHS, LHS);
    return DAG.getNode(AMDGPUISD::FMIN_LEGACY, DL, VT, LHS, RHS);
  }
  case ISD::SETGT:
  case ISD::SETGE:
  case ISD::SETOGE:
  case ISD::SETOGT: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG &&
        !DCI.isCalledByLegalizer())
      return SDValue();

    if (LHS == True)
      return DAG.getNode(AMDGPUISD::FMAX_LEGACY, DL, VT, LHS, RHS);
    return DAG.getNode(AMDGPUISD::FMIN_LEGACY, DL, VT, RHS, LHS);
  }
  case ISD::SETCC_INVALID:
    llvm_unreachable("Invalid setcc condcode!");
  }
  return SDValue();
}

std::pair<SDValue, SDValue>
AMDGPUTargetLowering::split64BitValue(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, Zero);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, One);

  return std::make_pair(Lo, Hi);
}

SDValue AMDGPUTargetLowering::getLoHalf64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);
  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, Zero);
}

SDValue AMDGPUTargetLowering::getHiHalf64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, One);
}

SDValue AMDGPUTargetLowering::SplitVectorLoad(const SDValue Op,
                                              SelectionDAG &DAG) const {
  LoadSDNode *Load = cast<LoadSDNode>(Op);
  EVT VT = Op.getValueType();


  // If this is a 2 element vector, we really want to scalarize and not create
  // weird 1 element vectors.
  if (VT.getVectorNumElements() == 2)
    return scalarizeVectorLoad(Load, DAG);

  SDValue BasePtr = Load->getBasePtr();
  EVT PtrVT = BasePtr.getValueType();
  EVT MemVT = Load->getMemoryVT();
  SDLoc SL(Op);

  const MachinePointerInfo &SrcValue = Load->getMemOperand()->getPointerInfo();

  EVT LoVT, HiVT;
  EVT LoMemVT, HiMemVT;
  SDValue Lo, Hi;

  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VT);
  std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemVT);
  std::tie(Lo, Hi) = DAG.SplitVector(Op, SL, LoVT, HiVT);

  unsigned Size = LoMemVT.getStoreSize();
  unsigned BaseAlign = Load->getAlignment();
  unsigned HiAlign = MinAlign(BaseAlign, Size);

  SDValue LoLoad = DAG.getExtLoad(Load->getExtensionType(), SL, LoVT,
                                  Load->getChain(), BasePtr, SrcValue, LoMemVT,
                                  BaseAlign, Load->getMemOperand()->getFlags());
  SDValue HiPtr = DAG.getNode(ISD::ADD, SL, PtrVT, BasePtr,
                              DAG.getConstant(Size, SL, PtrVT));
  SDValue HiLoad =
      DAG.getExtLoad(Load->getExtensionType(), SL, HiVT, Load->getChain(),
                     HiPtr, SrcValue.getWithOffset(LoMemVT.getStoreSize()),
                     HiMemVT, HiAlign, Load->getMemOperand()->getFlags());

  SDValue Ops[] = {
    DAG.getNode(ISD::CONCAT_VECTORS, SL, VT, LoLoad, HiLoad),
    DAG.getNode(ISD::TokenFactor, SL, MVT::Other,
                LoLoad.getValue(1), HiLoad.getValue(1))
  };

  return DAG.getMergeValues(Ops, SL);
}

SDValue AMDGPUTargetLowering::SplitVectorStore(SDValue Op,
                                               SelectionDAG &DAG) const {
  StoreSDNode *Store = cast<StoreSDNode>(Op);
  SDValue Val = Store->getValue();
  EVT VT = Val.getValueType();

  // If this is a 2 element vector, we really want to scalarize and not create
  // weird 1 element vectors.
  if (VT.getVectorNumElements() == 2)
    return scalarizeVectorStore(Store, DAG);

  EVT MemVT = Store->getMemoryVT();
  SDValue Chain = Store->getChain();
  SDValue BasePtr = Store->getBasePtr();
  SDLoc SL(Op);

  EVT LoVT, HiVT;
  EVT LoMemVT, HiMemVT;
  SDValue Lo, Hi;

  std::tie(LoVT, HiVT) = DAG.GetSplitDestVTs(VT);
  std::tie(LoMemVT, HiMemVT) = DAG.GetSplitDestVTs(MemVT);
  std::tie(Lo, Hi) = DAG.SplitVector(Val, SL, LoVT, HiVT);

  EVT PtrVT = BasePtr.getValueType();
  SDValue HiPtr = DAG.getNode(ISD::ADD, SL, PtrVT, BasePtr,
                              DAG.getConstant(LoMemVT.getStoreSize(), SL,
                                              PtrVT));

  const MachinePointerInfo &SrcValue = Store->getMemOperand()->getPointerInfo();
  unsigned BaseAlign = Store->getAlignment();
  unsigned Size = LoMemVT.getStoreSize();
  unsigned HiAlign = MinAlign(BaseAlign, Size);

  SDValue LoStore =
      DAG.getTruncStore(Chain, SL, Lo, BasePtr, SrcValue, LoMemVT, BaseAlign,
                        Store->getMemOperand()->getFlags());
  SDValue HiStore =
      DAG.getTruncStore(Chain, SL, Hi, HiPtr, SrcValue.getWithOffset(Size),
                        HiMemVT, HiAlign, Store->getMemOperand()->getFlags());

  return DAG.getNode(ISD::TokenFactor, SL, MVT::Other, LoStore, HiStore);
}

// This is a shortcut for integer division because we have fast i32<->f32
// conversions, and fast f32 reciprocal instructions. The fractional part of a
// float is enough to accurately represent up to a 24-bit signed integer.
SDValue AMDGPUTargetLowering::LowerDIVREM24(SDValue Op, SelectionDAG &DAG,
                                            bool Sign) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  MVT IntVT = MVT::i32;
  MVT FltVT = MVT::f32;

  unsigned LHSSignBits = DAG.ComputeNumSignBits(LHS);
  if (LHSSignBits < 9)
    return SDValue();

  unsigned RHSSignBits = DAG.ComputeNumSignBits(RHS);
  if (RHSSignBits < 9)
    return SDValue();

  unsigned BitSize = VT.getSizeInBits();
  unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
  unsigned DivBits = BitSize - SignBits;
  if (Sign)
    ++DivBits;

  ISD::NodeType ToFp = Sign ? ISD::SINT_TO_FP : ISD::UINT_TO_FP;
  ISD::NodeType ToInt = Sign ? ISD::FP_TO_SINT : ISD::FP_TO_UINT;

  SDValue jq = DAG.getConstant(1, DL, IntVT);

  if (Sign) {
    // char|short jq = ia ^ ib;
    jq = DAG.getNode(ISD::XOR, DL, VT, LHS, RHS);

    // jq = jq >> (bitsize - 2)
    jq = DAG.getNode(ISD::SRA, DL, VT, jq,
                     DAG.getConstant(BitSize - 2, DL, VT));

    // jq = jq | 0x1
    jq = DAG.getNode(ISD::OR, DL, VT, jq, DAG.getConstant(1, DL, VT));
  }

  // int ia = (int)LHS;
  SDValue ia = LHS;

  // int ib, (int)RHS;
  SDValue ib = RHS;

  // float fa = (float)ia;
  SDValue fa = DAG.getNode(ToFp, DL, FltVT, ia);

  // float fb = (float)ib;
  SDValue fb = DAG.getNode(ToFp, DL, FltVT, ib);

  SDValue fq = DAG.getNode(ISD::FMUL, DL, FltVT,
                           fa, DAG.getNode(AMDGPUISD::RCP, DL, FltVT, fb));

  // fq = trunc(fq);
  fq = DAG.getNode(ISD::FTRUNC, DL, FltVT, fq);

  // float fqneg = -fq;
  SDValue fqneg = DAG.getNode(ISD::FNEG, DL, FltVT, fq);

  // float fr = mad(fqneg, fb, fa);
  SDValue fr = DAG.getNode(ISD::FMAD, DL, FltVT, fqneg, fb, fa);

  // int iq = (int)fq;
  SDValue iq = DAG.getNode(ToInt, DL, IntVT, fq);

  // fr = fabs(fr);
  fr = DAG.getNode(ISD::FABS, DL, FltVT, fr);

  // fb = fabs(fb);
  fb = DAG.getNode(ISD::FABS, DL, FltVT, fb);

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);

  // int cv = fr >= fb;
  SDValue cv = DAG.getSetCC(DL, SetCCVT, fr, fb, ISD::SETOGE);

  // jq = (cv ? jq : 0);
  jq = DAG.getNode(ISD::SELECT, DL, VT, cv, jq, DAG.getConstant(0, DL, VT));

  // dst = iq + jq;
  SDValue Div = DAG.getNode(ISD::ADD, DL, VT, iq, jq);

  // Rem needs compensation, it's easier to recompute it
  SDValue Rem = DAG.getNode(ISD::MUL, DL, VT, Div, RHS);
  Rem = DAG.getNode(ISD::SUB, DL, VT, LHS, Rem);

  // Truncate to number of bits this divide really is.
  if (Sign) {
    SDValue InRegSize
      = DAG.getValueType(EVT::getIntegerVT(*DAG.getContext(), DivBits));
    Div = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Div, InRegSize);
    Rem = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Rem, InRegSize);
  } else {
    SDValue TruncMask = DAG.getConstant((UINT64_C(1) << DivBits) - 1, DL, VT);
    Div = DAG.getNode(ISD::AND, DL, VT, Div, TruncMask);
    Rem = DAG.getNode(ISD::AND, DL, VT, Rem, TruncMask);
  }

  return DAG.getMergeValues({ Div, Rem }, DL);
}

void AMDGPUTargetLowering::LowerUDIVREM64(SDValue Op,
                                      SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &Results) const {
  assert(Op.getValueType() == MVT::i64);

  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  EVT HalfVT = VT.getHalfSizedIntegerVT(*DAG.getContext());

  SDValue one = DAG.getConstant(1, DL, HalfVT);
  SDValue zero = DAG.getConstant(0, DL, HalfVT);

  //HiLo split
  SDValue LHS = Op.getOperand(0);
  SDValue LHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, zero);
  SDValue LHS_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, one);

  SDValue RHS = Op.getOperand(1);
  SDValue RHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, zero);
  SDValue RHS_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, one);

  if (VT == MVT::i64 &&
    DAG.MaskedValueIsZero(RHS, APInt::getHighBitsSet(64, 32)) &&
    DAG.MaskedValueIsZero(LHS, APInt::getHighBitsSet(64, 32))) {

    SDValue Res = DAG.getNode(ISD::UDIVREM, DL, DAG.getVTList(HalfVT, HalfVT),
                              LHS_Lo, RHS_Lo);

    SDValue DIV = DAG.getBuildVector(MVT::v2i32, DL, {Res.getValue(0), zero});
    SDValue REM = DAG.getBuildVector(MVT::v2i32, DL, {Res.getValue(1), zero});

    Results.push_back(DAG.getNode(ISD::BITCAST, DL, MVT::i64, DIV));
    Results.push_back(DAG.getNode(ISD::BITCAST, DL, MVT::i64, REM));
    return;
  }

  // Get Speculative values
  SDValue DIV_Part = DAG.getNode(ISD::UDIV, DL, HalfVT, LHS_Hi, RHS_Lo);
  SDValue REM_Part = DAG.getNode(ISD::UREM, DL, HalfVT, LHS_Hi, RHS_Lo);

  SDValue REM_Lo = DAG.getSelectCC(DL, RHS_Hi, zero, REM_Part, LHS_Hi, ISD::SETEQ);
  SDValue REM = DAG.getBuildVector(MVT::v2i32, DL, {REM_Lo, zero});
  REM = DAG.getNode(ISD::BITCAST, DL, MVT::i64, REM);

  SDValue DIV_Hi = DAG.getSelectCC(DL, RHS_Hi, zero, DIV_Part, zero, ISD::SETEQ);
  SDValue DIV_Lo = zero;

  const unsigned halfBitWidth = HalfVT.getSizeInBits();

  for (unsigned i = 0; i < halfBitWidth; ++i) {
    const unsigned bitPos = halfBitWidth - i - 1;
    SDValue POS = DAG.getConstant(bitPos, DL, HalfVT);
    // Get value of high bit
    SDValue HBit = DAG.getNode(ISD::SRL, DL, HalfVT, LHS_Lo, POS);
    HBit = DAG.getNode(ISD::AND, DL, HalfVT, HBit, one);
    HBit = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, HBit);

    // Shift
    REM = DAG.getNode(ISD::SHL, DL, VT, REM, DAG.getConstant(1, DL, VT));
    // Add LHS high bit
    REM = DAG.getNode(ISD::OR, DL, VT, REM, HBit);

    SDValue BIT = DAG.getConstant(1ULL << bitPos, DL, HalfVT);
    SDValue realBIT = DAG.getSelectCC(DL, REM, RHS, BIT, zero, ISD::SETUGE);

    DIV_Lo = DAG.getNode(ISD::OR, DL, HalfVT, DIV_Lo, realBIT);

    // Update REM
    SDValue REM_sub = DAG.getNode(ISD::SUB, DL, VT, REM, RHS);
    REM = DAG.getSelectCC(DL, REM, RHS, REM_sub, REM, ISD::SETUGE);
  }

  SDValue DIV = DAG.getBuildVector(MVT::v2i32, DL, {DIV_Lo, DIV_Hi});
  DIV = DAG.getNode(ISD::BITCAST, DL, MVT::i64, DIV);
  Results.push_back(DIV);
  Results.push_back(REM);
}

SDValue AMDGPUTargetLowering::LowerUDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  if (VT == MVT::i64) {
    SmallVector<SDValue, 2> Results;
    LowerUDIVREM64(Op, DAG, Results);
    return DAG.getMergeValues(Results, DL);
  }

  if (VT == MVT::i32) {
    if (SDValue Res = LowerDIVREM24(Op, DAG, false))
      return Res;
  }

  SDValue Num = Op.getOperand(0);
  SDValue Den = Op.getOperand(1);

  // RCP =  URECIP(Den) = 2^32 / Den + e
  // e is rounding error.
  SDValue RCP = DAG.getNode(AMDGPUISD::URECIP, DL, VT, Den);

  // RCP_LO = mul(RCP, Den) */
  SDValue RCP_LO = DAG.getNode(ISD::MUL, DL, VT, RCP, Den);

  // RCP_HI = mulhu (RCP, Den) */
  SDValue RCP_HI = DAG.getNode(ISD::MULHU, DL, VT, RCP, Den);

  // NEG_RCP_LO = -RCP_LO
  SDValue NEG_RCP_LO = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT),
                                                     RCP_LO);

  // ABS_RCP_LO = (RCP_HI == 0 ? NEG_RCP_LO : RCP_LO)
  SDValue ABS_RCP_LO = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, DL, VT),
                                           NEG_RCP_LO, RCP_LO,
                                           ISD::SETEQ);
  // Calculate the rounding error from the URECIP instruction
  // E = mulhu(ABS_RCP_LO, RCP)
  SDValue E = DAG.getNode(ISD::MULHU, DL, VT, ABS_RCP_LO, RCP);

  // RCP_A_E = RCP + E
  SDValue RCP_A_E = DAG.getNode(ISD::ADD, DL, VT, RCP, E);

  // RCP_S_E = RCP - E
  SDValue RCP_S_E = DAG.getNode(ISD::SUB, DL, VT, RCP, E);

  // Tmp0 = (RCP_HI == 0 ? RCP_A_E : RCP_SUB_E)
  SDValue Tmp0 = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, DL, VT),
                                     RCP_A_E, RCP_S_E,
                                     ISD::SETEQ);
  // Quotient = mulhu(Tmp0, Num)
  SDValue Quotient = DAG.getNode(ISD::MULHU, DL, VT, Tmp0, Num);

  // Num_S_Remainder = Quotient * Den
  SDValue Num_S_Remainder = DAG.getNode(ISD::MUL, DL, VT, Quotient, Den);

  // Remainder = Num - Num_S_Remainder
  SDValue Remainder = DAG.getNode(ISD::SUB, DL, VT, Num, Num_S_Remainder);

  // Remainder_GE_Den = (Remainder >= Den ? -1 : 0)
  SDValue Remainder_GE_Den = DAG.getSelectCC(DL, Remainder, Den,
                                                 DAG.getConstant(-1, DL, VT),
                                                 DAG.getConstant(0, DL, VT),
                                                 ISD::SETUGE);
  // Remainder_GE_Zero = (Num >= Num_S_Remainder ? -1 : 0)
  SDValue Remainder_GE_Zero = DAG.getSelectCC(DL, Num,
                                                  Num_S_Remainder,
                                                  DAG.getConstant(-1, DL, VT),
                                                  DAG.getConstant(0, DL, VT),
                                                  ISD::SETUGE);
  // Tmp1 = Remainder_GE_Den & Remainder_GE_Zero
  SDValue Tmp1 = DAG.getNode(ISD::AND, DL, VT, Remainder_GE_Den,
                                               Remainder_GE_Zero);

  // Calculate Division result:

  // Quotient_A_One = Quotient + 1
  SDValue Quotient_A_One = DAG.getNode(ISD::ADD, DL, VT, Quotient,
                                       DAG.getConstant(1, DL, VT));

  // Quotient_S_One = Quotient - 1
  SDValue Quotient_S_One = DAG.getNode(ISD::SUB, DL, VT, Quotient,
                                       DAG.getConstant(1, DL, VT));

  // Div = (Tmp1 == 0 ? Quotient : Quotient_A_One)
  SDValue Div = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, DL, VT),
                                     Quotient, Quotient_A_One, ISD::SETEQ);

  // Div = (Remainder_GE_Zero == 0 ? Quotient_S_One : Div)
  Div = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, DL, VT),
                            Quotient_S_One, Div, ISD::SETEQ);

  // Calculate Rem result:

  // Remainder_S_Den = Remainder - Den
  SDValue Remainder_S_Den = DAG.getNode(ISD::SUB, DL, VT, Remainder, Den);

  // Remainder_A_Den = Remainder + Den
  SDValue Remainder_A_Den = DAG.getNode(ISD::ADD, DL, VT, Remainder, Den);

  // Rem = (Tmp1 == 0 ? Remainder : Remainder_S_Den)
  SDValue Rem = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, DL, VT),
                                    Remainder, Remainder_S_Den, ISD::SETEQ);

  // Rem = (Remainder_GE_Zero == 0 ? Remainder_A_Den : Rem)
  Rem = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, DL, VT),
                            Remainder_A_Den, Rem, ISD::SETEQ);
  SDValue Ops[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Ops, DL);
}

SDValue AMDGPUTargetLowering::LowerSDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue NegOne = DAG.getConstant(-1, DL, VT);

  if (VT == MVT::i32) {
    if (SDValue Res = LowerDIVREM24(Op, DAG, true))
      return Res;
  }

  if (VT == MVT::i64 &&
      DAG.ComputeNumSignBits(LHS) > 32 &&
      DAG.ComputeNumSignBits(RHS) > 32) {
    EVT HalfVT = VT.getHalfSizedIntegerVT(*DAG.getContext());

    //HiLo split
    SDValue LHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, Zero);
    SDValue RHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, Zero);
    SDValue DIVREM = DAG.getNode(ISD::SDIVREM, DL, DAG.getVTList(HalfVT, HalfVT),
                                 LHS_Lo, RHS_Lo);
    SDValue Res[2] = {
      DAG.getNode(ISD::SIGN_EXTEND, DL, VT, DIVREM.getValue(0)),
      DAG.getNode(ISD::SIGN_EXTEND, DL, VT, DIVREM.getValue(1))
    };
    return DAG.getMergeValues(Res, DL);
  }

  SDValue LHSign = DAG.getSelectCC(DL, LHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue RHSign = DAG.getSelectCC(DL, RHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue DSign = DAG.getNode(ISD::XOR, DL, VT, LHSign, RHSign);
  SDValue RSign = LHSign; // Remainder sign is the same as LHS

  LHS = DAG.getNode(ISD::ADD, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::ADD, DL, VT, RHS, RHSign);

  LHS = DAG.getNode(ISD::XOR, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::XOR, DL, VT, RHS, RHSign);

  SDValue Div = DAG.getNode(ISD::UDIVREM, DL, DAG.getVTList(VT, VT), LHS, RHS);
  SDValue Rem = Div.getValue(1);

  Div = DAG.getNode(ISD::XOR, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::XOR, DL, VT, Rem, RSign);

  Div = DAG.getNode(ISD::SUB, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::SUB, DL, VT, Rem, RSign);

  SDValue Res[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Res, DL);
}

// (frem x, y) -> (fsub x, (fmul (ftrunc (fdiv x, y)), y))
SDValue AMDGPUTargetLowering::LowerFREM(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  // TODO: Should this propagate fast-math-flags?

  SDValue Div = DAG.getNode(ISD::FDIV, SL, VT, X, Y);
  SDValue Floor = DAG.getNode(ISD::FTRUNC, SL, VT, Div);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, VT, Floor, Y);

  return DAG.getNode(ISD::FSUB, SL, VT, X, Mul);
}

SDValue AMDGPUTargetLowering::LowerFCEIL(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  // result = trunc(src)
  // if (src > 0.0 && src != result)
  //   result += 1.0

  SDValue Trunc = DAG.getNode(ISD::FTRUNC, SL, MVT::f64, Src);

  const SDValue Zero = DAG.getConstantFP(0.0, SL, MVT::f64);
  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f64);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f64);

  SDValue Lt0 = DAG.getSetCC(SL, SetCCVT, Src, Zero, ISD::SETOGT);
  SDValue NeTrunc = DAG.getSetCC(SL, SetCCVT, Src, Trunc, ISD::SETONE);
  SDValue And = DAG.getNode(ISD::AND, SL, SetCCVT, Lt0, NeTrunc);

  SDValue Add = DAG.getNode(ISD::SELECT, SL, MVT::f64, And, One, Zero);
  // TODO: Should this propagate fast-math-flags?
  return DAG.getNode(ISD::FADD, SL, MVT::f64, Trunc, Add);
}

static SDValue extractF64Exponent(SDValue Hi, const SDLoc &SL,
                                  SelectionDAG &DAG) {
  const unsigned FractBits = 52;
  const unsigned ExpBits = 11;

  SDValue ExpPart = DAG.getNode(AMDGPUISD::BFE_U32, SL, MVT::i32,
                                Hi,
                                DAG.getConstant(FractBits - 32, SL, MVT::i32),
                                DAG.getConstant(ExpBits, SL, MVT::i32));
  SDValue Exp = DAG.getNode(ISD::SUB, SL, MVT::i32, ExpPart,
                            DAG.getConstant(1023, SL, MVT::i32));

  return Exp;
}

SDValue AMDGPUTargetLowering::LowerFTRUNC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  assert(Op.getValueType() == MVT::f64);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue VecSrc = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Src);

  // Extract the upper half, since this is where we will find the sign and
  // exponent.
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecSrc, One);

  SDValue Exp = extractF64Exponent(Hi, SL, DAG);

  const unsigned FractBits = 52;

  // Extract the sign bit.
  const SDValue SignBitMask = DAG.getConstant(UINT32_C(1) << 31, SL, MVT::i32);
  SDValue SignBit = DAG.getNode(ISD::AND, SL, MVT::i32, Hi, SignBitMask);

  // Extend back to to 64-bits.
  SDValue SignBit64 = DAG.getBuildVector(MVT::v2i32, SL, {Zero, SignBit});
  SignBit64 = DAG.getNode(ISD::BITCAST, SL, MVT::i64, SignBit64);

  SDValue BcInt = DAG.getNode(ISD::BITCAST, SL, MVT::i64, Src);
  const SDValue FractMask
    = DAG.getConstant((UINT64_C(1) << FractBits) - 1, SL, MVT::i64);

  SDValue Shr = DAG.getNode(ISD::SRA, SL, MVT::i64, FractMask, Exp);
  SDValue Not = DAG.getNOT(SL, Shr, MVT::i64);
  SDValue Tmp0 = DAG.getNode(ISD::AND, SL, MVT::i64, BcInt, Not);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::i32);

  const SDValue FiftyOne = DAG.getConstant(FractBits - 1, SL, MVT::i32);

  SDValue ExpLt0 = DAG.getSetCC(SL, SetCCVT, Exp, Zero, ISD::SETLT);
  SDValue ExpGt51 = DAG.getSetCC(SL, SetCCVT, Exp, FiftyOne, ISD::SETGT);

  SDValue Tmp1 = DAG.getNode(ISD::SELECT, SL, MVT::i64, ExpLt0, SignBit64, Tmp0);
  SDValue Tmp2 = DAG.getNode(ISD::SELECT, SL, MVT::i64, ExpGt51, BcInt, Tmp1);

  return DAG.getNode(ISD::BITCAST, SL, MVT::f64, Tmp2);
}

SDValue AMDGPUTargetLowering::LowerFRINT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  assert(Op.getValueType() == MVT::f64);

  APFloat C1Val(APFloat::IEEEdouble, "0x1.0p+52");
  SDValue C1 = DAG.getConstantFP(C1Val, SL, MVT::f64);
  SDValue CopySign = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f64, C1, Src);

  // TODO: Should this propagate fast-math-flags?

  SDValue Tmp1 = DAG.getNode(ISD::FADD, SL, MVT::f64, Src, CopySign);
  SDValue Tmp2 = DAG.getNode(ISD::FSUB, SL, MVT::f64, Tmp1, CopySign);

  SDValue Fabs = DAG.getNode(ISD::FABS, SL, MVT::f64, Src);

  APFloat C2Val(APFloat::IEEEdouble, "0x1.fffffffffffffp+51");
  SDValue C2 = DAG.getConstantFP(C2Val, SL, MVT::f64);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f64);
  SDValue Cond = DAG.getSetCC(SL, SetCCVT, Fabs, C2, ISD::SETOGT);

  return DAG.getSelect(SL, MVT::f64, Cond, Src, Tmp2);
}

SDValue AMDGPUTargetLowering::LowerFNEARBYINT(SDValue Op, SelectionDAG &DAG) const {
  // FNEARBYINT and FRINT are the same, except in their handling of FP
  // exceptions. Those aren't really meaningful for us, and OpenCL only has
  // rint, so just treat them as equivalent.
  return DAG.getNode(ISD::FRINT, SDLoc(Op), Op.getValueType(), Op.getOperand(0));
}

// XXX - May require not supporting f32 denormals?
SDValue AMDGPUTargetLowering::LowerFROUND32(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);

  SDValue T = DAG.getNode(ISD::FTRUNC, SL, MVT::f32, X);

  // TODO: Should this propagate fast-math-flags?

  SDValue Diff = DAG.getNode(ISD::FSUB, SL, MVT::f32, X, T);

  SDValue AbsDiff = DAG.getNode(ISD::FABS, SL, MVT::f32, Diff);

  const SDValue Zero = DAG.getConstantFP(0.0, SL, MVT::f32);
  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);
  const SDValue Half = DAG.getConstantFP(0.5, SL, MVT::f32);

  SDValue SignOne = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f32, One, X);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);

  SDValue Cmp = DAG.getSetCC(SL, SetCCVT, AbsDiff, Half, ISD::SETOGE);

  SDValue Sel = DAG.getNode(ISD::SELECT, SL, MVT::f32, Cmp, SignOne, Zero);

  return DAG.getNode(ISD::FADD, SL, MVT::f32, T, Sel);
}

SDValue AMDGPUTargetLowering::LowerFROUND64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);

  SDValue L = DAG.getNode(ISD::BITCAST, SL, MVT::i64, X);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  const SDValue NegOne = DAG.getConstant(-1, SL, MVT::i32);
  const SDValue FiftyOne = DAG.getConstant(51, SL, MVT::i32);
  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::i32);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, X);

  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BC, One);

  SDValue Exp = extractF64Exponent(Hi, SL, DAG);

  const SDValue Mask = DAG.getConstant(INT64_C(0x000fffffffffffff), SL,
                                       MVT::i64);

  SDValue M = DAG.getNode(ISD::SRA, SL, MVT::i64, Mask, Exp);
  SDValue D = DAG.getNode(ISD::SRA, SL, MVT::i64,
                          DAG.getConstant(INT64_C(0x0008000000000000), SL,
                                          MVT::i64),
                          Exp);

  SDValue Tmp0 = DAG.getNode(ISD::AND, SL, MVT::i64, L, M);
  SDValue Tmp1 = DAG.getSetCC(SL, SetCCVT,
                              DAG.getConstant(0, SL, MVT::i64), Tmp0,
                              ISD::SETNE);

  SDValue Tmp2 = DAG.getNode(ISD::SELECT, SL, MVT::i64, Tmp1,
                             D, DAG.getConstant(0, SL, MVT::i64));
  SDValue K = DAG.getNode(ISD::ADD, SL, MVT::i64, L, Tmp2);

  K = DAG.getNode(ISD::AND, SL, MVT::i64, K, DAG.getNOT(SL, M, MVT::i64));
  K = DAG.getNode(ISD::BITCAST, SL, MVT::f64, K);

  SDValue ExpLt0 = DAG.getSetCC(SL, SetCCVT, Exp, Zero, ISD::SETLT);
  SDValue ExpGt51 = DAG.getSetCC(SL, SetCCVT, Exp, FiftyOne, ISD::SETGT);
  SDValue ExpEqNegOne = DAG.getSetCC(SL, SetCCVT, NegOne, Exp, ISD::SETEQ);

  SDValue Mag = DAG.getNode(ISD::SELECT, SL, MVT::f64,
                            ExpEqNegOne,
                            DAG.getConstantFP(1.0, SL, MVT::f64),
                            DAG.getConstantFP(0.0, SL, MVT::f64));

  SDValue S = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f64, Mag, X);

  K = DAG.getNode(ISD::SELECT, SL, MVT::f64, ExpLt0, S, K);
  K = DAG.getNode(ISD::SELECT, SL, MVT::f64, ExpGt51, X, K);

  return K;
}

SDValue AMDGPUTargetLowering::LowerFROUND(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32)
    return LowerFROUND32(Op, DAG);

  if (VT == MVT::f64)
    return LowerFROUND64(Op, DAG);

  llvm_unreachable("unhandled type");
}

SDValue AMDGPUTargetLowering::LowerFFLOOR(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  // result = trunc(src);
  // if (src < 0.0 && src != result)
  //   result += -1.0.

  SDValue Trunc = DAG.getNode(ISD::FTRUNC, SL, MVT::f64, Src);

  const SDValue Zero = DAG.getConstantFP(0.0, SL, MVT::f64);
  const SDValue NegOne = DAG.getConstantFP(-1.0, SL, MVT::f64);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f64);

  SDValue Lt0 = DAG.getSetCC(SL, SetCCVT, Src, Zero, ISD::SETOLT);
  SDValue NeTrunc = DAG.getSetCC(SL, SetCCVT, Src, Trunc, ISD::SETONE);
  SDValue And = DAG.getNode(ISD::AND, SL, SetCCVT, Lt0, NeTrunc);

  SDValue Add = DAG.getNode(ISD::SELECT, SL, MVT::f64, And, NegOne, Zero);
  // TODO: Should this propagate fast-math-flags?
  return DAG.getNode(ISD::FADD, SL, MVT::f64, Trunc, Add);
}

SDValue AMDGPUTargetLowering::LowerCTLZ(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  bool ZeroUndef = Op.getOpcode() == ISD::CTLZ_ZERO_UNDEF;

  if (ZeroUndef && Src.getValueType() == MVT::i32)
    return DAG.getNode(AMDGPUISD::FFBH_U32, SL, MVT::i32, Src);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Src);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, Zero);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, One);

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(),
                                   *DAG.getContext(), MVT::i32);

  SDValue Hi0 = DAG.getSetCC(SL, SetCCVT, Hi, Zero, ISD::SETEQ);

  SDValue CtlzLo = DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SL, MVT::i32, Lo);
  SDValue CtlzHi = DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SL, MVT::i32, Hi);

  const SDValue Bits32 = DAG.getConstant(32, SL, MVT::i32);
  SDValue Add = DAG.getNode(ISD::ADD, SL, MVT::i32, CtlzLo, Bits32);

  // ctlz(x) = hi_32(x) == 0 ? ctlz(lo_32(x)) + 32 : ctlz(hi_32(x))
  SDValue NewCtlz = DAG.getNode(ISD::SELECT, SL, MVT::i32, Hi0, Add, CtlzHi);

  if (!ZeroUndef) {
    // Test if the full 64-bit input is zero.

    // FIXME: DAG combines turn what should be an s_and_b64 into a v_or_b32,
    // which we probably don't want.
    SDValue Lo0 = DAG.getSetCC(SL, SetCCVT, Lo, Zero, ISD::SETEQ);
    SDValue SrcIsZero = DAG.getNode(ISD::AND, SL, SetCCVT, Lo0, Hi0);

    // TODO: If i64 setcc is half rate, it can result in 1 fewer instruction
    // with the same cycles, otherwise it is slower.
    // SDValue SrcIsZero = DAG.getSetCC(SL, SetCCVT, Src,
    // DAG.getConstant(0, SL, MVT::i64), ISD::SETEQ);

    const SDValue Bits32 = DAG.getConstant(64, SL, MVT::i32);

    // The instruction returns -1 for 0 input, but the defined intrinsic
    // behavior is to return the number of bits.
    NewCtlz = DAG.getNode(ISD::SELECT, SL, MVT::i32,
                          SrcIsZero, Bits32, NewCtlz);
  }

  return DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i64, NewCtlz);
}

SDValue AMDGPUTargetLowering::LowerINT_TO_FP32(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  // Unsigned
  // cul2f(ulong u)
  //{
  //  uint lz = clz(u);
  //  uint e = (u != 0) ? 127U + 63U - lz : 0;
  //  u = (u << lz) & 0x7fffffffffffffffUL;
  //  ulong t = u & 0xffffffffffUL;
  //  uint v = (e << 23) | (uint)(u >> 40);
  //  uint r = t > 0x8000000000UL ? 1U : (t == 0x8000000000UL ? v & 1U : 0U);
  //  return as_float(v + r);
  //}
  // Signed
  // cl2f(long l)
  //{
  //  long s = l >> 63;
  //  float r = cul2f((l + s) ^ s);
  //  return s ? -r : r;
  //}

  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  SDValue L = Src;

  SDValue S;
  if (Signed) {
    const SDValue SignBit = DAG.getConstant(63, SL, MVT::i64);
    S = DAG.getNode(ISD::SRA, SL, MVT::i64, L, SignBit);

    SDValue LPlusS = DAG.getNode(ISD::ADD, SL, MVT::i64, L, S);
    L = DAG.getNode(ISD::XOR, SL, MVT::i64, LPlusS, S);
  }

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(),
                                   *DAG.getContext(), MVT::f32);


  SDValue ZeroI32 = DAG.getConstant(0, SL, MVT::i32);
  SDValue ZeroI64 = DAG.getConstant(0, SL, MVT::i64);
  SDValue LZ = DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SL, MVT::i64, L);
  LZ = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, LZ);

  SDValue K = DAG.getConstant(127U + 63U, SL, MVT::i32);
  SDValue E = DAG.getSelect(SL, MVT::i32,
    DAG.getSetCC(SL, SetCCVT, L, ZeroI64, ISD::SETNE),
    DAG.getNode(ISD::SUB, SL, MVT::i32, K, LZ),
    ZeroI32);

  SDValue U = DAG.getNode(ISD::AND, SL, MVT::i64,
    DAG.getNode(ISD::SHL, SL, MVT::i64, L, LZ),
    DAG.getConstant((-1ULL) >> 1, SL, MVT::i64));

  SDValue T = DAG.getNode(ISD::AND, SL, MVT::i64, U,
                          DAG.getConstant(0xffffffffffULL, SL, MVT::i64));

  SDValue UShl = DAG.getNode(ISD::SRL, SL, MVT::i64,
                             U, DAG.getConstant(40, SL, MVT::i64));

  SDValue V = DAG.getNode(ISD::OR, SL, MVT::i32,
    DAG.getNode(ISD::SHL, SL, MVT::i32, E, DAG.getConstant(23, SL, MVT::i32)),
    DAG.getNode(ISD::TRUNCATE, SL, MVT::i32,  UShl));

  SDValue C = DAG.getConstant(0x8000000000ULL, SL, MVT::i64);
  SDValue RCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETUGT);
  SDValue TCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETEQ);

  SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue VTrunc1 = DAG.getNode(ISD::AND, SL, MVT::i32, V, One);

  SDValue R = DAG.getSelect(SL, MVT::i32,
    RCmp,
    One,
    DAG.getSelect(SL, MVT::i32, TCmp, VTrunc1, ZeroI32));
  R = DAG.getNode(ISD::ADD, SL, MVT::i32, V, R);
  R = DAG.getNode(ISD::BITCAST, SL, MVT::f32, R);

  if (!Signed)
    return R;

  SDValue RNeg = DAG.getNode(ISD::FNEG, SL, MVT::f32, R);
  return DAG.getSelect(SL, MVT::f32, DAG.getSExtOrTrunc(S, SL, SetCCVT), RNeg, R);
}

SDValue AMDGPUTargetLowering::LowerINT_TO_FP64(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Src);

  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BC,
                           DAG.getConstant(0, SL, MVT::i32));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BC,
                           DAG.getConstant(1, SL, MVT::i32));

  SDValue CvtHi = DAG.getNode(Signed ? ISD::SINT_TO_FP : ISD::UINT_TO_FP,
                              SL, MVT::f64, Hi);

  SDValue CvtLo = DAG.getNode(ISD::UINT_TO_FP, SL, MVT::f64, Lo);

  SDValue LdExp = DAG.getNode(AMDGPUISD::LDEXP, SL, MVT::f64, CvtHi,
                              DAG.getConstant(32, SL, MVT::i32));
  // TODO: Should this propagate fast-math-flags?
  return DAG.getNode(ISD::FADD, SL, MVT::f64, LdExp, CvtLo);
}

SDValue AMDGPUTargetLowering::LowerUINT_TO_FP(SDValue Op,
                                               SelectionDAG &DAG) const {
  assert(Op.getOperand(0).getValueType() == MVT::i64 &&
         "operation should be legal");

  EVT DestVT = Op.getValueType();

  if (DestVT == MVT::f32)
    return LowerINT_TO_FP32(Op, DAG, false);

  assert(DestVT == MVT::f64);
  return LowerINT_TO_FP64(Op, DAG, false);
}

SDValue AMDGPUTargetLowering::LowerSINT_TO_FP(SDValue Op,
                                              SelectionDAG &DAG) const {
  assert(Op.getOperand(0).getValueType() == MVT::i64 &&
         "operation should be legal");

  EVT DestVT = Op.getValueType();
  if (DestVT == MVT::f32)
    return LowerINT_TO_FP32(Op, DAG, true);

  assert(DestVT == MVT::f64);
  return LowerINT_TO_FP64(Op, DAG, true);
}

SDValue AMDGPUTargetLowering::LowerFP64_TO_INT(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  SDLoc SL(Op);

  SDValue Src = Op.getOperand(0);

  SDValue Trunc = DAG.getNode(ISD::FTRUNC, SL, MVT::f64, Src);

  SDValue K0 = DAG.getConstantFP(BitsToDouble(UINT64_C(0x3df0000000000000)), SL,
                                 MVT::f64);
  SDValue K1 = DAG.getConstantFP(BitsToDouble(UINT64_C(0xc1f0000000000000)), SL,
                                 MVT::f64);
  // TODO: Should this propagate fast-math-flags?
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f64, Trunc, K0);

  SDValue FloorMul = DAG.getNode(ISD::FFLOOR, SL, MVT::f64, Mul);


  SDValue Fma = DAG.getNode(ISD::FMA, SL, MVT::f64, FloorMul, K1, Trunc);

  SDValue Hi = DAG.getNode(Signed ? ISD::FP_TO_SINT : ISD::FP_TO_UINT, SL,
                           MVT::i32, FloorMul);
  SDValue Lo = DAG.getNode(ISD::FP_TO_UINT, SL, MVT::i32, Fma);

  SDValue Result = DAG.getBuildVector(MVT::v2i32, SL, {Lo, Hi});

  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Result);
}

SDValue AMDGPUTargetLowering::LowerFP_TO_SINT(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDValue Src = Op.getOperand(0);

  if (Op.getValueType() == MVT::i64 && Src.getValueType() == MVT::f64)
    return LowerFP64_TO_INT(Op, DAG, true);

  return SDValue();
}

SDValue AMDGPUTargetLowering::LowerFP_TO_UINT(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDValue Src = Op.getOperand(0);

  if (Op.getValueType() == MVT::i64 && Src.getValueType() == MVT::f64)
    return LowerFP64_TO_INT(Op, DAG, false);

  return SDValue();
}

SDValue AMDGPUTargetLowering::LowerSIGN_EXTEND_INREG(SDValue Op,
                                                     SelectionDAG &DAG) const {
  EVT ExtraVT = cast<VTSDNode>(Op.getOperand(1))->getVT();
  MVT VT = Op.getSimpleValueType();
  MVT ScalarVT = VT.getScalarType();

  assert(VT.isVector());

  SDValue Src = Op.getOperand(0);
  SDLoc DL(Op);

  // TODO: Don't scalarize on Evergreen?
  unsigned NElts = VT.getVectorNumElements();
  SmallVector<SDValue, 8> Args;
  DAG.ExtractVectorElements(Src, Args, 0, NElts);

  SDValue VTOp = DAG.getValueType(ExtraVT.getScalarType());
  for (unsigned I = 0; I < NElts; ++I)
    Args[I] = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, ScalarVT, Args[I], VTOp);

  return DAG.getBuildVector(VT, DL, Args);
}

//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//

static bool isU24(SDValue Op, SelectionDAG &DAG) {
  APInt KnownZero, KnownOne;
  EVT VT = Op.getValueType();
  DAG.computeKnownBits(Op, KnownZero, KnownOne);

  return (VT.getSizeInBits() - KnownZero.countLeadingOnes()) <= 24;
}

static bool isI24(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // In order for this to be a signed 24-bit value, bit 23, must
  // be a sign bit.
  return VT.getSizeInBits() >= 24 && // Types less than 24-bit should be treated
                                     // as unsigned 24-bit values.
         (VT.getSizeInBits() - DAG.ComputeNumSignBits(Op)) < 24;
}

static bool simplifyI24(SDValue Op, TargetLowering::DAGCombinerInfo &DCI) {

  SelectionDAG &DAG = DCI.DAG;
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT VT = Op.getValueType();

  APInt Demanded = APInt::getLowBitsSet(VT.getSizeInBits(), 24);
  APInt KnownZero, KnownOne;
  TargetLowering::TargetLoweringOpt TLO(DAG, true, true);
  if (TLI.SimplifyDemandedBits(Op, Demanded, KnownZero, KnownOne, TLO)) {
    DCI.CommitTargetLoweringOpt(TLO);
    return true;
  }

  return false;
}

template <typename IntTy>
static SDValue constantFoldBFE(SelectionDAG &DAG, IntTy Src0, uint32_t Offset,
                               uint32_t Width, const SDLoc &DL) {
  if (Width + Offset < 32) {
    uint32_t Shl = static_cast<uint32_t>(Src0) << (32 - Offset - Width);
    IntTy Result = static_cast<IntTy>(Shl) >> (32 - Width);
    return DAG.getConstant(Result, DL, MVT::i32);
  }

  return DAG.getConstant(Src0 >> Offset, DL, MVT::i32);
}

static bool hasVolatileUser(SDNode *Val) {
  for (SDNode *U : Val->uses()) {
    if (MemSDNode *M = dyn_cast<MemSDNode>(U)) {
      if (M->isVolatile())
        return true;
    }
  }

  return false;
}

bool AMDGPUTargetLowering::shouldCombineMemoryType(EVT VT) const {
  // i32 vectors are the canonical memory type.
  if (VT.getScalarType() == MVT::i32 || isTypeLegal(VT))
    return false;

  if (!VT.isByteSized())
    return false;

  unsigned Size = VT.getStoreSize();

  if ((Size == 1 || Size == 2 || Size == 4) && !VT.isVector())
    return false;

  if (Size == 3 || (Size > 4 && (Size % 4 != 0)))
    return false;

  return true;
}

// Replace load of an illegal type with a store of a bitcast to a friendlier
// type.
SDValue AMDGPUTargetLowering::performLoadCombine(SDNode *N,
                                                 DAGCombinerInfo &DCI) const {
  if (!DCI.isBeforeLegalize())
    return SDValue();

  LoadSDNode *LN = cast<LoadSDNode>(N);
  if (LN->isVolatile() || !ISD::isNormalLoad(LN) || hasVolatileUser(LN))
    return SDValue();

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = LN->getMemoryVT();

  unsigned Size = VT.getStoreSize();
  unsigned Align = LN->getAlignment();
  if (Align < Size && isTypeLegal(VT)) {
    bool IsFast;
    unsigned AS = LN->getAddressSpace();

    // Expand unaligned loads earlier than legalization. Due to visitation order
    // problems during legalization, the emitted instructions to pack and unpack
    // the bytes again are not eliminated in the case of an unaligned copy.
    if (!allowsMisalignedMemoryAccesses(VT, AS, Align, &IsFast)) {
      SDValue Ops[2];
      std::tie(Ops[0], Ops[1]) = expandUnalignedLoad(LN, DAG);
      return DAG.getMergeValues(Ops, SDLoc(N));
    }

    if (!IsFast)
      return SDValue();
  }

  if (!shouldCombineMemoryType(VT))
    return SDValue();

  EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);

  SDValue NewLoad
    = DAG.getLoad(NewVT, SL, LN->getChain(),
                  LN->getBasePtr(), LN->getMemOperand());

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, VT, NewLoad);
  DCI.CombineTo(N, BC, NewLoad.getValue(1));
  return SDValue(N, 0);
}

// Replace store of an illegal type with a store of a bitcast to a friendlier
// type.
SDValue AMDGPUTargetLowering::performStoreCombine(SDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  if (!DCI.isBeforeLegalize())
    return SDValue();

  StoreSDNode *SN = cast<StoreSDNode>(N);
  if (SN->isVolatile() || !ISD::isNormalStore(SN))
    return SDValue();

  EVT VT = SN->getMemoryVT();
  unsigned Size = VT.getStoreSize();

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;
  unsigned Align = SN->getAlignment();
  if (Align < Size && isTypeLegal(VT)) {
    bool IsFast;
    unsigned AS = SN->getAddressSpace();

    // Expand unaligned stores earlier than legalization. Due to visitation
    // order problems during legalization, the emitted instructions to pack and
    // unpack the bytes again are not eliminated in the case of an unaligned
    // copy.
    if (!allowsMisalignedMemoryAccesses(VT, AS, Align, &IsFast))
      return expandUnalignedStore(SN, DAG);

    if (!IsFast)
      return SDValue();
  }

  if (!shouldCombineMemoryType(VT))
    return SDValue();

  EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);
  SDValue Val = SN->getValue();

  //DCI.AddToWorklist(Val.getNode());

  bool OtherUses = !Val.hasOneUse();
  SDValue CastVal = DAG.getNode(ISD::BITCAST, SL, NewVT, Val);
  if (OtherUses) {
    SDValue CastBack = DAG.getNode(ISD::BITCAST, SL, VT, CastVal);
    DAG.ReplaceAllUsesOfValueWith(Val, CastBack);
  }

  return DAG.getStore(SN->getChain(), SL, CastVal,
                      SN->getBasePtr(), SN->getMemOperand());
}

// TODO: Should repeat for other bit ops.
SDValue AMDGPUTargetLowering::performAndCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  if (N->getValueType(0) != MVT::i64)
    return SDValue();

  // Break up 64-bit and of a constant into two 32-bit ands. This will typically
  // happen anyway for a VALU 64-bit and. This exposes other 32-bit integer
  // combine opportunities since most 64-bit operations are decomposed this way.
  // TODO: We won't want this for SALU especially if it is an inline immediate.
  const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  uint64_t Val = RHS->getZExtValue();
  if (Lo_32(Val) != 0 && Hi_32(Val) != 0 && !RHS->hasOneUse()) {
    // If either half of the constant is 0, this is really a 32-bit and, so
    // split it. If we can re-use the full materialized constant, keep it.
    return SDValue();
  }

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;

  SDValue Lo, Hi;
  std::tie(Lo, Hi) = split64BitValue(N->getOperand(0), DAG);

  SDValue LoRHS = DAG.getConstant(Lo_32(Val), SL, MVT::i32);
  SDValue HiRHS = DAG.getConstant(Hi_32(Val), SL, MVT::i32);

  SDValue LoAnd = DAG.getNode(ISD::AND, SL, MVT::i32, Lo, LoRHS);
  SDValue HiAnd = DAG.getNode(ISD::AND, SL, MVT::i32, Hi, HiRHS);

  // Re-visit the ands. It's possible we eliminated one of them and it could
  // simplify the vector.
  DCI.AddToWorklist(Lo.getNode());
  DCI.AddToWorklist(Hi.getNode());

  SDValue Vec = DAG.getBuildVector(MVT::v2i32, SL, {LoAnd, HiAnd});
  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
}

SDValue AMDGPUTargetLowering::performShlCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  if (N->getValueType(0) != MVT::i64)
    return SDValue();

  // i64 (shl x, C) -> (build_pair 0, (shl x, C -32))

  // On some subtargets, 64-bit shift is a quarter rate instruction. In the
  // common case, splitting this into a move and a 32-bit shift is faster and
  // the same code size.
  const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  unsigned RHSVal = RHS->getZExtValue();
  if (RHSVal < 32)
    return SDValue();

  SDValue LHS = N->getOperand(0);

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;

  SDValue ShiftAmt = DAG.getConstant(RHSVal - 32, SL, MVT::i32);

  SDValue Lo = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, LHS);
  SDValue NewShift = DAG.getNode(ISD::SHL, SL, MVT::i32, Lo, ShiftAmt);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);

  SDValue Vec = DAG.getBuildVector(MVT::v2i32, SL, {Zero, NewShift});
  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
}

SDValue AMDGPUTargetLowering::performSraCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  if (N->getValueType(0) != MVT::i64)
    return SDValue();

  const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  unsigned RHSVal = RHS->getZExtValue();

  // (sra i64:x, 32) -> build_pair x, (sra hi_32(x), 31)
  if (RHSVal == 32) {
    SDValue Hi = getHiHalf64(N->getOperand(0), DAG);
    SDValue NewShift = DAG.getNode(ISD::SRA, SL, MVT::i32, Hi,
                                   DAG.getConstant(31, SL, MVT::i32));

    SDValue BuildVec = DAG.getBuildVector(MVT::v2i32, SL, {Hi, NewShift});
    return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildVec);
  }

  // (sra i64:x, 63) -> build_pair (sra hi_32(x), 31), (sra hi_32(x), 31)
  if (RHSVal == 63) {
    SDValue Hi = getHiHalf64(N->getOperand(0), DAG);
    SDValue NewShift = DAG.getNode(ISD::SRA, SL, MVT::i32, Hi,
                                   DAG.getConstant(31, SL, MVT::i32));
    SDValue BuildVec = DAG.getBuildVector(MVT::v2i32, SL, {NewShift, NewShift});
    return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildVec);
  }

  return SDValue();
}

SDValue AMDGPUTargetLowering::performSrlCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  if (N->getValueType(0) != MVT::i64)
    return SDValue();

  const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  unsigned ShiftAmt = RHS->getZExtValue();
  if (ShiftAmt < 32)
    return SDValue();

  // srl i64:x, C for C >= 32
  // =>
  //   build_pair (srl hi_32(x), C - 32), 0

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue One = DAG.getConstant(1, SL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, SL, MVT::i32);

  SDValue VecOp = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, N->getOperand(0));
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32,
                           VecOp, One);

  SDValue NewConst = DAG.getConstant(ShiftAmt - 32, SL, MVT::i32);
  SDValue NewShift = DAG.getNode(ISD::SRL, SL, MVT::i32, Hi, NewConst);

  SDValue BuildPair = DAG.getBuildVector(MVT::v2i32, SL, {NewShift, Zero});

  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildPair);
}

// We need to specifically handle i64 mul here to avoid unnecessary conversion
// instructions. If we only match on the legalized i64 mul expansion,
// SimplifyDemandedBits will be unable to remove them because there will be
// multiple uses due to the separate mul + mulh[su].
static SDValue getMul24(SelectionDAG &DAG, const SDLoc &SL,
                        SDValue N0, SDValue N1, unsigned Size, bool Signed) {
  if (Size <= 32) {
    unsigned MulOpc = Signed ? AMDGPUISD::MUL_I24 : AMDGPUISD::MUL_U24;
    return DAG.getNode(MulOpc, SL, MVT::i32, N0, N1);
  }

  // Because we want to eliminate extension instructions before the
  // operation, we need to create a single user here (i.e. not the separate
  // mul_lo + mul_hi) so that SimplifyDemandedBits will deal with it.

  unsigned MulOpc = Signed ? AMDGPUISD::MUL_LOHI_I24 : AMDGPUISD::MUL_LOHI_U24;

  SDValue Mul = DAG.getNode(MulOpc, SL,
                            DAG.getVTList(MVT::i32, MVT::i32), N0, N1);

  return DAG.getNode(ISD::BUILD_PAIR, SL, MVT::i64,
                     Mul.getValue(0), Mul.getValue(1));
}

SDValue AMDGPUTargetLowering::performMulCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  unsigned Size = VT.getSizeInBits();
  if (VT.isVector() || Size > 64)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue Mul;

  if (Subtarget->hasMulU24() && isU24(N0, DAG) && isU24(N1, DAG)) {
    N0 = DAG.getZExtOrTrunc(N0, DL, MVT::i32);
    N1 = DAG.getZExtOrTrunc(N1, DL, MVT::i32);
    Mul = getMul24(DAG, DL, N0, N1, Size, false);
  } else if (Subtarget->hasMulI24() && isI24(N0, DAG) && isI24(N1, DAG)) {
    N0 = DAG.getSExtOrTrunc(N0, DL, MVT::i32);
    N1 = DAG.getSExtOrTrunc(N1, DL, MVT::i32);
    Mul = getMul24(DAG, DL, N0, N1, Size, true);
  } else {
    return SDValue();
  }

  // We need to use sext even for MUL_U24, because MUL_U24 is used
  // for signed multiply of 8 and 16-bit types.
  return DAG.getSExtOrTrunc(Mul, DL, VT);
}

SDValue AMDGPUTargetLowering::performMulhsCombine(SDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  if (!Subtarget->hasMulI24() || VT.isVector())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  if (!isI24(N0, DAG) || !isI24(N1, DAG))
    return SDValue();

  N0 = DAG.getSExtOrTrunc(N0, DL, MVT::i32);
  N1 = DAG.getSExtOrTrunc(N1, DL, MVT::i32);

  SDValue Mulhi = DAG.getNode(AMDGPUISD::MULHI_I24, DL, MVT::i32, N0, N1);
  DCI.AddToWorklist(Mulhi.getNode());
  return DAG.getSExtOrTrunc(Mulhi, DL, VT);
}

SDValue AMDGPUTargetLowering::performMulhuCombine(SDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  if (!Subtarget->hasMulU24() || VT.isVector() || VT.getSizeInBits() > 32)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  if (!isU24(N0, DAG) || !isU24(N1, DAG))
    return SDValue();

  N0 = DAG.getZExtOrTrunc(N0, DL, MVT::i32);
  N1 = DAG.getZExtOrTrunc(N1, DL, MVT::i32);

  SDValue Mulhi = DAG.getNode(AMDGPUISD::MULHI_U24, DL, MVT::i32, N0, N1);
  DCI.AddToWorklist(Mulhi.getNode());
  return DAG.getZExtOrTrunc(Mulhi, DL, VT);
}

SDValue AMDGPUTargetLowering::performMulLoHi24Combine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // Simplify demanded bits before splitting into multiple users.
  if (simplifyI24(N0, DCI) || simplifyI24(N1, DCI))
    return SDValue();

  bool Signed = (N->getOpcode() == AMDGPUISD::MUL_LOHI_I24);

  unsigned MulLoOpc = Signed ? AMDGPUISD::MUL_I24 : AMDGPUISD::MUL_U24;
  unsigned MulHiOpc = Signed ? AMDGPUISD::MULHI_I24 : AMDGPUISD::MULHI_U24;

  SDLoc SL(N);

  SDValue MulLo = DAG.getNode(MulLoOpc, SL, MVT::i32, N0, N1);
  SDValue MulHi = DAG.getNode(MulHiOpc, SL, MVT::i32, N0, N1);
  return DAG.getMergeValues({ MulLo, MulHi }, SL);
}

static bool isNegativeOne(SDValue Val) {
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val))
    return C->isAllOnesValue();
  return false;
}

static bool isCtlzOpc(unsigned Opc) {
  return Opc == ISD::CTLZ || Opc == ISD::CTLZ_ZERO_UNDEF;
}

// Get FFBH node if the incoming op may have been type legalized from a smaller
// type VT.
// Need to match pre-legalized type because the generic legalization inserts the
// add/sub between the select and compare.
static SDValue getFFBH_U32(const TargetLowering &TLI, SelectionDAG &DAG,
                           const SDLoc &SL, SDValue Op) {
  EVT VT = Op.getValueType();
  EVT LegalVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
  if (LegalVT != MVT::i32)
    return SDValue();

  if (VT != MVT::i32)
    Op = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Op);

  SDValue FFBH = DAG.getNode(AMDGPUISD::FFBH_U32, SL, MVT::i32, Op);
  if (VT != MVT::i32)
    FFBH = DAG.getNode(ISD::TRUNCATE, SL, VT, FFBH);

  return FFBH;
}

// The native instructions return -1 on 0 input. Optimize out a select that
// produces -1 on 0.
//
// TODO: If zero is not undef, we could also do this if the output is compared
// against the bitwidth.
//
// TODO: Should probably combine against FFBH_U32 instead of ctlz directly.
SDValue AMDGPUTargetLowering::performCtlzCombine(const SDLoc &SL, SDValue Cond,
                                                 SDValue LHS, SDValue RHS,
                                                 DAGCombinerInfo &DCI) const {
  ConstantSDNode *CmpRhs = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
  if (!CmpRhs || !CmpRhs->isNullValue())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  ISD::CondCode CCOpcode = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
  SDValue CmpLHS = Cond.getOperand(0);

  // select (setcc x, 0, eq), -1, (ctlz_zero_undef x) -> ffbh_u32 x
  if (CCOpcode == ISD::SETEQ &&
      isCtlzOpc(RHS.getOpcode()) &&
      RHS.getOperand(0) == CmpLHS &&
      isNegativeOne(LHS)) {
    return getFFBH_U32(*this, DAG, SL, CmpLHS);
  }

  // select (setcc x, 0, ne), (ctlz_zero_undef x), -1 -> ffbh_u32 x
  if (CCOpcode == ISD::SETNE &&
      isCtlzOpc(LHS.getOpcode()) &&
      LHS.getOperand(0) == CmpLHS &&
      isNegativeOne(RHS)) {
    return getFFBH_U32(*this, DAG, SL, CmpLHS);
  }

  return SDValue();
}

SDValue AMDGPUTargetLowering::performSelectCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  SDValue Cond = N->getOperand(0);
  if (Cond.getOpcode() != ISD::SETCC)
    return SDValue();

  EVT VT = N->getValueType(0);
  SDValue LHS = Cond.getOperand(0);
  SDValue RHS = Cond.getOperand(1);
  SDValue CC = Cond.getOperand(2);

  SDValue True = N->getOperand(1);
  SDValue False = N->getOperand(2);

  if (VT == MVT::f32 && Cond.hasOneUse()) {
    SDValue MinMax
      = CombineFMinMaxLegacy(SDLoc(N), VT, LHS, RHS, True, False, CC, DCI);
    // Revisit this node so we can catch min3/max3/med3 patterns.
    //DCI.AddToWorklist(MinMax.getNode());
    return MinMax;
  }

  // There's no reason to not do this if the condition has other uses.
  return performCtlzCombine(SDLoc(N), Cond, True, False, DCI);
}

SDValue AMDGPUTargetLowering::PerformDAGCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  switch(N->getOpcode()) {
  default:
    break;
  case ISD::BITCAST: {
    EVT DestVT = N->getValueType(0);
    if (DestVT.getSizeInBits() != 64 && !DestVT.isVector())
      break;

    // Fold bitcasts of constants.
    //
    // v2i32 (bitcast i64:k) -> build_vector lo_32(k), hi_32(k)
    // TODO: Generalize and move to DAGCombiner
    SDValue Src = N->getOperand(0);
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Src)) {
      assert(Src.getValueType() == MVT::i64);
      SDLoc SL(N);
      uint64_t CVal = C->getZExtValue();
      return DAG.getNode(ISD::BUILD_VECTOR, SL, DestVT,
                         DAG.getConstant(Lo_32(CVal), SL, MVT::i32),
                         DAG.getConstant(Hi_32(CVal), SL, MVT::i32));
    }

    if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Src)) {
      const APInt &Val = C->getValueAPF().bitcastToAPInt();
      SDLoc SL(N);
      uint64_t CVal = Val.getZExtValue();
      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32,
                                DAG.getConstant(Lo_32(CVal), SL, MVT::i32),
                                DAG.getConstant(Hi_32(CVal), SL, MVT::i32));

      return DAG.getNode(ISD::BITCAST, SL, DestVT, Vec);
    }

    break;
  }
  case ISD::SHL: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    return performShlCombine(N, DCI);
  }
  case ISD::SRL: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    return performSrlCombine(N, DCI);
  }
  case ISD::SRA: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    return performSraCombine(N, DCI);
  }
  case ISD::AND: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;

    return performAndCombine(N, DCI);
  }
  case ISD::MUL:
    return performMulCombine(N, DCI);
  case ISD::MULHS:
    return performMulhsCombine(N, DCI);
  case ISD::MULHU:
    return performMulhuCombine(N, DCI);
  case AMDGPUISD::MUL_I24:
  case AMDGPUISD::MUL_U24:
  case AMDGPUISD::MULHI_I24:
  case AMDGPUISD::MULHI_U24: {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    simplifyI24(N0, DCI);
    simplifyI24(N1, DCI);
    return SDValue();
  }
  case AMDGPUISD::MUL_LOHI_I24:
  case AMDGPUISD::MUL_LOHI_U24:
    return performMulLoHi24Combine(N, DCI);
  case ISD::SELECT:
    return performSelectCombine(N, DCI);
  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    assert(!N->getValueType(0).isVector() &&
           "Vector handling of BFE not implemented");
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(N->getOperand(2));
    if (!Width)
      break;

    uint32_t WidthVal = Width->getZExtValue() & 0x1f;
    if (WidthVal == 0)
      return DAG.getConstant(0, DL, MVT::i32);

    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!Offset)
      break;

    SDValue BitsFrom = N->getOperand(0);
    uint32_t OffsetVal = Offset->getZExtValue() & 0x1f;

    bool Signed = N->getOpcode() == AMDGPUISD::BFE_I32;

    if (OffsetVal == 0) {
      // This is already sign / zero extended, so try to fold away extra BFEs.
      unsigned SignBits =  Signed ? (32 - WidthVal + 1) : (32 - WidthVal);

      unsigned OpSignBits = DAG.ComputeNumSignBits(BitsFrom);
      if (OpSignBits >= SignBits)
        return BitsFrom;

      EVT SmallVT = EVT::getIntegerVT(*DAG.getContext(), WidthVal);
      if (Signed) {
        // This is a sign_extend_inreg. Replace it to take advantage of existing
        // DAG Combines. If not eliminated, we will match back to BFE during
        // selection.

        // TODO: The sext_inreg of extended types ends, although we can could
        // handle them in a single BFE.
        return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, BitsFrom,
                           DAG.getValueType(SmallVT));
      }

      return DAG.getZeroExtendInReg(BitsFrom, DL, SmallVT);
    }

    if (ConstantSDNode *CVal = dyn_cast<ConstantSDNode>(BitsFrom)) {
      if (Signed) {
        return constantFoldBFE<int32_t>(DAG,
                                        CVal->getSExtValue(),
                                        OffsetVal,
                                        WidthVal,
                                        DL);
      }

      return constantFoldBFE<uint32_t>(DAG,
                                       CVal->getZExtValue(),
                                       OffsetVal,
                                       WidthVal,
                                       DL);
    }

    if ((OffsetVal + WidthVal) >= 32) {
      SDValue ShiftVal = DAG.getConstant(OffsetVal, DL, MVT::i32);
      return DAG.getNode(Signed ? ISD::SRA : ISD::SRL, DL, MVT::i32,
                         BitsFrom, ShiftVal);
    }

    if (BitsFrom.hasOneUse()) {
      APInt Demanded = APInt::getBitsSet(32,
                                         OffsetVal,
                                         OffsetVal + WidthVal);

      APInt KnownZero, KnownOne;
      TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                            !DCI.isBeforeLegalizeOps());
      const TargetLowering &TLI = DAG.getTargetLoweringInfo();
      if (TLO.ShrinkDemandedConstant(BitsFrom, Demanded) ||
          TLI.SimplifyDemandedBits(BitsFrom, Demanded,
                                   KnownZero, KnownOne, TLO)) {
        DCI.CommitTargetLoweringOpt(TLO);
      }
    }

    break;
  }
  case ISD::LOAD:
    return performLoadCombine(N, DCI);
  case ISD::STORE:
    return performStoreCombine(N, DCI);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

void AMDGPUTargetLowering::getOriginalFunctionArgs(
                               SelectionDAG &DAG,
                               const Function *F,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SmallVectorImpl<ISD::InputArg> &OrigIns) const {

  for (unsigned i = 0, e = Ins.size(); i < e; ++i) {
    if (Ins[i].ArgVT == Ins[i].VT) {
      OrigIns.push_back(Ins[i]);
      continue;
    }

    EVT VT;
    if (Ins[i].ArgVT.isVector() && !Ins[i].VT.isVector()) {
      // Vector has been split into scalars.
      VT = Ins[i].ArgVT.getVectorElementType();
    } else if (Ins[i].VT.isVector() && Ins[i].ArgVT.isVector() &&
               Ins[i].ArgVT.getVectorElementType() !=
               Ins[i].VT.getVectorElementType()) {
      // Vector elements have been promoted
      VT = Ins[i].ArgVT;
    } else {
      // Vector has been spilt into smaller vectors.
      VT = Ins[i].VT;
    }

    ISD::InputArg Arg(Ins[i].Flags, VT, VT, Ins[i].Used,
                      Ins[i].OrigArgIndex, Ins[i].PartOffset);
    OrigIns.push_back(Arg);
  }
}

SDValue AMDGPUTargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                                  const TargetRegisterClass *RC,
                                                   unsigned Reg, EVT VT) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned VirtualRegister;
  if (!MRI.isLiveIn(Reg)) {
    VirtualRegister = MRI.createVirtualRegister(RC);
    MRI.addLiveIn(Reg, VirtualRegister);
  } else {
    VirtualRegister = MRI.getLiveInVirtReg(Reg);
  }
  return DAG.getRegister(VirtualRegister, VT);
}

uint32_t AMDGPUTargetLowering::getImplicitParameterOffset(
    const AMDGPUMachineFunction *MFI, const ImplicitParameter Param) const {
  uint64_t ArgOffset = alignTo(MFI->getABIArgOffset(), 4);
  switch (Param) {
  case GRID_DIM:
    return ArgOffset;
  case GRID_OFFSET:
    return ArgOffset + 4;
  }
  llvm_unreachable("unexpected implicit parameter type");
}

#define NODE_NAME_CASE(node) case AMDGPUISD::node: return #node;

const char* AMDGPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((AMDGPUISD::NodeType)Opcode) {
  case AMDGPUISD::FIRST_NUMBER: break;
  // AMDIL DAG nodes
  NODE_NAME_CASE(CALL);
  NODE_NAME_CASE(UMUL);
  NODE_NAME_CASE(BRANCH_COND);

  // AMDGPU DAG nodes
  NODE_NAME_CASE(ENDPGM)
  NODE_NAME_CASE(RETURN)
  NODE_NAME_CASE(DWORDADDR)
  NODE_NAME_CASE(FRACT)
  NODE_NAME_CASE(SETCC)
  NODE_NAME_CASE(CLAMP)
  NODE_NAME_CASE(COS_HW)
  NODE_NAME_CASE(SIN_HW)
  NODE_NAME_CASE(FMAX_LEGACY)
  NODE_NAME_CASE(FMIN_LEGACY)
  NODE_NAME_CASE(FMAX3)
  NODE_NAME_CASE(SMAX3)
  NODE_NAME_CASE(UMAX3)
  NODE_NAME_CASE(FMIN3)
  NODE_NAME_CASE(SMIN3)
  NODE_NAME_CASE(UMIN3)
  NODE_NAME_CASE(FMED3)
  NODE_NAME_CASE(SMED3)
  NODE_NAME_CASE(UMED3)
  NODE_NAME_CASE(URECIP)
  NODE_NAME_CASE(DIV_SCALE)
  NODE_NAME_CASE(DIV_FMAS)
  NODE_NAME_CASE(DIV_FIXUP)
  NODE_NAME_CASE(TRIG_PREOP)
  NODE_NAME_CASE(RCP)
  NODE_NAME_CASE(RSQ)
  NODE_NAME_CASE(RCP_LEGACY)
  NODE_NAME_CASE(RSQ_LEGACY)
  NODE_NAME_CASE(FMUL_LEGACY)
  NODE_NAME_CASE(RSQ_CLAMP)
  NODE_NAME_CASE(LDEXP)
  NODE_NAME_CASE(FP_CLASS)
  NODE_NAME_CASE(DOT4)
  NODE_NAME_CASE(CARRY)
  NODE_NAME_CASE(BORROW)
  NODE_NAME_CASE(BFE_U32)
  NODE_NAME_CASE(BFE_I32)
  NODE_NAME_CASE(BFI)
  NODE_NAME_CASE(BFM)
  NODE_NAME_CASE(FFBH_U32)
  NODE_NAME_CASE(FFBH_I32)
  NODE_NAME_CASE(MUL_U24)
  NODE_NAME_CASE(MUL_I24)
  NODE_NAME_CASE(MULHI_U24)
  NODE_NAME_CASE(MULHI_I24)
  NODE_NAME_CASE(MUL_LOHI_U24)
  NODE_NAME_CASE(MUL_LOHI_I24)
  NODE_NAME_CASE(MAD_U24)
  NODE_NAME_CASE(MAD_I24)
  NODE_NAME_CASE(TEXTURE_FETCH)
  NODE_NAME_CASE(EXPORT)
  NODE_NAME_CASE(CONST_ADDRESS)
  NODE_NAME_CASE(REGISTER_LOAD)
  NODE_NAME_CASE(REGISTER_STORE)
  NODE_NAME_CASE(LOAD_INPUT)
  NODE_NAME_CASE(SAMPLE)
  NODE_NAME_CASE(SAMPLEB)
  NODE_NAME_CASE(SAMPLED)
  NODE_NAME_CASE(SAMPLEL)
  NODE_NAME_CASE(CVT_F32_UBYTE0)
  NODE_NAME_CASE(CVT_F32_UBYTE1)
  NODE_NAME_CASE(CVT_F32_UBYTE2)
  NODE_NAME_CASE(CVT_F32_UBYTE3)
  NODE_NAME_CASE(BUILD_VERTICAL_VECTOR)
  NODE_NAME_CASE(CONST_DATA_PTR)
  NODE_NAME_CASE(PC_ADD_REL_OFFSET)
  NODE_NAME_CASE(KILL)
  case AMDGPUISD::FIRST_MEM_OPCODE_NUMBER: break;
  NODE_NAME_CASE(SENDMSG)
  NODE_NAME_CASE(INTERP_MOV)
  NODE_NAME_CASE(INTERP_P1)
  NODE_NAME_CASE(INTERP_P2)
  NODE_NAME_CASE(STORE_MSKOR)
  NODE_NAME_CASE(LOAD_CONSTANT)
  NODE_NAME_CASE(TBUFFER_STORE_FORMAT)
  NODE_NAME_CASE(ATOMIC_CMP_SWAP)
  NODE_NAME_CASE(ATOMIC_INC)
  NODE_NAME_CASE(ATOMIC_DEC)
  case AMDGPUISD::LAST_AMDGPU_ISD_NUMBER: break;
  }
  return nullptr;
}

SDValue AMDGPUTargetLowering::getRsqrtEstimate(SDValue Operand,
                                               DAGCombinerInfo &DCI,
                                               unsigned &RefinementSteps,
                                               bool &UseOneConstNR) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = Operand.getValueType();

  if (VT == MVT::f32) {
    RefinementSteps = 0;
    return DAG.getNode(AMDGPUISD::RSQ, SDLoc(Operand), VT, Operand);
  }

  // TODO: There is also f64 rsq instruction, but the documentation is less
  // clear on its precision.

  return SDValue();
}

SDValue AMDGPUTargetLowering::getRecipEstimate(SDValue Operand,
                                               DAGCombinerInfo &DCI,
                                               unsigned &RefinementSteps) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = Operand.getValueType();

  if (VT == MVT::f32) {
    // Reciprocal, < 1 ulp error.
    //
    // This reciprocal approximation converges to < 0.5 ulp error with one
    // newton rhapson performed with two fused multiple adds (FMAs).

    RefinementSteps = 0;
    return DAG.getNode(AMDGPUISD::RCP, SDLoc(Operand), VT, Operand);
  }

  // TODO: There is also f64 rcp instruction, but the documentation is less
  // clear on its precision.

  return SDValue();
}

void AMDGPUTargetLowering::computeKnownBitsForTargetNode(
  const SDValue Op,
  APInt &KnownZero,
  APInt &KnownOne,
  const SelectionDAG &DAG,
  unsigned Depth) const {

  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0); // Don't know anything.

  APInt KnownZero2;
  APInt KnownOne2;
  unsigned Opc = Op.getOpcode();

  switch (Opc) {
  default:
    break;
  case AMDGPUISD::CARRY:
  case AMDGPUISD::BORROW: {
    KnownZero = APInt::getHighBitsSet(32, 31);
    break;
  }

  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    ConstantSDNode *CWidth = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!CWidth)
      return;

    unsigned BitWidth = 32;
    uint32_t Width = CWidth->getZExtValue() & 0x1f;

    if (Opc == AMDGPUISD::BFE_U32)
      KnownZero = APInt::getHighBitsSet(BitWidth, BitWidth - Width);

    break;
  }
  }
}

unsigned AMDGPUTargetLowering::ComputeNumSignBitsForTargetNode(
  SDValue Op,
  const SelectionDAG &DAG,
  unsigned Depth) const {
  switch (Op.getOpcode()) {
  case AMDGPUISD::BFE_I32: {
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!Width)
      return 1;

    unsigned SignBits = 32 - Width->getZExtValue() + 1;
    if (!isNullConstant(Op.getOperand(1)))
      return SignBits;

    // TODO: Could probably figure something out with non-0 offsets.
    unsigned Op0SignBits = DAG.ComputeNumSignBits(Op.getOperand(0), Depth + 1);
    return std::max(SignBits, Op0SignBits);
  }

  case AMDGPUISD::BFE_U32: {
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    return Width ? 32 - (Width->getZExtValue() & 0x1f) : 1;
  }

  case AMDGPUISD::CARRY:
  case AMDGPUISD::BORROW:
    return 31;

  default:
    return 1;
  }
}
