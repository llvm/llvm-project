//===- NVVMTcgenOps.cpp - NVVM dialect tcgen05 ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Tcgen` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMTcgenOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <cassert>
#include <cmath>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Helper/Utility methods
//===----------------------------------------------------------------------===//

// Helper method to convert CtaGroupKind in NVVM Dialect to CtaGroupKind in LLVM
static llvm::nvvm::CTAGroupKind
getNVVMCtaGroupKind(NVVM::CTAGroupKind ctaGroup) {
  switch (ctaGroup) {
  case NVVM::CTAGroupKind::CTA_1:
    return llvm::nvvm::CTAGroupKind::CG_1;
  case NVVM::CTAGroupKind::CTA_2:
    return llvm::nvvm::CTAGroupKind::CG_2;
  }
  llvm_unreachable("unsupported cta_group value");
}

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

LogicalResult NVVM::Tcgen05CpOp::verify() {
  auto mc = getMulticast();

  using SH = Tcgen05CpShape;
  using MC = Tcgen05CpMulticast;
  switch (getShape()) {
  case SH::SHAPE_128x256b:
  case SH::SHAPE_128x128b:
  case SH::SHAPE_4x256b:
    if (mc != MC::NONE)
      return emitError("Invalid multicast type for tcgen05.cp Op");
    break;
  case SH::SHAPE_64x128b:
    if (mc != MC::WARPX2_01_23 && mc != MC::WARPX2_02_13)
      return emitError("Shape 64x128b requires multicast warpx2_01_23 or "
                       "warpx2_02_13 for tcgen05.cp Op");
    break;
  case SH::SHAPE_32x128b:
    if (mc != MC::WARPX4)
      return emitError(
          "Shape 32x128b requires multicast warpx4 for tcgen05.cp Op");
    break;
  }
  return success();
}

/// Packs the given `field` into the `result`.
/// The `result` is 64-bits and each `field` can be 32-bits or narrower.
static llvm::Value *
packValInto64Bits(llvm::IRBuilderBase &builder,
                  llvm::Value *result, // the `result` (unset bits are zero)
                  llvm::Value *field,  // `field` to pack into `result`
                  unsigned sizeInBits, // Size of `field` in bits
                  unsigned start) {    // Starting bit within `result`
  field = builder.CreateZExtOrBitCast(field, builder.getInt32Ty());

  unsigned mask = (sizeInBits < 32 ? ((1u << sizeInBits) - 1) : 0xffffffffu);
  if (mask != 0xffffffffu)
    field = builder.CreateAnd(field, builder.getInt32(mask));

  field = builder.CreateZExtOrBitCast(field, builder.getInt64Ty());
  field = builder.CreateShl(field, start);

  return builder.CreateOr(result, field);
}

void Tcgen05MmaSmemDescOp::createSmemDescriptor(Operation &op,
                                                LLVM::ModuleTranslation &mt,
                                                llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::Tcgen05MmaSmemDescOp>(op);
  llvm::Value *smemDesc = builder.getInt64(0);

  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getStartAddr()), 14, 0);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getLeadingDimOffset()), 14, 16);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getStrideDimOffset()), 14, 32);

  smemDesc = packValInto64Bits(builder, smemDesc, builder.getInt32(1), 3, 46);
  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getBaseOffset()), 3, 49);
  smemDesc = packValInto64Bits(
      builder, smemDesc, mt.lookupValue(thisOp.getLeadingDimMode()), 1, 52);
  smemDesc = packValInto64Bits(builder, smemDesc,
                               mt.lookupValue(thisOp.getSwizzleMode()), 3, 61);

  mt.mapValue(thisOp.getRes()) = smemDesc;
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05AllocOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::Tcgen05AllocOp>(op);
  bool is2CTAMode = curOp.getGroup() == CTAGroupKind::CTA_2;

  llvm::Intrinsic::ID id = is2CTAMode ? llvm::Intrinsic::nvvm_tcgen05_alloc_cg2
                                      : llvm::Intrinsic::nvvm_tcgen05_alloc_cg1;

  return {id,
          {mt.lookupValue(curOp.getAddr()), mt.lookupValue(curOp.getNCols()),
           builder.getInt1(curOp.getIsExclusive())}};
}

mlir::NVVM::IDArgPair Tcgen05DeallocOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::Tcgen05DeallocOp>(op);
  llvm::Intrinsic::ID id = (curOp.getGroup() == CTAGroupKind::CTA_1)
                               ? llvm::Intrinsic::nvvm_tcgen05_dealloc_cg1
                               : llvm::Intrinsic::nvvm_tcgen05_dealloc_cg2;

  return {id,
          {mt.lookupValue(curOp.getTaddr()), mt.lookupValue(curOp.getNCols()),
           builder.getInt1(curOp.getIsExclusive())}};
}

llvm::Intrinsic::ID
Tcgen05CommitOp::getIntrinsicIDAndArgs(Operation &op,
                                       LLVM::ModuleTranslation &mt,
                                       llvm::SmallVector<llvm::Value *> &args) {
  auto curOp = cast<NVVM::Tcgen05CommitOp>(op);
  bool hasMulticast = static_cast<bool>(curOp.getMulticastMask());
  bool is2CTAMode = curOp.getGroup() == CTAGroupKind::CTA_2;
  bool hasSmemARead = curOp.getSmemARead();
  unsigned index = (static_cast<unsigned>(hasSmemARead) << 1) |
                   static_cast<unsigned>(is2CTAMode);

  using namespace llvm::Intrinsic;
  static constexpr ID IDs[] = {
      nvvm_tcgen05_commit_cg1,
      nvvm_tcgen05_commit_cg2,
      nvvm_tcgen05_commit_smem_a_read_cg1,
      nvvm_tcgen05_commit_smem_a_read_cg2,
  };

  static constexpr ID multicastIDs[] = {
      nvvm_tcgen05_commit_mc_cg1,
      nvvm_tcgen05_commit_mc_cg2,
      nvvm_tcgen05_commit_smem_a_read_mc_cg1,
      nvvm_tcgen05_commit_smem_a_read_mc_cg2,
  };

  ID id = hasMulticast ? multicastIDs[index] : IDs[index];
  // Fill the Intrinsic Args
  args.push_back(mt.lookupValue(curOp.getAddr()));
  if (hasMulticast)
    args.push_back(mt.lookupValue(curOp.getMulticastMask()));

  return id;
}

#define TCGEN05_CP_IMPL(shape_mc, src_fmt, cg)                                 \
  llvm::Intrinsic::nvvm_tcgen05_cp##shape_mc##src_fmt##cg

#define TCGEN05_CP_2CTA(shape_mc, src_fmt, is_2cta)                            \
  is_2cta ? TCGEN05_CP_IMPL(shape_mc, src_fmt, _cg2)                           \
          : TCGEN05_CP_IMPL(shape_mc, src_fmt, _cg1)

#define GET_TCGEN05_CP_ID(shape_mc, src_fmt, is_2cta)                          \
  [&]() -> auto {                                                              \
    if ((src_fmt) == Tcgen05CpSrcFormat::B6x16_P32)                            \
      return TCGEN05_CP_2CTA(shape_mc, _b6x16_p32, is_2cta);                   \
    if ((src_fmt) == Tcgen05CpSrcFormat::B4x16_P64)                            \
      return TCGEN05_CP_2CTA(shape_mc, _b4x16_p64, is_2cta);                   \
    return TCGEN05_CP_2CTA(shape_mc, , is_2cta);                               \
  }()

llvm::Intrinsic::ID Tcgen05CpOp::getIntrinsicID(Operation &op) {
  auto curOp = cast<NVVM::Tcgen05CpOp>(op);
  bool is2CTA = curOp.getGroup() == CTAGroupKind::CTA_2;
  auto srcFmt = curOp.getSrcFormat();
  auto mc = curOp.getMulticast();

  switch (curOp.getShape()) {
  case Tcgen05CpShape::SHAPE_128x256b:
    return GET_TCGEN05_CP_ID(_128x256b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_128x128b:
    return GET_TCGEN05_CP_ID(_128x128b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_4x256b:
    return GET_TCGEN05_CP_ID(_4x256b, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_32x128b:
    return GET_TCGEN05_CP_ID(_32x128b_warpx4, srcFmt, is2CTA);
  case Tcgen05CpShape::SHAPE_64x128b:
    return (mc == Tcgen05CpMulticast::WARPX2_01_23)
               ? GET_TCGEN05_CP_ID(_64x128b_warpx2_01_23, srcFmt, is2CTA)
               : GET_TCGEN05_CP_ID(_64x128b_warpx2_02_13, srcFmt, is2CTA);
  }
  llvm_unreachable("Invalid shape in tcgen05 cp Op");
}

// Returns the valid vector length for a given shape and vector length, the
// function models the table mentioned in the tcgen05.{ld, st} Op description
static unsigned isValidVectorLength(NVVM::Tcgen05LdStShape shape,
                                    unsigned vecLen) {
  if (shape == NVVM::Tcgen05LdStShape::SHAPE_16X128B)
    return vecLen >= 2;
  if (shape == NVVM::Tcgen05LdStShape::SHAPE_16X256B)
    return vecLen >= 4;
  return true;
}

LogicalResult Tcgen05LdOp::verify() {
  LogicalResult result = success();
  if (getShape() == NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && !getOffset())
    result = emitError("shape 16x32bx2 requires offset argument");

  if (getShape() != NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && getOffset())
    result = emitError("offset argument is only supported for shape 16x32bx2");

  auto resTy = getRes().getType();
  unsigned resLen = isa<VectorType>(resTy)
                        ? llvm::cast<VectorType>(resTy).getNumElements()
                        : 1;
  if (!isValidVectorLength(getShape(), resLen))
    result = emitError(llvm::formatv("invalid result type length {0} for shape "
                                     "{1} in tcgen05.ld Op",
                                     resLen, stringifyEnum(getShape())));

  return result;
}

LogicalResult Tcgen05StOp::verify() {
  LogicalResult result = success();
  if (getShape() == NVVM::Tcgen05LdStShape::SHAPE_16X32BX2 && !getOffset())
    result = emitError("shape 16x32bx2 requires offset argument");

  auto valTy = getVal().getType();
  unsigned valLen = isa<VectorType>(valTy)
                        ? llvm::cast<VectorType>(valTy).getNumElements()
                        : 1;
  if (!isValidVectorLength(getShape(), valLen))
    result = emitError(llvm::formatv("invalid input length {0} for shape "
                                     "{1} in tcgen05.st Op",
                                     valLen, stringifyEnum(getShape())));

  return result;
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma functions
//===----------------------------------------------------------------------===//

static llvm::nvvm::Tcgen05MMAKind
getNVVMTcgen05MMAKind(NVVM::Tcgen05MMAKind kind) {
  switch (kind) {
  case NVVM::Tcgen05MMAKind::F16:
    return llvm::nvvm::Tcgen05MMAKind::F16;
  case NVVM::Tcgen05MMAKind::TF32:
    return llvm::nvvm::Tcgen05MMAKind::TF32;
  case NVVM::Tcgen05MMAKind::F8F6F4:
    return llvm::nvvm::Tcgen05MMAKind::F8F6F4;
  case NVVM::Tcgen05MMAKind::I8:
    return llvm::nvvm::Tcgen05MMAKind::I8;
  case NVVM::Tcgen05MMAKind::TI16:
    return llvm::nvvm::Tcgen05MMAKind::TI16;
  case NVVM::Tcgen05MMAKind::MXF8F6F4:
  case NVVM::Tcgen05MMAKind::MXF4:
  case NVVM::Tcgen05MMAKind::MXF4NVF4:
    // Block-scale kinds are handled by the tcgen05.mma.block_scale
    // lowering paths and are not valid for plain tcgen05.mma.
    llvm_unreachable("Unsupported tcgen05.mma kind");
  }
}

mlir::NVVM::IDArgPair
Tcgen05MMAOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                    llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  const bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));

  using EnableAShiftArray = std::array<llvm::Intrinsic::ID, 2>;
  using CtaGroupArray = std::array<EnableAShiftArray, 2>;
  using IsATensorArray = std::array<CtaGroupArray, 2>;
  using HasScaleInputDArray = std::array<IsATensorArray, 2>;
  using HasDisableOutputLaneArray = std::array<HasScaleInputDArray, 2>;

  // [hasDisableOutputLane][hasScaleInputD][isATensor][CtaGroup][EnableAShift]
  static constexpr HasDisableOutputLaneArray tcgen05MMAIDs = {
      {  // without diable output lane
       {{// without scale input D
         {{
             // shared
             {{// cg1
               {llvm::Intrinsic::nvvm_tcgen05_mma_shared, notIntrinsic},
               // cg2
               {llvm::Intrinsic::nvvm_tcgen05_mma_shared, notIntrinsic}}},
             {{// tensor
               {
                   // cg1
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor_ashift,
               },
               {
                   // cg2
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_tensor_ashift,
               }}},
         }},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_scale_d, notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_scale_d, notIntrinsic}}},
           {{// tensor
             {
                 // cg1
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d_ashift,
             },
             {
                 // cg2
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_tensor_scale_d_ashift,
             }}}}}}},
       // with disable output lane
       {{    // without scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_shared_disable_output_lane_cg2,
              notIntrinsic}}},
           {{// cg1
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg1,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg1_ashift,
             },
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_disable_output_lane_cg2_ashift,
             }}}}},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_shared_scale_d_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_shared_scale_d_disable_output_lane_cg2,
              notIntrinsic}}},
           // tensor
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg1,
              llvm::Intrinsic::
                  nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg1_ashift},
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_tensor_scale_d_disable_output_lane_cg2_ashift,
             }}}}}}}}};

  llvm::Value *ScaleInputD = mt.lookupValue(thisOp.getScaleInputD());
  bool hasScaleInputD = ScaleInputD != nullptr;

  llvm::Value *DisableOutputLane =
      mt.lookupValue(thisOp.getDisableOutputLane());
  bool hasDisableOutputLane = DisableOutputLane != nullptr;

  const unsigned ctaGroup =
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()));

  llvm::Intrinsic::ID ID =
      tcgen05MMAIDs[hasDisableOutputLane][hasScaleInputD][isATensor]
                   [ctaGroup - 1][thisOp.getAShift()];

  assert(ID != notIntrinsic && "Invalid intrinsic for Tcgen05MMAOp.");

  if (hasScaleInputD)
    args.push_back(ScaleInputD);

  if (hasDisableOutputLane)
    args.push_back(DisableOutputLane);

  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMTcgen05MMAKind(thisOp.getKind()))));

  if (!hasDisableOutputLane)
    args.push_back(builder.getInt32(ctaGroup));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  return {ID, args};
}

static LogicalResult
verifyTcgen05MMAOp(bool isATensor, mlir::Value disableOutputLane,
                   NVVM::CTAGroupKind ctaGroup, bool hasAShift,
                   NVVM::Tcgen05MMACollectorOp collectorOp, Location loc) {

  if (disableOutputLane) {
    mlir::VectorType disableOutputLaneType =
        cast<mlir::VectorType>(disableOutputLane.getType());
    if ((ctaGroup == NVVM::CTAGroupKind::CTA_1 &&
         disableOutputLaneType.getNumElements() != 4) ||
        (ctaGroup == NVVM::CTAGroupKind::CTA_2 &&
         disableOutputLaneType.getNumElements() != 8))
      return emitError(loc) << "Disable Output Lane of length "
                            << disableOutputLaneType.getNumElements()
                            << " is incompatible with CtaGroupAttr";
  }

  if (hasAShift && !isATensor)
    return emitError(
        loc, "A-shift can be applied only when matrix A is in tensor memory");

  if (hasAShift == true && (collectorOp == Tcgen05MMACollectorOp::FILL ||
                            collectorOp == Tcgen05MMACollectorOp::USE))
    return emitError(
        loc, "Cannot use collector buffer operation fill or use with ashift");

  return success();
}

LogicalResult Tcgen05MMAOp::verify() {
  return verifyTcgen05MMAOp(isa<LLVM::LLVMPointerType>(getMatrixA().getType()),
                            getDisableOutputLane(), getCtaGroup(), getAShift(),
                            getCollectorOp(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.sp functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMASparseOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMASparseOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));

  using EnableAShiftArray = std::array<llvm::Intrinsic::ID, 2>;
  using CtaGroupArray = std::array<EnableAShiftArray, 2>;
  using IsATensorArray = std::array<CtaGroupArray, 2>;
  using HasScaleInputDArray = std::array<IsATensorArray, 2>;
  using HasDisableOutputLaneArray = std::array<HasScaleInputDArray, 2>;

  // [hasDisableOutputLane][hasScaleInputD][isATensor][CtaGroup][EnableAShift]
  static constexpr HasDisableOutputLaneArray tcgen05MMASparseIDs = {
      {  // without diable output lane
       {{// without scale input D
         {{
             // shared
             {{// cg1
               {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared, notIntrinsic},
               // cg2
               {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared, notIntrinsic}}},
             {{// tensor
               {
                   // cg1
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_ashift,
               },
               {
                   // cg2
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor,
                   llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_ashift,
               }}},
         }},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared_scale_d,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::nvvm_tcgen05_mma_sp_shared_scale_d,
              notIntrinsic}}},
           {{// tensor
             {
                 // cg1
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d_ashift,
             },
             {
                 // cg2
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d,
                 llvm::Intrinsic::nvvm_tcgen05_mma_sp_tensor_scale_d_ashift,
             }}}}}}},
       // with disable output lane
       {{    // without scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_disable_output_lane_cg2,
              notIntrinsic}}},
           {{// cg1
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg1,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg1_ashift,
             },
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_disable_output_lane_cg2_ashift,
             }}}}},
         // with scale input D
         {{  // shared
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_scale_d_disable_output_lane_cg1,
              notIntrinsic},
             // cg2
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_shared_scale_d_disable_output_lane_cg2,
              notIntrinsic}}},
           // tensor
           {{// cg1
             {llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg1,
              llvm::Intrinsic::
                  nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg1_ashift},
             // cg2
             {
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg2,
                 llvm::Intrinsic::
                     nvvm_tcgen05_mma_sp_tensor_scale_d_disable_output_lane_cg2_ashift,
             }}}}}}}}};

  llvm::Value *ScaleInputD = mt.lookupValue(thisOp.getScaleInputD());
  bool hasScaleInputD = ScaleInputD != nullptr;

  llvm::Value *DisableOutputLane =
      mt.lookupValue(thisOp.getDisableOutputLane());
  bool hasDisableOutputLane = DisableOutputLane != nullptr;

  unsigned ctaGroup =
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()));

  llvm::Intrinsic::ID ID =
      tcgen05MMASparseIDs[hasDisableOutputLane][hasScaleInputD][isATensor]
                         [ctaGroup - 1][thisOp.getAShift()];

  assert(ID != notIntrinsic && "Invalid intrinsic for Tcgen05MMASparseOp.");

  if (hasScaleInputD)
    args.push_back(ScaleInputD);

  if (hasDisableOutputLane)
    args.push_back(DisableOutputLane);

  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMTcgen05MMAKind(thisOp.getKind()))));

  if (!hasDisableOutputLane)
    args.push_back(builder.getInt32(ctaGroup));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  return {ID, args};
}

LogicalResult Tcgen05MMASparseOp::verify() {
  return verifyTcgen05MMAOp(isa<LLVM::LLVMPointerType>(getMatrixA().getType()),
                            getDisableOutputLane(), getCtaGroup(), getAShift(),
                            getCollectorOp(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.block_scale functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMABlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMABlockScaleOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getScaleA()));
  args.push_back(mt.lookupValue(thisOp.getScaleB()));
  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  auto kind = thisOp.getKind();
  auto blockScale = thisOp.getBlockScale();
  llvm::Intrinsic::ID ID = [&]() {
    if (kind == NVVM::Tcgen05MMAKind::MXF8F6F4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_tensor_mxf8f6f4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_shared_mxf8f6f4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf8f6f4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf8f6f4_block_scale_block32;
      }
    } else if (kind == NVVM::Tcgen05MMAKind::MXF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor
                   ? llvm::Intrinsic::nvvm_tcgen05_mma_tensor_mxf4_block_scale
                   : llvm::Intrinsic::nvvm_tcgen05_mma_shared_mxf4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_tensor_mxf4_block_scale_block32
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_shared_mxf4_block_scale_block32;
      }
    } else if (kind == NVVM::Tcgen05MMAKind::MXF4NVF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf4nvf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf4nvf4_block_scale_block32;

      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_tensor_mxf4nvf4_block_scale_block16
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_shared_mxf4nvf4_block_scale_block16;
      }
    }
    llvm_unreachable("Invalid tcgen05.mma.block_scale attributes");
  }();

  return {ID, args};
}

static LogicalResult verifyTcgen05MMABlockScaleOp(
    NVVM::Tcgen05MMACollectorOp collectorOp, NVVM::Tcgen05MMAKind kind,
    NVVM::Tcgen05MMABlockScale blockScale, Location loc) {
  if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT &&
      kind == NVVM::Tcgen05MMAKind::MXF4NVF4)
    return emitError(loc, "mxf4nvf4 requires block scale attribute");

  if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16 &&
      kind != NVVM::Tcgen05MMAKind::MXF4NVF4)
    return emitError(loc,
                     llvm::formatv("{} kind does not support block16 attribute",
                                   stringifyEnum(kind)));

  return success();
}

LogicalResult Tcgen05MMABlockScaleOp::verify() {
  return verifyTcgen05MMABlockScaleOp(getCollectorOp(), getKind(),
                                      getBlockScale(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.sp.block_scale functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMASparseBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMASparseBlockScaleOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));
  args.push_back(mt.lookupValue(thisOp.getScaleA()));
  args.push_back(mt.lookupValue(thisOp.getScaleB()));
  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  auto kind = thisOp.getKind();
  auto blockScale = thisOp.getBlockScale();
  llvm::Intrinsic::ID ID = [&]() {
    if (kind == NVVM::Tcgen05MMAKind::MXF8F6F4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_tensor_mxf8f6f4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_shared_mxf8f6f4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf8f6f4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf8f6f4_block_scale_block32;
      }
    } else if (kind == NVVM::Tcgen05MMAKind::MXF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::DEFAULT) {
        return isATensor ? llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_tensor_mxf4_block_scale
                         : llvm::Intrinsic::
                               nvvm_tcgen05_mma_sp_shared_mxf4_block_scale;
      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4_block_scale_block32;
      }
    } else if (kind == NVVM::Tcgen05MMAKind::MXF4NVF4) {
      if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK32) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4nvf4_block_scale_block32
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4nvf4_block_scale_block32;

      } else if (blockScale == NVVM::Tcgen05MMABlockScale::BLOCK16) {
        return isATensor
                   ? llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_tensor_mxf4nvf4_block_scale_block16
                   : llvm::Intrinsic::
                         nvvm_tcgen05_mma_sp_shared_mxf4nvf4_block_scale_block16;
      }
    }
    llvm_unreachable("Invalid tcgen05.mma.sp.block_scale attributes");
  }();

  return {ID, args};
}

LogicalResult Tcgen05MMASparseBlockScaleOp::verify() {
  return verifyTcgen05MMABlockScaleOp(getCollectorOp(), getKind(),
                                      getBlockScale(), getLoc());
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.ws functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMAWsOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAWsOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));

  mlir::Value ZeroColMask = thisOp.getZeroColMask();
  llvm::Intrinsic::ID ID = notIntrinsic;
  if (ZeroColMask) {
    args.push_back(mt.lookupValue(ZeroColMask));
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_tensor_zero_col_mask
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_shared_zero_col_mask;
  } else
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_tensor
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_shared;

  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMTcgen05MMAKind(thisOp.getKind()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorBBuffer())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.ws.sp functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMAWsSparseOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {

  auto thisOp = cast<NVVM::Tcgen05MMAWsSparseOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getSparseMetadata()));

  mlir::Value ZeroColMask = thisOp.getZeroColMask();
  llvm::Intrinsic::ID ID = notIntrinsic;
  if (ZeroColMask) {
    args.push_back(mt.lookupValue(ZeroColMask));
    ID = isATensor
             ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_tensor_zero_col_mask
             : llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_shared_zero_col_mask;
  } else
    ID = isATensor ? llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_tensor
                   : llvm::Intrinsic::nvvm_tcgen05_mma_ws_sp_shared;

  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMTcgen05MMAKind(thisOp.getKind()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorBBuffer())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOp())));

  return {ID, args};
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.decompress_b functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMADecompressBOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<Tcgen05MMADecompressBOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  const bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getDecompressBMetadata()));

  llvm::Value *DisableOutputLane =
      mt.lookupValue(thisOp.getDisableOutputLane());
  bool hasDisableOutputLane = DisableOutputLane != nullptr;

  NVVM::CTAGroupKind ctaGroup = thisOp.getCtaGroup();

  using namespace llvm::Intrinsic;
  ID intrinsicID = not_intrinsic;

  if (hasDisableOutputLane) {
    if (ctaGroup == NVVM::CTAGroupKind::CTA_1) {
      intrinsicID =
          isATensor
              ? nvvm_tcgen05_mma_tensor_f8f6f4_disable_output_lane_cg1_decompress_b
              : nvvm_tcgen05_mma_shared_f8f6f4_disable_output_lane_cg1_decompress_b;
    } else if (ctaGroup == NVVM::CTAGroupKind::CTA_2) {
      intrinsicID =
          isATensor
              ? nvvm_tcgen05_mma_tensor_f8f6f4_disable_output_lane_cg2_decompress_b
              : nvvm_tcgen05_mma_shared_f8f6f4_disable_output_lane_cg2_decompress_b;
    } else {
      llvm_unreachable("Unknown ctaGroup for tcgen05.mma.decompress_b");
    }
  } else {
    intrinsicID = isATensor ? nvvm_tcgen05_mma_tensor_f8f6f4_decompress_b
                            : nvvm_tcgen05_mma_shared_f8f6f4_decompress_b;
  }

  assert(intrinsicID != not_intrinsic &&
         "Invalid intrinsic for Tcgen05MMADecompressBOp.");

  if (hasDisableOutputLane)
    args.push_back(DisableOutputLane);
  else
    args.push_back(
        builder.getInt32(static_cast<unsigned>(getNVVMCtaGroupKind(ctaGroup))));

  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpA())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  return {intrinsicID, args};
}

LogicalResult Tcgen05MMADecompressBOp::verify() {
  mlir::Value disableOutputLane = getDisableOutputLane();

  if (disableOutputLane) {
    NVVM::CTAGroupKind ctaGroup = getCtaGroup();

    mlir::VectorType disableOutputLaneType =
        cast<mlir::VectorType>(disableOutputLane.getType());
    if ((ctaGroup == NVVM::CTAGroupKind::CTA_1 &&
         disableOutputLaneType.getNumElements() != 4) ||
        (ctaGroup == NVVM::CTAGroupKind::CTA_2 &&
         disableOutputLaneType.getNumElements() != 8))
      return emitOpError() << "Disable Output Lane of length "
                           << disableOutputLaneType.getNumElements()
                           << " is incompatible with CtaGroupAttr";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.mma.block_scale.decompress_b functions
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair Tcgen05MMABlockScaleDecompressBOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<Tcgen05MMABlockScaleDecompressBOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  args.push_back(mt.lookupValue(thisOp.getMatrixD()));

  llvm::Value *A = mt.lookupValue(thisOp.getMatrixA());
  const bool isATensor = isa<llvm::PointerType>(A->getType());
  args.push_back(A);

  args.push_back(mt.lookupValue(thisOp.getMatrixB()));
  args.push_back(mt.lookupValue(thisOp.getIdesc()));
  args.push_back(mt.lookupValue(thisOp.getEnableInputD()));
  args.push_back(mt.lookupValue(thisOp.getScaleA()));
  args.push_back(mt.lookupValue(thisOp.getScaleB()));
  args.push_back(mt.lookupValue(thisOp.getDecompressBMetadata()));
  args.push_back(builder.getInt32(
      static_cast<unsigned>(getNVVMCtaGroupKind(thisOp.getCtaGroup()))));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpA())));
  args.push_back(
      builder.getInt32(static_cast<unsigned>(thisOp.getCollectorOpB())));

  llvm::Intrinsic::ID intrinsicID =
      isATensor
          ? llvm::Intrinsic::
                nvvm_tcgen05_mma_tensor_mxf8f6f4_block_scale_block32_decompress_b
          : llvm::Intrinsic::
                nvvm_tcgen05_mma_shared_mxf8f6f4_block_scale_block32_decompress_b;

  return {intrinsicID, args};
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.ld.red functions
//===----------------------------------------------------------------------===//

#define TCGEN05LDRED(SHAPE, NUM, TYPE)                                         \
  llvm::Intrinsic::nvvm_tcgen05_ld_red_##SHAPE##_##NUM##_##TYPE

mlir::NVVM::IDArgPair NVVM::Tcgen05LdRedOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::Tcgen05LdRedOp>(op);
  llvm::SmallVector<llvm::Value *> args;

  mlir::VectorType VecResTy =
      cast<mlir::VectorType>(thisOp.getData().getType());
  unsigned Num = VecResTy.getNumElements();
  bool IsFloat = thisOp.getRedVal().getType().isF32();

  llvm::Intrinsic::ID Shape32x32b[][2] = {
      {notIntrinsic, notIntrinsic},
      {TCGEN05LDRED(32x32b, x2, i32), TCGEN05LDRED(32x32b, x2, f32)},
      {TCGEN05LDRED(32x32b, x4, i32), TCGEN05LDRED(32x32b, x4, f32)},
      {TCGEN05LDRED(32x32b, x8, i32), TCGEN05LDRED(32x32b, x8, f32)},
      {TCGEN05LDRED(32x32b, x16, i32), TCGEN05LDRED(32x32b, x16, f32)},
      {TCGEN05LDRED(32x32b, x32, i32), TCGEN05LDRED(32x32b, x32, f32)},
      {TCGEN05LDRED(32x32b, x64, i32), TCGEN05LDRED(32x32b, x64, f32)},
      {TCGEN05LDRED(32x32b, x128, i32), TCGEN05LDRED(32x32b, x128, f32)},
  };

  llvm::Intrinsic::ID Shape16x32bx2[][2] = {
      {notIntrinsic, notIntrinsic},
      {TCGEN05LDRED(16x32bx2, x2, i32), TCGEN05LDRED(16x32bx2, x2, f32)},
      {TCGEN05LDRED(16x32bx2, x4, i32), TCGEN05LDRED(16x32bx2, x4, f32)},
      {TCGEN05LDRED(16x32bx2, x8, i32), TCGEN05LDRED(16x32bx2, x8, f32)},
      {TCGEN05LDRED(16x32bx2, x16, i32), TCGEN05LDRED(16x32bx2, x16, f32)},
      {TCGEN05LDRED(16x32bx2, x32, i32), TCGEN05LDRED(16x32bx2, x32, f32)},
      {TCGEN05LDRED(16x32bx2, x64, i32), TCGEN05LDRED(16x32bx2, x64, f32)},
      {TCGEN05LDRED(16x32bx2, x128, i32), TCGEN05LDRED(16x32bx2, x128, f32)},
  };

  NVVM::Tcgen05LdStShape shape = thisOp.getShape();
  unsigned ID = [&]() {
    // `num` contains the length of vector and log2 of `num` returns the index
    // into the shape array
    unsigned idx = std::log2(Num);
    switch (shape) {
    case NVVM::Tcgen05LdStShape::SHAPE_32X32B:
      return Shape32x32b[idx][IsFloat];
    case NVVM::Tcgen05LdStShape::SHAPE_16X32BX2:
      return Shape16x32bx2[idx][IsFloat];
    default:
      llvm_unreachable("unhandled tcgen05.ld lowering");
    }
  }();

  args.push_back(mt.lookupValue(thisOp.getAddr()));

  if (shape == NVVM::Tcgen05LdStShape::SHAPE_16X32BX2)
    args.push_back(mt.lookupValue(thisOp.getOffset()));

  args.push_back(
      builder.getInt32(thisOp.getOp() == NVVM::ReductionKind::MIN ? 0 : 1));

  if (IsFloat) {
    args.push_back(builder.getInt1(static_cast<unsigned>(thisOp.getAbs())));
    args.push_back(builder.getInt1(static_cast<unsigned>(thisOp.getNan())));
  }
  return {ID, args};
}

LogicalResult Tcgen05LdRedOp::verify() {
  VectorType data = cast<VectorType>(getData().getType());
  Type redVal = getRedVal().getType();

  if (data.getElementType() != redVal)
    return emitError(
        "type of reduction value and element type of vector data should match");

  if (getOp() != NVVM::ReductionKind::MIN &&
      getOp() != NVVM::ReductionKind::MAX)
    return emitError("only min and max reduction kinds are supported");

  if (redVal.isInteger() && (getAbs() || getNan())) {
    return emitError("abs or nan is only applicable for f32 type");
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMTcgenOps.cpp.inc"
