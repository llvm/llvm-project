//===- NVVMToLLVMIRTranslation.cpp - Translate NVVM to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR NVVM dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

#define REDUX_F32_ID_IMPL(op, abs, hasNaN)                                     \
  hasNaN ? llvm::Intrinsic::nvvm_redux_sync_f##op##abs##_NaN                   \
         : llvm::Intrinsic::nvvm_redux_sync_f##op##abs

#define GET_REDUX_F32_ID(op, hasAbs, hasNaN)                                   \
  hasAbs ? REDUX_F32_ID_IMPL(op, _abs, hasNaN) : REDUX_F32_ID_IMPL(op, , hasNaN)

static llvm::Intrinsic::ID getReduxIntrinsicId(llvm::Type *resultType,
                                               NVVM::ReduxKind kind,
                                               bool hasAbs, bool hasNaN) {
  if (!(resultType->isIntegerTy(32) || resultType->isFloatTy()))
    llvm_unreachable("unsupported data type for redux");

  switch (kind) {
  case NVVM::ReduxKind::ADD:
    return llvm::Intrinsic::nvvm_redux_sync_add;
  case NVVM::ReduxKind::UMAX:
    return llvm::Intrinsic::nvvm_redux_sync_umax;
  case NVVM::ReduxKind::UMIN:
    return llvm::Intrinsic::nvvm_redux_sync_umin;
  case NVVM::ReduxKind::AND:
    return llvm::Intrinsic::nvvm_redux_sync_and;
  case NVVM::ReduxKind::OR:
    return llvm::Intrinsic::nvvm_redux_sync_or;
  case NVVM::ReduxKind::XOR:
    return llvm::Intrinsic::nvvm_redux_sync_xor;
  case NVVM::ReduxKind::MAX:
    return llvm::Intrinsic::nvvm_redux_sync_max;
  case NVVM::ReduxKind::MIN:
    return llvm::Intrinsic::nvvm_redux_sync_min;
  case NVVM::ReduxKind::FMIN:
    return GET_REDUX_F32_ID(min, hasAbs, hasNaN);
  case NVVM::ReduxKind::FMAX:
    return GET_REDUX_F32_ID(max, hasAbs, hasNaN);
  }
  llvm_unreachable("unknown redux kind");
}

static llvm::Intrinsic::ID getShflIntrinsicId(llvm::Type *resultType,
                                              NVVM::ShflKind kind,
                                              bool withPredicate) {

  if (withPredicate) {
    resultType = cast<llvm::StructType>(resultType)->getElementType(0);
    switch (kind) {
    case NVVM::ShflKind::bfly:
      return resultType->isFloatTy()
                 ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32p
                 : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32p;
    case NVVM::ShflKind::up:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_up_f32p
                                     : llvm::Intrinsic::nvvm_shfl_sync_up_i32p;
    case NVVM::ShflKind::down:
      return resultType->isFloatTy()
                 ? llvm::Intrinsic::nvvm_shfl_sync_down_f32p
                 : llvm::Intrinsic::nvvm_shfl_sync_down_i32p;
    case NVVM::ShflKind::idx:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_idx_f32p
                                     : llvm::Intrinsic::nvvm_shfl_sync_idx_i32p;
    }
  } else {
    switch (kind) {
    case NVVM::ShflKind::bfly:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_bfly_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_bfly_i32;
    case NVVM::ShflKind::up:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_up_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_up_i32;
    case NVVM::ShflKind::down:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_down_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_down_i32;
    case NVVM::ShflKind::idx:
      return resultType->isFloatTy() ? llvm::Intrinsic::nvvm_shfl_sync_idx_f32
                                     : llvm::Intrinsic::nvvm_shfl_sync_idx_i32;
    }
  }
  llvm_unreachable("unknown shuffle kind");
}

static llvm::Intrinsic::ID getMatchSyncIntrinsicId(Type valType,
                                                   NVVM::MatchSyncKind kind) {
  switch (kind) {
  case NVVM::MatchSyncKind::any:
    return valType.isInteger(32) ? llvm::Intrinsic::nvvm_match_any_sync_i32
                                 : llvm::Intrinsic::nvvm_match_any_sync_i64;
  case NVVM::MatchSyncKind::all:
    // match.all instruction has two variants -- one returns a single value,
    // another returns a pair {value, predicate}. We currently only implement
    // the latter as that's the variant exposed by CUDA API.
    return valType.isInteger(32) ? llvm::Intrinsic::nvvm_match_all_sync_i32p
                                 : llvm::Intrinsic::nvvm_match_all_sync_i64p;
  }
  llvm_unreachable("unsupported match sync kind");
}

static llvm::Intrinsic::ID getVoteSyncIntrinsicId(NVVM::VoteSyncKind kind) {
  switch (kind) {
  case NVVM::VoteSyncKind::any:
    return llvm::Intrinsic::nvvm_vote_any_sync;
  case NVVM::VoteSyncKind::all:
    return llvm::Intrinsic::nvvm_vote_all_sync;
  case NVVM::VoteSyncKind::ballot:
    return llvm::Intrinsic::nvvm_vote_ballot_sync;
  case NVVM::VoteSyncKind::uni:
    return llvm::Intrinsic::nvvm_vote_uni_sync;
  }
  llvm_unreachable("unsupported vote kind");
}

static llvm::Intrinsic::ID
getLdMatrixIntrinsicId(NVVM::MMALayout layout, int32_t num,
                       NVVM::LdStMatrixShapeAttr shape,
                       NVVM::LdStMatrixEltType eltType) {
  if (shape.getM() == 8 && shape.getN() == 8) {
    switch (num) {
    case 1:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_b16
                 : llvm::Intrinsic::
                       nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16;
    case 2:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_b16
                 : llvm::Intrinsic::
                       nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16;
    case 4:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16
                 : llvm::Intrinsic::
                       nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16;
    }
  } else if (shape.getM() == 8 && shape.getN() == 16) {
    if (eltType == NVVM::LdStMatrixEltType::B8X16_B6X16_P32) {
      switch (num) {
      case 1:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x1_b8x16_b6x16_p32;
      case 2:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x2_b8x16_b6x16_p32;
      case 4:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x4_b8x16_b6x16_p32;
      }
    } else if (eltType == NVVM::LdStMatrixEltType::B8X16_B4X16_P64) {
      switch (num) {
      case 1:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x1_b8x16_b4x16_p64;
      case 2:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x2_b8x16_b4x16_p64;
      case 4:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m8n16_x4_b8x16_b4x16_p64;
      }
    }
  } else if (shape.getM() == 16 && shape.getN() == 16) {
    if (eltType == NVVM::LdStMatrixEltType::B8) {
      switch (num) {
      case 1:
        return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8;
      case 2:
        return llvm::Intrinsic::nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8;
      }
    } else if (eltType == NVVM::LdStMatrixEltType::B8X16_B6X16_P32) {
      switch (num) {
      case 1:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8x16_b6x16_p32;
      case 2:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8x16_b6x16_p32;
      }
    } else if (eltType == NVVM::LdStMatrixEltType::B8X16_B4X16_P64) {
      switch (num) {
      case 1:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8x16_b4x16_p64;
      case 2:
        return llvm::Intrinsic::
            nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8x16_b4x16_p64;
      }
    }
  }
  llvm_unreachable("unknown ldmatrix kind");
}

/// Return the intrinsic ID associated with stmatrix for the given paramters.
static llvm::Intrinsic::ID
getStMatrixIntrinsicId(NVVM::MMALayout layout, int32_t num,
                       NVVM::LdStMatrixShapeAttr shape,
                       NVVM::LdStMatrixEltType eltType) {
  if (shape.getM() == 8 && shape.getN() == 8) {
    switch (num) {
    case 1:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_b16
                 : llvm::Intrinsic::
                       nvvm_stmatrix_sync_aligned_m8n8_x1_trans_b16;
    case 2:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_b16
                 : llvm::Intrinsic::
                       nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16;
    case 4:
      return (layout == NVVM::MMALayout::row)
                 ? llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_b16
                 : llvm::Intrinsic::
                       nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16;
    }
  } else if (shape.getM() == 16 && shape.getN() == 8) {
    switch (num) {
    case 1:
      return llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x1_trans_b8;
    case 2:
      return llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x2_trans_b8;
    case 4:
      return llvm::Intrinsic::nvvm_stmatrix_sync_aligned_m16n8_x4_trans_b8;
    }
  }
  llvm_unreachable("unknown stmatrix kind");
}

/// Return the intrinsic ID associated with st.bulk for the given address type.
static llvm::Intrinsic::ID
getStBulkIntrinsicId(LLVM::LLVMPointerType addrType) {
  bool isSharedMemory =
      addrType.getAddressSpace() == NVVM::NVVMMemorySpace::kSharedMemorySpace;
  return isSharedMemory ? llvm::Intrinsic::nvvm_st_bulk_shared_cta
                        : llvm::Intrinsic::nvvm_st_bulk;
}

static unsigned getUnidirectionalFenceProxyID(NVVM::ProxyKind fromProxy,
                                              NVVM::ProxyKind toProxy,
                                              NVVM::MemScopeKind scope,
                                              bool isRelease) {
  if (fromProxy == NVVM::ProxyKind::GENERIC &&
      toProxy == NVVM::ProxyKind::TENSORMAP) {
    switch (scope) {
    case NVVM::MemScopeKind::CTA: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_cta;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_cta;
    }
    case NVVM::MemScopeKind::CLUSTER: {
      if (isRelease)
        return llvm::Intrinsic::
            nvvm_fence_proxy_tensormap_generic_release_cluster;
      return llvm::Intrinsic::
          nvvm_fence_proxy_tensormap_generic_acquire_cluster;
    }
    case NVVM::MemScopeKind::GPU: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_gpu;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_gpu;
    }
    case NVVM::MemScopeKind::SYS: {
      if (isRelease)
        return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_release_sys;
      return llvm::Intrinsic::nvvm_fence_proxy_tensormap_generic_acquire_sys;
    }
    }
    llvm_unreachable("Unknown scope for uni-directional fence.proxy operation");
  }
  llvm_unreachable("Unsupported proxy kinds");
}

#define TCGEN05LD(SHAPE, NUM) llvm::Intrinsic::nvvm_tcgen05_ld_##SHAPE##_##NUM

static llvm::Intrinsic::ID
getTcgen05LdIntrinsicID(mlir::NVVM::Tcgen05LdStShape shape, uint32_t num) {
  llvm::Intrinsic::ID Shape16x64b[] = {
      TCGEN05LD(16x64b, x1),  TCGEN05LD(16x64b, x2),   TCGEN05LD(16x64b, x4),
      TCGEN05LD(16x64b, x8),  TCGEN05LD(16x64b, x16),  TCGEN05LD(16x64b, x32),
      TCGEN05LD(16x64b, x64), TCGEN05LD(16x64b, x128),
  };

  llvm::Intrinsic::ID Shape16x128b[] = {
      TCGEN05LD(16x128b, x1),  TCGEN05LD(16x128b, x2),  TCGEN05LD(16x128b, x4),
      TCGEN05LD(16x128b, x8),  TCGEN05LD(16x128b, x16), TCGEN05LD(16x128b, x32),
      TCGEN05LD(16x128b, x64),
  };

  llvm::Intrinsic::ID Shape16x256b[] = {
      TCGEN05LD(16x256b, x1), TCGEN05LD(16x256b, x2),  TCGEN05LD(16x256b, x4),
      TCGEN05LD(16x256b, x8), TCGEN05LD(16x256b, x16), TCGEN05LD(16x256b, x32),
  };

  llvm::Intrinsic::ID Shape16x32bx2[] = {
      TCGEN05LD(16x32bx2, x1),  TCGEN05LD(16x32bx2, x2),
      TCGEN05LD(16x32bx2, x4),  TCGEN05LD(16x32bx2, x8),
      TCGEN05LD(16x32bx2, x16), TCGEN05LD(16x32bx2, x32),
      TCGEN05LD(16x32bx2, x64), TCGEN05LD(16x32bx2, x128),
  };

  llvm::Intrinsic::ID Shape32x32b[] = {
      TCGEN05LD(32x32b, x1),  TCGEN05LD(32x32b, x2),   TCGEN05LD(32x32b, x4),
      TCGEN05LD(32x32b, x8),  TCGEN05LD(32x32b, x16),  TCGEN05LD(32x32b, x32),
      TCGEN05LD(32x32b, x64), TCGEN05LD(32x32b, x128),
  };

  // `num` contains the length of vector and log2 of `num` returns the index
  // into the shape array
  unsigned Idx = std::log2(num);

  switch (shape) {
  case NVVM::Tcgen05LdStShape::SHAPE_16X64B:
    return Shape16x64b[Idx];
  case NVVM::Tcgen05LdStShape::SHAPE_16X128B:
    return Shape16x128b[Idx - 1];
  case NVVM::Tcgen05LdStShape::SHAPE_16X256B:
    return Shape16x256b[Idx - 2];
  case NVVM::Tcgen05LdStShape::SHAPE_32X32B:
    return Shape32x32b[Idx];
  case NVVM::Tcgen05LdStShape::SHAPE_16X32BX2:
    return Shape16x32bx2[Idx];
  }
  llvm_unreachable("unhandled tcgen05.ld lowering");
}

#define TCGEN05ST(SHAPE, NUM) llvm::Intrinsic::nvvm_tcgen05_st_##SHAPE##_##NUM

static llvm::Intrinsic::ID
getTcgen05StIntrinsicID(mlir::NVVM::Tcgen05LdStShape shape, uint32_t num) {
  llvm::Intrinsic::ID Shape16x64b[] = {
      TCGEN05ST(16x64b, x1),  TCGEN05ST(16x64b, x2),   TCGEN05ST(16x64b, x4),
      TCGEN05ST(16x64b, x8),  TCGEN05ST(16x64b, x16),  TCGEN05ST(16x64b, x32),
      TCGEN05ST(16x64b, x64), TCGEN05ST(16x64b, x128),
  };

  llvm::Intrinsic::ID Shape16x128b[] = {
      TCGEN05ST(16x128b, x1),  TCGEN05ST(16x128b, x2),  TCGEN05ST(16x128b, x4),
      TCGEN05ST(16x128b, x8),  TCGEN05ST(16x128b, x16), TCGEN05ST(16x128b, x32),
      TCGEN05ST(16x128b, x64),
  };

  llvm::Intrinsic::ID Shape16x256b[] = {
      TCGEN05ST(16x256b, x1), TCGEN05ST(16x256b, x2),  TCGEN05ST(16x256b, x4),
      TCGEN05ST(16x256b, x8), TCGEN05ST(16x256b, x16), TCGEN05ST(16x256b, x32),
  };

  llvm::Intrinsic::ID Shape16x32bx2[] = {
      TCGEN05ST(16x32bx2, x1),  TCGEN05ST(16x32bx2, x2),
      TCGEN05ST(16x32bx2, x4),  TCGEN05ST(16x32bx2, x8),
      TCGEN05ST(16x32bx2, x16), TCGEN05ST(16x32bx2, x32),
      TCGEN05ST(16x32bx2, x64), TCGEN05ST(16x32bx2, x128),
  };

  llvm::Intrinsic::ID Shape32x32b[] = {
      TCGEN05ST(32x32b, x1),  TCGEN05ST(32x32b, x2),   TCGEN05ST(32x32b, x4),
      TCGEN05ST(32x32b, x8),  TCGEN05ST(32x32b, x16),  TCGEN05ST(32x32b, x32),
      TCGEN05ST(32x32b, x64), TCGEN05ST(32x32b, x128),
  };

  // `num` contains the length of vector and log2 of `num` returns the index
  // into the shape array
  unsigned Idx = std::log2(num);

  switch (shape) {
  case NVVM::Tcgen05LdStShape::SHAPE_16X64B:
    return Shape16x64b[Idx];
  case NVVM::Tcgen05LdStShape::SHAPE_16X128B:
    return Shape16x128b[Idx - 1];
  case NVVM::Tcgen05LdStShape::SHAPE_16X256B:
    return Shape16x256b[Idx - 2];
  case NVVM::Tcgen05LdStShape::SHAPE_32X32B:
    return Shape32x32b[Idx];
  case NVVM::Tcgen05LdStShape::SHAPE_16X32BX2:
    return Shape16x32bx2[Idx];
  }
  llvm_unreachable("unhandled tcgen05.st lowering");
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the NVVM dialect to LLVM IR.
class NVVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/NVVMConversions.inc"

    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return failure();
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());

    if (attribute.getName() == NVVM::NVVMDialect::getMaxntidAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.maxntid", attr);
    } else if (attribute.getName() == NVVM::NVVMDialect::getReqntidAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.reqntid", attr);
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getClusterDimAttrName()) {
      if (!isa<DenseI32ArrayAttr>(attribute.getValue()))
        return failure();
      auto values = cast<DenseI32ArrayAttr>(attribute.getValue());
      const std::string attr = llvm::formatv(
          "{0:$[,]}", llvm::make_range(values.asArrayRef().begin(),
                                       values.asArrayRef().end()));
      llvmFunc->addFnAttr("nvvm.cluster_dim", attr);
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getClusterMaxBlocksAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.maxclusterrank", llvm::utostr(value.getInt()));
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getMinctasmAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.minctasm", llvm::utostr(value.getInt()));
    } else if (attribute.getName() == NVVM::NVVMDialect::getMaxnregAttrName()) {
      auto value = dyn_cast<IntegerAttr>(attribute.getValue());
      llvmFunc->addFnAttr("nvvm.maxnreg", llvm::utostr(value.getInt()));
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getKernelFuncAttrName()) {
      llvmFunc->setCallingConv(llvm::CallingConv::PTX_Kernel);
    } else if (attribute.getName() ==
               NVVM::NVVMDialect::getBlocksAreClustersAttrName()) {
      llvmFunc->addFnAttr("nvvm.blocksareclusters");
    }

    return success();
  }

  LogicalResult
  convertParameterAttr(LLVMFuncOp funcOp, int argIdx, NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const final {

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc =
        moduleTranslation.lookupFunction(funcOp.getName());

    if (attribute.getName() == NVVM::NVVMDialect::getGridConstantAttrName()) {
      llvmFunc->addParamAttr(
          argIdx, llvm::Attribute::get(llvmContext, "nvvm.grid_constant"));
    }
    return success();
  }
};
} // namespace

void mlir::registerNVVMDialectTranslation(DialectRegistry &registry) {
  registry.insert<NVVM::NVVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    dialect->addInterfaces<NVVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerNVVMDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
