//===- MemoryAccessOpInterfacesImpl.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement memref dialect interfaces that enable manipulating memref indexing
// in passes like FoldMemRefAliasOps.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/MemoryAccessOpInterfacesImpl.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/MemorySpaceUtils.h"
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::amdgpu;
using namespace mlir::memref;

namespace {
template <typename OpTy>
struct TransposeLoadAccess final
    : IndexedAccessOpInterface::ExternalModel<TransposeLoadAccess<OpTy>, OpTy> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<OpTy>(op).getSrc());
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<OpTy>(op).getSrcIndices();
  }

  SmallVector<int64_t> getAccessedShape(Operation *op) const {
    return {cast<VectorType>(cast<OpTy>(op).getResult().getType())
                .getNumElements()};
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto accessOp = cast<OpTy>(op);
    rewriter.modifyOpInPlace(accessOp, [&]() {
      accessOp.getSrcMutable().assign(newMemref);
      accessOp.getSrcIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};

template <typename OpTy>
struct BaseAndIndicesAccess final
    : IndexedAccessOpInterface::ExternalModel<BaseAndIndicesAccess<OpTy>,
                                              OpTy> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<OpTy>(op).getBase());
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<OpTy>(op).getIndices();
  }

  SmallVector<int64_t> getAccessedShape(Operation *) const { return {}; }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto accessOp = cast<OpTy>(op);
    rewriter.modifyOpInPlace(accessOp, [&]() {
      accessOp.getBaseMutable().assign(newMemref);
      accessOp.getIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};

template <typename OpTy>
struct DescriptorAtomicBarrierAccess final
    : IndexedAccessOpInterface::ExternalModel<
          DescriptorAtomicBarrierAccess<OpTy>, OpTy> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    Value memref = cast<OpTy>(op).getAtomicBarrierAddress();
    if (!memref)
      return {};
    return cast<TypedValue<MemRefType>>(memref);
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<OpTy>(op).getAtomicBarrierIndices();
  }

  SmallVector<int64_t> getAccessedShape(Operation *) const { return {}; }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto accessOp = cast<OpTy>(op);
    rewriter.modifyOpInPlace(accessOp, [&]() {
      accessOp.getAtomicBarrierAddressMutable().assign(newMemref);
      accessOp.getAtomicBarrierIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};

struct GatherToLDSCopy final
    : IndexedMemCopyOpInterface::ExternalModel<GatherToLDSCopy, GatherToLDSOp> {
  TypedValue<MemRefType> getSrc(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<GatherToLDSOp>(op).getSrc());
  }

  Operation::operand_range getSrcIndices(Operation *op) const {
    return cast<GatherToLDSOp>(op).getSrcIndices();
  }

  TypedValue<MemRefType> getDst(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<GatherToLDSOp>(op).getDst());
  }

  Operation::operand_range getDstIndices(Operation *op) const {
    return cast<GatherToLDSOp>(op).getDstIndices();
  }

  void setMemrefsAndIndices(Operation *op, RewriterBase &rewriter, Value newSrc,
                            ValueRange newSrcIndices, Value newDst,
                            ValueRange newDstIndices) const {
    auto copyOp = cast<GatherToLDSOp>(op);
    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(newSrc);
      copyOp.getSrcIndicesMutable().assign(newSrcIndices);
      copyOp.getDstMutable().assign(newDst);
      copyOp.getDstIndicesMutable().assign(newDstIndices);
    });
  }

  bool hasInboundsSrcIndices(Operation *op) const {
    MemRefType srcType = cast<GatherToLDSOp>(op).getSrc().getType();
    return !isFatRawBufferMemorySpace(srcType.getMemorySpace());
  }

  bool hasInboundsDstIndices(Operation *) const { return true; }
};

struct GlobalLoadAsyncToLDSCopy final
    : IndexedMemCopyOpInterface::ExternalModel<GlobalLoadAsyncToLDSCopy,
                                               GlobalLoadAsyncToLDSOp> {
  TypedValue<MemRefType> getSrc(Operation *op) const {
    return cast<TypedValue<MemRefType>>(
        cast<GlobalLoadAsyncToLDSOp>(op).getSrc());
  }

  Operation::operand_range getSrcIndices(Operation *op) const {
    return cast<GlobalLoadAsyncToLDSOp>(op).getSrcIndices();
  }

  TypedValue<MemRefType> getDst(Operation *op) const {
    return cast<TypedValue<MemRefType>>(
        cast<GlobalLoadAsyncToLDSOp>(op).getDst());
  }

  Operation::operand_range getDstIndices(Operation *op) const {
    return cast<GlobalLoadAsyncToLDSOp>(op).getDstIndices();
  }

  void setMemrefsAndIndices(Operation *op, RewriterBase &rewriter, Value newSrc,
                            ValueRange newSrcIndices, Value newDst,
                            ValueRange newDstIndices) const {
    auto copyOp = cast<GlobalLoadAsyncToLDSOp>(op);
    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(newSrc);
      copyOp.getSrcIndicesMutable().assign(newSrcIndices);
      copyOp.getDstMutable().assign(newDst);
      copyOp.getDstIndicesMutable().assign(newDstIndices);
    });
  }

  bool hasInboundsSrcIndices(Operation *) const { return true; }

  bool hasInboundsDstIndices(Operation *op) const {
    // Masked lanes may carry out-of-bounds destination indices; lowering
    // replaces their destination pointer with -1 before the instruction uses
    // it.
    return !cast<GlobalLoadAsyncToLDSOp>(op).getMask();
  }
};

template <typename OpTy>
struct DmaBaseCopy final
    : IndexedMemCopyOpInterface::ExternalModel<DmaBaseCopy<OpTy>, OpTy> {
  TypedValue<MemRefType> getSrc(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<OpTy>(op).getGlobal());
  }

  Operation::operand_range getSrcIndices(Operation *op) const {
    return cast<OpTy>(op).getGlobalIndices();
  }

  TypedValue<MemRefType> getDst(Operation *op) const {
    return cast<TypedValue<MemRefType>>(cast<OpTy>(op).getLds());
  }

  Operation::operand_range getDstIndices(Operation *op) const {
    return cast<OpTy>(op).getLdsIndices();
  }

  void setMemrefsAndIndices(Operation *op, RewriterBase &rewriter, Value newSrc,
                            ValueRange newSrcIndices, Value newDst,
                            ValueRange newDstIndices) const {
    auto copyOp = cast<OpTy>(op);
    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getGlobalMutable().assign(newSrc);
      copyOp.getGlobalIndicesMutable().assign(newSrcIndices);
      copyOp.getLdsMutable().assign(newDst);
      copyOp.getLdsIndicesMutable().assign(newDstIndices);
    });
  }

  bool hasInboundsSrcIndices(Operation *) const { return true; }

  bool hasInboundsDstIndices(Operation *) const { return true; }
};
} // namespace

void mlir::amdgpu::registerMemoryAccessOpInterfacesExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, amdgpu::AMDGPUDialect *) {
    TransposeLoadOp::attachInterface<TransposeLoadAccess<TransposeLoadOp>>(
        *ctx);
    GlobalTransposeLoadOp::attachInterface<
        TransposeLoadAccess<GlobalTransposeLoadOp>>(*ctx);
    MakeDmaDescriptorOp::attachInterface<
        DescriptorAtomicBarrierAccess<MakeDmaDescriptorOp>>(*ctx);
    MakeGatherDmaDescriptorOp::attachInterface<
        DescriptorAtomicBarrierAccess<MakeGatherDmaDescriptorOp>>(*ctx);
    DsBarrierInitOp::attachInterface<BaseAndIndicesAccess<DsBarrierInitOp>>(
        *ctx);
    DsBarrierPollStateOp::attachInterface<
        BaseAndIndicesAccess<DsBarrierPollStateOp>>(*ctx);
    DsAsyncBarrierArriveOp::attachInterface<
        BaseAndIndicesAccess<DsAsyncBarrierArriveOp>>(*ctx);
    DsBarrierArriveOp::attachInterface<BaseAndIndicesAccess<DsBarrierArriveOp>>(
        *ctx);
    GatherToLDSOp::attachInterface<GatherToLDSCopy>(*ctx);
    GlobalLoadAsyncToLDSOp::attachInterface<GlobalLoadAsyncToLDSCopy>(*ctx);
    MakeDmaBaseOp::attachInterface<DmaBaseCopy<MakeDmaBaseOp>>(*ctx);
    MakeGatherDmaBaseOp::attachInterface<DmaBaseCopy<MakeGatherDmaBaseOp>>(
        *ctx);
  });
}
