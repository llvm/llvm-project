//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::ml_program;

namespace mlir {
namespace ml_program {
namespace {

template <typename Interface, typename Op>
struct ExternalModelBase
    : public BufferizableOpInterface::ExternalModel<Interface, Op> {

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *, OpResult,
                                const AnalysisState &) const {
    return BufferRelation::Unknown;
  }
};

/// Bufferization of ml_program.global into a memref.global
struct GlobalOpInterface
    : public ExternalModelBase<GlobalOpInterface, GlobalOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  bool hasTensorSemantics(Operation *) const { return true; }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &) const {
    auto globalOp = cast<GlobalOp>(op);
    if (!globalOp.getValue().has_value())
      return globalOp.emitError("global op must have a value");

    auto tensorType = cast<TensorType>(globalOp.getType());
    auto memrefType = getMemRefTypeWithStaticIdentityLayout(tensorType);

    replaceOpWithNewBufferizedOp<memref::GlobalOp>(
        rewriter, globalOp, globalOp.getSymName(),
        /*sym_visibility=*/globalOp.getSymVisibilityAttr(),
        /*type=*/cast<MemRefType>(memrefType),
        /*initial_value=*/globalOp.getValue().value(),
        /*constant=*/!globalOp.getIsMutable(),
        /*alignment=*/nullptr);

    return success();
  }
};

/// Bufferization of ml_program.global_load into a memref.get_global
struct GlobalLoadOpInterface
    : public ExternalModelBase<GlobalLoadOpInterface, GlobalLoadOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  bool isWritable(Operation *, Value, const AnalysisState &) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &) const {
    auto globalLoadOp = cast<GlobalLoadOp>(op);

    auto tensorType = cast<TensorType>(globalLoadOp.getType());
    auto memrefType = getMemRefTypeWithStaticIdentityLayout(tensorType);

    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, globalLoadOp, memrefType,
        globalLoadOp.getGlobalAttr().getLeafReference());

    return success();
  }
};

/// Bufferization of ml_program.global_store into a memref.get_global and
/// memcpy
struct GlobalStoreOpInterface
    : public ExternalModelBase<GlobalStoreOpInterface, GlobalStoreOp> {

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto globalStoreOp = cast<GlobalStoreOp>(op);

    auto tensorType = cast<TensorType>(globalStoreOp.getValue().getType());
    auto memrefType = getMemRefTypeWithStaticIdentityLayout(tensorType);

    auto loc = globalStoreOp.getLoc();
    auto targetMemref = rewriter.create<memref::GetGlobalOp>(
        loc, memrefType, globalStoreOp.getGlobalAttr().getLeafReference());

    auto sourceMemref = getBuffer(rewriter, globalStoreOp.getValue(), options);
    if (failed(sourceMemref)) {
      return failure();
    }

    auto memcpy =
        options.createMemCpy(rewriter, loc, sourceMemref.value(), targetMemref);
    if (failed(memcpy)) {
      return failure();
    }
    rewriter.eraseOp(globalStoreOp);

    return success();
  }
};
} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, MLProgramDialect *) {
    GlobalOp::attachInterface<GlobalOpInterface>(*ctx);
    GlobalLoadOp::attachInterface<GlobalLoadOpInterface>(*ctx);
    GlobalStoreOp::attachInterface<GlobalStoreOpInterface>(*ctx);
  });
}
} // namespace ml_program
} // namespace mlir
