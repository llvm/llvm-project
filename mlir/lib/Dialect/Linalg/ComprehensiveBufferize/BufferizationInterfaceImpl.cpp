//===- BufferizationInterfaceImpl.cpp - Bufferization Impl. of Interface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizationInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace comprehensive_bufferize;

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace bufferization_ext {

// TODO: These ops should implement BufferizableOpInterface directly when moved
// to the Bufferization dialect.

/// Bufferization of bufferization.to_memref. to_memref(to_tensor(x)) is folded
/// to x. Other to_memref ops are ignored during bufferization.
///
/// ToMemrefOp casts a tensor into a memref. The resulting memref is the memory
/// location of the incoming tensor once it will be bufferized. In the anlysis,
/// the incoming tensor is assumed to bufferize to a memory read and to an
/// inplace memory write, since it is unknown what will happen to the resulting
/// memref.
///
/// Note: ToMemrefOp / ToTensorOp are temporary ops that are inserted at the
/// bufferization boundary. When bufferization is complete, there should be no
/// such ops left over. If `allowUnknownOps`, such ops may be part of the
/// resulting IR, but such IR may no longer be bufferizable by Comprehensive
/// Bufferize.
struct ToMemrefOpInterface
    : public BufferizableOpInterface::ExternalModel<ToMemrefOpInterface,
                                                    bufferization::ToMemrefOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    // It is unknown whether the resulting memref will be read or not.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    // It is unknown whether the resulting MemRef will be written or not.
    return true;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const BufferizationState &state) const {
    // ToMemrefOps always bufferize inplace.
    // TODO: Remove ToMemrefOps from the analysis.
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto toMemrefOp = cast<bufferization::ToMemrefOp>(op);

    // Fold to_memref(to_tensor(x)) to x. Insert a cast if necessary.
    if (auto toTensorOp =
            toMemrefOp.tensor().getDefiningOp<bufferization::ToTensorOp>()) {
      Value buffer = toTensorOp.memref();

      // Insert cast in case to_memref(to_tensor(x))'s type is different from
      // x's type.
      if (toTensorOp.memref().getType() != toMemrefOp.getType())
        buffer = rewriter.create<memref::CastOp>(toMemrefOp.getLoc(), buffer,
                                                 toMemrefOp.getType());
      replaceOpWithBufferizedValues(rewriter, toMemrefOp, buffer);
      return success();
    }

    return failure();
  }
};

/// Bufferization of bufferization.to_tensor. Such ops cannot be bufferized.
/// However, other ops that are using to_tensor's result will eventually be
/// bufferized. At that point, they will start using to_tensor's memref operand.
/// Once all users of to_tensor are bufferized, the op will not have any users
/// anymore and DCE away.
///
/// ToTensorOp conceptually loads a tensor from a memory location. The analysis
/// has no information about the memref that is loaded from by ToTensorOp. We
/// have to assume that the loaded tensor may after bufferization potentially
/// alias with any other bufferized tensor. Since ToTensorOp and ToMemrefOp have
/// no aliasing OpOperand/OpResult pairs, this cannot be encoded directly in the
/// analysis. However, declaring ToTensorOp results as not writable enforces a
/// buffer copy and has the same effect.
struct ToTensorOpInterface
    : public BufferizableOpInterface::ExternalModel<ToTensorOpInterface,
                                                    bufferization::ToTensorOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    return failure();
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    // It is unknown whether the memref operand is writable or not.
    return false;
  }
};

} // namespace bufferization_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::bufferization_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<bufferization::ToMemrefOp,
                          bufferization_ext::ToMemrefOpInterface>();
  registry.addOpInterface<bufferization::ToTensorOp,
                          bufferization_ext::ToTensorOpInterface>();
}
