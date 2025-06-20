//===- IRDLExtensionOps.cpp - IRDL extension for the Transform dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.cpp.inc"

namespace mlir::transform {

DiagnosedSilenceableFailure
IRDLCollectMatchingOp::apply(TransformRewriter &rewriter,
                             TransformResults &results, TransformState &state) {
  auto dialect = cast<irdl::DialectOp>(getBody().front().front());
  Block &body = dialect.getBody().front();
  irdl::OperationOp operation = *body.getOps<irdl::OperationOp>().begin();
  auto verifier = irdl::createVerifier(
      operation,
      DenseMap<irdl::TypeOp, std::unique_ptr<DynamicTypeDefinition>>(),
      DenseMap<irdl::AttributeOp, std::unique_ptr<DynamicAttrDefinition>>());

  auto handlerID = getContext()->getDiagEngine().registerHandler(
      [](Diagnostic &) { return success(); });
  SmallVector<Operation *> matched;
  for (Operation *payload : state.getPayloadOps(getRoot())) {
    payload->walk([&](Operation *target) {
      if (succeeded(verifier(target))) {
        matched.push_back(target);
      }
    });
  }
  getContext()->getDiagEngine().eraseHandler(handlerID);
  results.set(cast<OpResult>(getMatched()), matched);
  return DiagnosedSilenceableFailure::success();
}

void IRDLCollectMatchingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRootMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

LogicalResult IRDLCollectMatchingOp::verify() {
  Block &bodyBlock = getBody().front();
  if (!llvm::hasSingleElement(bodyBlock))
    return emitOpError() << "expects a single operation in the body";

  auto dialect = dyn_cast<irdl::DialectOp>(bodyBlock.front());
  if (!dialect) {
    return emitOpError() << "expects the body operation to be "
                         << irdl::DialectOp::getOperationName();
  }

  // TODO: relax this by taking a symbol name of the operation to match, note
  // that symbol name is also the name of the operation and we may want to
  // divert from that to have constraints on-the-fly using IRDL.
  auto irdlOperations = dialect.getOps<irdl::OperationOp>();
  if (!llvm::hasSingleElement(irdlOperations))
    return emitOpError() << "expects IRDL to contain exactly one operation";

  if (!dialect.getOps<irdl::TypeOp>().empty() ||
      !dialect.getOps<irdl::AttributeOp>().empty()) {
    return emitOpError() << "IRDL types and attributes are not yet supported";
  }

  return success();
}

} // namespace mlir::transform
