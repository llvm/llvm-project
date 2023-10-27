//===- PDLExtension.cpp - PDL extension for the Transform dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/PDLExtension/PDLExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
/// Implementation of the TransformHandleTypeInterface for the PDL
/// OperationType. Accepts any payload operation.
struct PDLOperationTypeTransformHandleTypeInterfaceImpl
    : public transform::TransformHandleTypeInterface::ExternalModel<
          PDLOperationTypeTransformHandleTypeInterfaceImpl,
          pdl::OperationType> {

  /// Accept any operation.
  DiagnosedSilenceableFailure
  checkPayload(Type type, Location loc, ArrayRef<Operation *> payload) const {
    return DiagnosedSilenceableFailure::success();
  }
};
} // namespace

namespace {
/// PDL extension of the Transform dialect. This provides transform operations
/// that connect to PDL matching as well as interfaces for PDL types to be used
/// with Transform dialect operations.
class PDLExtension : public transform::TransformDialectExtension<PDLExtension> {
public:
  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.cpp.inc"
        >();

    addDialectDataInitializer<transform::PDLMatchHooks>(
        [](transform::PDLMatchHooks &) {});

    // Declare PDL as dependent so we can attach an interface to its type in the
    // later step.
    declareDependentDialect<pdl::PDLDialect>();

    // PDLInterp is only relevant if we actually apply the transform IR so
    // declare it as generated.
    declareGeneratedDialect<pdl_interp::PDLInterpDialect>();

    // Make PDL OperationType usable as a transform dialect type.
    addCustomInitializationStep([](MLIRContext *context) {
      pdl::OperationType::attachInterface<
          PDLOperationTypeTransformHandleTypeInterfaceImpl>(*context);
    });
  }
};
} // namespace

void mlir::transform::registerPDLExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<PDLExtension>();
}
