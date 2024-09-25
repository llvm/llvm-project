//===-- MyExtension.cpp - Transform dialect tutorial ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 2 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#include "MyExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

// Define a new transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class MyExtension
    : public ::mlir::transform::TransformDialectExtension<MyExtension> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyExtension)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void MyExtension::init() {
  // Similarly to dialects, an extension can declare a dependent dialect. This
  // dialect will be loaded along with the extension and, therefore, along with
  // the Transform dialect. Only declare as dependent the dialects that contain
  // the attributes or types used by transform operations. Do NOT declare as
  // dependent the dialects produced during the transformation.
  // declareDependentDialect<MyDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  // In the following chapter, we will be add operations that generate function
  // calls and structured control flow operations, so let's declare the
  // corresponding dialects as generated.
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "MyExtension.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "MyExtension.cpp.inc"

static void updateCallee(mlir::func::CallOp call, llvm::StringRef newTarget) {
  call.setCallee(newTarget);
}

// Implementation of our transform dialect operation.
// This operation returns a tri-state result that can be one of:
// - success when the transformation succeeded;
// - definite failure when the transformation failed in such a way that
//   following transformations are impossible or undesirable, typically it could
//   have left payload IR in an invalid state; it is expected that a diagnostic
//   is emitted immediately before returning the definite error;
// - silenceable failure when the transformation failed but following
//   transformations are still applicable, typically this means a precondition
//   for the transformation is not satisfied and the payload IR has not been
//   modified. The silenceable failure additionally carries a Diagnostic that
//   can be emitted to the user.
::mlir::DiagnosedSilenceableFailure mlir::transform::ChangeCallTargetOp::apply(
    // The rewriter that should be used when modifying IR.
    ::mlir::transform::TransformRewriter &rewriter,
    // The list of payload IR entities that will be associated with the
    // transform IR values defined by this transform operation. In this case, it
    // can remain empty as there are no results.
    ::mlir::transform::TransformResults &results,
    // The transform application state. This object can be used to query the
    // current associations between transform IR values and payload IR entities.
    // It can also carry additional user-defined state.
    ::mlir::transform::TransformState &state) {

  // First, we need to obtain the list of payload operations that are associated
  // with the operand handle.
  auto payload = state.getPayloadOps(getCall());

  // Then, we iterate over the list of operands and call the actual IR-mutating
  // function. We also check the preconditions here.
  for (Operation *payloadOp : payload) {
    auto call = dyn_cast<::mlir::func::CallOp>(payloadOp);
    if (!call) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "only applies to func.call payloads";
      diag.attachNote(payloadOp->getLoc()) << "offending payload";
      return diag;
    }

    updateCallee(call, getNewTarget());
  }

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::ChangeCallTargetOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  // Indicate that the `call` handle is only read by this operation because the
  // associated operation is not erased but rather modified in-place, so the
  // reference to it remains valid.
  onlyReadsHandle(getCallMutable(), effects);

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}

void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
