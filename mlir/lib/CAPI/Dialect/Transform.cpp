//===- Transform.cpp - C Interface for Transform dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Transform.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Dialect/Transform.h"
#include "mlir/CAPI/Interfaces.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Transform, transform,
                                      transform::TransformDialect)

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyOpType(MlirType type) {
  return isa<transform::AnyOpType>(unwrap(type));
}

MlirTypeID mlirTransformAnyOpTypeGetTypeID(void) {
  return wrap(transform::AnyOpType::getTypeID());
}

MlirType mlirTransformAnyOpTypeGet(MlirContext ctx) {
  return wrap(transform::AnyOpType::get(unwrap(ctx)));
}

MlirStringRef mlirTransformAnyOpTypeGetName(void) {
  return wrap(transform::AnyOpType::name);
}

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyParamType(MlirType type) {
  return isa<transform::AnyParamType>(unwrap(type));
}

MlirTypeID mlirTransformAnyParamTypeGetTypeID(void) {
  return wrap(transform::AnyParamType::getTypeID());
}

MlirType mlirTransformAnyParamTypeGet(MlirContext ctx) {
  return wrap(transform::AnyParamType::get(unwrap(ctx)));
}

MlirStringRef mlirTransformAnyParamTypeGetName(void) {
  return wrap(transform::AnyParamType::name);
}

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformAnyValueType(MlirType type) {
  return isa<transform::AnyValueType>(unwrap(type));
}

MlirTypeID mlirTransformAnyValueTypeGetTypeID(void) {
  return wrap(transform::AnyValueType::getTypeID());
}

MlirType mlirTransformAnyValueTypeGet(MlirContext ctx) {
  return wrap(transform::AnyValueType::get(unwrap(ctx)));
}

MlirStringRef mlirTransformAnyValueTypeGetName(void) {
  return wrap(transform::AnyValueType::name);
}

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformOperationType(MlirType type) {
  return isa<transform::OperationType>(unwrap(type));
}

MlirTypeID mlirTransformOperationTypeGetTypeID(void) {
  return wrap(transform::OperationType::getTypeID());
}

MlirType mlirTransformOperationTypeGet(MlirContext ctx,
                                       MlirStringRef operationName) {
  return wrap(
      transform::OperationType::get(unwrap(ctx), unwrap(operationName)));
}

MlirStringRef mlirTransformOperationTypeGetName(void) {
  return wrap(transform::OperationType::name);
}

MlirStringRef mlirTransformOperationTypeGetOperationName(MlirType type) {
  return wrap(cast<transform::OperationType>(unwrap(type)).getOperationName());
}

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

bool mlirTypeIsATransformParamType(MlirType type) {
  return isa<transform::ParamType>(unwrap(type));
}

MlirTypeID mlirTransformParamTypeGetTypeID(void) {
  return wrap(transform::ParamType::getTypeID());
}

MlirType mlirTransformParamTypeGet(MlirContext ctx, MlirType type) {
  return wrap(transform::ParamType::get(unwrap(ctx), unwrap(type)));
}

MlirStringRef mlirTransformParamTypeGetName(void) {
  return wrap(transform::ParamType::name);
}

MlirType mlirTransformParamTypeGetType(MlirType type) {
  return wrap(cast<transform::ParamType>(unwrap(type)).getType());
}

//===---------------------------------------------------------------------===//
// TransformRewriter
//===---------------------------------------------------------------------===//

/// Casts a `MlirTransformRewriter` to a `MlirRewriterBase`.
MlirRewriterBase mlirTransformRewriterAsBase(MlirTransformRewriter rewriter) {
  mlir::transform::TransformRewriter *t = unwrap(rewriter);
  mlir::RewriterBase *base = static_cast<mlir::RewriterBase *>(t);
  return wrap(base);
}

//===---------------------------------------------------------------------===//
// TransformResults
//===---------------------------------------------------------------------===//

void mlirTransformResultsSetOps(MlirTransformResults results, MlirValue result,
                                intptr_t numOps, MlirOperation *ops) {
  SmallVector<Operation *> opsVec;
  opsVec.reserve(numOps);
  for (intptr_t i = 0; i < numOps; ++i)
    opsVec.push_back(unwrap(ops[i]));
  unwrap(results)->set(cast<OpResult>(unwrap(result)), opsVec);
}

void mlirTransformResultsSetValues(MlirTransformResults results,
                                   MlirValue result, intptr_t numValues,
                                   MlirValue *values) {
  SmallVector<Value> valuesVec;
  valuesVec.reserve(numValues);
  for (intptr_t i = 0; i < numValues; ++i)
    valuesVec.push_back(unwrap(values[i]));
  unwrap(results)->setValues(cast<OpResult>(unwrap(result)), valuesVec);
}

void mlirTransformResultsSetParams(MlirTransformResults results,
                                   MlirValue result, intptr_t numParams,
                                   MlirAttribute *params) {
  SmallVector<Attribute> paramsVec;
  paramsVec.reserve(numParams);
  for (intptr_t i = 0; i < numParams; ++i)
    paramsVec.push_back(unwrap(params[i]));
  unwrap(results)->setParams(cast<OpResult>(unwrap(result)), paramsVec);
}

//===---------------------------------------------------------------------===//
// TransformState
//===---------------------------------------------------------------------===//

void mlirTransformStateForEachPayloadOp(MlirTransformState state,
                                        MlirValue value,
                                        MlirOperationCallback callback,
                                        void *userData) {
  for (Operation *op : unwrap(state)->getPayloadOps(unwrap(value)))
    callback(wrap(op), userData);
}

void mlirTransformStateForEachPayloadValue(MlirTransformState state,
                                           MlirValue value,
                                           MlirValueCallback callback,
                                           void *userData) {
  for (Value val : unwrap(state)->getPayloadValues(unwrap(value)))
    callback(wrap(val), userData);
}

void mlirTransformStateForEachParam(MlirTransformState state, MlirValue value,
                                    MlirAttributeCallback callback,
                                    void *userData) {
  for (Attribute attr : unwrap(state)->getParams(unwrap(value)))
    callback(wrap(attr), userData);
}

//===---------------------------------------------------------------------===//
// TransformOpInterface
//===---------------------------------------------------------------------===//

MlirTypeID mlirTransformOpInterfaceTypeID(void) {
  return wrap(transform::TransformOpInterface::getInterfaceID());
}

/// Fallback model for the TransformOpInterface that uses C API callbacks.
class TransformOpInterfaceFallbackModel
    : public mlir::transform::TransformOpInterface::FallbackModel<
          TransformOpInterfaceFallbackModel> {
public:
  /// Sets the callbacks that this FallbackModel will use.
  /// NB: the callbacks can only be set through this method as the
  /// RegisteredOperationName::attachInterface mechanism default-constructs
  /// the FallbackModel without being able to provide arguments.
  void setCallbacks(MlirTransformOpInterfaceCallbacks callbacks) {
    this->callbacks = callbacks;
  }

  ~TransformOpInterfaceFallbackModel() {
    if (callbacks.destruct)
      callbacks.destruct(callbacks.userData);
  }

  static TypeID getInterfaceID() {
    return transform::TransformOpInterface::getInterfaceID();
  }

  static bool classof(const mlir::transform::detail::
                          TransformOpInterfaceInterfaceTraits::Concept *op) {
    // Enable casting back to the FallbackModel from the Interface. This is
    // necessary as attachInterface(...) default-constructs the FallbackModel
    // without being able to pass in the callbacks and returns just the Concept.
    return true;
  }

  ::mlir::DiagnosedSilenceableFailure
  apply(Operation *op, ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state) const {
    assert(callbacks.apply && "apply callback not set");

    MlirDiagnosedSilenceableFailure status =
        callbacks.apply(wrap(op), wrap(&rewriter), wrap(&transformResults),
                        wrap(&state), callbacks.userData);

    switch (status) {
    case MlirDiagnosedSilenceableFailureSuccess:
      return DiagnosedSilenceableFailure::success();
    case MlirDiagnosedSilenceableFailureSilenceableFailure:
      // TODO: enable passing diagnostic info from C API to C++ API.
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(
          *(op->emitError()
            << "TransformOpInterfaceFallbackModel: silenceable failure")
               .getUnderlyingDiagnostic()));
    case MlirDiagnosedSilenceableFailureDefiniteFailure:
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    llvm_unreachable("unknown transform status");
  }

  bool allowsRepeatedHandleOperands(Operation *op) const {
    assert(callbacks.allowsRepeatedHandleOperands &&
           "allowsRepeatedHandleOperands callback not set");
    return callbacks.allowsRepeatedHandleOperands(wrap(op), callbacks.userData);
  }

private:
  MlirTransformOpInterfaceCallbacks callbacks;
};

/// Attach a TransformOpInterface FallbackModel to the given named operation.
/// The FallbackModel uses the provided callbacks to implement the interface.
void mlirTransformOpInterfaceAttachFallbackModel(
    MlirContext ctx, MlirStringRef opName,
    MlirTransformOpInterfaceCallbacks callbacks) {
  // Look up the operation definition in the context.
  std::optional<RegisteredOperationName> opInfo =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(ctx));

  assert(opInfo.has_value() && "operation not found in context");

  // NB: the following default-constructs the FallbackModel _without_ being able
  // to provide arguments.
  opInfo->attachInterface<TransformOpInterfaceFallbackModel>();
  // Cast to get the underlying FallbackModel and set the callbacks.
  auto *model = cast<TransformOpInterfaceFallbackModel>(
      opInfo->getInterface<TransformOpInterfaceFallbackModel>());

  assert(model && "Failed to get TransformOpInterfaceFallbackModel");
  model->setCallbacks(callbacks);
}

//===---------------------------------------------------------------------===//
// MemoryEffectsOpInterface helpers
//===---------------------------------------------------------------------===//

/// Set the effect for the operands to only read the transform handles.
void mlirTransformOnlyReadsHandle(MlirOpOperand *operands, intptr_t numOperands,
                                  MlirMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> operandArray(unwrap(*operands), numOperands);
  transform::onlyReadsHandle(operandArray, *unwrap(effects));
}

/// Set the effect for the operands to consuming the transform handles.
void mlirTransformConsumesHandle(MlirOpOperand *operands, intptr_t numOperands,
                                 MlirMemoryEffectInstancesList effects) {
  MutableArrayRef<OpOperand> operandArray(unwrap(*operands), numOperands);
  transform::consumesHandle(operandArray, *unwrap(effects));
}

/// Set the effect for the results to that they produce transform handles.
void mlirTransformProducesHandle(MlirValue *results, intptr_t numResults,
                                 MlirMemoryEffectInstancesList effects) {
  // NB: calling `producesHandle()` `numResults` as we cannot cast array of
  // `OpResult`s to a single `ResultRange` (and neither is `ResultRange` exposed
  // to Python). `producesHandle` iterates over the given `ResultRange` anyway.
  SmallVectorImpl<MemoryEffects::EffectInstance> &effectList = *unwrap(effects);
  for (intptr_t i = 0; i < numResults; ++i) {
    auto opResult = cast<OpResult>(unwrap(results[i]));
    transform::producesHandle(ResultRange(opResult), effectList);
  }
}

/// Set the effect of potentially modifying payload IR.
void mlirTransformModifiesPayload(MlirMemoryEffectInstancesList effects) {
  transform::modifiesPayload(*unwrap(effects));
}

/// Set the effect of potentially reading payload IR.
void mlirTransformOnlyReadsPayload(MlirMemoryEffectInstancesList effects) {
  transform::onlyReadsPayload(*unwrap(effects));
}
