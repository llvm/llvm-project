//===- TransformDialect.cpp - Transform Dialect Definition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

#include "mlir/Dialect/Transform/IR/TransformDialect.cpp.inc"

#ifndef NDEBUG
void transform::detail::checkImplementsTransformOpInterface(
    StringRef name, MLIRContext *context) {
  // Since the operation is being inserted into the Transform dialect and the
  // dialect does not implement the interface fallback, only check for the op
  // itself having the interface implementation.
  RegisteredOperationName opName =
      *RegisteredOperationName::lookup(name, context);
  assert((opName.hasInterface<TransformOpInterface>() ||
          opName.hasTrait<OpTrait::IsTerminator>()) &&
         "non-terminator ops injected into the transform dialect must "
         "implement TransformOpInterface");
  assert(opName.hasInterface<MemoryEffectOpInterface>() &&
         "ops injected into the transform dialect must implement "
         "MemoryEffectsOpInterface");
}

void transform::detail::checkImplementsTransformTypeInterface(
    TypeID typeID, MLIRContext *context) {
  const auto &abstractType = AbstractType::lookup(typeID, context);
  assert(abstractType.hasInterface(TransformTypeInterface::getInterfaceID()));
}
#endif // NDEBUG

namespace {
struct PDLOperationTypeTransformTypeInterfaceImpl
    : public transform::TransformTypeInterface::ExternalModel<
          PDLOperationTypeTransformTypeInterfaceImpl, pdl::OperationType> {
  DiagnosedSilenceableFailure
  checkPayload(Type type, Location loc, ArrayRef<Operation *> payload) const {
    return DiagnosedSilenceableFailure::success();
  }
};
} // namespace

void transform::TransformDialect::initialize() {
  // Using the checked versions to enable the same assertions as for the ops
  // from extensions.
  addOperationsChecked<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"
      >();
  initializeTypes();

  pdl::OperationType::attachInterface<
      PDLOperationTypeTransformTypeInterfaceImpl>(*getContext());
}

void transform::TransformDialect::mergeInPDLMatchHooks(
    llvm::StringMap<PDLConstraintFunction> &&constraintFns) {
  // Steal the constraint functions form the given map.
  for (auto &it : constraintFns)
    pdlMatchHooks.registerConstraintFunction(it.getKey(), std::move(it.second));
}

const llvm::StringMap<PDLConstraintFunction> &
transform::TransformDialect::getPDLConstraintHooks() const {
  return pdlMatchHooks.getConstraintFunctions();
}

Type transform::TransformDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&keyword)))
    return nullptr;

  auto it = typeParsingHooks.find(keyword);
  if (it == typeParsingHooks.end()) {
    parser.emitError(loc) << "unknown type mnemonic: " << keyword;
    return nullptr;
  }

  return it->getValue()(parser);
}

void transform::TransformDialect::printType(Type type,
                                            DialectAsmPrinter &printer) const {
  auto it = typePrintingHooks.find(type.getTypeID());
  assert(it != typePrintingHooks.end() && "printing unknown type");
  it->getSecond()(type, printer);
}

void transform::TransformDialect::reportDuplicateTypeRegistration(
    StringRef mnemonic) {
  std::string buffer;
  llvm::raw_string_ostream msg(buffer);
  msg << "extensible dialect type '" << mnemonic
      << "' is already registered with a different implementation";
  msg.flush();
  llvm::report_fatal_error(StringRef(buffer));
}

void transform::TransformDialect::reportDuplicateOpRegistration(
    StringRef opName) {
  std::string buffer;
  llvm::raw_string_ostream msg(buffer);
  msg << "extensible dialect operation '" << opName
      << "' is already registered with a mismatching TypeID";
  msg.flush();
  llvm::report_fatal_error(StringRef(buffer));
}

#include "mlir/Dialect/Transform/IR/TransformDialectEnums.cpp.inc"
