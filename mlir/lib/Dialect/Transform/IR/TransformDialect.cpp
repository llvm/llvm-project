//===- TransformDialect.cpp - Transform Dialect Definition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;

#include "mlir/Dialect/Transform/IR/TransformDialect.cpp.inc"

namespace {
/// This interface enables inlining of `transform.named_sequence` operations
/// into the body of other `transform.named_sequence` operations. The dialect
/// does not allow inlining into any other context.
struct TransformInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// A call may be inlined when its callee is a `transform.named_sequence`.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return isa<transform::NamedSequenceOp>(callable);
  }

  /// A region may be inlined into another region only when both are bodies of
  /// `transform.named_sequence` operations: this restricts inlining to the
  /// "named sequence into named sequence" case.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return isa_and_nonnull<transform::NamedSequenceOp>(dest->getParentOp()) &&
           isa_and_nonnull<transform::NamedSequenceOp>(src->getParentOp());
  }

  /// Any operation is legal to inline into the body of a
  /// `transform.named_sequence`. Whether a particular operation is actually
  /// valid in that context is enforced by the regular op verifiers.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return isa_and_nonnull<transform::NamedSequenceOp>(dest->getParentOp());
  }

  /// Replace the `transform.yield` terminator of an inlined single-block
  /// region by directly forwarding its operands to the values that used to be
  /// produced by the call site.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto yieldOp = cast<transform::YieldOp>(op);
    assert(yieldOp.getNumOperands() == valuesToRepl.size() &&
           "mismatched yield/call result count");
    for (auto [from, to] : llvm::zip(valuesToRepl, yieldOp.getOperands()))
      from.replaceAllUsesWith(to);
  }
};
} // namespace

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
void transform::detail::checkImplementsTransformOpInterface(
    StringRef name, MLIRContext *context) {
  // Since the operation is being inserted into the Transform dialect and the
  // dialect does not implement the interface fallback, only check for the op
  // itself having the interface implementation.
  RegisteredOperationName opName =
      *RegisteredOperationName::lookup(name, context);
  assert((opName.hasInterface<TransformOpInterface>() ||
          opName.hasInterface<PatternDescriptorOpInterface>() ||
          opName.hasInterface<ConversionPatternDescriptorOpInterface>() ||
          opName.hasInterface<TypeConverterBuilderOpInterface>() ||
          opName.hasTrait<OpTrait::IsTerminator>() ||
          opName.hasInterface<NormalFormCheckedOpInterface>()) &&
         "non-terminator ops injected into the transform dialect must "
         "implement TransformOpInterface or PatternDescriptorOpInterface or "
         "ConversionPatternDescriptorOpInterface");
  if (!opName.hasInterface<PatternDescriptorOpInterface>() &&
      !opName.hasInterface<ConversionPatternDescriptorOpInterface>() &&
      !opName.hasInterface<TypeConverterBuilderOpInterface>() &&
      !opName.hasInterface<NormalFormCheckedOpInterface>()) {
    assert(opName.hasInterface<MemoryEffectOpInterface>() &&
           "ops injected into the transform dialect must implement "
           "MemoryEffectsOpInterface");
  }
}

void transform::detail::checkImplementsTransformHandleTypeInterface(
    TypeID typeID, MLIRContext *context) {
  const auto &abstractType = AbstractType::lookup(typeID, context);
  assert((abstractType.hasInterface(
              TransformHandleTypeInterface::getInterfaceID()) ||
          abstractType.hasInterface(
              TransformParamTypeInterface::getInterfaceID()) ||
          abstractType.hasInterface(
              TransformValueHandleTypeInterface::getInterfaceID())) &&
         "expected Transform dialect type to implement one of the three "
         "interfaces");
}
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

void transform::TransformDialect::initialize() {
  // Using the checked versions to enable the same assertions as for the ops
  // from extensions.
  addOperationsChecked<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"
      >();
  initializeAttributes();
  initializeTypes();
  initializeLibraryModule();
  addInterfaces<TransformInlinerInterface>();
}

Attribute transform::TransformDialect::parseAttribute(DialectAsmParser &parser,
                                                      Type type) const {
  StringRef keyword;
  SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeyword(&keyword)))
    return nullptr;

  auto it = attributeParsingHooks.find(keyword);
  if (it == attributeParsingHooks.end()) {
    parser.emitError(loc) << "unknown attribute mnemonic: " << keyword;
    return nullptr;
  }

  return it->getValue()(parser, type);
}

void transform::TransformDialect::printAttribute(
    Attribute attribute, DialectAsmPrinter &printer) const {
  auto it = attributePrintingHooks.find(attribute.getTypeID());
  assert(it != attributePrintingHooks.end() && "printing unknown attribute");
  it->getSecond()(attribute, printer);
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

LogicalResult transform::TransformDialect::loadIntoLibraryModule(
    ::mlir::OwningOpRef<::mlir::ModuleOp> &&library) {
  return detail::mergeSymbolsInto(getLibraryModule(), std::move(library));
}

void transform::TransformDialect::initializeLibraryModule() {
  MLIRContext *context = getContext();
  auto loc =
      FileLineColLoc::get(context, "<transform-dialect-library-module>", 0, 0);
  libraryModule = ModuleOp::create(loc, "__transform_library");
  libraryModule.get()->setAttr(TransformDialect::kWithNamedSequenceAttrName,
                               UnitAttr::get(context));
}

void transform::TransformDialect::reportDuplicateAttributeRegistration(
    StringRef attrName) {
  std::string buffer;
  llvm::raw_string_ostream msg(buffer);
  msg << "extensible dialect attribute '" << attrName
      << "' is already registered with a different implementation";
  llvm::report_fatal_error(StringRef(buffer));
}

void transform::TransformDialect::reportDuplicateTypeRegistration(
    StringRef mnemonic) {
  std::string buffer;
  llvm::raw_string_ostream msg(buffer);
  msg << "extensible dialect type '" << mnemonic
      << "' is already registered with a different implementation";
  llvm::report_fatal_error(StringRef(buffer));
}

void transform::TransformDialect::reportDuplicateOpRegistration(
    StringRef opName) {
  std::string buffer;
  llvm::raw_string_ostream msg(buffer);
  msg << "extensible dialect operation '" << opName
      << "' is already registered with a mismatching TypeID";
  llvm::report_fatal_error(StringRef(buffer));
}

LogicalResult transform::TransformDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute attribute) {
  if (attribute.getName().getValue() == kWithNamedSequenceAttrName) {
    if (!op->hasTrait<OpTrait::SymbolTable>()) {
      return emitError(op->getLoc()) << attribute.getName()
                                     << " attribute can only be attached to "
                                        "operations with symbol tables";
    }

    // Pre-verify calls and callables because call graph construction below
    // assumes they are valid, but this verifier runs before verifying the
    // nested operations.
    WalkResult walkResult = op->walk([](Operation *nested) {
      if (!isa<CallableOpInterface, CallOpInterface>(nested))
        return WalkResult::advance();

      if (failed(verify(nested, /*verifyRecursively=*/false)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();

    return detail::verifyNoRecursionInCallGraph(op);
  }
  if (attribute.getName().getValue() == kTargetTagAttrName) {
    if (!llvm::isa<StringAttr>(attribute.getValue())) {
      return op->emitError()
             << attribute.getName() << " attribute must be a string";
    }
    return success();
  }
  if (attribute.getName().getValue() == kArgConsumedAttrName ||
      attribute.getName().getValue() == kArgReadOnlyAttrName) {
    if (!llvm::isa<UnitAttr>(attribute.getValue())) {
      return op->emitError()
             << attribute.getName() << " must be a unit attribute";
    }
    return success();
  }
  if (attribute.getName().getValue() ==
      FindPayloadReplacementOpInterface::kSilenceTrackingFailuresAttrName) {
    if (!llvm::isa<UnitAttr>(attribute.getValue())) {
      return op->emitError()
             << attribute.getName() << " must be a unit attribute";
    }
    return success();
  }
  return emitError(op->getLoc())
         << "unknown attribute: " << attribute.getName();
}
