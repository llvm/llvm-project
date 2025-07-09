//===- TransformDialect.cpp - Transform Dialect Definition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/IR/Utils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/Transform/IR/TransformDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Transform/IR/TransformAttrs.cpp.inc"

#ifndef NDEBUG
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
          opName.hasTrait<OpTrait::IsTerminator>()) &&
         "non-terminator ops injected into the transform dialect must "
         "implement TransformOpInterface or PatternDescriptorOpInterface or "
         "ConversionPatternDescriptorOpInterface");
  if (!opName.hasInterface<PatternDescriptorOpInterface>() &&
      !opName.hasInterface<ConversionPatternDescriptorOpInterface>() &&
      !opName.hasInterface<TypeConverterBuilderOpInterface>()) {
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
#endif // NDEBUG

void transform::TransformDialect::initialize() {
  // Using the checked versions to enable the same assertions as for the ops
  // from extensions.
  addOperationsChecked<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"
      >();
  initializeTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Transform/IR/TransformAttrs.cpp.inc"
      >();
  initializeLibraryModule();
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

    const mlir::CallGraph callgraph(op);
    for (auto scc = llvm::scc_begin(&callgraph); !scc.isAtEnd(); ++scc) {
      if (!scc.hasCycle())
        continue;

      // Need to check this here additionally because this verification may run
      // before we check the nested operations.
      if ((*scc->begin())->isExternal())
        return op->emitOpError() << "contains a call to an external operation, "
                                    "which is not allowed";

      Operation *first = (*scc->begin())->getCallableRegion()->getParentOp();
      InFlightDiagnostic diag = emitError(first->getLoc())
                                << "recursion not allowed in named sequences";
      for (auto it = std::next(scc->begin()); it != scc->end(); ++it) {
        // Need to check this here additionally because this verification may
        // run before we check the nested operations.
        if ((*it)->isExternal()) {
          return op->emitOpError() << "contains a call to an external "
                                      "operation, which is not allowed";
        }

        Operation *current = (*it)->getCallableRegion()->getParentOp();
        diag.attachNote(current->getLoc()) << "operation on recursion stack";
      }
      return diag;
    }
    return success();
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
