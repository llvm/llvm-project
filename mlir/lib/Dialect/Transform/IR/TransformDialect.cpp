//===- TransformDialect.cpp - Transform Dialect Definition ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

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

/// A handle used to reference external elements instances.
using TransformDialectResourceBlobHandle =
    mlir::DialectResourceBlobHandle<mlir::transform::TransformDialect>;

struct TransformResourceBlobManagerInterface
    : public ResourceBlobManagerDialectInterfaceBase<
          TransformDialectResourceBlobHandle> {
  using ResourceBlobManagerDialectInterfaceBase<
      TransformDialectResourceBlobHandle>::
      ResourceBlobManagerDialectInterfaceBase;
};

struct TransformOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  TransformOpAsmInterface(Dialect *dialect,
                          TransformResourceBlobManagerInterface &mgr)
      : OpAsmDialectInterface(dialect), blobManager(mgr) {}

  ~TransformOpAsmInterface() override {
    for (auto op : orderedLibraryModules)
      op->erase();
  }

  //===------------------------------------------------------------------===//
  // Resources
  //===------------------------------------------------------------------===//

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<TransformDialectResourceBlobHandle>(handle).getKey().str();
  }

  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobManager.insert(key);
  }

  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    // If its a string, then treat it as a filename.
    // TODO: Could be extended for blob resources where the file is encoded.
    if (entry.getKind() != AsmResourceEntryKind::String)
      return failure();

    FailureOr<std::string> name = entry.parseAsString();
    if (failed(name))
      return failure();

    auto fileOr = llvm::MemoryBuffer::getFile(*name);
    if (!fileOr) {
      return entry.emitError()
             << "failed to load resource: " << fileOr.getError().message();
    }

    // Parse the module in the source file.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(fileOr.get()), llvm::SMLoc());
    auto transformModule = OwningOpRef<ModuleOp>(
        parseSourceFile<ModuleOp>(sourceMgr, getContext()));
    if (!transformModule)
      return entry.emitError() << "failed to parse Transform module";

    orderedLibraryNames.push_back({entry.getKey().str(), *name});
    registerLibraryModule(entry.getKey(), std::move(transformModule));
    return success();
  }

  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    // Only print for top-level libraries, print without considering what is
    // referenced to capture state.
    if (op->getParentOp() == nullptr) {
      // On top-level op print libraries additionally.
      for (auto it : orderedLibraryNames)
        provider.buildString(std::get<0>(it), std::get<1>(it));
    }
  }

  /// Returns a range of registered library modules.
  ArrayRef<ModuleOp> getLibraryModules() { return orderedLibraryModules; }

  void registerLibraryModule(StringRef key,
                             OwningOpRef<ModuleOp> &&library) const {
    libraryModules[key] = orderedLibraryModules.emplace_back(library.release());
  }

  void registerLibraryModule(OwningOpRef<ModuleOp> &&library) const {
    orderedLibraryModules.push_back(library.release());
  }

  Operation *getRegisteredTransform(StringRef resource, SymbolRefAttr ref) {
    auto it = libraryModules.find(resource);
    if (it == libraryModules.end()) {
      return nullptr;
    }

    return SymbolTable::lookupSymbolIn(it->second, ref);
  }

private:
  /// The blob manager for the dialect.
  TransformResourceBlobManagerInterface &blobManager;

  /// Library modules registered.
  mutable llvm::StringMap<ModuleOp> libraryModules;
  /// Keep a sorted list too for iteration.
  mutable llvm::SmallVector<ModuleOp, 2> orderedLibraryModules;
  /// Keep list of external files used for printing again.
  mutable llvm::SmallVector<std::pair<std::string, std::string>, 2>
      orderedLibraryNames;
};

ArrayRef<ModuleOp> transform::TransformDialect::getLibraryModules() {
  return getRegisteredInterface<TransformOpAsmInterface>()->getLibraryModules();
}

void transform::TransformDialect::registerLibraryModule(
    OwningOpRef<ModuleOp> &&library) {
  getRegisteredInterface<TransformOpAsmInterface>()->registerLibraryModule(
      std::move(library));
}

Operation *
transform::TransformDialect::getRegisteredTransform(StringRef resource, SymbolRefAttr ref) {
  auto &interface = *getRegisteredInterface<TransformOpAsmInterface>();
  return interface.getRegisteredTransform(resource, ref);
}

void transform::TransformDialect::initialize() {
  // Using the checked versions to enable the same assertions as for the ops
  // from extensions.
  addOperationsChecked<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"
      >();
  initializeTypes();

  auto &blobInterface = addInterface<TransformResourceBlobManagerInterface>();
  addInterface<TransformOpAsmInterface>(blobInterface);
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
  if (attribute.getName().getValue() == kSilenceTrackingFailuresAttrName) {
    if (!llvm::isa<UnitAttr>(attribute.getValue())) {
      return op->emitError()
             << attribute.getName() << " must be a unit attribute";
    }
    return success();
  }
  return emitError(op->getLoc())
         << "unknown attribute: " << attribute.getName();
}
