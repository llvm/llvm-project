//===- LLVMAttrs.cpp - LLVM Attributes registration -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attribute details for the LLVM IR dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect
//===----------------------------------------------------------------------===//

void LLVMDialect::registerAttributes() {
  addAttributes<DistinctSequenceAttr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"
      >();
}

Attribute LLVMDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef mnemonic;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &mnemonic, type, attr);
  if (result.has_value())
    return attr;

  if (mnemonic == DistinctSequenceAttr::getMnemonic())
    return DistinctSequenceAttr::parse(parser, type);

  llvm_unreachable("unhandled LLVM attribute kind");
}

void LLVMDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (succeeded(generatedAttributePrinter(attr, os)))
    return;

  if (auto composite = dyn_cast<DistinctSequenceAttr>(attr))
    composite.print(os);
  else
    llvm_unreachable("unhandled LLVM attribute kind");
}

//===----------------------------------------------------------------------===//
// DINodeAttr
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DIBasicTypeAttr, DICompileUnitAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DIFileAttr, DILexicalBlockAttr,
                   DILexicalBlockFileAttr, DILocalVariableAttr, DINamespaceAttr,
                   DINullTypeAttr, DISubprogramAttr, DISubrangeAttr,
                   DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DICompositeTypeAttr, DIFileAttr,
                   DILocalScopeAttr, DINamespaceAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DITypeAttr
//===----------------------------------------------------------------------===//

bool DITypeAttr::classof(Attribute attr) {
  return llvm::isa<DINullTypeAttr, DIBasicTypeAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

MemoryEffectsAttr MemoryEffectsAttr::get(MLIRContext *context,
                                         ArrayRef<ModRefInfo> memInfoArgs) {
  if (memInfoArgs.empty())
    return MemoryEffectsAttr::get(context, ModRefInfo::ModRef,
                                  ModRefInfo::ModRef, ModRefInfo::ModRef);
  if (memInfoArgs.size() == 3)
    return MemoryEffectsAttr::get(context, memInfoArgs[0], memInfoArgs[1],
                                  memInfoArgs[2]);
  return {};
}

bool MemoryEffectsAttr::isReadWrite() {
  if (this->getArgMem() != ModRefInfo::ModRef)
    return false;
  if (this->getInaccessibleMem() != ModRefInfo::ModRef)
    return false;
  if (this->getOther() != ModRefInfo::ModRef)
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// DistinctSequenceAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {
namespace detail {
/// Attribute storage class of the distinct sequence attribute that stores the
/// sequence scope and a mutable state that is used to get the next identifier.
class DistinctSequenceAttrStorage : public AttributeStorage {
public:
  using KeyTy = SymbolRefAttr;

  DistinctSequenceAttrStorage(SymbolRefAttr scope) : scope(scope) {}

  static DistinctSequenceAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<DistinctSequenceAttrStorage>())
        DistinctSequenceAttrStorage(key);
  }

  /// Stores the next identifier value that matches the state variable to
  /// `nextID` and post increments the state (incrementing is thread safe since
  /// the storage uniquer acquires a mutex before calling the mutate method).
  LogicalResult mutate(AttributeStorageAllocator &allocator, int64_t *nextID) {
    *nextID = state++;
    return success();
  }

  /// Sets the state to `state` after parsing an attribute instance (setting
  /// the state is thread safe since the storage uniquer acquires a mutex before
  /// calling the mutate method).
  LogicalResult mutate(AttributeStorageAllocator &allocator, int64_t state) {
    this->state = state;
    return success();
  }

  /// Returns the scope of the sequence.
  SymbolRefAttr getScope() const { return scope; }

  /// Returns the state of the sequence.
  int64_t getState() const { return state; }

  /// Compares the non-mutable part of the attribute.
  bool operator==(const KeyTy &other) const { return scope == other; }

private:
  SymbolRefAttr scope;
  int64_t state = 0;
};
} // namespace detail
} // namespace LLVM
} // namespace mlir

DistinctSequenceAttr DistinctSequenceAttr::get(SymbolRefAttr scope) {
  return Base::get(scope.getContext(), scope);
}

SymbolRefAttr DistinctSequenceAttr::getScope() const {
  return getImpl()->getScope();
}

int64_t DistinctSequenceAttr::getState() const { return getImpl()->getState(); }

int64_t DistinctSequenceAttr::getNextID() {
  int64_t nextID;
  (void)Base::mutate(&nextID);
  return nextID;
}

Attribute DistinctSequenceAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // A helper function to parse a struct parameter.
  auto parseParameter =
      [&](StringRef name, StringRef type, bool &seen,
          function_ref<ParseResult()> parseFn) -> ParseResult {
    if (seen) {
      return parser.emitError(parser.getCurrentLocation())
             << "struct has duplicate parameter '" << name << "'";
    }
    if (failed(parseFn())) {
      return parser.emitError(parser.getCurrentLocation())
             << "failed to parse DistinctSequenceAttr parameter '" << name
             << "' which is to be a '" << type << "'";
    }
    seen = true;
    return success();
  };

  std::pair<SymbolRefAttr, bool> scope = {nullptr, false};
  std::pair<int64_t, bool> state = {0, false};
  do {
    std::string keyword;
    if (failed(parser.parseKeywordOrString(&keyword))) {
      parser.emitError(parser.getCurrentLocation())
          << "expected a parameter name in struct";
      return {};
    }
    if (parser.parseEqual())
      return {};

    if (keyword == "scope") {
      if (failed(parseParameter(keyword, "SymbolRefAttr", scope.second, [&]() {
            return parser.parseAttribute<SymbolRefAttr>(scope.first);
          })))
        return {};
    } else if (keyword == "state") {
      if (failed(parseParameter(keyword, "int64_t", state.second, [&]() {
            return parser.parseInteger(state.first);
          })))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "expected a parameter name in struct";
      return {};
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (!scope.second) {
    parser.emitError(parser.getCurrentLocation())
        << "struct is missing required parameter 'scope'";
    return {};
  }
  if (!state.second) {
    parser.emitError(parser.getCurrentLocation())
        << "struct is missing required parameter 'state'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  DistinctSequenceAttr distinctSeqAttr = get(scope.first);
  (void)distinctSeqAttr.mutate(state.first);
  return distinctSeqAttr;
}

void DistinctSequenceAttr::print(AsmPrinter &os) const {
  os << DistinctSequenceAttr::getMnemonic() << "<";
  os << "scope = ";
  os.printAttribute(getImpl()->getScope());
  os << ", state = " << getImpl()->getState();
  os << ">";
}

LogicalResult
mlir::LLVM::verifyAccessGroups(Operation *op,
                               ArrayRef<AccessGroupAttr> accessGroups) {
  // Search the top-level symbol table.
  Operation *topLevelSymbolTable = SymbolTable::getNearestSymbolTable(op);
  Operation *parentOp = topLevelSymbolTable->getParentOp();
  while (parentOp && parentOp->hasTrait<OpTrait::SymbolTable>()) {
    topLevelSymbolTable = parentOp;
    parentOp = parentOp->getParentOp();
  }

  // Verify the access group sequence scope matches the parent operation and the
  // unique identifiers are smaller than the sequence state.
  for (AccessGroupAttr group : accessGroups) {
    DistinctSequenceAttr sequence = group.getElemOf();
    Operation *scopeOp = SymbolTable::lookupNearestSymbolFrom(
        topLevelSymbolTable, sequence.getScope());
    if (scopeOp != op->getParentOp()) {
      return op->emitOpError()
             << "expected distinct sequence scope '" << sequence.getScope()
             << "' to resolve to the parent operation";
    }
    if (group.getId() >= sequence.getState()) {
      return op->emitOpError() << "expected access group id '" << group.getId()
                               << "' to be lower than the sequence state '"
                               << sequence.getState() << "'";
    }
  }
  return success();
}
