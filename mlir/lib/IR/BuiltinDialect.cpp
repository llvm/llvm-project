//===- BuiltinDialect.cpp - MLIR Builtin Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect that contains all of the attributes,
// operations, and types that are necessary for the validity of the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Builtin Dialect
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.cpp.inc"

namespace {
struct BuiltinOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  LogicalResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (attr.isa<AffineMapAttr>()) {
      os << "map";
      return success();
    }
    if (attr.isa<IntegerSetAttr>()) {
      os << "set";
      return success();
    }
    if (attr.isa<LocationAttr>()) {
      os << "loc";
      return success();
    }
    return failure();
  }

  LogicalResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = type.dyn_cast<TupleType>()) {
      if (tupleType.size() > 16) {
        os << "tuple";
        return success();
      }
    }
    return failure();
  }
};
} // end anonymous namespace.

void BuiltinDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerLocationAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/IR/BuiltinOps.cpp.inc"
      >();
  addInterfaces<BuiltinOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, "func");
  OpBuilder builder(location->getContext());
  FuncOp::build(builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::makeArrayRef(attrRef));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  FuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                           /*resultAttrs=*/llvm::None);
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(FuncOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult verify(FuncOp op) {
  // If this function is external there is nothing to do.
  if (op.isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncOp::cloneInto(FuncOp dest, BlockAndValueMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<Identifier, Attribute> newAttrs;
  for (const auto &attr : dest->getAttrs())
    newAttrs.insert(attr);
  for (const auto &attr : (*this)->getAttrs())
    newAttrs.insert(attr);
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs.takeVector()));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncOp FuncOp::clone(BlockAndValueMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
FuncOp FuncOp::clone() {
  BlockAndValueMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     Optional<StringRef> name) {
  state.addRegion()->emplaceBlock();
  if (name) {
    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(*name)));
  }
}

/// Construct a module from the given context.
ModuleOp ModuleOp::create(Location loc, Optional<StringRef> name) {
  OpBuilder builder(loc->getContext());
  return builder.create<ModuleOp>(loc, name);
}

DataLayoutSpecInterface ModuleOp::getDataLayoutSpec() {
  // Take the first and only (if present) attribute that implements the
  // interface. This needs a linear search, but is called only once per data
  // layout object construction that is used for repeated queries.
  for (Attribute attr : llvm::make_second_range(getOperation()->getAttrs())) {
    if (auto spec = attr.dyn_cast<DataLayoutSpecInterface>())
      return spec;
  }
  return {};
}

static LogicalResult verify(ModuleOp op) {
  // Check that none of the attributes are non-dialect attributes, except for
  // the symbol related attributes.
  for (auto attr : op->getAttrs()) {
    if (!attr.first.strref().contains('.') &&
        !llvm::is_contained(
            ArrayRef<StringRef>{mlir::SymbolTable::getSymbolAttrName(),
                                mlir::SymbolTable::getVisibilityAttrName()},
            attr.first.strref()))
      return op.emitOpError() << "can only contain attributes with "
                                 "dialect-prefixed names, found: '"
                              << attr.first << "'";
  }

  // Check that there is at most one data layout spec attribute.
  StringRef layoutSpecAttrName;
  DataLayoutSpecInterface layoutSpec;
  for (const NamedAttribute &na : op->getAttrs()) {
    if (auto spec = na.second.dyn_cast<DataLayoutSpecInterface>()) {
      if (layoutSpec) {
        InFlightDiagnostic diag =
            op.emitOpError() << "expects at most one data layout attribute";
        diag.attachNote() << "'" << layoutSpecAttrName
                          << "' is a data layout attribute";
        diag.attachNote() << "'" << na.first << "' is a data layout attribute";
      }
      layoutSpecAttrName = na.first.strref();
      layoutSpec = spec;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

LogicalResult
UnrealizedConversionCastOp::fold(ArrayRef<Attribute> attrOperands,
                                 SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = inputs();
  if (operands.empty())
    return failure();

  // Check that the input is a cast with results that all feed into this
  // operation, and operand types that directly match the result types of this
  // operation.
  ResultRange results = outputs();
  Value firstInput = operands.front();
  auto inputOp = firstInput.getDefiningOp<UnrealizedConversionCastOp>();
  if (!inputOp || inputOp.getResults() != operands ||
      inputOp.getOperandTypes() != results.getTypes())
    return failure();

  // If everything matches up, we can fold the passthrough.
  foldResults.append(inputOp->operand_begin(), inputOp->operand_end());
  return success();
}

bool UnrealizedConversionCastOp::areCastCompatible(TypeRange inputs,
                                                   TypeRange outputs) {
  // `UnrealizedConversionCastOp` is agnostic of the input/output types.
  return true;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/IR/BuiltinOps.cpp.inc"
