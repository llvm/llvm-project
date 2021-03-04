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
  addTypes<ComplexType, BFloat16Type, Float16Type, Float32Type, Float64Type,
           Float80Type, Float128Type, FunctionType, IndexType, IntegerType,
           MemRefType, UnrankedMemRefType, NoneType, OpaqueType,
           RankedTensorType, TupleType, UnrankedTensorType, VectorType>();
  addAttributes<AffineMapAttr, ArrayAttr, DenseIntOrFPElementsAttr,
                DenseStringElementsAttr, DictionaryAttr, FloatAttr,
                SymbolRefAttr, IntegerAttr, IntegerSetAttr, OpaqueAttr,
                OpaqueElementsAttr, SparseElementsAttr, StringAttr, TypeAttr,
                UnitAttr>();
  addAttributes<CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc, OpaqueLoc,
                UnknownLoc>();
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
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (DictionaryAttr argDict = argAttrs[i])
      state.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return impl::parseFunctionLikeOp(parser, result, /*allowVariadic=*/false,
                                   buildFuncType);
}

static void print(FuncOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  impl::printFunctionLikeOp(p, op, fnType.getInputs(), /*isVariadic=*/false,
                            fnType.getResults());
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
  FunctionType newType = getType();

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  bool isExternalFn = isExternal();
  if (!isExternalFn) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(getContext(), inputTypes, newType.getResults());
  }

  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());
  newFunc.setType(newType);

  /// Set the argument attributes for arguments that aren't being replaced.
  for (unsigned i = 0, e = getNumArguments(), destI = 0; i != e; ++i)
    if (isExternalFn || !mapper.contains(getArgument(i)))
      newFunc.setArgAttrs(destI++, getArgAttrs(i));

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
  ensureTerminator(*state.addRegion(), builder, state.location);
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
