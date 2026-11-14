//===- LLVMDialect.cpp - LLVM IR dialect registration ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR and registers it. The ops are
// defined by the source file of their op partition, e.g. LLVMArithOps.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#include "LLVMDialectBytecode.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void LLVMDialect::initialize() {
  registerAttributes();

  // clang-format off
  addTypes<LLVMVoidType,
           LLVMLabelType,
           LLVMMetadataType>();
  // clang-format on
  registerTypes();

  registerLLVMDialectOperations(this);

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();
  declarePromisedInterface<DialectInlinerInterface, LLVMDialect>();
  detail::addBytecodeInterface(this);
}

LogicalResult LLVMDialect::verifyDataLayoutString(
    StringRef descr, llvm::function_ref<void(const Twine &)> reportError) {
  llvm::Expected<llvm::DataLayout> maybeDataLayout =
      llvm::DataLayout::parse(descr);
  if (maybeDataLayout)
    return success();

  std::string message;
  llvm::raw_string_ostream messageStream(message);
  llvm::logAllUnhandledErrors(maybeDataLayout.takeError(), messageStream);
  reportError("invalid data layout descriptor: " + message);
  return failure();
}

/// Verify LLVM dialect attributes.
LogicalResult LLVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // If the data layout attribute is present, it must use the LLVM data layout
  // syntax. Try parsing it and report errors in case of failure. Users of this
  // attribute may assume it is well-formed and can pass it to the (asserting)
  // llvm::DataLayout constructor.
  if (attr.getName() != LLVM::LLVMDialect::getDataLayoutAttrName())
    return success();
  if (auto stringAttr = llvm::dyn_cast<StringAttr>(attr.getValue()))
    return verifyDataLayoutString(
        stringAttr.getValue(),
        [op](const Twine &message) { op->emitOpError() << message.str(); });

  return op->emitOpError() << "expected '"
                           << LLVM::LLVMDialect::getDataLayoutAttrName()
                           << "' to be a string attributes";
}

LogicalResult LLVMDialect::verifyParameterAttribute(Operation *op,
                                                    Type paramType,
                                                    NamedAttribute paramAttr) {
  // LLVM attribute may be attached to a result of operation that has not been
  // converted to LLVM dialect yet, so the result may have a type with unknown
  // representation in LLVM dialect type space. In this case we cannot verify
  // whether the attribute may be
  bool verifyValueType = isCompatibleType(paramType);
  StringAttr name = paramAttr.getName();

  auto checkUnitAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<UnitAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be a unit attribute";
    return success();
  };
  auto checkTypeAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<TypeAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be a type attribute";
    return success();
  };
  auto checkIntegerAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<IntegerAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be an integer attribute";
    return success();
  };
  auto checkPointerType = [&]() -> LogicalResult {
    if (!llvm::isa<LLVMPointerType>(paramType))
      return op->emitError()
             << name << " attribute attached to non-pointer LLVM type";
    return success();
  };
  auto checkIntegerType = [&]() -> LogicalResult {
    if (!llvm::isa<IntegerType>(paramType))
      return op->emitError()
             << name << " attribute attached to non-integer LLVM type";
    return success();
  };
  auto checkPointerTypeMatches = [&]() -> LogicalResult {
    if (failed(checkPointerType()))
      return failure();

    return success();
  };

  // Check a unit attribute that is attached to a pointer value.
  if (name == LLVMDialect::getNoAliasAttrName() ||
      name == LLVMDialect::getReadonlyAttrName() ||
      name == LLVMDialect::getReadnoneAttrName() ||
      name == LLVMDialect::getWriteOnlyAttrName() ||
      name == LLVMDialect::getNestAttrName() ||
      name == LLVMDialect::getNoCaptureAttrName() ||
      name == LLVMDialect::getNoFreeAttrName() ||
      name == LLVMDialect::getNoFreeObjAttrName() ||
      name == LLVMDialect::getNonNullAttrName()) {
    if (failed(checkUnitAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerType()))
      return failure();
    return success();
  }

  // Check a type attribute that is attached to a pointer value.
  if (name == LLVMDialect::getStructRetAttrName() ||
      name == LLVMDialect::getByValAttrName() ||
      name == LLVMDialect::getByRefAttrName() ||
      name == LLVMDialect::getElementTypeAttrName() ||
      name == LLVMDialect::getInAllocaAttrName() ||
      name == LLVMDialect::getPreallocatedAttrName()) {
    if (failed(checkTypeAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerTypeMatches()))
      return failure();
    return success();
  }

  // Check a unit attribute that is attached to an integer value.
  if (name == LLVMDialect::getSExtAttrName() ||
      name == LLVMDialect::getZExtAttrName() ||
      name == LLVMDialect::getNoExtAttrName()) {
    if (failed(checkUnitAttrType()))
      return failure();
    if (verifyValueType && failed(checkIntegerType()))
      return failure();
    return success();
  }

  // Check an integer attribute that is attached to a pointer value.
  if (name == LLVMDialect::getAlignAttrName() ||
      name == LLVMDialect::getDereferenceableAttrName() ||
      name == LLVMDialect::getDereferenceableOrNullAttrName()) {
    if (failed(checkIntegerAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerType()))
      return failure();
    return success();
  }

  // Check an integer attribute that is attached to a pointer value.
  if (name == LLVMDialect::getStackAlignmentAttrName()) {
    if (failed(checkIntegerAttrType()))
      return failure();
    return success();
  }

  // Check a unit attribute that can be attached to arbitrary types.
  if (name == LLVMDialect::getNoUndefAttrName() ||
      name == LLVMDialect::getInRegAttrName() ||
      name == LLVMDialect::getReturnedAttrName())
    return checkUnitAttrType();

  return success();
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIdx,
                                                    unsigned argIdx,
                                                    NamedAttribute argAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();
  Type argType = funcOp.getArgumentTypes()[argIdx];

  return verifyParameterAttribute(op, argType, argAttr);
}

LogicalResult LLVMDialect::verifyRegionResultAttribute(Operation *op,
                                                       unsigned regionIdx,
                                                       unsigned resIdx,
                                                       NamedAttribute resAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();
  Type resType = funcOp.getResultTypes()[resIdx];

  // Check to see if this function has a void return with a result attribute
  // to it. It isn't clear what semantics we would assign to that.
  if (llvm::isa<LLVMVoidType>(resType))
    return op->emitError() << "cannot attach result attributes to functions "
                              "with a void return";

  // Check to see if this attribute is allowed as a result attribute. Only
  // explicitly forbidden LLVM attributes will cause an error.
  auto name = resAttr.getName();
  if (name == LLVMDialect::getAllocAlignAttrName() ||
      name == LLVMDialect::getAllocatedPointerAttrName() ||
      name == LLVMDialect::getByValAttrName() ||
      name == LLVMDialect::getByRefAttrName() ||
      name == LLVMDialect::getInAllocaAttrName() ||
      name == LLVMDialect::getNestAttrName() ||
      name == LLVMDialect::getNoCaptureAttrName() ||
      name == LLVMDialect::getNoFreeAttrName() ||
      name == LLVMDialect::getPreallocatedAttrName() ||
      name == LLVMDialect::getReadnoneAttrName() ||
      name == LLVMDialect::getReadonlyAttrName() ||
      name == LLVMDialect::getReturnedAttrName() ||
      name == LLVMDialect::getStackAlignmentAttrName() ||
      name == LLVMDialect::getStructRetAttrName() ||
      name == LLVMDialect::getWriteOnlyAttrName())
    return op->emitError() << name << " is not a valid result attribute";
  return verifyParameterAttribute(op, resType, resAttr);
}

Operation *LLVMDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // If this was folded from an operation other than llvm.mlir.constant, it
  // should be materialized as such. Note that an llvm.mlir.zero may fold into
  // a builtin zero attribute and thus will materialize as a llvm.mlir.constant.
  if (auto symbol = dyn_cast<FlatSymbolRefAttr>(value))
    if (isa<LLVM::LLVMPointerType>(type))
      return LLVM::AddressOfOp::create(builder, loc, type, symbol);
  if (isa<LLVM::UndefAttr>(value))
    return LLVM::UndefOp::create(builder, loc, type);
  if (isa<LLVM::PoisonAttr>(value))
    return LLVM::PoisonOp::create(builder, loc, type);
  if (isa<LLVM::ZeroAttr>(value))
    return LLVM::ZeroOp::create(builder, loc, type);
  if (isa<LLVM::MDStringAttr, LLVM::MDConstantAttr, LLVM::MDGlobalValueAttr,
          LLVM::MDNodeAttr>(value))
    if (isa<LLVM::LLVMMetadataType>(type))
      return LLVM::MetadataAsValueOp::create(builder, loc, type, value);
  // Otherwise try materializing it as a regular llvm.mlir.constant op.
  return LLVM::ConstantOp::materialize(builder, value, type, loc);
}
