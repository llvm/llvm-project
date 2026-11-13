//===- LLVMConstantOps.cpp - LLVM dialect constant ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Constant` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"

#include "mlir/Dialect/LLVMIR/LLVMControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// LLVM::AddressOfOp.
//===----------------------------------------------------------------------===//

GlobalOp AddressOfOp::getGlobal(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<GlobalOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

LLVMFuncOp AddressOfOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

AliasOp AddressOfOp::getAlias(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<AliasOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

IFuncOp AddressOfOp::getIFunc(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<IFuncOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

LogicalResult
AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol =
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr());

  auto global = dyn_cast_or_null<GlobalOp>(symbol);
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);
  auto alias = dyn_cast_or_null<AliasOp>(symbol);
  auto ifunc = dyn_cast_or_null<IFuncOp>(symbol);

  if (!global && !function && !alias && !ifunc)
    return emitOpError("must reference a global defined by 'llvm.mlir.global', "
                       "'llvm.mlir.alias' or 'llvm.func' or 'llvm.mlir.ifunc'");

  LLVMPointerType type = getType();
  if ((global && global.getAddrSpace() != type.getAddressSpace()) ||
      (alias && alias.getAddrSpace() != type.getAddressSpace()))
    return emitOpError("pointer address space must match address space of the "
                       "referenced global or alias");

  return success();
}

// AddressOfOp constant-folds to the global symbol name.
OpFoldResult LLVM::AddressOfOp::fold(FoldAdaptor) {
  return getGlobalNameAttr();
}

//===----------------------------------------------------------------------===//
// LLVM::DSOLocalEquivalentOp
//===----------------------------------------------------------------------===//

LLVMFuncOp
DSOLocalEquivalentOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getFunctionNameAttr()));
}

AliasOp DSOLocalEquivalentOp::getAlias(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<AliasOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getFunctionNameAttr()));
}

LogicalResult
DSOLocalEquivalentOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this),
                                                 getFunctionNameAttr());
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);
  auto alias = dyn_cast_or_null<AliasOp>(symbol);

  if (!function && !alias)
    return emitOpError(
        "must reference a global defined by 'llvm.func' or 'llvm.mlir.alias'");

  if (alias) {
    if (alias.getInitializer()
            .walk([&](AddressOfOp addrOp) {
              if (addrOp.getGlobal(symbolTable))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return emitOpError("must reference an alias to a function");
  }

  if ((function && function.getLinkage() == LLVM::Linkage::ExternWeak) ||
      (alias && alias.getLinkage() == LLVM::Linkage::ExternWeak))
    return emitOpError(
        "target function with 'extern_weak' linkage not allowed");

  return success();
}

/// Fold a dso_local_equivalent operation to a dedicated dso_local_equivalent
/// attribute.
OpFoldResult DSOLocalEquivalentOp::fold(FoldAdaptor) {
  return DSOLocalEquivalentAttr::get(getContext(), getFunctionNameAttr());
}

//===----------------------------------------------------------------------===//
// UndefOp.
//===----------------------------------------------------------------------===//

/// Fold an undef operation to a dedicated undef attribute.
OpFoldResult LLVM::UndefOp::fold(FoldAdaptor) {
  return LLVM::UndefAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// PoisonOp.
//===----------------------------------------------------------------------===//

/// Fold a poison operation to a dedicated poison attribute.
OpFoldResult LLVM::PoisonOp::fold(FoldAdaptor) {
  return LLVM::PoisonAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// MetadataAsValueOp.
//===----------------------------------------------------------------------===//

/// Fold a metadata-as-value operation to its wrapped metadata attribute.
OpFoldResult LLVM::MetadataAsValueOp::fold(FoldAdaptor) {
  return getMetadataAttr();
}

//===----------------------------------------------------------------------===//
// ZeroOp.
//===----------------------------------------------------------------------===//

LogicalResult LLVM::ZeroOp::verify() {
  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType()))
    if (!targetExtType.hasProperty(LLVM::LLVMTargetExtType::HasZeroInit))
      return emitOpError()
             << "target extension type does not support zero-initializer";

  return success();
}

/// Fold a zero operation to a builtin zero attribute when possible and fall
/// back to a dedicated zero attribute.
OpFoldResult LLVM::ZeroOp::fold(FoldAdaptor) {
  OpFoldResult result = Builder(getContext()).getZeroAttr(getType());
  if (result)
    return result;
  return LLVM::ZeroAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// ConstantOp.
//===----------------------------------------------------------------------===//

/// Compute the total number of elements in the given type, also taking into
/// account nested types. Supported types are `VectorType` and `LLVMArrayType`.
/// Everything else is treated as a scalar.
static int64_t getNumElements(Type t) {
  if (auto vecType = dyn_cast<VectorType>(t)) {
    assert(!vecType.isScalable() &&
           "number of elements of a scalable vector type is unknown");
    return vecType.getNumElements() * getNumElements(vecType.getElementType());
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t))
    return arrayType.getNumElements() *
           getNumElements(arrayType.getElementType());
  return 1;
}

/// Determine the element type of `type`. Supported types are `VectorType`,
/// `TensorType`, and `LLVMArrayType`. Everything else is treated as a scalar.
Type LLVM::getConstantElementType(Type type) {
  while (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    type = arrayType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  if (auto tenType = dyn_cast<TensorType>(type))
    return tenType.getElementType();
  return type;
}

/// Check if the given type is a scalable vector type or a vector/array type
/// that contains a nested scalable vector type.
static bool hasScalableVectorType(Type t) {
  if (auto vecType = dyn_cast<VectorType>(t)) {
    if (vecType.isScalable())
      return true;
    return hasScalableVectorType(vecType.getElementType());
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t))
    return hasScalableVectorType(arrayType.getElementType());
  return false;
}

/// Verifies the constant array represented by `arrayAttr` matches the provided
/// `arrayType`.
static LogicalResult verifyStructArrayConstant(LLVM::ConstantOp op,
                                               LLVM::LLVMArrayType arrayType,
                                               ArrayAttr arrayAttr, int dim) {
  if (arrayType.getNumElements() != arrayAttr.size())
    return op.emitOpError()
           << "array attribute size does not match array type size in "
              "dimension "
           << dim << ": " << arrayAttr.size() << " vs. "
           << arrayType.getNumElements();

  llvm::DenseSet<Attribute> elementsVerified;

  // Recursively verify sub-dimensions for multidimensional arrays.
  if (auto subArrayType =
          dyn_cast<LLVM::LLVMArrayType>(arrayType.getElementType())) {
    for (auto [idx, elementAttr] : llvm::enumerate(arrayAttr))
      if (elementsVerified.insert(elementAttr).second) {
        if (isa<LLVM::ZeroAttr, LLVM::UndefAttr>(elementAttr))
          continue;
        auto subArrayAttr = dyn_cast<ArrayAttr>(elementAttr);
        if (!subArrayAttr)
          return op.emitOpError()
                 << "nested attribute for sub-array in dimension " << dim
                 << " at index " << idx
                 << " must be a zero, or undef, or array attribute";
        if (failed(verifyStructArrayConstant(op, subArrayType, subArrayAttr,
                                             dim + 1)))
          return failure();
      }
    return success();
  }

  // Forbid usages of ArrayAttr for simple array types that should use
  // DenseElementsAttr instead. Note that there would be a use case for such
  // array types when one element value is obtained via a ptr-to-int conversion
  // from a symbol and cannot be represented in a DenseElementsAttr, but no MLIR
  // user needs this so far, and it seems better to avoid people misusing the
  // ArrayAttr for simple types.
  Type elementType = arrayType.getElementType();
  if (isa<LLVM::LLVMPointerType>(elementType)) {
    for (auto [idx, elementAttr] : llvm::enumerate(arrayAttr)) {
      if (isa<FlatSymbolRefAttr, LLVM::ZeroAttr, LLVM::UndefAttr,
              LLVM::PoisonAttr>(elementAttr))
        continue;
      return op.emitOpError()
             << "pointer array element at index " << idx
             << " must be a flat symbol reference, zero, undef, or poison";
    }
    return success();
  }
  auto structType = dyn_cast<LLVM::LLVMStructType>(elementType);
  if (!structType)
    return op.emitOpError() << "for array with an array attribute must have a "
                               "struct element type";

  // Shallow verification that leaf attributes are appropriate as struct initial
  // value.
  size_t numStructElements = structType.getBody().size();
  for (auto [idx, elementAttr] : llvm::enumerate(arrayAttr)) {
    if (elementsVerified.insert(elementAttr).second) {
      if (isa<LLVM::ZeroAttr, LLVM::UndefAttr>(elementAttr))
        continue;
      auto subArrayAttr = dyn_cast<ArrayAttr>(elementAttr);
      if (!subArrayAttr)
        return op.emitOpError()
               << "nested attribute for struct element at index " << idx
               << " must be a zero, or undef, or array attribute";
      if (subArrayAttr.size() != numStructElements)
        return op.emitOpError()
               << "nested array attribute size for struct element at index "
               << idx << " must match struct size: " << subArrayAttr.size()
               << " vs. " << numStructElements;
    }
  }

  return success();
}

LogicalResult LLVM::ConstantOp::verify() {
  if (StringAttr sAttr = llvm::dyn_cast<StringAttr>(getValue())) {
    auto arrayType = llvm::dyn_cast<LLVMArrayType>(getType());
    if (!arrayType || arrayType.getNumElements() != sAttr.getValue().size() ||
        !arrayType.getElementType().isInteger(8)) {
      return emitOpError() << "expected array type of "
                           << sAttr.getValue().size()
                           << " i8 elements for the string constant";
    }
    return success();
  }
  if (auto structType = dyn_cast<LLVMStructType>(getType())) {
    auto arrayAttr = dyn_cast<ArrayAttr>(getValue());
    if (!arrayAttr)
      return emitOpError() << "expected array attribute for struct type";

    ArrayRef<Type> elementTypes = structType.getBody();
    if (arrayAttr.size() != elementTypes.size()) {
      return emitOpError() << "expected array attribute of size "
                           << elementTypes.size();
    }
    for (auto [i, attr, type] : llvm::enumerate(arrayAttr, elementTypes)) {
      if (!type.isSignlessIntOrIndexOrFloat()) {
        return emitOpError() << "expected struct element types to be floating "
                                "point type or integer type";
      }
      if (!isa<FloatAttr, IntegerAttr>(attr)) {
        return emitOpError() << "expected element of array attribute to be "
                                "floating point or integer";
      }
      if (cast<TypedAttr>(attr).getType() != type)
        return emitOpError()
               << "struct element at index " << i << " is of wrong type";
    }

    return success();
  }
  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType()))
    return emitOpError() << "does not support target extension type.";

  // Check that an attribute whose element type has floating point semantics
  // `attributeFloatSemantics` is compatible with a type whose element type
  // is `constantElementType`.
  //
  // Requirement is that either
  // 1) They have identical floating point types.
  // 2) `constantElementType` is an integer type of the same width as the float
  //     attribute. This is to support builtin MLIR float types without LLVM
  //     equivalents, see comments in getLLVMConstant for more details.
  auto verifyFloatSemantics =
      [this](const llvm::fltSemantics &attributeFloatSemantics,
             Type constantElementType) -> LogicalResult {
    if (auto floatType = dyn_cast<FloatType>(constantElementType)) {
      if (&floatType.getFloatSemantics() != &attributeFloatSemantics) {
        return emitOpError()
               << "attribute and type have different float semantics";
      }
      return success();
    }
    unsigned floatWidth = APFloat::getSizeInBits(attributeFloatSemantics);
    if (isa<IntegerType>(constantElementType)) {
      if (!constantElementType.isInteger(floatWidth))
        return emitOpError() << "expected integer type of width " << floatWidth;

      return success();
    }
    return success();
  };

  // Check that an integer attribute whose element type is `attributeIntType`
  // is compatible with a type whose element type is `constantElementType`.
  //
  // Contrary to floats, integers must match exactly. An integer attribute
  // carries no information that the corresponding LLVM type cannot represent,
  // so any difference in width, signedness, or in `index` versus a fixed-width
  // integer indicates a malformed constant. Note that `index` never reaches
  // this check as a constant type, since it is not LLVM dialect-compatible.
  auto verifyIntegerSemantics = [this](Type attributeIntType,
                                       Type constantElementType,
                                       StringRef description) -> LogicalResult {
    if (attributeIntType != constantElementType)
      return emitOpError() << "attribute and type have different integer "
                           << description << "s: " << attributeIntType
                           << " vs. " << constantElementType;
    return success();
  };

  // Verification of IntegerAttr, FloatAttr, ElementsAttr, ArrayAttr.
  if (auto intAttr = dyn_cast<IntegerAttr>(getValue())) {
    if (!llvm::isa<IntegerType>(getType()))
      return emitOpError() << "expected integer type";
    return verifyIntegerSemantics(intAttr.getType(), getType(), "type");
  } else if (auto floatAttr = dyn_cast<FloatAttr>(getValue())) {
    return verifyFloatSemantics(floatAttr.getValue().getSemantics(), getType());
  } else if (auto elementsAttr = dyn_cast<ElementsAttr>(getValue())) {
    // Check that the element type of the attribute is compatible with the
    // element type of the constant. Shared by the scalable and the fixed-size
    // paths, since element types must agree either way.
    auto verifyElementTypes = [&](ElementsAttr attr) -> LogicalResult {
      Type attrElmType = LLVM::getConstantElementType(attr.getType());
      Type resultElmType = LLVM::getConstantElementType(getType());
      if (auto floatType = dyn_cast<FloatType>(attrElmType))
        return verifyFloatSemantics(floatType.getFloatSemantics(),
                                    resultElmType);

      if (isa<IntegerType, IndexType>(attrElmType)) {
        if (!isa<IntegerType>(resultElmType))
          return emitOpError(
              "expected integer element type for integer elements attribute");
        return verifyIntegerSemantics(attrElmType, resultElmType,
                                      "element type");
      }
      return success();
    };

    if (hasScalableVectorType(getType())) {
      // The exact number of elements of a scalable vector is unknown, so we
      // allow only splat attributes.
      auto splatElementsAttr = dyn_cast<SplatElementsAttr>(getValue());
      if (!splatElementsAttr)
        return emitOpError()
               << "scalable vector type requires a splat attribute";
      return verifyElementTypes(splatElementsAttr);
    }
    if (!isa<VectorType, LLVM::LLVMArrayType>(getType()))
      return emitOpError() << "expected vector or array type";

    // The number of elements of the attribute and the type must match.
    int64_t attrNumElements = elementsAttr.getNumElements();
    if (getNumElements(getType()) != attrNumElements) {
      return emitOpError()
             << "type and attribute have a different number of elements: "
             << getNumElements(getType()) << " vs. " << attrNumElements;
    }

    return verifyElementTypes(elementsAttr);
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(getValue())) {

    // The case where the constant is LLVMStructType has already been handled.
    auto arrayType = dyn_cast<LLVM::LLVMArrayType>(getType());
    if (!arrayType)
      return emitOpError()
             << "expected array or struct type for array attribute";

    // When the attribute is an ArrayAttr, check that its nesting matches the
    // corresponding ArrayType or VectorType nesting.
    return verifyStructArrayConstant(*this, arrayType, arrayAttr, /*dim=*/0);
  } else {
    return emitOpError()
           << "only supports integer, float, string or elements attributes";
  }
}

bool LLVM::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  auto typedAttr = dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type || !isCompatibleType(type))
    return false;
  // The value's type must be an LLVM compatible type.
  if (!isCompatibleType(type))
    return false;
  // TODO: Add support for additional attributes kinds once needed.
  return isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

ConstantOp LLVM::ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
  if (isBuildableWith(value, type))
    return LLVM::ConstantOp::create(builder, loc, cast<TypedAttr>(value));
  return nullptr;
}

// Constant op constant-folds to its value.
OpFoldResult LLVM::ConstantOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// BlockAddressOp
//===----------------------------------------------------------------------===//

LogicalResult
BlockAddressOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this),
                                                 getBlockAddr().getFunction());
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);

  if (!function)
    return emitOpError("must reference a function defined by 'llvm.func'");

  return success();
}

LLVMFuncOp BlockAddressOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getBlockAddr().getFunction()));
}

BlockTagOp BlockAddressOp::getBlockTagOp() {
  Operation *sym = mlir::SymbolTable::lookupNearestSymbolFrom(
      parentLLVMModule(*this), getBlockAddr().getFunction());
  if (!sym)
    return nullptr;
  auto funcOp = dyn_cast<LLVMFuncOp>(sym);
  if (!funcOp)
    return nullptr;
  BlockTagOp blockTagOp = nullptr;
  funcOp.walk([&](LLVM::BlockTagOp labelOp) {
    if (labelOp.getTag() == getBlockAddr().getTag()) {
      blockTagOp = labelOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return blockTagOp;
}

LogicalResult BlockAddressOp::verify() {
  if (!getBlockTagOp())
    return emitOpError(
        "expects an existing block label target in the referenced function");

  return success();
}

/// Fold a blockaddress operation to a dedicated blockaddress
/// attribute.
OpFoldResult BlockAddressOp::fold(FoldAdaptor) { return getBlockAddr(); }

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.cpp.inc"
