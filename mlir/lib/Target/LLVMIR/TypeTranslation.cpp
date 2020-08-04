//===- TypeTranslation.cpp - type translation between MLIR LLVM & LLVM IR -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace mlir;

namespace {
/// Support for translating MLIR LLVM dialect types to LLVM IR.
class TypeToLLVMIRTranslator {
public:
  /// Constructs a class creating types in the given LLVM context.
  TypeToLLVMIRTranslator(llvm::LLVMContext &context) : context(context) {}

  /// Translates a single type.
  llvm::Type *translateType(LLVM::LLVMTypeNew type) {
    // If the conversion is already known, just return it.
    if (knownTranslations.count(type))
      return knownTranslations.lookup(type);

    // Dispatch to an appropriate function.
    llvm::Type *translated =
        llvm::TypeSwitch<LLVM::LLVMTypeNew, llvm::Type *>(type)
            .Case([this](LLVM::LLVMVoidType) {
              return llvm::Type::getVoidTy(context);
            })
            .Case([this](LLVM::LLVMHalfType) {
              return llvm::Type::getHalfTy(context);
            })
            .Case([this](LLVM::LLVMBFloatType) {
              return llvm::Type::getBFloatTy(context);
            })
            .Case([this](LLVM::LLVMFloatType) {
              return llvm::Type::getFloatTy(context);
            })
            .Case([this](LLVM::LLVMDoubleType) {
              return llvm::Type::getDoubleTy(context);
            })
            .Case([this](LLVM::LLVMFP128Type) {
              return llvm::Type::getFP128Ty(context);
            })
            .Case([this](LLVM::LLVMX86FP80Type) {
              return llvm::Type::getX86_FP80Ty(context);
            })
            .Case([this](LLVM::LLVMPPCFP128Type) {
              return llvm::Type::getPPC_FP128Ty(context);
            })
            .Case([this](LLVM::LLVMX86MMXType) {
              return llvm::Type::getX86_MMXTy(context);
            })
            .Case([this](LLVM::LLVMTokenType) {
              return llvm::Type::getTokenTy(context);
            })
            .Case([this](LLVM::LLVMLabelType) {
              return llvm::Type::getLabelTy(context);
            })
            .Case([this](LLVM::LLVMMetadataType) {
              return llvm::Type::getMetadataTy(context);
            })
            .Case<LLVM::LLVMArrayType, LLVM::LLVMIntegerType,
                  LLVM::LLVMFunctionType, LLVM::LLVMPointerType,
                  LLVM::LLVMStructType, LLVM::LLVMFixedVectorType,
                  LLVM::LLVMScalableVectorType>(
                [this](auto array) { return translate(array); })
            .Default([](LLVM::LLVMTypeNew t) -> llvm::Type * {
              llvm_unreachable("unknown LLVM dialect type");
            });

    // Cache the result of the conversion and return.
    knownTranslations.try_emplace(type, translated);
    return translated;
  }

private:
  /// Translates the given array type.
  llvm::Type *translate(LLVM::LLVMArrayType type) {
    return llvm::ArrayType::get(translateType(type.getElementType()),
                                type.getNumElements());
  }

  /// Translates the given function type.
  llvm::Type *translate(LLVM::LLVMFunctionType type) {
    SmallVector<llvm::Type *, 8> paramTypes;
    translateTypes(type.getParams(), paramTypes);
    return llvm::FunctionType::get(translateType(type.getReturnType()),
                                   paramTypes, type.isVarArg());
  }

  /// Translates the given integer type.
  llvm::Type *translate(LLVM::LLVMIntegerType type) {
    return llvm::IntegerType::get(context, type.getBitWidth());
  }

  /// Translates the given pointer type.
  llvm::Type *translate(LLVM::LLVMPointerType type) {
    return llvm::PointerType::get(translateType(type.getElementType()),
                                  type.getAddressSpace());
  }

  /// Translates the given structure type, supports both identified and literal
  /// structs. This will _create_ a new identified structure every time, use
  /// `convertType` if a structure with the same name must be looked up instead.
  llvm::Type *translate(LLVM::LLVMStructType type) {
    SmallVector<llvm::Type *, 8> subtypes;
    if (!type.isIdentified()) {
      translateTypes(type.getBody(), subtypes);
      return llvm::StructType::get(context, subtypes, type.isPacked());
    }

    llvm::StructType *structType =
        llvm::StructType::create(context, type.getName());
    // Mark the type we just created as known so that recursive calls can pick
    // it up and use directly.
    knownTranslations.try_emplace(type, structType);
    if (type.isOpaque())
      return structType;

    translateTypes(type.getBody(), subtypes);
    structType->setBody(subtypes, type.isPacked());
    return structType;
  }

  /// Translates the given fixed-vector type.
  llvm::Type *translate(LLVM::LLVMFixedVectorType type) {
    return llvm::FixedVectorType::get(translateType(type.getElementType()),
                                      type.getNumElements());
  }

  /// Translates the given scalable-vector type.
  llvm::Type *translate(LLVM::LLVMScalableVectorType type) {
    return llvm::ScalableVectorType::get(translateType(type.getElementType()),
                                         type.getMinNumElements());
  }

  /// Translates a list of types.
  void translateTypes(ArrayRef<LLVM::LLVMTypeNew> types,
                      SmallVectorImpl<llvm::Type *> &result) {
    result.reserve(result.size() + types.size());
    for (auto type : types)
      result.push_back(translateType(type));
  }

  /// Reference to the context in which the LLVM IR types are created.
  llvm::LLVMContext &context;

  /// Map of known translation. This serves a double purpose: caches translation
  /// results to avoid repeated recursive calls and makes sure identified
  /// structs with the same name (that is, equal) are resolved to an existing
  /// type instead of creating a new type.
  llvm::DenseMap<LLVM::LLVMTypeNew, llvm::Type *> knownTranslations;
};
} // end namespace

/// Translates a type from MLIR LLVM dialect to LLVM IR. This does not maintain
/// the mapping for identified structs so new structs will be created with
/// auto-renaming on each call. This is intended exclusively for testing.
llvm::Type *mlir::LLVM::translateTypeToLLVMIR(LLVM::LLVMTypeNew type,
                                              llvm::LLVMContext &context) {
  return TypeToLLVMIRTranslator(context).translateType(type);
}

namespace {
/// Support for translating LLVM IR types to MLIR LLVM dialect types.
class TypeFromLLVMIRTranslator {
public:
  /// Constructs a class creating types in the given MLIR context.
  TypeFromLLVMIRTranslator(MLIRContext &context) : context(context) {}

  /// Translates the given type.
  LLVM::LLVMTypeNew translateType(llvm::Type *type) {
    if (knownTranslations.count(type))
      return knownTranslations.lookup(type);

    LLVM::LLVMTypeNew translated =
        llvm::TypeSwitch<llvm::Type *, LLVM::LLVMTypeNew>(type)
            .Case<llvm::ArrayType, llvm::FunctionType, llvm::IntegerType,
                  llvm::PointerType, llvm::StructType, llvm::FixedVectorType,
                  llvm::ScalableVectorType>(
                [this](auto *type) { return translate(type); })
            .Default([this](llvm::Type *type) {
              return translatePrimitiveType(type);
            });
    knownTranslations.try_emplace(type, translated);
    return translated;
  }

private:
  /// Translates the given primitive, i.e. non-parametric in MLIR nomenclature,
  /// type.
  LLVM::LLVMTypeNew translatePrimitiveType(llvm::Type *type) {
    if (type->isVoidTy())
      return LLVM::LLVMVoidType::get(&context);
    if (type->isHalfTy())
      return LLVM::LLVMHalfType::get(&context);
    if (type->isBFloatTy())
      return LLVM::LLVMBFloatType::get(&context);
    if (type->isFloatTy())
      return LLVM::LLVMFloatType::get(&context);
    if (type->isDoubleTy())
      return LLVM::LLVMDoubleType::get(&context);
    if (type->isFP128Ty())
      return LLVM::LLVMFP128Type::get(&context);
    if (type->isX86_FP80Ty())
      return LLVM::LLVMX86FP80Type::get(&context);
    if (type->isPPC_FP128Ty())
      return LLVM::LLVMPPCFP128Type::get(&context);
    if (type->isX86_MMXTy())
      return LLVM::LLVMX86MMXType::get(&context);
    if (type->isLabelTy())
      return LLVM::LLVMLabelType::get(&context);
    if (type->isMetadataTy())
      return LLVM::LLVMMetadataType::get(&context);
    llvm_unreachable("not a primitive type");
  }

  /// Translates the given array type.
  LLVM::LLVMTypeNew translate(llvm::ArrayType *type) {
    return LLVM::LLVMArrayType::get(translateType(type->getElementType()),
                                    type->getNumElements());
  }

  /// Translates the given function type.
  LLVM::LLVMTypeNew translate(llvm::FunctionType *type) {
    SmallVector<LLVM::LLVMTypeNew, 8> paramTypes;
    translateTypes(type->params(), paramTypes);
    return LLVM::LLVMFunctionType::get(translateType(type->getReturnType()),
                                       paramTypes, type->isVarArg());
  }

  /// Translates the given integer type.
  LLVM::LLVMTypeNew translate(llvm::IntegerType *type) {
    return LLVM::LLVMIntegerType::get(&context, type->getBitWidth());
  }

  /// Translates the given pointer type.
  LLVM::LLVMTypeNew translate(llvm::PointerType *type) {
    return LLVM::LLVMPointerType::get(translateType(type->getElementType()),
                                      type->getAddressSpace());
  }

  /// Translates the given structure type.
  LLVM::LLVMTypeNew translate(llvm::StructType *type) {
    SmallVector<LLVM::LLVMTypeNew, 8> subtypes;
    if (type->isLiteral()) {
      translateTypes(type->subtypes(), subtypes);
      return LLVM::LLVMStructType::getLiteral(&context, subtypes,
                                              type->isPacked());
    }

    if (type->isOpaque())
      return LLVM::LLVMStructType::getOpaque(type->getName(), &context);

    LLVM::LLVMStructType translated =
        LLVM::LLVMStructType::getIdentified(&context, type->getName());
    knownTranslations.try_emplace(type, translated);
    translateTypes(type->subtypes(), subtypes);
    LogicalResult bodySet = translated.setBody(subtypes, type->isPacked());
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;
    return translated;
  }

  /// Translates the given fixed-vector type.
  LLVM::LLVMTypeNew translate(llvm::FixedVectorType *type) {
    return LLVM::LLVMFixedVectorType::get(translateType(type->getElementType()),
                                          type->getNumElements());
  }

  /// Translates the given scalable-vector type.
  LLVM::LLVMTypeNew translate(llvm::ScalableVectorType *type) {
    return LLVM::LLVMScalableVectorType::get(
        translateType(type->getElementType()), type->getMinNumElements());
  }

  /// Translates a list of types.
  void translateTypes(ArrayRef<llvm::Type *> types,
                      SmallVectorImpl<LLVM::LLVMTypeNew> &result) {
    result.reserve(result.size() + types.size());
    for (llvm::Type *type : types)
      result.push_back(translateType(type));
  }

  /// Map of known translations. Serves as a cache and as recursion stopper for
  /// translating recursive structs.
  llvm::DenseMap<llvm::Type *, LLVM::LLVMTypeNew> knownTranslations;

  /// The context in which MLIR types are created.
  MLIRContext &context;
};
} // end namespace

/// Translates a type from LLVM IR to MLIR LLVM dialect. This is intended
/// exclusively for testing.
LLVM::LLVMTypeNew mlir::LLVM::translateTypeFromLLVMIR(llvm::Type *type,
                                                      MLIRContext &context) {
  return TypeFromLLVMIRTranslator(context).translateType(type);
}
