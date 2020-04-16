//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffects.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"

namespace llvm {
class Type;
class LLVMContext;
namespace sys {
template <bool mt_only>
class SmartMutex;
} // end namespace sys
} // end namespace llvm

namespace mlir {
namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMTypeStorage;
struct LLVMDialectImpl;
} // namespace detail

class LLVMType : public mlir::Type::TypeBase<LLVMType, mlir::Type,
                                             detail::LLVMTypeStorage> {
public:
  enum Kind {
    LLVM_TYPE = FIRST_LLVM_TYPE,
  };

  using Base::Base;

  static bool kindof(unsigned kind) { return kind == LLVM_TYPE; }

  LLVMDialect &getDialect();
  llvm::Type *getUnderlyingType() const;

  /// Utilities to identify types.
  bool isHalfTy() { return getUnderlyingType()->isHalfTy(); }
  bool isFloatTy() { return getUnderlyingType()->isFloatTy(); }
  bool isDoubleTy() { return getUnderlyingType()->isDoubleTy(); }
  bool isIntegerTy() { return getUnderlyingType()->isIntegerTy(); }
  bool isIntegerTy(unsigned bitwidth) {
    return getUnderlyingType()->isIntegerTy(bitwidth);
  }

  /// Array type utilities.
  LLVMType getArrayElementType();
  unsigned getArrayNumElements();
  bool isArrayTy();

  /// Vector type utilities.
  LLVMType getVectorElementType();
  unsigned getVectorNumElements();
  bool isVectorTy();

  /// Function type utilities.
  LLVMType getFunctionParamType(unsigned argIdx);
  unsigned getFunctionNumParams();
  LLVMType getFunctionResultType();
  bool isFunctionTy();

  /// Pointer type utilities.
  LLVMType getPointerTo(unsigned addrSpace = 0);
  LLVMType getPointerElementTy();
  bool isPointerTy();

  /// Struct type utilities.
  LLVMType getStructElementType(unsigned i);
  unsigned getStructNumElements();
  bool isStructTy();

  /// Utilities used to generate floating point types.
  static LLVMType getDoubleTy(LLVMDialect *dialect);
  static LLVMType getFloatTy(LLVMDialect *dialect);
  static LLVMType getHalfTy(LLVMDialect *dialect);
  static LLVMType getFP128Ty(LLVMDialect *dialect);
  static LLVMType getX86_FP80Ty(LLVMDialect *dialect);

  /// Utilities used to generate integer types.
  static LLVMType getIntNTy(LLVMDialect *dialect, unsigned numBits);
  static LLVMType getInt1Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/1);
  }
  static LLVMType getInt8Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/8);
  }
  static LLVMType getInt8PtrTy(LLVMDialect *dialect) {
    return getInt8Ty(dialect).getPointerTo();
  }
  static LLVMType getInt16Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/16);
  }
  static LLVMType getInt32Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/32);
  }
  static LLVMType getInt64Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/64);
  }

  /// Utilities used to generate other miscellaneous types.
  static LLVMType getArrayTy(LLVMType elementType, uint64_t numElements);
  static LLVMType getFunctionTy(LLVMType result, ArrayRef<LLVMType> params,
                                bool isVarArg);
  static LLVMType getFunctionTy(LLVMType result, bool isVarArg) {
    return getFunctionTy(result, llvm::None, isVarArg);
  }
  static LLVMType getStructTy(LLVMDialect *dialect, ArrayRef<LLVMType> elements,
                              bool isPacked = false);
  static LLVMType getStructTy(LLVMDialect *dialect, bool isPacked = false) {
    return getStructTy(dialect, llvm::None, isPacked);
  }
  template <typename... Args>
  static typename std::enable_if<llvm::are_base_of<LLVMType, Args...>::value,
                                 LLVMType>::type
  getStructTy(LLVMType elt1, Args... elts) {
    SmallVector<LLVMType, 8> fields({elt1, elts...});
    return getStructTy(&elt1.getDialect(), fields);
  }
  static LLVMType getVectorTy(LLVMType elementType, unsigned numElements);

  /// Void type utilities.
  static LLVMType getVoidTy(LLVMDialect *dialect);
  bool isVoidTy();

  // Creation and setting of LLVM's identified struct types
  static LLVMType createStructTy(LLVMDialect *dialect,
                                 ArrayRef<LLVMType> elements,
                                 Optional<StringRef> name,
                                 bool isPacked = false);

  static LLVMType createStructTy(LLVMDialect *dialect,
                                 Optional<StringRef> name) {
    return createStructTy(dialect, llvm::None, name);
  }

  static LLVMType createStructTy(ArrayRef<LLVMType> elements,
                                 Optional<StringRef> name,
                                 bool isPacked = false) {
    assert(!elements.empty() &&
           "This method may not be invoked with an empty list");
    LLVMType ele0 = elements.front();
    return createStructTy(&ele0.getDialect(), elements, name, isPacked);
  }

  template <typename... Args>
  static typename std::enable_if_t<llvm::are_base_of<LLVMType, Args...>::value,
                                   LLVMType>
  createStructTy(StringRef name, LLVMType elt1, Args... elts) {
    SmallVector<LLVMType, 8> fields({elt1, elts...});
    Optional<StringRef> opt_name(name);
    return createStructTy(&elt1.getDialect(), fields, opt_name);
  }

  static LLVMType setStructTyBody(LLVMType structType,
                                  ArrayRef<LLVMType> elements,
                                  bool isPacked = false);

  template <typename... Args>
  static typename std::enable_if_t<llvm::are_base_of<LLVMType, Args...>::value,
                                   LLVMType>
  setStructTyBody(LLVMType structType, LLVMType elt1, Args... elts) {
    SmallVector<LLVMType, 8> fields({elt1, elts...});
    return setStructTyBody(structType, fields);
  }

private:
  friend LLVMDialect;

  /// Get an LLVMType with a pre-existing llvm type.
  static LLVMType get(MLIRContext *context, llvm::Type *llvmType);

  /// Get an LLVMType with an llvm type that may cause changes to the underlying
  /// llvm context when constructed.
  static LLVMType getLocked(LLVMDialect *dialect,
                            function_ref<llvm::Type *()> typeBuilder);
};

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc"

/// Create an LLVM global containing the string "value" at the module containing
/// surrounding the insertion point of builder. Obtain the address of that
/// global and use it to compute the address of the first character in the
/// string (operations inserted at the builder insertion point).
Value createGlobalString(Location loc, OpBuilder &builder, StringRef name,
                         StringRef value, LLVM::Linkage linkage,
                         LLVM::LLVMDialect *llvmDialect);

/// LLVM requires some operations to be inside of a Module operation. This
/// function confirms that the Operation has the desired properties.
bool satisfiesLLVMModule(Operation *op);

/// Clones the given module into the provided context. This is implemented by
/// transforming the module into bitcode and then reparsing the bitcode in the
/// provided context.
std::unique_ptr<llvm::Module>
cloneModuleIntoNewContext(llvm::LLVMContext *context, llvm::Module *module);

} // end namespace LLVM
} // end namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
