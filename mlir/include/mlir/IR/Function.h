//===- Function.h - MLIR Function Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions are the basic unit of composition in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
//===--------------------------------------------------------------------===//
// Function Operation.
//===--------------------------------------------------------------------===//

/// FuncOp represents a function, or an operation containing one region that
/// forms a CFG(Control Flow Graph). The region of a function is not allowed to
/// implicitly capture global values, and all external references must use
/// Function arguments or attributes that establish a symbolic connection(e.g.
/// symbols referenced by name via a string attribute).
class FuncOp
    : public Op<FuncOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                OpTrait::OneRegion, OpTrait::IsIsolatedFromAbove,
                OpTrait::FunctionLike, OpTrait::AutomaticAllocationScope,
                OpTrait::AffineScope, CallableOpInterface::Trait,
                SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;

  static StringRef getOperationName() { return "func"; }

  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs = {});
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       iterator_range<dialect_attr_iterator> attrs);
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<MutableDictionaryAttr> argAttrs);

  static void build(OpBuilder &builder, OperationState &result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs = {},
                    ArrayRef<MutableDictionaryAttr> argAttrs = {});

  /// Operation hooks.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  /// Erase a single argument at `argIndex`.
  void eraseArgument(unsigned argIndex) { eraseArguments({argIndex}); }
  /// Erases the arguments listed in `argIndices`.
  /// `argIndices` is allowed to have duplicates and can be in any order.
  void eraseArguments(ArrayRef<unsigned> argIndices);

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  FuncOp clone(BlockAndValueMapping &mapper);
  FuncOp clone();

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

  //===--------------------------------------------------------------------===//
  // CallableOpInterface
  //===--------------------------------------------------------------------===//

  /// Returns the region on the current operation that is callable. This may
  /// return null in the case of an external callable object, e.g. an external
  /// function.
  Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

  /// Returns the results types that the callable region produces when executed.
  ArrayRef<Type> getCallableResults() { return getType().getResults(); }

private:
  // This trait needs access to the hooks defined below.
  friend class OpTrait::FunctionLike<FuncOp>;

  /// Returns the number of arguments. This is a hook for OpTrait::FunctionLike.
  unsigned getNumFuncArguments() { return getType().getInputs().size(); }

  /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
  unsigned getNumFuncResults() { return getType().getResults().size(); }

  /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
  /// attribute is present and checks if it holds a function type.  Ensures
  /// getType, getNumFuncArguments, and getNumFuncResults can be called safely.
  LogicalResult verifyType() {
    auto type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
      return emitOpError("requires '" + getTypeAttrName() +
                         "' attribute of function type");
    return success();
  }
};
} // end namespace mlir

namespace llvm {

// Functions hash just like pointers.
template <> struct DenseMapInfo<mlir::FuncOp> {
  static mlir::FuncOp getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static mlir::FuncOp getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::FuncOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::FuncOp LHS, mlir::FuncOp RHS) { return LHS == RHS; }
};

/// Allow stealing the low bits of FuncOp.
template <> struct PointerLikeTypeTraits<mlir::FuncOp> {
public:
  static inline void *getAsVoidPointer(mlir::FuncOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::FuncOp getFromVoidPointer(void *P) {
    return mlir::FuncOp::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // namespace llvm

#endif // MLIR_IR_FUNCTION_H
