//===- ModuleImport.h - LLVM to MLIR conversion -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the import of an LLVM IR module into an LLVM dialect
// module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_MODULEIMPORT_H
#define MLIR_TARGET_LLVMIR_MODULEIMPORT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

namespace llvm {
class BasicBlock;
class CallBase;
class Function;
class Instruction;
class Value;
} // namespace llvm

namespace mlir {
namespace LLVM {

namespace detail {
class DebugImporter;
} // namespace detail

/// Module import implementation class that provides methods to import globals
/// and functions from an LLVM module into an MLIR module. It holds mappings
/// between the original and translated globals, basic blocks, and values used
/// during the translation. Additionally, it keeps track of the current constant
/// insertion point since LLVM immediate values translate to MLIR operations
/// that are introduced at the beginning of the region.
class ModuleImport {
public:
  ModuleImport(ModuleOp mlirModule, std::unique_ptr<llvm::Module> llvmModule);

  /// Stores the mapping between an LLVM value and its MLIR counterpart.
  void mapValue(llvm::Value *llvm, Value mlir) { mapValue(llvm) = mlir; }

  /// Provides write-once access to store the MLIR value corresponding to the
  /// given LLVM value.
  Value &mapValue(llvm::Value *value) {
    Value &mlir = valueMapping[value];
    assert(mlir == nullptr &&
           "attempting to map a value that is already mapped");
    return mlir;
  }

  /// Returns the MLIR value mapped to the given LLVM value.
  Value lookupValue(llvm::Value *value) { return valueMapping.lookup(value); }

  /// Stores the mapping between an LLVM block and its MLIR counterpart.
  void mapBlock(llvm::BasicBlock *llvm, Block *mlir) {
    auto result = blockMapping.try_emplace(llvm, mlir);
    (void)result;
    assert(result.second && "attempting to map a block that is already mapped");
  }

  /// Returns the MLIR block mapped to the given LLVM block.
  Block *lookupBlock(llvm::BasicBlock *block) const {
    return blockMapping.lookup(block);
  }

  /// Converts an LLVM value to an MLIR value, or returns failure if the
  /// conversion fails. Uses the `convertConstant` method to translate constant
  /// LLVM values.
  FailureOr<Value> convertValue(llvm::Value *value);

  /// Converts a range of LLVM values to a range of MLIR values using the
  /// `convertValue` method, or returns failure if the conversion fails.
  FailureOr<SmallVector<Value>> convertValues(ArrayRef<llvm::Value *> values);

  /// Converts `value` to an integer attribute. Asserts if the matching fails.
  IntegerAttr matchIntegerAttr(llvm::Value *value);

  /// Converts `value` to a local variable attribute. Asserts if the matching
  /// fails.
  DILocalVariableAttr matchLocalVariableAttr(llvm::Value *value);

  /// Translates the debug location.
  Location translateLoc(llvm::DILocation *loc);

  /// Converts the type from LLVM to MLIR LLVM dialect.
  Type convertType(llvm::Type *type) {
    return typeTranslator.translateType(type);
  }

  /// Converts an LLVM intrinsic to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertIntrinsic(OpBuilder &odsBuilder, llvm::CallInst *inst,
                                 llvm::Intrinsic::ID intrinsicID);

  /// Converts an LLVM instruction to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertOperation(OpBuilder &odsBuilder,
                                 llvm::Instruction *inst);

  /// Imports `func` into the current module.
  LogicalResult processFunction(llvm::Function *func);

  /// Converts function attributes of LLVM Function \p func
  /// into LLVM dialect attributes of LLVMFuncOp \p funcOp.
  void processFunctionAttributes(llvm::Function *func, LLVMFuncOp funcOp);

  /// Imports `globalVar` as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *globalVar);

  /// Converts all functions of the LLVM module to MLIR functions.
  LogicalResult convertFunctions();

  /// Converts all global variables of the LLVM module to MLIR global variables.
  LogicalResult convertGlobals();

private:
  /// Clears the block and value mapping before processing a new region.
  void clearBlockAndValueMapping() {
    valueMapping.clear();
    blockMapping.clear();
  }
  /// Sets the constant insertion point to the start of the given block.
  void setConstantInsertionPointToStart(Block *block) {
    constantInsertionBlock = block;
    constantInsertionOp = nullptr;
  }

  /// Sets the fastmath flags attribute for the imported operation `op` given
  /// the original instruction `inst`. Asserts if the operation does not
  /// implement the fastmath interface.
  void setFastmathFlagsAttr(llvm::Instruction *inst, Operation *op) const;
  /// Returns personality of `func` as a FlatSymbolRefAttr.
  FlatSymbolRefAttr getPersonalityAsAttr(llvm::Function *func);
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates valueMapping[inst] with the result of the
  /// imported operation.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// Converts the `branch` arguments in the order of the phi's found in
  /// `target` and appends them to the `blockArguments` to attach to the
  /// generated branch operation. The `blockArguments` thus have the same order
  /// as the phi's in `target`.
  LogicalResult convertBranchArgs(llvm::Instruction *branch,
                                  llvm::BasicBlock *target,
                                  SmallVectorImpl<Value> &blockArguments);
  /// Appends the converted result type and operands of `callInst` to the
  /// `types` and `operands` arrays. For indirect calls, the method additionally
  /// inserts the called function at the beginning of the `operands` array.
  LogicalResult convertCallTypeAndOperands(llvm::CallBase *callInst,
                                           SmallVectorImpl<Type> &types,
                                           SmallVectorImpl<Value> &operands);
  /// Returns the builtin type equivalent to be used in attributes for the given
  /// LLVM IR dialect type.
  Type getStdTypeForAttr(Type type);
  /// Returns `value` as an attribute to attach to a GlobalOp.
  Attribute getConstantAsAttr(llvm::Constant *value);
  /// Returns the topologically sorted set of transitive dependencies needed to
  /// convert the given constant.
  SetVector<llvm::Constant *> getConstantsToConvert(llvm::Constant *constant);
  /// Converts an LLVM constant to an MLIR value, or returns failure if the
  /// conversion fails. The MLIR value may be produced by a ConstantOp,
  /// AddressOfOp, NullOp, or a side-effect free operation (for ConstantExprs or
  /// ConstantGEPs).
  FailureOr<Value> convertConstant(llvm::Constant *constant);
  /// Converts an LLVM constant and its transitive constant dependencies to MLIR
  /// operations by converting them in topological order using the
  /// `convertConstant` method, or returns failure if the conversion of any of
  /// them fails. All operations are inserted at the start of the current
  /// function entry block.
  FailureOr<Value> convertConstantExpr(llvm::Constant *constant);

  /// Builder pointing at where the next instruction should be generated.
  OpBuilder builder;
  /// Block to insert the next constant into.
  Block *constantInsertionBlock = nullptr;
  /// Operation to insert the next constant after.
  Operation *constantInsertionOp = nullptr;
  /// Operation to insert the next global after.
  Operation *globalInsertionOp = nullptr;
  /// The current context.
  MLIRContext *context;
  /// The MLIR module being created.
  ModuleOp mlirModule;
  /// The LLVM module being imported.
  std::unique_ptr<llvm::Module> llvmModule;

  /// Function-local mapping between original and imported block.
  DenseMap<llvm::BasicBlock *, Block *> blockMapping;
  /// Function-local mapping between original and imported values.
  DenseMap<llvm::Value *, Value> valueMapping;
  /// Uniquing map of GlobalVariables.
  DenseMap<llvm::GlobalVariable *, GlobalOp> globals;
  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  /// Stateful debug information importer.
  std::unique_ptr<detail::DebugImporter> debugImporter;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_MODULEIMPORT_H
