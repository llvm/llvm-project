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
#include "mlir/Target/LLVMIR/LLVMImportInterface.h"
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
class LoopAnnotationImporter;
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

  /// Calls the LLVMImportInterface initialization that queries the registered
  /// dialect interfaces for the supported LLVM IR intrinsics and metadata kinds
  /// and builds the dispatch tables. Returns failure if multiple dialect
  /// interfaces translate the same LLVM IR intrinsic.
  LogicalResult initializeImportInterface() { return iface.initializeImport(); }

  /// Converts all functions of the LLVM module to MLIR functions.
  LogicalResult convertFunctions();

  /// Converts all global variables of the LLVM module to MLIR global variables.
  LogicalResult convertGlobals();

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

  /// Stores a mapping between an LLVM instruction and the imported MLIR
  /// operation if the operation returns no result. Asserts if the operation
  /// returns a result and should be added to valueMapping instead.
  void mapNoResultOp(llvm::Instruction *llvm, Operation *mlir) {
    mapNoResultOp(llvm) = mlir;
  }

  /// Provides write-once access to store the MLIR operation corresponding to
  /// the given LLVM instruction if the operation returns no result. Asserts if
  /// the operation returns a result and should be added to valueMapping
  /// instead.
  Operation *&mapNoResultOp(llvm::Instruction *inst) {
    Operation *&mlir = noResultOpMapping[inst];
    assert(inst->getType()->isVoidTy() &&
           "attempting to map an operation that returns a result");
    assert(mlir == nullptr &&
           "attempting to map an operation that is already mapped");
    return mlir;
  }

  /// Returns the MLIR operation mapped to the given LLVM instruction. Queries
  /// valueMapping and noResultOpMapping to support operations with and without
  /// result.
  Operation *lookupOperation(llvm::Instruction *inst) {
    if (Value value = lookupValue(inst))
      return value.getDefiningOp();
    return noResultOpMapping.lookup(inst);
  }

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

  /// Converts an LLVM metadata value to an MLIR value, or returns failure if
  /// the conversion fails. Uses the `convertConstant` method to translate
  /// constant LLVM values.
  FailureOr<Value> convertMetadataValue(llvm::Value *value);

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

  /// Imports `func` into the current module.
  LogicalResult processFunction(llvm::Function *func);

  /// Converts function attributes of LLVM Function `func` into LLVM dialect
  /// attributes of LLVMFuncOp `funcOp`.
  void processFunctionAttributes(llvm::Function *func, LLVMFuncOp funcOp);

  /// Sets the fastmath flags attribute for the imported operation `op` given
  /// the original instruction `inst`. Asserts if the operation does not
  /// implement the fastmath interface.
  void setFastmathFlagsAttr(llvm::Instruction *inst, Operation *op) const;

  /// Converts all LLVM metadata nodes that translate to operations nested in a
  /// global metadata operation, such as alias analysis or access group
  /// metadata, and builds a map from the metadata nodes to the symbols pointing
  /// to the converted operations. Returns success if all conversions succeed
  /// and failure otherwise.
  // Note: All metadata is nested inside a single global metadata operation to
  // minimize the number of symbols that pollute the global namespace.
  LogicalResult convertMetadata();

  /// Returns the MLIR symbol reference mapped to the given LLVM TBAA
  /// metadata `node`.
  SymbolRefAttr lookupTBAAAttr(const llvm::MDNode *node) const {
    return tbaaMapping.lookup(node);
  }

  /// Returns the symbol references pointing to the access group operations that
  /// map to the access group nodes starting from the access group metadata
  /// `node`. Returns failure, if any of the symbol references cannot be found.
  FailureOr<SmallVector<SymbolRefAttr>>
  lookupAccessGroupAttrs(const llvm::MDNode *node) const;

  /// Returns the loop annotation attribute that corresponds to the given LLVM
  /// loop metadata `node`.
  LoopAnnotationAttr translateLoopAnnotationAttr(const llvm::MDNode *node,
                                                 Location loc) const;

  /// Returns the symbol references pointing to the alias scope operations that
  /// map to the alias scope nodes starting from the metadata `node`. Returns
  /// failure, if any of the symbol references cannot be found.
  FailureOr<SmallVector<SymbolRefAttr>>
  lookupAliasScopeAttrs(const llvm::MDNode *node) const;

private:
  /// Clears the block and value mapping before processing a new region.
  void clearBlockAndValueMapping() {
    valueMapping.clear();
    noResultOpMapping.clear();
    blockMapping.clear();
  }
  /// Sets the constant insertion point to the start of the given block.
  void setConstantInsertionPointToStart(Block *block) {
    constantInsertionBlock = block;
    constantInsertionOp = nullptr;
  }

  /// Converts an LLVM global variable into an MLIR LLVM dialect global
  /// operation if a conversion exists. Otherwise, returns failure.
  LogicalResult convertGlobal(llvm::GlobalVariable *globalVar);
  /// Imports the magic globals "global_ctors" and "global_dtors".
  LogicalResult convertGlobalCtorsAndDtors(llvm::GlobalVariable *globalVar);
  /// Returns personality of `func` as a FlatSymbolRefAttr.
  FlatSymbolRefAttr getPersonalityAsAttr(llvm::Function *func);
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Converts an LLVM intrinsic to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertIntrinsic(llvm::CallInst *inst);
  /// Converts an LLVM instruction to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertInstruction(llvm::Instruction *inst);
  /// Converts the metadata attached to the original instruction `inst` if
  /// a dialect interfaces supports the specific kind of metadata and attaches
  /// the resulting dialect attributes to the converted operation `op`. Emits a
  /// warning if the conversion of a supported metadata kind fails.
  void setNonDebugMetadataAttrs(llvm::Instruction *inst, Operation *op);
  /// Imports `inst` and populates valueMapping[inst] with the result of the
  /// imported operation or noResultOpMapping[inst] with the imported operation
  /// if it has no result.
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
  /// Converts the parameter attributes attached to `func` and adds them to the
  /// `funcOp`.
  void convertParameterAttributes(llvm::Function *func, LLVMFuncOp funcOp,
                                  OpBuilder &builder);
  /// Converts the AttributeSet of one parameter in LLVM IR to a corresponding
  /// DictionaryAttr for the LLVM dialect.
  DictionaryAttr convertParameterAttribute(llvm::AttributeSet llvmParamAttrs,
                                           OpBuilder &builder);
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
  /// Returns a global metadata operation that serves as a container for LLVM
  /// metadata that converts to MLIR operations. Creates the global metadata
  /// operation on the first invocation.
  MetadataOp getGlobalMetadataOp();
  /// Performs conversion of LLVM TBAA metadata starting from
  /// `node`. On exit from this function all nodes reachable
  /// from `node` are converted, and tbaaMapping map is updated
  /// (unless all dependencies have been converted by a previous
  /// invocation of this function).
  LogicalResult processTBAAMetadata(const llvm::MDNode *node);
  /// Converts all LLVM access groups starting from `node` to MLIR access group
  /// operations and stores a mapping from every nested access group node to the
  /// symbol pointing to the translated operation. Returns success if all
  /// conversions succeed and failure otherwise.
  LogicalResult processAccessGroupMetadata(const llvm::MDNode *node);
  /// Converts all LLVM alias scopes and domains starting from `node` to MLIR
  /// alias scope and domain operations and stores a mapping from every nested
  /// alias scope or alias domain node to the symbol pointing to the translated
  /// operation. Returns success if all conversions succeed and failure
  /// otherwise.
  LogicalResult processAliasScopeMetadata(const llvm::MDNode *node);

  /// Builder pointing at where the next instruction should be generated.
  OpBuilder builder;
  /// Block to insert the next constant into.
  Block *constantInsertionBlock = nullptr;
  /// Operation to insert the next constant after.
  Operation *constantInsertionOp = nullptr;
  /// Operation to insert the next global after.
  Operation *globalInsertionOp = nullptr;
  /// Operation to insert metadata operations into.
  MetadataOp globalMetadataOp = nullptr;
  /// The current context.
  MLIRContext *context;
  /// The MLIR module being created.
  ModuleOp mlirModule;
  /// The LLVM module being imported.
  std::unique_ptr<llvm::Module> llvmModule;

  /// A dialect interface collection used for dispatching the import to specific
  /// dialects.
  LLVMImportInterface iface;

  /// Function-local mapping between original and imported block.
  DenseMap<llvm::BasicBlock *, Block *> blockMapping;
  /// Function-local mapping between original and imported values.
  DenseMap<llvm::Value *, Value> valueMapping;
  /// Function-local mapping between original instructions and imported
  /// operations for all operations that return no result. All operations that
  /// return a result have a valueMapping entry instead.
  DenseMap<llvm::Instruction *, Operation *> noResultOpMapping;
  /// Mapping between LLVM alias scope and domain metadata nodes and symbol
  /// references to the LLVM dialect operations corresponding to these nodes.
  DenseMap<const llvm::MDNode *, SymbolRefAttr> aliasScopeMapping;
  /// Mapping between LLVM TBAA metadata nodes and symbol references to the LLVM
  /// dialect TBAA operations corresponding to these nodes.
  DenseMap<const llvm::MDNode *, SymbolRefAttr> tbaaMapping;
  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  /// Stateful debug information importer.
  std::unique_ptr<detail::DebugImporter> debugImporter;
  /// Loop annotation importer.
  std::unique_ptr<detail::LoopAnnotationImporter> loopAnnotationImporter;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_MODULEIMPORT_H
