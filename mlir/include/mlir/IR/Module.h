//===- Module.h - MLIR Module Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module is the top-level container for code in an MLIR program.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MODULE_H
#define MLIR_IR_MODULE_H

#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class ModuleTerminatorOp;

//===----------------------------------------------------------------------===//
// Module Operation.
//===----------------------------------------------------------------------===//

/// ModuleOp represents a module, or an operation containing one region with a
/// single block containing opaque operations. The region of a module is not
/// allowed to implicitly capture global values, and all external references
/// must use symbolic references via attributes(e.g. via a string name).
class ModuleOp
    : public Op<
          ModuleOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
          OpTrait::IsIsolatedFromAbove, OpTrait::AffineScope,
          OpTrait::SymbolTable,
          OpTrait::SingleBlockImplicitTerminator<ModuleTerminatorOp>::Impl,
          SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;

  static StringRef getOperationName() { return "module"; }

  static void build(OpBuilder &builder, OperationState &result,
                    Optional<StringRef> name = llvm::None);

  /// Construct a module from the given location with an optional name.
  static ModuleOp create(Location loc, Optional<StringRef> name = llvm::None);

  /// Operation hooks.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  /// Return body of this module.
  Region &getBodyRegion();
  Block *getBody();

  /// Return the name of this module if present.
  Optional<StringRef> getName();

  /// Print the this module in the custom top-level form.
  void print(raw_ostream &os, OpPrintingFlags flags = llvm::None);
  void print(raw_ostream &os, AsmState &state,
             OpPrintingFlags flags = llvm::None);
  void dump();

  //===--------------------------------------------------------------------===//
  // Body Management.
  //===--------------------------------------------------------------------===//

  /// Iteration over the operations in the module.
  using iterator = Block::iterator;

  iterator begin() { return getBody()->begin(); }
  iterator end() { return getBody()->end(); }
  Operation &front() { return *begin(); }

  /// This returns a range of operations of the given type 'T' held within the
  /// module.
  template <typename T> iterator_range<Block::op_iterator<T>> getOps() {
    return getBody()->getOps<T>();
  }

  /// Insert the operation into the back of the body, before the terminator.
  void push_back(Operation *op) {
    insert(Block::iterator(getBody()->getTerminator()), op);
  }

  /// Insert the operation at the given insertion point. Note: The operation is
  /// never inserted after the terminator, even if the insertion point is end().
  void insert(Operation *insertPt, Operation *op) {
    insert(Block::iterator(insertPt), op);
  }
  void insert(Block::iterator insertPt, Operation *op) {
    auto *body = getBody();
    if (insertPt == body->end())
      insertPt = Block::iterator(body->getTerminator());
    body->getOperations().insert(insertPt, op);
  }

  //===--------------------------------------------------------------------===//
  // SymbolOpInterface Methods
  //===--------------------------------------------------------------------===//

  /// A ModuleOp may optionally define a symbol.
  bool isOptionalSymbol() { return true; }
};

/// The ModuleTerminatorOp is a special terminator operation for the body of a
/// ModuleOp, it has no semantic meaning beyond keeping the body of a ModuleOp
/// well-formed.
///
/// This operation does _not_ have a custom syntax. However, ModuleOp will omit
/// the terminator in their custom syntax for brevity.
class ModuleTerminatorOp
    : public Op<ModuleTerminatorOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                OpTrait::HasParent<ModuleOp>::Impl, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "module_terminator"; }
  static void build(OpBuilder &, OperationState &) {}
};

/// This class acts as an owning reference to a module, and will automatically
/// destroy the held module if valid.
class OwningModuleRef {
public:
  OwningModuleRef(std::nullptr_t = nullptr) {}
  OwningModuleRef(ModuleOp module) : module(module) {}
  OwningModuleRef(OwningModuleRef &&other) : module(other.release()) {}
  ~OwningModuleRef() {
    if (module)
      module.erase();
  }

  // Assign from another module reference.
  OwningModuleRef &operator=(OwningModuleRef &&other) {
    if (module)
      module.erase();
    module = other.release();
    return *this;
  }

  /// Allow accessing the internal module.
  ModuleOp get() const { return module; }
  ModuleOp operator*() const { return module; }
  ModuleOp *operator->() { return &module; }
  explicit operator bool() const { return module; }

  /// Release the referenced module.
  ModuleOp release() {
    ModuleOp released;
    std::swap(released, module);
    return released;
  }

private:
  ModuleOp module;
};

} // end namespace mlir

namespace llvm {

/// Allow stealing the low bits of ModuleOp.
template <> struct PointerLikeTypeTraits<mlir::ModuleOp> {
public:
  static inline void *getAsVoidPointer(mlir::ModuleOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::ModuleOp getFromVoidPointer(void *P) {
    return mlir::ModuleOp::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable = 3;
};

} // end namespace llvm

#endif // MLIR_IR_MODULE_H
