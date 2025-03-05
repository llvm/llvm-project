//===- LinkerInterface.h - MLIR Linker Interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces and utilities necessary for dialects
// to hook into mlir linker.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
#define MLIR_LINKER_LINKAGEDIALECTINTERFACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir {
namespace link {

//===----------------------------------------------------------------------===//
// LinkerInterface
//===----------------------------------------------------------------------===//

struct DenseMapOperationKey {
  Operation *op;
};

class LinkerSummaryState {
public:
  LinkerSummaryState() : type(TypeID::get<LinkerSummaryState>()) {}

  virtual ~LinkerSummaryState() = default;

  LinkerSummaryState(const LinkerSummaryState &) = delete;
  LinkerSummaryState &operator=(const LinkerSummaryState &) = delete;

  static bool classof(const LinkerSummaryState *base) { return true; }

  TypeID getType() const { return type; }

protected:
  explicit LinkerSummaryState(TypeID type) : type(type) {}

private:
  TypeID type;
};

class LinkerInterface : public DialectInterface::Base<LinkerInterface> {
public:
  LinkerInterface(Dialect *dialect) : Base(dialect) {}

  // TODO: Should be moved to SymbolLinkerInterface
  virtual bool isDeclaration(GlobalValueLinkageOpInterface op) const {
    return false;
  }

  // TODO: Should not exist (too llvm specific)
  bool isDeclarationForLinker(GlobalValueLinkageOpInterface op) const {
    if (op.hasAvailableExternallyLinkage())
      return true;
    return isDeclaration(op);
  }

  virtual std::unique_ptr<LinkerSummaryState>
  summarize(std::vector<ModuleOp> &modules) const {
    return nullptr;
  }

  virtual LogicalResult link(ModuleOp dst, ModuleOp src,
                             const LinkerSummaryState *state) const {
    return failure();
  }
};

struct LinkableOp {
  LinkableOp() = default;

  explicit LinkableOp(Operation *op)
      : op(op), linker(op ? dyn_cast_or_null<LinkerInterface>(op->getDialect())
                          : nullptr) {}

  explicit LinkableOp(DenseMapOperationKey op) : op(op.op), linker() {}

  operator bool() const { return op; }

  Operation *getOperation() const { return op; }

protected:
  Operation *op = nullptr;
  LinkerInterface *linker = nullptr;
};

template <typename Op>
struct LinkableOpDenseMapInfo {
  static Op getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<::mlir::Operation *>::getEmptyKey();
    return {::mlir::link::DenseMapOperationKey{pointer}};
  }

  static Op getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<::mlir::Operation *>::getEmptyKey();
    return {::mlir::link::DenseMapOperationKey{pointer}};
  }

  static unsigned getHashValue(const Op &val) {
    return DenseMapInfo<::mlir::Operation *>::getHashValue(val.getOperation());
  }

  static bool isEqual(const Op &lhs, const Op &rhs) {
    return lhs.getOperation() == rhs.getOperation();
  }
};

template <typename Interface>
struct OpInterface : LinkableOp {
  OpInterface() = default;
  OpInterface(Interface op) : LinkableOp(op) {}
  OpInterface(DenseMapOperationKey op) : LinkableOp(op) {}

  Interface interface() const { return cast<Interface>(op); }
  Interface operator*() const { return interface(); }

  operator Interface() { return cast<Interface>(op); }
  operator Interface() const { return cast<Interface>(op); }
};

template <typename Interface>
struct GlobalValueBase : OpInterface<Interface> {
  using Base = OpInterface<Interface>;
  using Base::Base;
  using Base::interface;

  bool isDeclaration() const {
    return this->linker->isDeclaration(interface());
  }

  bool isDeclarationForLinker() const {
    return this->linker->isDeclarationForLinker(interface());
  }

  bool hasExternalLinkage() const { return interface().hasExternalLinkage(); }

  bool hasAvailableExternallyLinkage() const {
    return interface().hasAvailableExternallyLinkage();
  }

  bool hasLinkOnceLinkage() const { return interface().hasLinkOnceLinkage(); }

  bool hasLinkOnceAnyLinkage() const {
    return interface().hasLinkOnceAnyLinkage();
  }

  bool hasLinkOnceODRLinkage() const {
    return interface().hasLinkOnceODRLinkage();
  }

  bool hasWeakLinkage() const { return interface().hasWeakLinkage(); }

  bool hasWeakAnyLinkage() const { return interface().hasWeakAnyLinkage(); }

  bool hasWeakODRLinkage() const { return interface().hasWeakODRLinkage(); }

  bool hasAppendingLinkage() const { return interface().hasAppendingLinkage(); }

  bool hasInternalLinkage() const { return interface().hasInternalLinkage(); }

  bool hasPrivateLinkage() const { return interface().hasPrivateLinkage(); }

  bool hasLocalLinkage() const { return interface().hasLocalLinkage(); }

  bool hasExternalWeakLinkage() const {
    return interface().hasExternalWeakLinkage();
  }

  bool hasCommonLinkage() const { return interface().hasCommonLinkage(); }

  Linkage getLinkage() const { return interface().getLinkage(); }

  void setLinkage(Linkage linkage) { interface().setLinkage(linkage); }

  StringRef getLinkedName() const { return interface().getLinkedName(); }

  void setLinkedName(StringRef name) { llvm_unreachable("Not implemented"); }
};

template <typename Interface>
struct GlobalObjectBase : GlobalValueBase<Interface> {
  using Base = GlobalValueBase<Interface>;
  using Base::Base;
  using Base::interface;
};

template <typename Interface>
struct GlobalAliasBase : GlobalValueBase<Interface> {
  using Base = GlobalValueBase<Interface>;
  using Base::Base;
  using Base::interface;

  Operation *getAliasee() const { llvm_unreachable("Not implemented"); }
};

template <typename Interface>
struct GlobalVariableBase : GlobalObjectBase<Interface> {
  using Base = GlobalObjectBase<Interface>;
  using Base::Base;
  using Base::interface;

  bool isConstant() const { return interface().isConstant(); }

  void setConstant(bool isConstant) { llvm_unreachable("Not implemented"); }

  // TODO fix this not to be optional
  std::optional<uint64_t> getAlignment() const {
    return interface().getAlignment();
  }

  // TODO fix this not to be optional
  void setAlignment(std::optional<uint64_t> alignment) {
    interface().setAlignment(alignment);
  }
};

template <typename Interface>
struct FunctionBase : GlobalObjectBase<Interface> {
  using Base = GlobalObjectBase<Interface>;
  using Base::Base;
  using Base::interface;
};

template <typename Interface>
struct GlobalIFuncBase : GlobalObjectBase<Interface> {
  using Base = GlobalObjectBase<Interface>;
  using Base::Base;
  using Base::interface;

  Operation *getResolver() const { llvm_unreachable("Not implemented"); }
};

struct GlobalValue : GlobalValueBase<GlobalValueLinkageOpInterface> {
  using GlobalValueBase::GlobalValueBase;
};

struct GlobalAlias : GlobalAliasBase<GlobalAliasLinkageOpInterface> {
  using GlobalAliasBase::GlobalAliasBase;

  operator GlobalValue() { return GlobalValue(interface()); }
  operator GlobalValue() const { return GlobalValue(interface()); }
};

struct GlobalVariable : GlobalVariableBase<GlobalVariableLinkageOpInterface> {
  using GlobalVariableBase::GlobalVariableBase;

  operator GlobalValue() { return GlobalValue(interface()); }
  operator GlobalValue() const { return GlobalValue(interface()); }
};

struct Function : FunctionBase<FunctionLinkageOpInterface> {
  using FunctionBase::FunctionBase;

  operator GlobalValue() { return GlobalValue(interface()); }
  operator GlobalValue() const { return GlobalValue(interface()); }
};

struct GlobalIFunc : GlobalIFuncBase<GlobalIFuncLinkageOpInterface> {
  using GlobalIFuncBase::GlobalIFuncBase;

  operator GlobalValue() { return GlobalValue(interface()); }
  operator GlobalValue() const { return GlobalValue(interface()); }
};

} // namespace link
} // namespace mlir

namespace llvm {

///
/// LinkableOp
///
template <typename T>
struct CastInfo<T, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, ::mlir::link::LinkableOp &,
                                     CastInfo<T, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return T::classof(op.getOperation());
  }

  static T doCast(::mlir::link::LinkableOp &op) { return T(op.getOperation()); }
};

template <>
struct CastInfo<::mlir::link::GlobalAlias, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<::mlir::link::GlobalAlias>,
      public DefaultDoCastIfPossible<
          ::mlir::link::GlobalAlias, ::mlir::link::LinkableOp &,
          CastInfo<::mlir::link::GlobalAlias, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return ::mlir::GlobalAliasLinkageOpInterface::classof(op.getOperation());
  }

  static ::mlir::link::GlobalAlias doCast(::mlir::link::LinkableOp &op) {
    return ::mlir::link::GlobalAlias(
        cast<::mlir::GlobalAliasLinkageOpInterface>(op.getOperation()));
  }
};

template <>
struct CastInfo<::mlir::link::GlobalVariable, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<::mlir::link::GlobalVariable>,
      public DefaultDoCastIfPossible<
          ::mlir::link::GlobalVariable, ::mlir::link::LinkableOp &,
          CastInfo<::mlir::link::GlobalVariable, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return ::mlir::GlobalVariableLinkageOpInterface::classof(op.getOperation());
  }

  static ::mlir::link::GlobalVariable doCast(::mlir::link::LinkableOp &op) {
    return ::mlir::link::GlobalVariable(
        cast<::mlir::GlobalVariableLinkageOpInterface>(op.getOperation()));
  }
};

template <>
struct CastInfo<::mlir::link::Function, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<::mlir::link::Function>,
      public DefaultDoCastIfPossible<
          ::mlir::link::Function, ::mlir::link::LinkableOp &,
          CastInfo<::mlir::link::Function, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return ::mlir::FunctionLinkageOpInterface::classof(op.getOperation());
  }

  static ::mlir::link::Function doCast(::mlir::link::LinkableOp &op) {
    return ::mlir::link::Function(
        cast<::mlir::FunctionLinkageOpInterface>(op.getOperation()));
  }
};

template <>
struct CastInfo<::mlir::link::GlobalIFunc, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<::mlir::link::GlobalIFunc>,
      public DefaultDoCastIfPossible<
          ::mlir::link::GlobalIFunc, ::mlir::link::LinkableOp &,
          CastInfo<::mlir::link::GlobalIFunc, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return ::mlir::GlobalIFuncLinkageOpInterface::classof(op.getOperation());
  }

  static ::mlir::link::GlobalIFunc doCast(::mlir::link::LinkableOp &op) {
    return ::mlir::link::GlobalIFunc(
        cast<::mlir::GlobalIFuncLinkageOpInterface>(op.getOperation()));
  }
};

template <typename T>
struct CastInfo<T, const ::mlir::link::LinkableOp>
    : public ConstStrippingForwardingCast<
          T, const ::mlir::link::LinkableOp,
          CastInfo<T, ::mlir::link::LinkableOp>> {};

template <>
struct DenseMapInfo<::mlir::link::LinkableOp>
    : public ::mlir::link::LinkableOpDenseMapInfo<::mlir::link::LinkableOp> {};

///
/// GlobalAlias
///

template <>
struct CastInfo<::mlir::link::GlobalAlias, ::mlir::Operation *>
    : public CastInfo<::mlir::GlobalAliasLinkageOpInterface,
                      ::mlir::Operation *> {};

template <typename T>
struct CastInfo<T, ::mlir::link::GlobalAlias>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::GlobalAlias>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::GlobalAlias>
    : public ::mlir::link::LinkableOpDenseMapInfo<::mlir::link::GlobalAlias> {};

///
/// GlobalValue
///

template <>
struct CastInfo<::mlir::link::GlobalValue, ::mlir::Operation *>
    : public CastInfo<::mlir::GlobalValueLinkageOpInterface,
                      ::mlir::Operation *> {};

template <typename T>
struct CastInfo<T, ::mlir::link::GlobalValue>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::GlobalValue>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::GlobalValue>
    : public ::mlir::link::LinkableOpDenseMapInfo<::mlir::link::GlobalValue> {};

///
/// GlobalVariable
///

template <>
struct CastInfo<::mlir::link::GlobalVariable, ::mlir::Operation *>
    : public CastInfo<::mlir::GlobalVariableLinkageOpInterface,
                      ::mlir::Operation *> {};

template <typename T>
struct CastInfo<T, ::mlir::link::GlobalVariable>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::GlobalVariable>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::GlobalVariable>
    : public ::mlir::link::LinkableOpDenseMapInfo<
          ::mlir::link::GlobalVariable> {};

///
/// Function
///

template <>
struct CastInfo<::mlir::link::Function, ::mlir::Operation *>
    : public CastInfo<::mlir::FunctionLinkageOpInterface, ::mlir::Operation *> {
};

template <typename T>
struct CastInfo<T, ::mlir::link::Function>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::Function>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::Function>
    : public ::mlir::link::LinkableOpDenseMapInfo<::mlir::link::Function> {};

} // namespace llvm

#endif // MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
