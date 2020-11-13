//===- Dialect.h - IR Dialect Description -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the 'dialect' abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECT_H
#define MLIR_IR_DIALECT_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

#include <map>

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
class DialectInterface;
class OpBuilder;
class Type;

using DialectAllocatorFunction = std::function<Dialect *(MLIRContext *)>;

/// Dialects are groups of MLIR operations, types and attributes, as well as
/// behavior associated with the entire group.  For example, hooks into other
/// systems for constant folding, interfaces, default named types for asm
/// printing, etc.
///
/// Instances of the dialect object are loaded in a specific MLIRContext.
///
class Dialect {
public:
  virtual ~Dialect();

  /// Utility function that returns if the given string is a valid dialect
  /// namespace.
  static bool isValidNamespace(StringRef str);

  MLIRContext *getContext() const { return context; }

  StringRef getNamespace() const { return name; }

  /// Returns the unique identifier that corresponds to this dialect.
  TypeID getTypeID() const { return dialectID; }

  /// Returns true if this dialect allows for unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowsUnknownOperations() const { return unknownOpsAllowed; }

  /// Return true if this dialect allows for unregistered types, i.e., types
  /// prefixed with the dialect namespace but not registered with addType.
  /// These are represented with OpaqueType.
  bool allowsUnknownTypes() const { return unknownTypesAllowed; }

  /// Registered hook to materialize a single constant operation from a given
  /// attribute value with the desired resultant type. This method should use
  /// the provided builder to create the operation without changing the
  /// insertion position. The generated operation is expected to be constant
  /// like, i.e. single result, zero operands, non side-effecting, etc. On
  /// success, this hook should return the value generated to represent the
  /// constant value. Otherwise, it should return null on failure.
  virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Parsing Hooks
  //===--------------------------------------------------------------------===//

  /// Parse an attribute registered to this dialect. If 'type' is nonnull, it
  /// refers to the expected type of the attribute.
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type) const;

  /// Print an attribute registered to this dialect. Note: The type of the
  /// attribute need not be printed by this method as it is always printed by
  /// the caller.
  virtual void printAttribute(Attribute, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered attribute printing hook");
  }

  /// Parse a type registered to this dialect.
  virtual Type parseType(DialectAsmParser &parser) const;

  /// Print a type registered to this dialect.
  virtual void printType(Type, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered type printing hook");
  }

  //===--------------------------------------------------------------------===//
  // Verification Hooks
  //===--------------------------------------------------------------------===//

  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionArgAttribute(Operation *,
                                                 unsigned regionIndex,
                                                 unsigned argIndex,
                                                 NamedAttribute);

  /// Verify an attribute from this dialect on the result at 'resultIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionResultAttribute(Operation *,
                                                    unsigned regionIndex,
                                                    unsigned resultIndex,
                                                    NamedAttribute);

  /// Verify an attribute from this dialect on the given operation. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute) {
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Interfaces
  //===--------------------------------------------------------------------===//

  /// Lookup an interface for the given ID if one is registered, otherwise
  /// nullptr.
  const DialectInterface *getRegisteredInterface(TypeID interfaceID) {
    auto it = registeredInterfaces.find(interfaceID);
    return it != registeredInterfaces.end() ? it->getSecond().get() : nullptr;
  }
  template <typename InterfaceT> const InterfaceT *getRegisteredInterface() {
    return static_cast<const InterfaceT *>(
        getRegisteredInterface(InterfaceT::getInterfaceID()));
  }

protected:
  /// The constructor takes a unique namespace for this dialect as well as the
  /// context to bind to.
  /// Note: The namespace must not contain '.' characters.
  /// Note: All operations belonging to this dialect must have names starting
  ///       with the namespace followed by '.'.
  /// Example:
  ///       - "tf" for the TensorFlow ops like "tf.add".
  Dialect(StringRef name, MLIRContext *context, TypeID id);

  /// This method is used by derived classes to add their operations to the set.
  ///
  template <typename... Args> void addOperations() {
    (void)std::initializer_list<int>{
        0, (AbstractOperation::insert<Args>(*this), 0)...};
  }

  /// Register a set of type classes with this dialect.
  template <typename... Args> void addTypes() {
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

  /// Register a set of attribute classes with this dialect.
  template <typename... Args> void addAttributes() {
    (void)std::initializer_list<int>{0, (addAttribute<Args>(), 0)...};
  }

  /// Enable support for unregistered operations.
  void allowUnknownOperations(bool allow = true) { unknownOpsAllowed = allow; }

  /// Enable support for unregistered types.
  void allowUnknownTypes(bool allow = true) { unknownTypesAllowed = allow; }

  /// Register a dialect interface with this dialect instance.
  void addInterface(std::unique_ptr<DialectInterface> interface);

  /// Register a set of dialect interfaces with this dialect instance.
  template <typename... Args> void addInterfaces() {
    (void)std::initializer_list<int>{
        0, (addInterface(std::make_unique<Args>(this)), 0)...};
  }

private:
  Dialect(const Dialect &) = delete;
  void operator=(Dialect &) = delete;

  /// Register an attribute instance with this dialect.
  template <typename T> void addAttribute() {
    // Add this attribute to the dialect and register it with the uniquer.
    addAttribute(T::getTypeID(), AbstractAttribute::get<T>(*this));
    detail::AttributeUniquer::registerAttribute<T>(context);
  }
  void addAttribute(TypeID typeID, AbstractAttribute &&attrInfo);

  /// Register a type instance with this dialect.
  template <typename T> void addType() {
    // Add this type to the dialect and register it with the uniquer.
    addType(T::getTypeID(), AbstractType::get<T>(*this));
    detail::TypeUniquer::registerType<T>(context);
  }
  void addType(TypeID typeID, AbstractType &&typeInfo);

  /// The namespace of this dialect.
  StringRef name;

  /// The unique identifier of the derived Op class, this is used in the context
  /// to allow registering multiple times the same dialect.
  TypeID dialectID;

  /// This is the context that owns this Dialect object.
  MLIRContext *context;

  /// Flag that specifies whether this dialect supports unregistered operations,
  /// i.e. operations prefixed with the dialect namespace but not registered
  /// with addOperation.
  bool unknownOpsAllowed = false;

  /// Flag that specifies whether this dialect allows unregistered types, i.e.
  /// types prefixed with the dialect namespace but not registered with addType.
  /// These types are represented with OpaqueType.
  bool unknownTypesAllowed = false;

  /// A collection of registered dialect interfaces.
  DenseMap<TypeID, std::unique_ptr<DialectInterface>> registeredInterfaces;

  friend void registerDialect();
  friend class MLIRContext;
};

/// The DialectRegistry maps a dialect namespace to a constructor for the
/// matching dialect.
/// This allows for decoupling the list of dialects "available" from the
/// dialects loaded in the Context. The parser in particular will lazily load
/// dialects in the Context as operations are encountered.
class DialectRegistry {
  using MapTy =
      std::map<std::string, std::pair<TypeID, DialectAllocatorFunction>>;

public:
  template <typename ConcreteDialect>
  void insert() {
    insert(TypeID::get<ConcreteDialect>(),
           ConcreteDialect::getDialectNamespace(),
           static_cast<DialectAllocatorFunction>(([](MLIRContext *ctx) {
             // Just allocate the dialect, the context
             // takes ownership of it.
             return ctx->getOrLoadDialect<ConcreteDialect>();
           })));
  }

  template <typename ConcreteDialect, typename OtherDialect,
            typename... MoreDialects>
  void insert() {
    insert<ConcreteDialect>();
    insert<OtherDialect, MoreDialects...>();
  }

  /// Add a new dialect constructor to the registry.
  void insert(TypeID typeID, StringRef name, DialectAllocatorFunction ctor);

  /// Load a dialect for this namespace in the provided context.
  Dialect *loadByName(StringRef name, MLIRContext *context);

  // Register all dialects available in the current registry with the registry
  // in the provided context.
  void appendTo(DialectRegistry &destination) {
    for (const auto &nameAndRegistrationIt : registry)
      destination.insert(nameAndRegistrationIt.second.first,
                         nameAndRegistrationIt.first,
                         nameAndRegistrationIt.second.second);
  }
  // Load all dialects available in the registry in the provided context.
  void loadAll(MLIRContext *context) {
    for (const auto &nameAndRegistrationIt : registry)
      nameAndRegistrationIt.second.second(context);
  }

  MapTy::const_iterator begin() const { return registry.begin(); }
  MapTy::const_iterator end() const { return registry.end(); }

private:
  MapTy registry;
};

} // namespace mlir

namespace llvm {
/// Provide isa functionality for Dialects.
template <typename T>
struct isa_impl<T, ::mlir::Dialect> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::TypeID::get<T>() == dialect.getTypeID();
  }
};
} // namespace llvm

#endif
