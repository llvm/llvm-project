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

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
class DialectInterface;
class OpBuilder;
class Type;

using DialectConstantDecodeHook =
    std::function<bool(const OpaqueElementsAttr, ElementsAttr &)>;
using DialectConstantFoldHook = std::function<LogicalResult(
    Operation *, ArrayRef<Attribute>, SmallVectorImpl<Attribute> &)>;
using DialectExtractElementHook =
    std::function<Attribute(const OpaqueElementsAttr, ArrayRef<uint64_t>)>;
using DialectAllocatorFunction = std::function<void(MLIRContext *)>;

/// Dialects are groups of MLIR operations and behavior associated with the
/// entire group.  For example, hooks into other systems for constant folding,
/// default named types for asm printing, etc.
///
/// Instances of the dialect object are global across all MLIRContext's that may
/// be active in the process.
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

  //===--------------------------------------------------------------------===//
  // Constant Hooks
  //===--------------------------------------------------------------------===//

  /// Registered fallback constant fold hook for the dialect. Like the constant
  /// fold hook of each operation, it attempts to constant fold the operation
  /// with the specified constant operand values - the elements in "operands"
  /// will correspond directly to the operands of the operation, but may be null
  /// if non-constant.  If constant folding is successful, this fills in the
  /// `results` vector.  If not, this returns failure and `results` is
  /// unspecified.
  DialectConstantFoldHook constantFoldHook =
      [](Operation *op, ArrayRef<Attribute> operands,
         SmallVectorImpl<Attribute> &results) { return failure(); };

  /// Registered hook to decode opaque constants associated with this
  /// dialect. The hook function attempts to decode an opaque constant tensor
  /// into a tensor with non-opaque content. If decoding is successful, this
  /// method returns false and sets 'output' attribute. If not, it returns true
  /// and leaves 'output' unspecified. The default hook fails to decode.
  DialectConstantDecodeHook decodeHook =
      [](const OpaqueElementsAttr input, ElementsAttr &output) { return true; };

  /// Registered hook to extract an element from an opaque constant associated
  /// with this dialect. If element has been successfully extracted, this
  /// method returns that element. If not, it returns an empty attribute.
  /// The default hook fails to extract an element.
  DialectExtractElementHook extractElementHook =
      [](const OpaqueElementsAttr input, ArrayRef<uint64_t> index) {
        return Attribute();
      };

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
        0, (addOperation(AbstractOperation::get<Args>(*this)), 0)...};
  }

  void addOperation(AbstractOperation opInfo);

  /// This method is used by derived classes to add their types to the set.
  template <typename... Args> void addTypes() {
    (void)std::initializer_list<int>{
        0, (addType(Args::getTypeID(), AbstractType::get<Args>(*this)), 0)...};
  }
  void addType(TypeID typeID, AbstractType &&typeInfo);

  /// This method is used by derived classes to add their attributes to the set.
  template <typename... Args> void addAttributes() {
    (void)std::initializer_list<int>{
        0,
        (addAttribute(Args::getTypeID(), AbstractAttribute::get<Args>(*this)),
         0)...};
  }
  void addAttribute(TypeID typeID, AbstractAttribute &&attrInfo);

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

  /// Registers a specific dialect creation function with the global registry.
  /// Used through the registerDialect template.
  /// Registrations are deduplicated by dialect TypeID and only the first
  /// registration will be used.
  static void
  registerDialectAllocator(TypeID typeID,
                           const DialectAllocatorFunction &function);
  template <typename ConcreteDialect>
  friend void registerDialect();
  friend class MLIRContext;
};

/// Registers all dialects and hooks from the global registries with the
/// specified MLIRContext.
/// Note: This method is not thread-safe.
void registerAllDialects(MLIRContext *context);

/// Utility to register a dialect. Client can register their dialect with the
/// global registry by calling registerDialect<MyDialect>();
/// Note: This method is not thread-safe.
template <typename ConcreteDialect> void registerDialect() {
  Dialect::registerDialectAllocator(
      TypeID::get<ConcreteDialect>(),
      [](MLIRContext *ctx) { ctx->getOrCreateDialect<ConcreteDialect>(); });
}

/// DialectRegistration provides a global initializer that registers a Dialect
/// allocation routine.
///
/// Usage:
///
///   // At namespace scope.
///   static DialectRegistration<MyDialect> Unused;
template <typename ConcreteDialect> struct DialectRegistration {
  DialectRegistration() { registerDialect<ConcreteDialect>(); }
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
