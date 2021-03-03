//===- OpClass.h - Helper classes for Op C++ code emission ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several classes for Op C++ code emission. They are only
// expected to be used by MLIR TableGen backends.
//
// We emit the op declaration and definition into separate files: *Ops.h.inc
// and *Ops.cpp.inc. The former is to be included in the dialect *Ops.h and
// the latter for dialect *Ops.cpp. This way provides a cleaner interface.
//
// In order to do this split, we need to track method signature and
// implementation logic separately. Signature information is used for both
// declaration and definition, while implementation logic is only for
// definition. So we have the following classes for C++ code emission.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_OPCLASS_H_
#define MLIR_TABLEGEN_OPCLASS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace mlir {
namespace tblgen {
class FmtObjectBase;

// Class for holding a single parameter of an op's method for C++ code emission.
class OpMethodParameter {
public:
  // Properties (qualifiers) for the parameter.
  enum Property {
    PP_None = 0x0,
    PP_Optional = 0x1,
  };

  OpMethodParameter(StringRef type, StringRef name, StringRef defaultValue = "",
                    Property properties = PP_None)
      : type(type), name(name), defaultValue(defaultValue),
        properties(properties) {}

  OpMethodParameter(StringRef type, StringRef name, Property property)
      : OpMethodParameter(type, name, "", property) {}

  // Writes the parameter as a part of a method declaration to `os`.
  void writeDeclTo(raw_ostream &os) const { writeTo(os, /*emitDefault=*/true); }

  // Writes the parameter as a part of a method definition to `os`
  void writeDefTo(raw_ostream &os) const { writeTo(os, /*emitDefault=*/false); }

  const std::string &getType() const { return type; }
  bool hasDefaultValue() const { return !defaultValue.empty(); }

private:
  void writeTo(raw_ostream &os, bool emitDefault) const;

  std::string type;
  std::string name;
  std::string defaultValue;
  Property properties;
};

// Base class for holding parameters of an op's method for C++ code emission.
class OpMethodParameters {
public:
  // Discriminator for LLVM-style RTTI.
  enum ParamsKind {
    // Separate type and name for each parameter is not known.
    PK_Unresolved,
    // Each parameter is resolved to a type and name.
    PK_Resolved,
  };

  OpMethodParameters(ParamsKind kind) : kind(kind) {}
  virtual ~OpMethodParameters() {}

  // LLVM-style RTTI support.
  ParamsKind getKind() const { return kind; }

  // Writes the parameters as a part of a method declaration to `os`.
  virtual void writeDeclTo(raw_ostream &os) const = 0;

  // Writes the parameters as a part of a method definition to `os`
  virtual void writeDefTo(raw_ostream &os) const = 0;

  // Factory methods to create the correct type of `OpMethodParameters`
  // object based on the arguments.
  static std::unique_ptr<OpMethodParameters> create();

  static std::unique_ptr<OpMethodParameters> create(StringRef params);

  static std::unique_ptr<OpMethodParameters>
  create(llvm::SmallVectorImpl<OpMethodParameter> &&params);

  static std::unique_ptr<OpMethodParameters>
  create(StringRef type, StringRef name, StringRef defaultValue = "");

private:
  const ParamsKind kind;
};

// Class for holding unresolved parameters.
class OpMethodUnresolvedParameters : public OpMethodParameters {
public:
  OpMethodUnresolvedParameters(StringRef params)
      : OpMethodParameters(PK_Unresolved), parameters(params) {}

  // write the parameters as a part of a method declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const override;

  // write the parameters as a part of a method definition to the given `os`
  void writeDefTo(raw_ostream &os) const override;

  // LLVM-style RTTI support.
  static bool classof(const OpMethodParameters *params) {
    return params->getKind() == PK_Unresolved;
  }

private:
  std::string parameters;
};

// Class for holding resolved parameters.
class OpMethodResolvedParameters : public OpMethodParameters {
public:
  OpMethodResolvedParameters() : OpMethodParameters(PK_Resolved) {}

  OpMethodResolvedParameters(llvm::SmallVectorImpl<OpMethodParameter> &&params)
      : OpMethodParameters(PK_Resolved) {
    for (OpMethodParameter &param : params)
      parameters.emplace_back(std::move(param));
  }

  OpMethodResolvedParameters(StringRef type, StringRef name,
                             StringRef defaultValue)
      : OpMethodParameters(PK_Resolved) {
    parameters.emplace_back(type, name, defaultValue);
  }

  // Returns the number of parameters.
  size_t getNumParameters() const { return parameters.size(); }

  // Returns if this method makes the `other` method redundant. Note that this
  // is more than just finding conflicting methods. This method determines if
  // the 2 set of parameters are conflicting and if so, returns true if this
  // method has a more general set of parameters that can replace all possible
  // calls to the `other` method.
  bool makesRedundant(const OpMethodResolvedParameters &other) const;

  // write the parameters as a part of a method declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const override;

  // write the parameters as a part of a method definition to the given `os`
  void writeDefTo(raw_ostream &os) const override;

  // LLVM-style RTTI support.
  static bool classof(const OpMethodParameters *params) {
    return params->getKind() == PK_Resolved;
  }

private:
  llvm::SmallVector<OpMethodParameter, 4> parameters;
};

// Class for holding the signature of an op's method for C++ code emission
class OpMethodSignature {
public:
  template <typename... Args>
  OpMethodSignature(StringRef retType, StringRef name, Args &&...args)
      : returnType(retType), methodName(name),
        parameters(OpMethodParameters::create(std::forward<Args>(args)...)) {}
  OpMethodSignature(OpMethodSignature &&) = default;

  // Returns if a method with this signature makes a method with `other`
  // signature redundant. Only supports resolved parameters.
  bool makesRedundant(const OpMethodSignature &other) const;

  // Returns the number of parameters (for resolved parameters).
  size_t getNumParameters() const {
    return cast<OpMethodResolvedParameters>(parameters.get())
        ->getNumParameters();
  }

  // Returns the name of the method.
  StringRef getName() const { return methodName; }

  // Writes the signature as a method declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;

  // Writes the signature as the start of a method definition to the given `os`.
  // `namePrefix` is the prefix to be prepended to the method name (typically
  // namespaces for qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  std::string returnType;
  std::string methodName;
  std::unique_ptr<OpMethodParameters> parameters;
};

// Class for holding the body of an op's method for C++ code emission
class OpMethodBody {
public:
  explicit OpMethodBody(bool declOnly);

  OpMethodBody &operator<<(Twine content);
  OpMethodBody &operator<<(int content);
  OpMethodBody &operator<<(const FmtObjectBase &content);

  void writeTo(raw_ostream &os) const;

private:
  // Whether this class should record method body.
  bool isEffective;
  std::string body;
};

// Class for holding an op's method for C++ code emission
class OpMethod {
public:
  // Properties (qualifiers) of class methods. Bitfield is used here to help
  // querying properties.
  enum Property {
    MP_None = 0x0,
    MP_Static = 0x1,
    MP_Constructor = 0x2,
    MP_Private = 0x4,
    MP_Declaration = 0x8,
    MP_Inline = 0x10,
    MP_Constexpr = 0x20 | MP_Inline,
    MP_StaticDeclaration = MP_Static | MP_Declaration,
  };

  template <typename... Args>
  OpMethod(StringRef retType, StringRef name, Property property, unsigned id,
           Args &&...args)
      : properties(property),
        methodSignature(retType, name, std::forward<Args>(args)...),
        methodBody(properties & MP_Declaration), id(id) {}

  OpMethod(OpMethod &&) = default;

  virtual ~OpMethod() = default;

  OpMethodBody &body() { return methodBody; }

  // Returns true if this is a static method.
  bool isStatic() const { return properties & MP_Static; }

  // Returns true if this is a private method.
  bool isPrivate() const { return properties & MP_Private; }

  // Returns true if this is an inline method.
  bool isInline() const { return properties & MP_Inline; }

  // Returns the name of this method.
  StringRef getName() const { return methodSignature.getName(); }

  // Returns the ID for this method
  unsigned getID() const { return id; }

  // Returns if this method makes the `other` method redundant.
  bool makesRedundant(const OpMethod &other) const {
    return methodSignature.makesRedundant(other.methodSignature);
  }

  // Writes the method as a declaration to the given `os`.
  virtual void writeDeclTo(raw_ostream &os) const;

  // Writes the method as a definition to the given `os`. `namePrefix` is the
  // prefix to be prepended to the method name (typically namespaces for
  // qualifying the method definition).
  virtual void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

protected:
  Property properties;
  OpMethodSignature methodSignature;
  OpMethodBody methodBody;
  const unsigned id;
};

// Class for holding an op's constructor method for C++ code emission.
class OpConstructor : public OpMethod {
public:
  template <typename... Args>
  OpConstructor(StringRef className, Property property, unsigned id,
                Args &&...args)
      : OpMethod("", className, property, id, std::forward<Args>(args)...) {}

  // Add member initializer to constructor initializing `name` with `value`.
  void addMemberInitializer(StringRef name, StringRef value);

  // Writes the method as a definition to the given `os`. `namePrefix` is the
  // prefix to be prepended to the method name (typically namespaces for
  // qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const override;

private:
  // Member initializers.
  std::string memberInitializers;
};

// A class used to emit C++ classes from Tablegen.  Contains a list of public
// methods and a list of private fields to be emitted.
class Class {
public:
  explicit Class(StringRef name);

  // Adds a new method to this class and prune redundant methods. Returns null
  // if the method was not added (because an existing method would make it
  // redundant), else returns a pointer to the added method. Note that this call
  // may also delete existing methods that are made redundant by a method to the
  // class.
  template <typename... Args>
  OpMethod *addMethodAndPrune(StringRef retType, StringRef name,
                              OpMethod::Property properties, Args &&...args) {
    auto newMethod = std::make_unique<OpMethod>(
        retType, name, properties, nextMethodID++, std::forward<Args>(args)...);
    return addMethodAndPrune(methods, std::move(newMethod));
  }

  template <typename... Args>
  OpMethod *addMethodAndPrune(StringRef retType, StringRef name,
                              Args &&...args) {
    return addMethodAndPrune(retType, name, OpMethod::MP_None,
                             std::forward<Args>(args)...);
  }

  template <typename... Args>
  OpConstructor *addConstructorAndPrune(Args &&...args) {
    auto newConstructor = std::make_unique<OpConstructor>(
        getClassName(), OpMethod::MP_Constructor, nextMethodID++,
        std::forward<Args>(args)...);
    return addMethodAndPrune(constructors, std::move(newConstructor));
  }

  // Creates a new field in this class.
  void newField(StringRef type, StringRef name, StringRef defaultValue = "");

  // Writes this op's class as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  // Writes the method definitions in this op's class to the given `os`.
  void writeDefTo(raw_ostream &os) const;

  // Returns the C++ class name of the op.
  StringRef getClassName() const { return className; }

protected:
  // Get a list of all the methods to emit, filtering out hidden ones.
  void forAllMethods(llvm::function_ref<void(const OpMethod &)> func) const {
    using ConsRef = const std::unique_ptr<OpConstructor> &;
    using MethodRef = const std::unique_ptr<OpMethod> &;
    llvm::for_each(constructors, [&](ConsRef ptr) { func(*ptr); });
    llvm::for_each(methods, [&](MethodRef ptr) { func(*ptr); });
  }

  // For deterministic code generation, keep methods sorted in the order in
  // which they were generated.
  template <typename MethodTy>
  struct MethodCompare {
    bool operator()(const std::unique_ptr<MethodTy> &x,
                    const std::unique_ptr<MethodTy> &y) const {
      return x->getID() < y->getID();
    }
  };

  template <typename MethodTy>
  using MethodSet =
      std::set<std::unique_ptr<MethodTy>, MethodCompare<MethodTy>>;

  template <typename MethodTy>
  MethodTy *addMethodAndPrune(MethodSet<MethodTy> &set,
                              std::unique_ptr<MethodTy> &&newMethod) {
    // Check if the new method will be made redundant by existing methods.
    for (auto &method : set)
      if (method->makesRedundant(*newMethod))
        return nullptr;

    // We can add this a method to the set. Prune any existing methods that will
    // be made redundant by adding this new method. Note that the redundant
    // check between two methods is more than a conflict check. makesRedundant()
    // below will check if the new method conflicts with an existing method and
    // if so, returns true if the new method makes the existing method redundant
    // because all calls to the existing method can be subsumed by the new
    // method. So makesRedundant() does a combined job of finding conflicts and
    // deciding which of the 2 conflicting methods survive.
    //
    // Note: llvm::erase_if does not work with sets of std::unique_ptr, so doing
    // it manually here.
    for (auto it = set.begin(), end = set.end(); it != end;) {
      if (newMethod->makesRedundant(*(it->get())))
        it = set.erase(it);
      else
        ++it;
    }

    MethodTy *ret = newMethod.get();
    set.insert(std::move(newMethod));
    return ret;
  }

  std::string className;
  MethodSet<OpConstructor> constructors;
  MethodSet<OpMethod> methods;
  unsigned nextMethodID = 0;
  SmallVector<std::string, 4> fields;
};

// Class for holding an op for C++ code emission
class OpClass : public Class {
public:
  explicit OpClass(StringRef name, StringRef extraClassDeclaration = "");

  // Adds an op trait.
  void addTrait(Twine trait);

  // Writes this op's class as a declaration to the given `os`.  Redefines
  // Class::writeDeclTo to also emit traits and extra class declarations.
  void writeDeclTo(raw_ostream &os) const;

private:
  StringRef extraClassDeclaration;
  SmallVector<std::string, 4> traitsVec;
  StringSet<> traitsSet;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_OPCLASS_H_
