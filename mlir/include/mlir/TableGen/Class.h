//===- Class.h - Helper classes for C++ code emission -----------*- C++ -*-===//
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

#ifndef MLIR_TABLEGEN_CLASS_H_
#define MLIR_TABLEGEN_CLASS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace mlir {
namespace tblgen {
class FmtObjectBase;

/// This class contains a single method parameter for a C++ function.
class MethodParameter {
public:
  /// Create a method parameter with a C++ type, parameter name, and an optional
  /// default value. Marking a parameter as "optional" is a cosmetic effect on
  /// the generated code.
  template <typename TypeT, typename NameT, typename DefaultT>
  MethodParameter(TypeT &&type, NameT &&name, DefaultT &&defaultValue,
                  bool optional = false)
      : type(stringify(std::forward<TypeT>(type))),
        name(stringify(std::forward<NameT>(name))),
        defaultValue(stringify(std::forward<DefaultT>(defaultValue))),
        optional(optional) {}

  /// Create a method parameter with a C++ type, parameter name, and no default
  /// value.
  template <typename TypeT, typename NameT>
  MethodParameter(TypeT &&type, NameT &&name, bool optional = false)
      : MethodParameter(std::forward<TypeT>(type), std::forward<NameT>(name),
                        /*defaultValue=*/"", optional) {}

  /// Write the parameter as part of a method declaration.
  void writeDeclTo(raw_ostream &os) const { writeTo(os, /*emitDefault=*/true); }
  /// Write the parameter as part of a method definition.
  void writeDefTo(raw_ostream &os) const { writeTo(os, /*emitDefault=*/false); }

  /// Get the C++ type.
  const std::string &getType() const { return type; }
  /// Returns true if the parameter has a default value.
  bool hasDefaultValue() const { return !defaultValue.empty(); }

private:
  void writeTo(raw_ostream &os, bool emitDefault) const;

  /// The C++ type.
  std::string type;
  /// The variable name.
  std::string name;
  /// An optional default value. The default value exists if the string is not
  /// empty.
  std::string defaultValue;
  /// Whether the parameter should be indicated as "optional".
  bool optional;
};

/// This class contains a list of method parameters for constructor, class
/// methods, and method signatures.
class MethodParameters {
public:
  /// Create a list of method parameters.
  MethodParameters(std::initializer_list<MethodParameter> parameters)
      : parameters(parameters) {}
  MethodParameters(SmallVector<MethodParameter> parameters)
      : parameters(std::move(parameters)) {}

  /// Write the parameters as part of a method declaration.
  void writeDeclTo(raw_ostream &os) const;
  /// Write the parameters as part of a method definition.
  void writeDefTo(raw_ostream &os) const;

  /// Determine whether this list of parameters "subsumes" another, which occurs
  /// when this parameter list is identical to the other and has zero or more
  /// additional default-valued parameters.
  bool subsumes(const MethodParameters &other) const;

  /// Return the number of parameters.
  unsigned getNumParameters() const { return parameters.size(); }

private:
  llvm::SmallVector<MethodParameter> parameters;
};

/// This class contains the signature of a C++ method, including the return
/// type. method name, and method parameters.
class MethodSignature {
public:
  MethodSignature(StringRef retType, StringRef name,
                  SmallVector<MethodParameter> &&parameters)
      : returnType(retType), methodName(name),
        parameters(std::move(parameters)) {}
  template <typename... Parameters>
  MethodSignature(StringRef retType, StringRef name, Parameters &&...parameters)
      : returnType(retType), methodName(name),
        parameters({std::forward<Parameters>(parameters)...}) {}

  /// Determine whether a method with this signature makes a method with
  /// `other` signature redundant. This occurs if the signatures have the same
  /// name and this signature's parameteres subsume the other's.
  ///
  /// A method that makes another method redundant with a different return type
  /// can replace the other, the assumption being that the subsuming method
  /// provides a more resolved return type, e.g. IntegerAttr vs. Attribute.
  bool makesRedundant(const MethodSignature &other) const;

  /// Get the name of the method.
  StringRef getName() const { return methodName; }

  /// Get the number of parameters.
  unsigned getNumParameters() const { return parameters.getNumParameters(); }

  /// Write the signature as part of a method declaration.
  void writeDeclTo(raw_ostream &os) const;

  /// Write the signature as part of a method definition. `namePrefix` is to be
  /// prepended to the method name (typically namespaces for qualifying the
  /// method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

private:
  /// The method's C++ return type.
  std::string returnType;
  /// The method name.
  std::string methodName;
  /// The method's parameter list.
  MethodParameters parameters;
};

/// Class for holding the body of an op's method for C++ code emission
class MethodBody {
public:
  explicit MethodBody(bool declOnly);

  MethodBody &operator<<(Twine content);
  MethodBody &operator<<(int content);
  MethodBody &operator<<(const FmtObjectBase &content);

  void writeTo(raw_ostream &os) const;

private:
  /// Whether this class should record method body.
  bool isEffective;
  /// The body of the method.
  std::string body;
};

/// Class for holding an op's method for C++ code emission
class Method {
public:
  /// Properties (qualifiers) of class methods. Bitfield is used here to help
  /// querying properties.
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
  Method(StringRef retType, StringRef name, Property property, Args &&...args)
      : properties(property),
        methodSignature(retType, name, std::forward<Args>(args)...),
        methodBody(properties & MP_Declaration) {}

  Method(Method &&) = default;
  Method &operator=(Method &&) = default;

  virtual ~Method() = default;

  MethodBody &body() { return methodBody; }

  /// Returns true if this is a static method.
  bool isStatic() const { return properties & MP_Static; }

  /// Returns true if this is a private method.
  bool isPrivate() const { return properties & MP_Private; }

  /// Returns true if this is an inline method.
  bool isInline() const { return properties & MP_Inline; }

  /// Returns the name of this method.
  StringRef getName() const { return methodSignature.getName(); }

  /// Returns if this method makes the `other` method redundant.
  bool makesRedundant(const Method &other) const {
    return methodSignature.makesRedundant(other.methodSignature);
  }

  /// Writes the method as a declaration to the given `os`.
  virtual void writeDeclTo(raw_ostream &os) const;

  /// Writes the method as a definition to the given `os`. `namePrefix` is the
  /// prefix to be prepended to the method name (typically namespaces for
  /// qualifying the method definition).
  virtual void writeDefTo(raw_ostream &os, StringRef namePrefix) const;

protected:
  /// A collection of method properties.
  Property properties;
  /// The signature of the method.
  MethodSignature methodSignature;
  /// The body of the method, if it has one.
  MethodBody methodBody;
};

} // end namespace tblgen
} // end namespace mlir

/// The OR of two method properties should return method properties. Ensure that
/// this function is visible to `Class`.
inline constexpr mlir::tblgen::Method::Property
operator|(mlir::tblgen::Method::Property lhs,
          mlir::tblgen::Method::Property rhs) {
  return mlir::tblgen::Method::Property(static_cast<unsigned>(lhs) |
                                        static_cast<unsigned>(rhs));
}

namespace mlir {
namespace tblgen {

/// Class for holding an op's constructor method for C++ code emission.
class Constructor : public Method {
public:
  template <typename... Parameters>
  Constructor(StringRef className, Property property,
              Parameters &&...parameters)
      : Method("", className, property,
               std::forward<Parameters>(parameters)...) {}

  /// Add member initializer to constructor initializing `name` with `value`.
  void addMemberInitializer(StringRef name, StringRef value);

  /// Writes the method as a definition to the given `os`. `namePrefix` is the
  /// prefix to be prepended to the method name (typically namespaces for
  /// qualifying the method definition).
  void writeDefTo(raw_ostream &os, StringRef namePrefix) const override;

private:
  /// Member initializers.
  std::string memberInitializers;
};

/// A class used to emit C++ classes from Tablegen.  Contains a list of public
/// methods and a list of private fields to be emitted.
class Class {
public:
  explicit Class(StringRef name);

  /// Add a new constructor to this class and prune and constructors made
  /// redundant by it. Returns null if the constructor was not added. Else,
  /// returns a pointer to the new constructor.
  template <typename... Parameters>
  Constructor *addConstructorAndPrune(Parameters &&...parameters) {
    return addConstructorAndPrune(
        Constructor(getClassName(), Method::MP_Constructor,
                    std::forward<Parameters>(parameters)...));
  }

  /// Add a new method to this class and prune any methods made redundant by it.
  /// Returns null if the method was not added (because an existing method would
  /// make it redundant). Else, returns a pointer to the new method.
  template <typename... Parameters>
  Method *addMethod(StringRef retType, StringRef name,
                    Method::Property properties, Parameters &&...parameters) {
    return addMethodAndPrune(Method(retType, name, properties,
                                    std::forward<Parameters>(parameters)...));
  }

  /// Add a method with statically-known properties.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *addMethod(StringRef retType, StringRef name,
                    Parameters &&...parameters) {
    return addMethod(retType, name, Properties,
                             std::forward<Parameters>(parameters)...);
  }

  /// Add a static method.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *addStaticMethod(StringRef retType, StringRef name,
                          Parameters &&...parameters) {
    return addMethod<Properties | Method::MP_Static>(
        retType, name, std::forward<Parameters>(parameters)...);
  }

  /// Add an inline static method.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *addStaticInlineMethod(StringRef retType, StringRef name,
                                Parameters &&...parameters) {
    return addMethod<Properties | Method::MP_Static | Method::MP_Inline>(
        retType, name, std::forward<Parameters>(parameters)...);
  }

  /// Add an inline method.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *addInlineMethod(StringRef retType, StringRef name,
                          Parameters &&...parameters) {
    return addMethod<Properties | Method::MP_Inline>(
        retType, name, std::forward<Parameters>(parameters)...);
  }

  /// Add a declaration for a method.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *declareMethod(StringRef retType, StringRef name,
                        Parameters &&...parameters) {
    return addMethod<Properties | Method::MP_Declaration>(
        retType, name, std::forward<Parameters>(parameters)...);
  }

  /// Add a declaration for a static method.
  template <Method::Property Properties = Method::MP_None,
            typename... Parameters>
  Method *declareStaticMethod(StringRef retType, StringRef name,
                              Parameters &&...parameters) {
    return addMethod<Properties | Method::MP_StaticDeclaration>(
        retType, name, std::forward<Parameters>(parameters)...);
  }

  /// Creates a new field in this class.
  void newField(StringRef type, StringRef name, StringRef defaultValue = "");

  /// Writes this op's class as a declaration to the given `os`.
  void writeDeclTo(raw_ostream &os) const;
  /// Writes the method definitions in this op's class to the given `os`.
  void writeDefTo(raw_ostream &os) const;

  /// Returns the C++ class name of the op.
  StringRef getClassName() const { return className; }

protected:
  /// Get a list of all the methods to emit, filtering out hidden ones.
  void forAllMethods(llvm::function_ref<void(const Method &)> func) const {
    llvm::for_each(constructors, [&](auto &ctor) { func(ctor); });
    llvm::for_each(methods, [&](auto &method) { func(method); });
  }

  /// Add a new constructor if it is not made redundant by any existing
  /// constructors and prune and existing constructors made redundant.
  Constructor *addConstructorAndPrune(Constructor &&newCtor);
  /// Add a new method if it is not made redundant by any existing methods and
  /// prune and existing methods made redundant.
  Method *addMethodAndPrune(Method &&newMethod);

  /// The C++ class name.
  std::string className;
  /// The list of constructors.
  std::vector<Constructor> constructors;
  /// The list of class methods.
  std::vector<Method> methods;
  /// The list of class members.
  SmallVector<std::string, 4> fields;
};

// Class for holding an op for C++ code emission
class OpClass : public Class {
public:
  explicit OpClass(StringRef name, StringRef extraClassDeclaration = "");

  /// Adds an op trait.
  void addTrait(Twine trait);

  /// Writes this op's class as a declaration to the given `os`.  Redefines
  /// Class::writeDeclTo to also emit traits and extra class declarations.
  void writeDeclTo(raw_ostream &os) const;

private:
  StringRef extraClassDeclaration;
  llvm::SetVector<std::string, SmallVector<std::string>, StringSet<>> traits;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CLASS_H_
