//===- Class.h - Helper classes for C++ code emission -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several classes for C++ code emission. They are only
// expected to be used by MLIR TableGen backends.
//
// We emit the declarations and definitions into separate files: *.h.inc and
// *.cpp.inc. The former is to be included in the dialect *.h and the latter for
// dialect *.cpp. This way provides a cleaner interface.
//
// In order to do this split, we need to track method signature and
// implementation logic separately. Signature information is used for both
// declaration and definition, while implementation logic is only for
// definition. So we have the following classes for C++ code emission.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CLASS_H_
#define MLIR_TABLEGEN_CLASS_H_

#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

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
  void writeDeclTo(raw_indented_ostream &os) const;
  /// Write the parameter as part of a method definition.
  void writeDefTo(raw_indented_ostream &os) const;

  /// Get the C++ type.
  StringRef getType() const { return type; }
  /// Returns true if the parameter has a default value.
  bool hasDefaultValue() const { return !defaultValue.empty(); }

private:
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
  void writeDeclTo(raw_indented_ostream &os) const;
  /// Write the parameters as part of a method definition.
  void writeDefTo(raw_indented_ostream &os) const;

  /// Determine whether this list of parameters "subsumes" another, which occurs
  /// when this parameter list is identical to the other and has zero or more
  /// additional default-valued parameters.
  bool subsumes(const MethodParameters &other) const;

  /// Return the number of parameters.
  unsigned getNumParameters() const { return parameters.size(); }

private:
  /// The list of parameters.
  SmallVector<MethodParameter> parameters;
};

/// This class contains the signature of a C++ method, including the return
/// type. method name, and method parameters.
class MethodSignature {
public:
  /// Create a method signature with a return type, a method name, and a list of
  /// parameters. Take ownership of the list.
  template <typename RetTypeT, typename NameT>
  MethodSignature(RetTypeT &&retType, NameT &&name,
                  SmallVector<MethodParameter> &&parameters)
      : returnType(stringify(std::forward<RetTypeT>(retType))),
        methodName(stringify(std::forward<NameT>(name))),
        parameters(std::move(parameters)) {}
  /// Create a method signature with a return type, a method name, and a list of
  /// parameters.
  template <typename RetTypeT, typename NameT>
  MethodSignature(RetTypeT &&retType, NameT &&name,
                  ArrayRef<MethodParameter> parameters)
      : MethodSignature(std::forward<RetTypeT>(retType),
                        std::forward<NameT>(name),
                        SmallVector<MethodParameter>(parameters.begin(),
                                                     parameters.end())) {}
  /// Create a method signature with a return type, a method name, and a
  /// variadic list of parameters.
  template <typename RetTypeT, typename NameT, typename... Parameters>
  MethodSignature(RetTypeT &&retType, NameT &&name, Parameters &&...parameters)
      : MethodSignature(std::forward<RetTypeT>(retType),
                        std::forward<NameT>(name),
                        ArrayRef<MethodParameter>(
                            {std::forward<Parameters>(parameters)...})) {}

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

  /// Get the return type of the method
  StringRef getReturnType() const { return returnType; }

  /// Get the number of parameters.
  unsigned getNumParameters() const { return parameters.getNumParameters(); }

  /// Write the signature as part of a method declaration.
  void writeDeclTo(raw_indented_ostream &os) const;

  /// Write the signature as part of a method definition. `namePrefix` is to be
  /// prepended to the method name (typically namespaces for qualifying the
  /// method definition).
  void writeDefTo(raw_indented_ostream &os, StringRef namePrefix) const;

private:
  /// The method's C++ return type.
  std::string returnType;
  /// The method name.
  std::string methodName;
  /// The method's parameter list.
  MethodParameters parameters;
};

/// This class contains the body of a C++ method.
class MethodBody {
public:
  /// Create a method body, indicating whether it should be elided for methods
  /// that are declaration-only.
  MethodBody(bool declOnly);

  /// Define a move constructor to correctly initialize the streams.
  MethodBody(MethodBody &&other)
      : declOnly(other.declOnly), body(std::move(other.body)), stringOs(body),
        os(stringOs) {}
  /// Define a move assignment operator. `raw_ostream` has deleted assignment
  /// operators, so reinitialize the whole object.
  MethodBody &operator=(MethodBody &&body) {
    this->~MethodBody();
    new (this) MethodBody(std::move(body));
    return *this;
  }

  /// Write a value to the method body.
  template <typename ValueT>
  MethodBody &operator<<(ValueT &&value) {
    if (!declOnly) {
      os << std::forward<ValueT>(value);
      os.flush();
    }
    return *this;
  }

  /// Write the method body to the output stream. The body can be written as
  /// part of the declaration of an inline method or just in the definition.
  void writeTo(raw_indented_ostream &os) const;

  /// Indent the output stream.
  MethodBody &indent() {
    os.indent();
    return *this;
  }
  /// Unindent the output stream.
  MethodBody &unindent() {
    os.unindent();
    return *this;
  }
  /// Create a delimited scope: immediately print `open`, indent if `indent` is
  /// true, and print `close` on object destruction.
  raw_indented_ostream::DelimitedScope
  scope(StringRef open = "", StringRef close = "", bool indent = false) {
    return os.scope(open, close, indent);
  }

  /// Get the underlying indented output stream.
  raw_indented_ostream &getStream() { return os; }

private:
  /// Whether the body should be elided.
  bool declOnly;
  /// The body data.
  std::string body;
  /// The string output stream.
  llvm::raw_string_ostream stringOs;
  /// An indented output stream for formatting input.
  raw_indented_ostream os;
};

/// A class declaration is a class element that appears as part of its
/// declaration.
class ClassDeclaration {
public:
  virtual ~ClassDeclaration() = default;

  /// Kinds for LLVM-style RTTI.
  enum Kind {
    Method,
    UsingDeclaration,
    VisibilityDeclaration,
    Field,
    ExtraClassDeclaration
  };
  /// Create a class declaration with a given kind.
  ClassDeclaration(Kind kind) : kind(kind) {}

  /// Get the class declaration kind.
  Kind getKind() const { return kind; }

  /// Write the declaration.
  virtual void writeDeclTo(raw_indented_ostream &os) const = 0;

  /// Write the definition, if any. `namePrefix` is the namespace prefix, which
  /// may contains a class name.
  virtual void writeDefTo(raw_indented_ostream &os,
                          StringRef namePrefix) const {}

private:
  /// The class declaration kind.
  Kind kind;
};

/// Base class for class declarations.
template <ClassDeclaration::Kind DeclKind>
class ClassDeclarationBase : public ClassDeclaration {
public:
  using Base = ClassDeclarationBase<DeclKind>;
  ClassDeclarationBase() : ClassDeclaration(DeclKind) {}

  static bool classof(const ClassDeclaration *other) {
    return other->getKind() == DeclKind;
  }
};

/// Class for holding an op's method for C++ code emission
class Method : public ClassDeclarationBase<ClassDeclaration::Method> {
public:
  /// Properties (qualifiers) of class methods. Bitfield is used here to help
  /// querying properties.
  enum Properties {
    None = 0x0,
    Static = 0x1,
    Constructor = 0x2,
    Private = 0x4,
    Declaration = 0x8,
    Inline = 0x10,
    ConstexprValue = 0x20,
    Const = 0x40,

    Constexpr = ConstexprValue | Inline,
    StaticDeclaration = Static | Declaration,
    StaticInline = Static | Inline,
    ConstInline = Const | Inline,
    ConstDeclaration = Const | Declaration
  };

  /// Create a method with a return type, a name, method properties, and a some
  /// parameters. The parameteres may be passed as a list or as a variadic pack.
  template <typename RetTypeT, typename NameT, typename... Args>
  Method(RetTypeT &&retType, NameT &&name, Properties properties,
         Args &&...args)
      : properties(properties),
        methodSignature(std::forward<RetTypeT>(retType),
                        std::forward<NameT>(name), std::forward<Args>(args)...),
        methodBody(properties & Declaration) {}
  /// Create a method with a return type, a name, method properties, and a list
  /// of parameters.
  Method(StringRef retType, StringRef name, Properties properties,
         std::initializer_list<MethodParameter> params)
      : properties(properties), methodSignature(retType, name, params),
        methodBody(properties & Declaration) {}

  // Define move constructor and assignment operator to prevent copying.
  Method(Method &&) = default;
  Method &operator=(Method &&) = default;

  /// Get the method body.
  MethodBody &body() { return methodBody; }

  /// Returns true if this is a static method.
  bool isStatic() const { return properties & Static; }

  /// Returns true if this is a private method.
  bool isPrivate() const { return properties & Private; }

  /// Returns true if this is an inline method.
  bool isInline() const { return properties & Inline; }

  /// Returns true if this is a constructor.
  bool isConstructor() const { return properties & Constructor; }

  /// Returns true if this class method is const.
  bool isConst() const { return properties & Const; }

  /// Returns the name of this method.
  StringRef getName() const { return methodSignature.getName(); }

  /// Returns the return type of this method
  StringRef getReturnType() const { return methodSignature.getReturnType(); }

  /// Returns if this method makes the `other` method redundant.
  bool makesRedundant(const Method &other) const {
    return methodSignature.makesRedundant(other.methodSignature);
  }

  /// Write the method declaration, including the definition if inline.
  void writeDeclTo(raw_indented_ostream &os) const override;

  /// Write the method definition. This is a no-op for inline methods.
  void writeDefTo(raw_indented_ostream &os,
                  StringRef namePrefix) const override;

protected:
  /// A collection of method properties.
  Properties properties;
  /// The signature of the method.
  MethodSignature methodSignature;
  /// The body of the method, if it has one.
  MethodBody methodBody;
};

/// This enum describes C++ inheritance visibility.
enum class Visibility { Public, Protected, Private };

/// Write "public", "protected", or "private".
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              mlir::tblgen::Visibility visibility);

// Class for holding an op's constructor method for C++ code emission.
class Constructor : public Method {
public:
  /// Create a constructor for a given class, with method properties, and
  /// parameters specified either as a list of a variadic pack.
  template <typename NameT, typename... Args>
  Constructor(NameT &&className, Properties properties, Args &&...args)
      : Method("", std::forward<NameT>(className), properties,
               std::forward<Args>(args)...) {}

  /// Add member initializer to constructor initializing `name` with `value`.
  template <typename NameT, typename ValueT>
  void addMemberInitializer(NameT &&name, ValueT &&value) {
    initializers.emplace_back(stringify(std::forward<NameT>(name)),
                              stringify(std::forward<ValueT>(value)));
  }

  /// Write the declaration of the constructor, and its definition if inline.
  void writeDeclTo(raw_indented_ostream &os) const override;

  /// Write the definition of the constructor if it is not inline.
  void writeDefTo(raw_indented_ostream &os,
                  StringRef namePrefix) const override;

  /// Return true if a method is a constructor.
  static bool classof(const ClassDeclaration *other) {
    return isa<Method>(other) && cast<Method>(other)->isConstructor();
  }

  /// Initialization of a class field in a constructor.
  class MemberInitializer {
  public:
    /// Create a member initializer in a constructor that initializes the class
    /// field `name` with `value`.
    MemberInitializer(std::string name, std::string value)
        : name(std::move(name)), value(std::move(value)) {}

    /// Write the member initializer.
    void writeTo(raw_indented_ostream &os) const;

  private:
    /// The name of the class field.
    std::string name;
    /// The value with which to initialize it.
    std::string value;
  };

private:
  /// The list of member initializers.
  SmallVector<MemberInitializer> initializers;
};

} // namespace tblgen
} // namespace mlir

/// The OR of two method properties should return method properties. Ensure that
/// this function is visible to `Class`.
inline constexpr mlir::tblgen::Method::Properties
operator|(mlir::tblgen::Method::Properties lhs,
          mlir::tblgen::Method::Properties rhs) {
  return mlir::tblgen::Method::Properties(static_cast<unsigned>(lhs) |
                                          static_cast<unsigned>(rhs));
}

inline constexpr mlir::tblgen::Method::Properties &
operator|=(mlir::tblgen::Method::Properties &lhs,
           mlir::tblgen::Method::Properties rhs) {
  return lhs = mlir::tblgen::Method::Properties(static_cast<unsigned>(lhs) |
                                                static_cast<unsigned>(rhs));
}

namespace mlir {
namespace tblgen {

/// This class describes a C++ parent class declaration.
class ParentClass {
public:
  /// Create a parent class with a class name and visibility.
  template <typename NameT>
  ParentClass(NameT &&name, Visibility visibility = Visibility::Public)
      : name(stringify(std::forward<NameT>(name))), visibility(visibility) {}

  /// Add a template parameter.
  template <typename ParamT>
  void addTemplateParam(ParamT param) {
    templateParams.insert(stringify(param));
  }
  /// Add a list of template parameters.
  template <typename ContainerT>
  void addTemplateParams(ContainerT &&container) {
    templateParams.insert(std::begin(container), std::end(container));
  }

  /// Write the parent class declaration.
  void writeTo(raw_indented_ostream &os) const;

private:
  /// The fully resolved C++ name of the parent class.
  std::string name;
  /// The visibility of the parent class.
  Visibility visibility;
  /// An optional list of class template parameters.
  SetVector<std::string, SmallVector<std::string>, StringSet<>> templateParams;
};

/// This class describes a using-declaration for a class. E.g.
///
///   using Op::Op;
///   using Adaptor = OpAdaptor;
///
class UsingDeclaration
    : public ClassDeclarationBase<ClassDeclaration::UsingDeclaration> {
public:
  /// Create a using declaration that either aliases `name` to `value` or
  /// inherits the parent methods `name.
  template <typename NameT, typename ValueT = std::string>
  UsingDeclaration(NameT &&name, ValueT &&value = "")
      : name(stringify(std::forward<NameT>(name))),
        value(stringify(std::forward<ValueT>(value))) {}

  /// Write the using declaration.
  void writeDeclTo(raw_indented_ostream &os) const override;

  /// Add a template parameter.
  template <typename ParamT>
  void addTemplateParam(ParamT param) {
    templateParams.insert(stringify(param));
  }

  /// Add a list of template parameters.
  template <typename ContainerT>
  void addTemplateParams(ContainerT &&container) {
    templateParams.insert(std::begin(container), std::end(container));
  }

private:
  /// The name of the declaration, or a resolved name to an inherited function.
  std::string name;
  /// The type that is being aliased. Leave empty for inheriting functions.
  std::string value;
  /// An optional list of class template parameters.
  /// This is simply a ordered list of parameter names that are then added as
  /// template type parameters when the using declaration is emitted.
  SetVector<std::string, SmallVector<std::string>, StringSet<>> templateParams;
};

/// This class describes a class field.
class Field : public ClassDeclarationBase<ClassDeclaration::Field> {
public:
  /// Create a class field with a type and variable name.
  template <typename TypeT, typename NameT>
  Field(TypeT &&type, NameT &&name)
      : type(stringify(std::forward<TypeT>(type))),
        name(stringify(std::forward<NameT>(name))) {}

  /// Write the declaration of the field.
  void writeDeclTo(raw_indented_ostream &os) const override;

private:
  /// The C++ type of the field.
  std::string type;
  /// The variable name of the class whether.
  std::string name;
};

/// A declaration for the visibility of subsequent declarations.
class VisibilityDeclaration
    : public ClassDeclarationBase<ClassDeclaration::VisibilityDeclaration> {
public:
  /// Create a declaration for the given visibility.
  VisibilityDeclaration(Visibility visibility) : visibility(visibility) {}

  /// Get the visibility.
  Visibility getVisibility() const { return visibility; }

  /// Write the visibility declaration.
  void writeDeclTo(raw_indented_ostream &os) const override;

private:
  /// The visibility of subsequent class declarations.
  Visibility visibility;
};

/// Unstructured extra class declarations and definitions, from TableGen
/// definitions. The default visibility of extra class declarations is up to the
/// owning class.
class ExtraClassDeclaration
    : public ClassDeclarationBase<ClassDeclaration::ExtraClassDeclaration> {
public:
  /// Create an extra class declaration.
  ExtraClassDeclaration(StringRef extraClassDeclaration,
                        StringRef extraClassDefinition = "")
      : extraClassDeclaration(extraClassDeclaration),
        extraClassDefinition(extraClassDefinition) {}

  /// Write the extra class declarations.
  void writeDeclTo(raw_indented_ostream &os) const override;

  /// Write the extra class definitions.
  void writeDefTo(raw_indented_ostream &os,
                  StringRef namePrefix) const override;

private:
  /// The string of the extra class declarations. It is re-indented before
  /// printed.
  StringRef extraClassDeclaration;
  /// The string of the extra class definitions. It is re-indented before
  /// printed.
  std::string extraClassDefinition;
};

/// A class used to emit C++ classes from Tablegen.  Contains a list of public
/// methods and a list of private fields to be emitted.
class Class {
public:
  virtual ~Class() = default;

  /// Explicitly delete the copy constructor. This is to work around a gcc-5 bug
  /// with std::is_trivially_move_constructible.
  Class(const Class &) = delete;

  /// Create a class with a name, and whether it should be declared as a `class`
  /// or `struct`. Also, prevent this from being mistaken as a move constructor
  /// candidate.
  template <typename NameT,
            typename = std::enable_if_t<!std::is_same<NameT, Class>::value>>
  Class(NameT &&name, bool isStruct = false)
      : className(stringify(std::forward<NameT>(name))), isStruct(isStruct) {}

  /// Add a new constructor to this class and prune and constructors made
  /// redundant by it. Returns null if the constructor was not added. Else,
  /// returns a pointer to the new constructor.
  template <Method::Properties Properties = Method::None, typename... Args>
  Constructor *addConstructor(Args &&...args) {
    Method::Properties defaultProperties = Method::Constructor;
    // If the class has template parameters, the constructor has to be defined
    // inline.
    if (!templateParams.empty())
      defaultProperties |= Method::Inline;
    return addConstructorAndPrune(Constructor(getClassName(),
                                              Properties | defaultProperties,
                                              std::forward<Args>(args)...));
  }

  /// Add a new method to this class and prune any methods made redundant by it.
  /// Returns null if the method was not added (because an existing method would
  /// make it redundant). Else, returns a pointer to the new method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT>
  Method *addMethod(RetTypeT &&retType, NameT &&name,
                    Method::Properties properties,
                    ArrayRef<MethodParameter> parameters) {
    // If the class has template parameters, the has to defined inline.
    if (!templateParams.empty())
      properties |= Method::Inline;
    return addMethodAndPrune(Method(std::forward<RetTypeT>(retType),
                                    std::forward<NameT>(name),
                                    Properties | properties, parameters));
  }

  /// Add a method with statically-known properties.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT>
  Method *addMethod(RetTypeT &&retType, NameT &&name,
                    ArrayRef<MethodParameter> parameters) {
    return addMethod(std::forward<RetTypeT>(retType), std::forward<NameT>(name),
                     Properties, parameters);
  }

  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addMethod(RetTypeT &&retType, NameT &&name,
                    Method::Properties properties, Args &&...args) {
    return addMethod(std::forward<RetTypeT>(retType), std::forward<NameT>(name),
                     properties | Properties, {std::forward<Args>(args)...});
  }

  /// Add a method with statically-known properties.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addMethod(RetTypeT &&retType, NameT &&name, Args &&...args) {
    return addMethod(std::forward<RetTypeT>(retType), std::forward<NameT>(name),
                     Properties, std::forward<Args>(args)...);
  }

  /// Add a static method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addStaticMethod(RetTypeT &&retType, NameT &&name, Args &&...args) {
    return addMethod<Properties | Method::Static>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add an inline static method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addStaticInlineMethod(RetTypeT &&retType, NameT &&name,
                                Args &&...args) {
    return addMethod<Properties | Method::StaticInline>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add an inline method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addInlineMethod(RetTypeT &&retType, NameT &&name, Args &&...args) {
    return addMethod<Properties | Method::Inline>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add a const method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *addConstMethod(RetTypeT &&retType, NameT &&name, Args &&...args) {
    return addMethod<Properties | Method::Const>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add a declaration for a method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *declareMethod(RetTypeT &&retType, NameT &&name, Args &&...args) {
    return addMethod<Properties | Method::Declaration>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add a declaration for a static method.
  template <Method::Properties Properties = Method::None, typename RetTypeT,
            typename NameT, typename... Args>
  Method *declareStaticMethod(RetTypeT &&retType, NameT &&name,
                              Args &&...args) {
    return addMethod<Properties | Method::StaticDeclaration>(
        std::forward<RetTypeT>(retType), std::forward<NameT>(name),
        std::forward<Args>(args)...);
  }

  /// Add a new field to the class. Class fields added this way are always
  /// private.
  template <typename TypeT, typename NameT>
  void addField(TypeT &&type, NameT &&name) {
    fields.emplace_back(std::forward<TypeT>(type), std::forward<NameT>(name));
  }

  /// Add a parent class.
  ParentClass &addParent(ParentClass parent);

  /// Add a template parameter.
  template <typename ParamT>
  void addTemplateParam(ParamT param) {
    templateParams.insert(stringify(param));
  }

  /// Add a list of template parameters.
  template <typename ContainerT>
  void addTemplateParams(ContainerT &&container) {
    templateParams.insert(std::begin(container), std::end(container));
  }

  /// Return the C++ name of the class.
  StringRef getClassName() const { return className; }

  /// Write the declaration of this class, all declarations, and definitions of
  /// inline functions. Wrap the output stream in an indented stream.
  void writeDeclTo(raw_ostream &rawOs) const {
    raw_indented_ostream os(rawOs);
    writeDeclTo(os);
  }
  /// Write the definitions of thiss class's out-of-line constructors and
  /// methods. Wrap the output stream in an indented stream.
  void writeDefTo(raw_ostream &rawOs) const {
    raw_indented_ostream os(rawOs);
    writeDefTo(os);
  }

  /// Write the declaration of this class, all declarations, and definitions of
  /// inline functions.
  void writeDeclTo(raw_indented_ostream &os) const;
  /// Write the definitions of thiss class's out-of-line constructors and
  /// methods.
  void writeDefTo(raw_indented_ostream &os) const;

  /// Add a declaration. The declaration is appended directly to the list of
  /// class declarations.
  template <typename DeclT, typename... Args>
  DeclT *declare(Args &&...args) {
    auto decl = std::make_unique<DeclT>(std::forward<Args>(args)...);
    auto *ret = decl.get();
    declarations.push_back(std::move(decl));
    return ret;
  }

  /// The declaration of a class needs to be "finalized".
  ///
  /// Class constructors, methods, and fields can be added in any order,
  /// regardless of whether they are public or private. These are stored in
  /// lists separate from list of declarations `declarations`.
  ///
  /// So that the generated C++ code is somewhat organised, public methods are
  /// declared together, and so are private methods and class fields. This
  /// function iterates through all the added methods and fields and organises
  /// them into the list of declarations, adding visibility declarations as
  /// needed, as follows:
  ///
  ///   1. public methods and constructors
  ///   2. private methods and constructors
  ///   3. class fields -- all are private
  ///
  /// `Class::finalize` clears the lists of pending methods and fields, and can
  /// be called multiple times.
  virtual void finalize();

protected:
  /// Add a new constructor if it is not made redundant by any existing
  /// constructors and prune and existing constructors made redundant.
  Constructor *addConstructorAndPrune(Constructor &&newCtor);
  /// Add a new method if it is not made redundant by any existing methods and
  /// prune and existing methods made redundant.
  Method *addMethodAndPrune(Method &&newMethod);

  /// Get the last visibility declaration.
  Visibility getLastVisibilityDecl() const;

  /// The C++ class name.
  std::string className;
  /// The list of parent classes.
  SmallVector<ParentClass> parents;
  /// The pending list of methods and constructors.
  std::vector<std::unique_ptr<Method>> methods;
  /// The pending list of private class fields.
  SmallVector<Field> fields;
  /// Whether this is a `class` or a `struct`.
  bool isStruct;

  /// A list of declarations in the class, emitted in order.
  std::vector<std::unique_ptr<ClassDeclaration>> declarations;

  /// An optional list of class template parameters.
  SetVector<std::string, SmallVector<std::string>, StringSet<>> templateParams;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CLASS_H_
