//===- Class.cpp - Helper classes for Op C++ code emission --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Class.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>

#define DEBUG_TYPE "mlir-tblgen-opclass"

using namespace mlir;
using namespace mlir::tblgen;

// Returns space to be emitted after the given C++ `type`. return "" if the
// ends with '&' or '*', or is empty, else returns " ".
static StringRef getSpaceAfterType(StringRef type) {
  return (type.empty() || type.endswith("&") || type.endswith("*")) ? "" : " ";
}

//===----------------------------------------------------------------------===//
// MethodParameter definitions
//===----------------------------------------------------------------------===//

void MethodParameter::writeTo(raw_ostream &os, bool emitDefault) const {
  if (optional)
    os << "/*optional*/";
  os << type << getSpaceAfterType(type) << name;
  if (emitDefault && hasDefaultValue())
    os << " = " << defaultValue;
}

//===----------------------------------------------------------------------===//
// MethodParameters definitions
//===----------------------------------------------------------------------===//

void MethodParameters::writeDeclTo(raw_ostream &os) const {
  llvm::interleaveComma(parameters, os,
                        [&os](auto &param) { param.writeDeclTo(os); });
}
void MethodParameters::writeDefTo(raw_ostream &os) const {
  llvm::interleaveComma(parameters, os,
                        [&os](auto &param) { param.writeDefTo(os); });
}

bool MethodParameters::subsumes(const MethodParameters &other) const {
  // These parameters do not subsume the others if there are fewer parameters
  // or their types do not match.
  if (parameters.size() < other.parameters.size())
    return false;
  if (!std::equal(
          other.parameters.begin(), other.parameters.end(), parameters.begin(),
          [](auto &lhs, auto &rhs) { return lhs.getType() == rhs.getType(); }))
    return false;

  // If all the common parameters have the same type, we can elide the other
  // method if this method has the same number of parameters as other or if the
  // first paramater after the common parameters has a default value (and, as
  // required by C++, subsequent parameters will have default values too).
  return parameters.size() == other.parameters.size() ||
         parameters[other.parameters.size()].hasDefaultValue();
}

//===----------------------------------------------------------------------===//
// MethodSignature definitions
//===----------------------------------------------------------------------===//

bool MethodSignature::makesRedundant(const MethodSignature &other) const {
  return methodName == other.methodName &&
         parameters.subsumes(other.parameters);
}

void MethodSignature::writeDeclTo(raw_ostream &os) const {
  os << returnType << getSpaceAfterType(returnType) << methodName << "(";
  parameters.writeDeclTo(os);
  os << ")";
}

void MethodSignature::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  os << returnType << getSpaceAfterType(returnType) << namePrefix
     << (namePrefix.empty() ? "" : "::") << methodName << "(";
  parameters.writeDefTo(os);
  os << ")";
}

//===----------------------------------------------------------------------===//
// MethodBody definitions
//===----------------------------------------------------------------------===//

MethodBody::MethodBody(bool declOnly) : isEffective(!declOnly) {}

MethodBody &MethodBody::operator<<(Twine content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

MethodBody &MethodBody::operator<<(int content) {
  if (isEffective)
    body.append(std::to_string(content));
  return *this;
}

MethodBody &MethodBody::operator<<(const FmtObjectBase &content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

void MethodBody::writeTo(raw_ostream &os) const {
  auto bodyRef = StringRef(body).drop_while([](char c) { return c == '\n'; });
  os << bodyRef;
  if (bodyRef.empty() || bodyRef.back() != '\n')
    os << "\n";
}

//===----------------------------------------------------------------------===//
// Method definitions
//===----------------------------------------------------------------------===//

void Method::writeDeclTo(raw_ostream &os) const {
  os.indent(2);
  if (isStatic())
    os << "static ";
  if ((properties & MP_Constexpr) == MP_Constexpr)
    os << "constexpr ";
  methodSignature.writeDeclTo(os);
  if (!isInline()) {
    os << ";";
  } else {
    os << " {\n";
    methodBody.writeTo(os.indent(2));
    os.indent(2) << "}";
  }
}

void Method::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  // Do not write definition if the method is decl only.
  if (properties & MP_Declaration)
    return;
  // Do not generate separate definition for inline method
  if (isInline())
    return;
  methodSignature.writeDefTo(os, namePrefix);
  os << " {\n";
  methodBody.writeTo(os);
  os << "}";
}

//===----------------------------------------------------------------------===//
// Constructor definitions
//===----------------------------------------------------------------------===//

void Constructor::addMemberInitializer(StringRef name, StringRef value) {
  memberInitializers.append(std::string(llvm::formatv(
      "{0}{1}({2})", memberInitializers.empty() ? " : " : ", ", name, value)));
}

void Constructor::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  // Do not write definition if the method is decl only.
  if (properties & MP_Declaration)
    return;

  methodSignature.writeDefTo(os, namePrefix);
  os << " " << memberInitializers << " {\n";
  methodBody.writeTo(os);
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// Class definitions
//===----------------------------------------------------------------------===//

Class::Class(StringRef name) : className(name) {}

void Class::newField(StringRef type, StringRef name, StringRef defaultValue) {
  std::string varName = formatv("{0} {1}", type, name).str();
  std::string field = defaultValue.empty()
                          ? varName
                          : formatv("{0} = {1}", varName, defaultValue).str();
  fields.push_back(std::move(field));
}

void Class::writeDeclTo(raw_ostream &os) const {
  bool hasPrivateMethod = false;
  os << "class " << className << " {\n";
  os << "public:\n";

  forAllMethods([&](const Method &method) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << '\n';
    } else {
      hasPrivateMethod = true;
    }
  });

  os << '\n';
  os << "private:\n";
  if (hasPrivateMethod) {
    forAllMethods([&](const Method &method) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << '\n';
      }
    });
    os << '\n';
  }

  for (const auto &field : fields)
    os.indent(2) << field << ";\n";
  os << "};\n";
}

void Class::writeDefTo(raw_ostream &os) const {
  forAllMethods([&](const Method &method) {
    method.writeDefTo(os, className);
    os << "\n";
  });
}

// Insert a new method into a list of methods, if it would not be pruned, and
// prune and existing methods.
template <typename ContainerT, typename MethodT>
MethodT *insertAndPrune(ContainerT &methods, MethodT newMethod) {
  if (llvm::any_of(methods, [&](auto &method) {
        return method.makesRedundant(newMethod);
      }))
    return nullptr;

  llvm::erase_if(
      methods, [&](auto &method) { return newMethod.makesRedundant(method); });
  methods.push_back(std::move(newMethod));
  return &methods.back();
}

Method *Class::addMethodAndPrune(Method &&newMethod) {
  return insertAndPrune(methods, std::move(newMethod));
}

Constructor *Class::addConstructorAndPrune(Constructor &&newCtor) {
  return insertAndPrune(constructors, std::move(newCtor));
}

//===----------------------------------------------------------------------===//
// OpClass definitions
//===----------------------------------------------------------------------===//

OpClass::OpClass(StringRef name, StringRef extraClassDeclaration)
    : Class(name), extraClassDeclaration(extraClassDeclaration) {}

void OpClass::addTrait(Twine trait) { traits.insert(trait.str()); }

void OpClass::writeDeclTo(raw_ostream &os) const {
  os << "class " << className << " : public ::mlir::Op<" << className;
  for (const auto &trait : traits)
    os << ", " << trait;
  os << "> {\npublic:\n"
     << "  using Op::Op;\n"
     << "  using Op::print;\n"
     << "  using Adaptor = " << className << "Adaptor;\n";

  bool hasPrivateMethod = false;
  forAllMethods([&](const Method &method) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << "\n";
    } else {
      hasPrivateMethod = true;
    }
  });

  // TODO: Add line control markers to make errors easier to debug.
  if (!extraClassDeclaration.empty())
    os << extraClassDeclaration << "\n";

  if (hasPrivateMethod) {
    os << "\nprivate:\n";
    forAllMethods([&](const Method &method) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << "\n";
      }
    });
  }

  os << "};\n";
}
