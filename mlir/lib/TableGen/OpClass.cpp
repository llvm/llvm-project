//===- OpClass.cpp - Helper classes for Op C++ code emission --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpClass.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// OpMethodSignature definitions
//===----------------------------------------------------------------------===//

tblgen::OpMethodSignature::OpMethodSignature(StringRef retType, StringRef name,
                                             StringRef params)
    : returnType(retType), methodName(name), parameters(params) {}

void tblgen::OpMethodSignature::writeDeclTo(raw_ostream &os) const {
  os << returnType << (elideSpaceAfterType(returnType) ? "" : " ") << methodName
     << "(" << parameters << ")";
}

void tblgen::OpMethodSignature::writeDefTo(raw_ostream &os,
                                           StringRef namePrefix) const {
  // We need to remove the default values for parameters in method definition.
  // TODO(antiagainst): We are using '=' and ',' as delimiters for parameter
  // initializers. This is incorrect for initializer list with more than one
  // element. Change to a more robust approach.
  auto removeParamDefaultValue = [](StringRef params) {
    std::string result;
    std::pair<StringRef, StringRef> parts;
    while (!params.empty()) {
      parts = params.split("=");
      result.append(result.empty() ? "" : ", ");
      result += parts.first;
      params = parts.second.split(",").second;
    }
    return result;
  };

  os << returnType << (elideSpaceAfterType(returnType) ? "" : " ") << namePrefix
     << (namePrefix.empty() ? "" : "::") << methodName << "("
     << removeParamDefaultValue(parameters) << ")";
}

bool tblgen::OpMethodSignature::elideSpaceAfterType(StringRef type) {
  return type.empty() || type.endswith("&") || type.endswith("*");
}

//===----------------------------------------------------------------------===//
// OpMethodBody definitions
//===----------------------------------------------------------------------===//

tblgen::OpMethodBody::OpMethodBody(bool declOnly) : isEffective(!declOnly) {}

tblgen::OpMethodBody &tblgen::OpMethodBody::operator<<(Twine content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

tblgen::OpMethodBody &tblgen::OpMethodBody::operator<<(int content) {
  if (isEffective)
    body.append(std::to_string(content));
  return *this;
}

tblgen::OpMethodBody &
tblgen::OpMethodBody::operator<<(const FmtObjectBase &content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

void tblgen::OpMethodBody::writeTo(raw_ostream &os) const {
  auto bodyRef = StringRef(body).drop_while([](char c) { return c == '\n'; });
  os << bodyRef;
  if (bodyRef.empty() || bodyRef.back() != '\n')
    os << "\n";
}

//===----------------------------------------------------------------------===//
// OpMethod definitions
//===----------------------------------------------------------------------===//

tblgen::OpMethod::OpMethod(StringRef retType, StringRef name, StringRef params,
                           OpMethod::Property property, bool declOnly)
    : properties(property), isDeclOnly(declOnly),
      methodSignature(retType, name, params), methodBody(declOnly) {}

tblgen::OpMethodBody &tblgen::OpMethod::body() { return methodBody; }

bool tblgen::OpMethod::isStatic() const { return properties & MP_Static; }

bool tblgen::OpMethod::isPrivate() const { return properties & MP_Private; }

void tblgen::OpMethod::writeDeclTo(raw_ostream &os) const {
  os.indent(2);
  if (isStatic())
    os << "static ";
  methodSignature.writeDeclTo(os);
  os << ";";
}

void tblgen::OpMethod::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  if (isDeclOnly)
    return;

  methodSignature.writeDefTo(os, namePrefix);
  os << " {\n";
  methodBody.writeTo(os);
  os << "}";
}

//===----------------------------------------------------------------------===//
// OpConstructor definitions
//===----------------------------------------------------------------------===//

void mlir::tblgen::OpConstructor::addMemberInitializer(StringRef name,
                                                       StringRef value) {
  memberInitializers.append(std::string(llvm::formatv(
      "{0}{1}({2})", memberInitializers.empty() ? " : " : ", ", name, value)));
}

void mlir::tblgen::OpConstructor::writeDefTo(raw_ostream &os,
                                             StringRef namePrefix) const {
  if (isDeclOnly)
    return;

  methodSignature.writeDefTo(os, namePrefix);
  os << " " << memberInitializers << " {\n";
  methodBody.writeTo(os);
  os << "}";
}

//===----------------------------------------------------------------------===//
// Class definitions
//===----------------------------------------------------------------------===//

tblgen::Class::Class(StringRef name) : className(name) {}

tblgen::OpMethod &tblgen::Class::newMethod(StringRef retType, StringRef name,
                                           StringRef params,
                                           OpMethod::Property property,
                                           bool declOnly) {
  methods.emplace_back(retType, name, params, property, declOnly);
  return methods.back();
}

tblgen::OpConstructor &tblgen::Class::newConstructor(StringRef params,
                                                     bool declOnly) {
  constructors.emplace_back("", getClassName(), params,
                            OpMethod::MP_Constructor, declOnly);
  return constructors.back();
}

void tblgen::Class::newField(StringRef type, StringRef name,
                             StringRef defaultValue) {
  std::string varName = formatv("{0} {1}", type, name).str();
  std::string field = defaultValue.empty()
                          ? varName
                          : formatv("{0} = {1}", varName, defaultValue).str();
  fields.push_back(std::move(field));
}

void tblgen::Class::writeDeclTo(raw_ostream &os) const {
  bool hasPrivateMethod = false;
  os << "class " << className << " {\n";
  os << "public:\n";
  for (const auto &method :
       llvm::concat<const OpMethod>(constructors, methods)) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << '\n';
    } else {
      hasPrivateMethod = true;
    }
  }
  os << '\n';
  os << "private:\n";
  if (hasPrivateMethod) {
    for (const auto &method :
         llvm::concat<const OpMethod>(constructors, methods)) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << '\n';
      }
    }
    os << '\n';
  }
  for (const auto &field : fields)
    os.indent(2) << field << ";\n";
  os << "};\n";
}

void tblgen::Class::writeDefTo(raw_ostream &os) const {
  for (const auto &method :
       llvm::concat<const OpMethod>(constructors, methods)) {
    method.writeDefTo(os, className);
    os << "\n\n";
  }
}

//===----------------------------------------------------------------------===//
// OpClass definitions
//===----------------------------------------------------------------------===//

tblgen::OpClass::OpClass(StringRef name, StringRef extraClassDeclaration)
    : Class(name), extraClassDeclaration(extraClassDeclaration) {}

void tblgen::OpClass::addTrait(Twine trait) {
  auto traitStr = trait.str();
  if (traitsSet.insert(traitStr).second)
    traitsVec.push_back(std::move(traitStr));
}

void tblgen::OpClass::writeDeclTo(raw_ostream &os) const {
  os << "class " << className << " : public Op<" << className;
  for (const auto &trait : traitsVec)
    os << ", " << trait;
  os << "> {\npublic:\n";
  os << "  using Op::Op;\n";
  os << "  using OperandAdaptor = " << className << "OperandAdaptor;\n";

  bool hasPrivateMethod = false;
  for (const auto &method : methods) {
    if (!method.isPrivate()) {
      method.writeDeclTo(os);
      os << "\n";
    } else {
      hasPrivateMethod = true;
    }
  }

  // TODO: Add line control markers to make errors easier to debug.
  if (!extraClassDeclaration.empty())
    os << extraClassDeclaration << "\n";

  if (hasPrivateMethod) {
    os << "\nprivate:\n";
    for (const auto &method : methods) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << "\n";
      }
    }
  }

  os << "};\n";
}
