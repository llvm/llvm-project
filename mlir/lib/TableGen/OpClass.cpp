//===- OpClass.cpp - Helper classes for Op C++ code emission --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpClass.h"

#include "mlir/TableGen/Format.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>

#define DEBUG_TYPE "mlir-tblgen-opclass"

using namespace mlir;
using namespace mlir::tblgen;

namespace {

// Returns space to be emitted after the given C++ `type`. return "" if the
// ends with '&' or '*', or is empty, else returns " ".
StringRef getSpaceAfterType(StringRef type) {
  return (type.empty() || type.endswith("&") || type.endswith("*")) ? "" : " ";
}

} // namespace

//===----------------------------------------------------------------------===//
// OpMethodParameter definitions
//===----------------------------------------------------------------------===//

void OpMethodParameter::writeTo(raw_ostream &os, bool emitDefault) const {
  if (properties & PP_Optional)
    os << "/*optional*/";
  os << type << getSpaceAfterType(type) << name;
  if (emitDefault && !defaultValue.empty())
    os << " = " << defaultValue;
}

//===----------------------------------------------------------------------===//
// OpMethodParameters definitions
//===----------------------------------------------------------------------===//

// Factory methods to construct the correct type of `OpMethodParameters`
// object based on the arguments.
std::unique_ptr<OpMethodParameters> OpMethodParameters::create() {
  return std::make_unique<OpMethodResolvedParameters>();
}

std::unique_ptr<OpMethodParameters>
OpMethodParameters::create(StringRef params) {
  return std::make_unique<OpMethodUnresolvedParameters>(params);
}

std::unique_ptr<OpMethodParameters>
OpMethodParameters::create(llvm::SmallVectorImpl<OpMethodParameter> &&params) {
  return std::make_unique<OpMethodResolvedParameters>(std::move(params));
}

std::unique_ptr<OpMethodParameters>
OpMethodParameters::create(StringRef type, StringRef name,
                           StringRef defaultValue) {
  return std::make_unique<OpMethodResolvedParameters>(type, name, defaultValue);
}

//===----------------------------------------------------------------------===//
// OpMethodUnresolvedParameters definitions
//===----------------------------------------------------------------------===//
void OpMethodUnresolvedParameters::writeDeclTo(raw_ostream &os) const {
  os << parameters;
}

void OpMethodUnresolvedParameters::writeDefTo(raw_ostream &os) const {
  // We need to remove the default values for parameters in method definition.
  // TODO: We are using '=' and ',' as delimiters for parameter
  // initializers. This is incorrect for initializer list with more than one
  // element. Change to a more robust approach.
  llvm::SmallVector<StringRef, 4> tokens;
  StringRef params = parameters;
  while (!params.empty()) {
    std::pair<StringRef, StringRef> parts = params.split("=");
    tokens.push_back(parts.first);
    params = parts.second.split(',').second;
  }
  llvm::interleaveComma(tokens, os, [&](StringRef token) { os << token; });
}

//===----------------------------------------------------------------------===//
// OpMethodResolvedParameters definitions
//===----------------------------------------------------------------------===//

// Returns true if a method with these parameters makes a method with parameters
// `other` redundant. This should return true only if all possible calls to the
// other method can be replaced by calls to this method.
bool OpMethodResolvedParameters::makesRedundant(
    const OpMethodResolvedParameters &other) const {
  const size_t otherNumParams = other.getNumParameters();
  const size_t thisNumParams = getNumParameters();

  // All calls to the other method can be replaced this method only if this
  // method has the same or more arguments number of arguments as the other, and
  // the common arguments have the same type.
  if (thisNumParams < otherNumParams)
    return false;
  for (int idx : llvm::seq<int>(0, otherNumParams))
    if (parameters[idx].getType() != other.parameters[idx].getType())
      return false;

  // If all the common arguments have the same type, we can elide the other
  // method if this method has the same number of arguments as other or the
  // first argument after the common ones has a default value (and by C++
  // requirement, all the later ones will also have a default value).
  return thisNumParams == otherNumParams ||
         parameters[otherNumParams].hasDefaultValue();
}

void OpMethodResolvedParameters::writeDeclTo(raw_ostream &os) const {
  llvm::interleaveComma(parameters, os, [&](const OpMethodParameter &param) {
    param.writeDeclTo(os);
  });
}

void OpMethodResolvedParameters::writeDefTo(raw_ostream &os) const {
  llvm::interleaveComma(parameters, os, [&](const OpMethodParameter &param) {
    param.writeDefTo(os);
  });
}

//===----------------------------------------------------------------------===//
// OpMethodSignature definitions
//===----------------------------------------------------------------------===//

// Returns if a method with this signature makes a method with `other` signature
// redundant. Only supports resolved parameters.
bool OpMethodSignature::makesRedundant(const OpMethodSignature &other) const {
  if (methodName != other.methodName)
    return false;
  auto *resolvedThis = dyn_cast<OpMethodResolvedParameters>(parameters.get());
  auto *resolvedOther =
      dyn_cast<OpMethodResolvedParameters>(other.parameters.get());
  if (resolvedThis && resolvedOther)
    return resolvedThis->makesRedundant(*resolvedOther);
  return false;
}

void OpMethodSignature::writeDeclTo(raw_ostream &os) const {
  os << returnType << getSpaceAfterType(returnType) << methodName << "(";
  parameters->writeDeclTo(os);
  os << ")";
}

void OpMethodSignature::writeDefTo(raw_ostream &os,
                                   StringRef namePrefix) const {
  os << returnType << getSpaceAfterType(returnType) << namePrefix
     << (namePrefix.empty() ? "" : "::") << methodName << "(";
  parameters->writeDefTo(os);
  os << ")";
}

//===----------------------------------------------------------------------===//
// OpMethodBody definitions
//===----------------------------------------------------------------------===//

OpMethodBody::OpMethodBody(bool declOnly) : isEffective(!declOnly) {}

OpMethodBody &OpMethodBody::operator<<(Twine content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

OpMethodBody &OpMethodBody::operator<<(int content) {
  if (isEffective)
    body.append(std::to_string(content));
  return *this;
}

OpMethodBody &OpMethodBody::operator<<(const FmtObjectBase &content) {
  if (isEffective)
    body.append(content.str());
  return *this;
}

void OpMethodBody::writeTo(raw_ostream &os) const {
  auto bodyRef = StringRef(body).drop_while([](char c) { return c == '\n'; });
  os << bodyRef;
  if (bodyRef.empty() || bodyRef.back() != '\n')
    os << "\n";
}

//===----------------------------------------------------------------------===//
// OpMethod definitions
//===----------------------------------------------------------------------===//

void OpMethod::writeDeclTo(raw_ostream &os) const {
  os.indent(2);
  if (isStatic())
    os << "static ";
  methodSignature.writeDeclTo(os);
  os << ";";
}

void OpMethod::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  // Do not write definition if the method is decl only.
  if (properties & MP_Declaration)
    return;
  methodSignature.writeDefTo(os, namePrefix);
  os << " {\n";
  methodBody.writeTo(os);
  os << "}";
}

//===----------------------------------------------------------------------===//
// OpConstructor definitions
//===----------------------------------------------------------------------===//

void OpConstructor::addMemberInitializer(StringRef name, StringRef value) {
  memberInitializers.append(std::string(llvm::formatv(
      "{0}{1}({2})", memberInitializers.empty() ? " : " : ", ", name, value)));
}

void OpConstructor::writeDefTo(raw_ostream &os, StringRef namePrefix) const {
  // Do not write definition if the method is decl only.
  if (properties & MP_Declaration)
    return;

  methodSignature.writeDefTo(os, namePrefix);
  os << " " << memberInitializers << " {\n";
  methodBody.writeTo(os);
  os << "}";
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

  forAllMethods([&](const OpMethod &method) {
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
    forAllMethods([&](const OpMethod &method) {
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
  forAllMethods([&](const OpMethod &method) {
    method.writeDefTo(os, className);
    os << "\n\n";
  });
}

//===----------------------------------------------------------------------===//
// OpClass definitions
//===----------------------------------------------------------------------===//

OpClass::OpClass(StringRef name, StringRef extraClassDeclaration)
    : Class(name), extraClassDeclaration(extraClassDeclaration) {}

void OpClass::addTrait(Twine trait) {
  auto traitStr = trait.str();
  if (traitsSet.insert(traitStr).second)
    traitsVec.push_back(std::move(traitStr));
}

void OpClass::writeDeclTo(raw_ostream &os) const {
  os << "class " << className << " : public ::mlir::Op<" << className;
  for (const auto &trait : traitsVec)
    os << ", " << trait;
  os << "> {\npublic:\n"
     << "  using Op::Op;\n"
     << "  using Op::print;\n"
     << "  using Adaptor = " << className << "Adaptor;\n";

  bool hasPrivateMethod = false;
  forAllMethods([&](const OpMethod &method) {
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
    forAllMethods([&](const OpMethod &method) {
      if (method.isPrivate()) {
        method.writeDeclTo(os);
        os << "\n";
      }
    });
  }

  os << "};\n";
}
