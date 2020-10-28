//===- TypeDef.cpp - TypeDef wrapper class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDef wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/TypeDef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

Dialect TypeDef::getDialect() const {
  auto *dialectDef =
      dyn_cast<llvm::DefInit>(def->getValue("dialect")->getValue());
  if (dialectDef == nullptr)
    return Dialect(nullptr);
  return Dialect(dialectDef->getDef());
}

StringRef TypeDef::getName() const { return def->getName(); }
StringRef TypeDef::getCppClassName() const {
  return def->getValueAsString("cppClassName");
}

bool TypeDef::hasDescription() const {
  const llvm::RecordVal *s = def->getValue("description");
  return s != nullptr && isa<llvm::StringInit>(s->getValue());
}

StringRef TypeDef::getDescription() const {
  return def->getValueAsString("description");
}

bool TypeDef::hasSummary() const {
  const llvm::RecordVal *s = def->getValue("summary");
  return s != nullptr && isa<llvm::StringInit>(s->getValue());
}

StringRef TypeDef::getSummary() const {
  return def->getValueAsString("summary");
}

StringRef TypeDef::getStorageClassName() const {
  return def->getValueAsString("storageClass");
}
StringRef TypeDef::getStorageNamespace() const {
  return def->getValueAsString("storageNamespace");
}

bool TypeDef::genStorageClass() const {
  return def->getValueAsBit("genStorageClass");
}
bool TypeDef::hasStorageCustomConstructor() const {
  return def->getValueAsBit("hasStorageCustomConstructor");
}
void TypeDef::getParameters(SmallVectorImpl<TypeParameter> &parameters) const {
  auto *parametersDag = def->getValueAsDag("parameters");
  if (parametersDag != nullptr) {
    size_t numParams = parametersDag->getNumArgs();
    for (unsigned i = 0; i < numParams; i++)
      parameters.push_back(TypeParameter(parametersDag, i));
  }
}
unsigned TypeDef::getNumParameters() const {
  auto *parametersDag = def->getValueAsDag("parameters");
  return parametersDag ? parametersDag->getNumArgs() : 0;
}
llvm::Optional<StringRef> TypeDef::getMnemonic() const {
  return def->getValueAsOptionalString("mnemonic");
}
llvm::Optional<StringRef> TypeDef::getPrinterCode() const {
  return def->getValueAsOptionalCode("printer");
}
llvm::Optional<StringRef> TypeDef::getParserCode() const {
  return def->getValueAsOptionalCode("parser");
}
bool TypeDef::genAccessors() const {
  return def->getValueAsBit("genAccessors");
}
bool TypeDef::genVerifyInvariantsDecl() const {
  return def->getValueAsBit("genVerifyInvariantsDecl");
}
llvm::Optional<StringRef> TypeDef::getExtraDecls() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}
llvm::ArrayRef<llvm::SMLoc> TypeDef::getLoc() const { return def->getLoc(); }
bool TypeDef::operator==(const TypeDef &other) const {
  return def == other.def;
}

bool TypeDef::operator<(const TypeDef &other) const {
  return getName() < other.getName();
}

StringRef TypeParameter::getName() const {
  return def->getArgName(num)->getValue();
}
llvm::Optional<StringRef> TypeParameter::getAllocator() const {
  llvm::Init *parameterType = def->getArg(num);
  if (isa<llvm::StringInit>(parameterType))
    return llvm::Optional<StringRef>();

  if (auto *typeParameter = dyn_cast<llvm::DefInit>(parameterType)) {
    llvm::RecordVal *code = typeParameter->getDef()->getValue("allocator");
    if (llvm::CodeInit *ci = dyn_cast<llvm::CodeInit>(code->getValue()))
      return ci->getValue();
    if (isa<llvm::UnsetInit>(code->getValue()))
      return llvm::Optional<StringRef>();

    llvm::PrintFatalError(
        typeParameter->getDef()->getLoc(),
        "Record `" + def->getArgName(num)->getValue() +
            "', field `printer' does not have a code initializer!");
  }

  llvm::PrintFatalError("Parameters DAG arguments must be either strings or "
                        "defs which inherit from TypeParameter\n");
}
StringRef TypeParameter::getCppType() const {
  auto *parameterType = def->getArg(num);
  if (auto *stringType = dyn_cast<llvm::StringInit>(parameterType))
    return stringType->getValue();
  if (auto *typeParameter = dyn_cast<llvm::DefInit>(parameterType))
    return typeParameter->getDef()->getValueAsString("cppType");
  llvm::PrintFatalError(
      "Parameters DAG arguments must be either strings or defs "
      "which inherit from TypeParameter\n");
}
llvm::Optional<StringRef> TypeParameter::getDescription() const {
  auto *parameterType = def->getArg(num);
  if (auto *typeParameter = dyn_cast<llvm::DefInit>(parameterType)) {
    const auto *desc = typeParameter->getDef()->getValue("description");
    if (llvm::StringInit *ci = dyn_cast<llvm::StringInit>(desc->getValue()))
      return ci->getValue();
  }
  return llvm::Optional<StringRef>();
}
StringRef TypeParameter::getSyntax() const {
  auto *parameterType = def->getArg(num);
  if (auto *stringType = dyn_cast<llvm::StringInit>(parameterType))
    return stringType->getValue();
  if (auto *typeParameter = dyn_cast<llvm::DefInit>(parameterType)) {
    const auto *syntax = typeParameter->getDef()->getValue("syntax");
    if (syntax && isa<llvm::StringInit>(syntax->getValue()))
      return dyn_cast<llvm::StringInit>(syntax->getValue())->getValue();
    return getCppType();
  }
  llvm::PrintFatalError("Parameters DAG arguments must be either strings or "
                        "defs which inherit from TypeParameter");
}
