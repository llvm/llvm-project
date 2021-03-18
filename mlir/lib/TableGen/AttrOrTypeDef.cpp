//===- AttrOrTypeDef.cpp - AttrOrTypeDef wrapper classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

/// Returns true if this builder is able to infer the MLIRContext parameter.
bool AttrOrTypeBuilder::hasInferredContextParameter() const {
  return def->getValueAsBit("hasInferredContextParam");
}

//===----------------------------------------------------------------------===//
// AttrOrTypeDef
//===----------------------------------------------------------------------===//

AttrOrTypeDef::AttrOrTypeDef(const llvm::Record *def) : def(def) {
  // Populate the builders.
  auto *builderList =
      dyn_cast_or_null<llvm::ListInit>(def->getValueInit("builders"));
  if (builderList && !builderList->empty()) {
    for (llvm::Init *init : builderList->getValues()) {
      AttrOrTypeBuilder builder(cast<llvm::DefInit>(init)->getDef(),
                                def->getLoc());

      // Ensure that all parameters have names.
      for (const AttrOrTypeBuilder::Parameter &param :
           builder.getParameters()) {
        if (!param.getName())
          PrintFatalError(def->getLoc(), "builder parameters must have a name");
      }
      builders.emplace_back(builder);
    }
  } else if (skipDefaultBuilders()) {
    PrintFatalError(
        def->getLoc(),
        "default builders are skipped and no custom builders provided");
  }
}

Dialect AttrOrTypeDef::getDialect() const {
  auto *dialect = dyn_cast<llvm::DefInit>(def->getValue("dialect")->getValue());
  return Dialect(dialect ? dialect->getDef() : nullptr);
}

StringRef AttrOrTypeDef::getName() const { return def->getName(); }

StringRef AttrOrTypeDef::getCppClassName() const {
  return def->getValueAsString("cppClassName");
}

StringRef AttrOrTypeDef::getCppBaseClassName() const {
  return def->getValueAsString("cppBaseClassName");
}

bool AttrOrTypeDef::hasDescription() const {
  const llvm::RecordVal *desc = def->getValue("description");
  return desc && isa<llvm::StringInit>(desc->getValue());
}

StringRef AttrOrTypeDef::getDescription() const {
  return def->getValueAsString("description");
}

bool AttrOrTypeDef::hasSummary() const {
  const llvm::RecordVal *summary = def->getValue("summary");
  return summary && isa<llvm::StringInit>(summary->getValue());
}

StringRef AttrOrTypeDef::getSummary() const {
  return def->getValueAsString("summary");
}

StringRef AttrOrTypeDef::getStorageClassName() const {
  return def->getValueAsString("storageClass");
}

StringRef AttrOrTypeDef::getStorageNamespace() const {
  return def->getValueAsString("storageNamespace");
}

bool AttrOrTypeDef::genStorageClass() const {
  return def->getValueAsBit("genStorageClass");
}

bool AttrOrTypeDef::hasStorageCustomConstructor() const {
  return def->getValueAsBit("hasStorageCustomConstructor");
}

void AttrOrTypeDef::getParameters(
    SmallVectorImpl<AttrOrTypeParameter> &parameters) const {
  if (auto *parametersDag = def->getValueAsDag("parameters")) {
    for (unsigned i = 0, e = parametersDag->getNumArgs(); i < e; ++i)
      parameters.push_back(AttrOrTypeParameter(parametersDag, i));
  }
}

unsigned AttrOrTypeDef::getNumParameters() const {
  auto *parametersDag = def->getValueAsDag("parameters");
  return parametersDag ? parametersDag->getNumArgs() : 0;
}

Optional<StringRef> AttrOrTypeDef::getMnemonic() const {
  return def->getValueAsOptionalString("mnemonic");
}

Optional<StringRef> AttrOrTypeDef::getPrinterCode() const {
  return def->getValueAsOptionalString("printer");
}

Optional<StringRef> AttrOrTypeDef::getParserCode() const {
  return def->getValueAsOptionalString("parser");
}

bool AttrOrTypeDef::genAccessors() const {
  return def->getValueAsBit("genAccessors");
}

bool AttrOrTypeDef::genVerifyDecl() const {
  return def->getValueAsBit("genVerifyDecl");
}

Optional<StringRef> AttrOrTypeDef::getExtraDecls() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? Optional<StringRef>() : value;
}

ArrayRef<llvm::SMLoc> AttrOrTypeDef::getLoc() const { return def->getLoc(); }

bool AttrOrTypeDef::skipDefaultBuilders() const {
  return def->getValueAsBit("skipDefaultBuilders");
}

bool AttrOrTypeDef::operator==(const AttrOrTypeDef &other) const {
  return def == other.def;
}

bool AttrOrTypeDef::operator<(const AttrOrTypeDef &other) const {
  return getName() < other.getName();
}

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

Optional<StringRef> AttrDef::getTypeBuilder() const {
  return def->getValueAsOptionalString("typeBuilder");
}

bool AttrDef::classof(const AttrOrTypeDef *def) {
  return def->getDef()->isSubClassOf("AttrDef");
}

//===----------------------------------------------------------------------===//
// AttrOrTypeParameter
//===----------------------------------------------------------------------===//

StringRef AttrOrTypeParameter::getName() const {
  return def->getArgName(index)->getValue();
}

Optional<StringRef> AttrOrTypeParameter::getAllocator() const {
  llvm::Init *parameterType = def->getArg(index);
  if (isa<llvm::StringInit>(parameterType))
    return Optional<StringRef>();
  if (auto *param = dyn_cast<llvm::DefInit>(parameterType))
    return param->getDef()->getValueAsOptionalString("allocator");
  llvm::PrintFatalError("Parameters DAG arguments must be either strings or "
                        "defs which inherit from AttrOrTypeParameter\n");
}

Optional<StringRef> AttrOrTypeParameter::getComparator() const {
  llvm::Init *parameterType = def->getArg(index);
  if (isa<llvm::StringInit>(parameterType))
    return Optional<StringRef>();
  if (auto *param = dyn_cast<llvm::DefInit>(parameterType))
    return param->getDef()->getValueAsOptionalString("comparator");
  llvm::PrintFatalError("Parameters DAG arguments must be either strings or "
                        "defs which inherit from AttrOrTypeParameter\n");
}

StringRef AttrOrTypeParameter::getCppType() const {
  auto *parameterType = def->getArg(index);
  if (auto *stringType = dyn_cast<llvm::StringInit>(parameterType))
    return stringType->getValue();
  if (auto *param = dyn_cast<llvm::DefInit>(parameterType))
    return param->getDef()->getValueAsString("cppType");
  llvm::PrintFatalError(
      "Parameters DAG arguments must be either strings or defs "
      "which inherit from AttrOrTypeParameter\n");
}

Optional<StringRef> AttrOrTypeParameter::getSummary() const {
  auto *parameterType = def->getArg(index);
  if (auto *param = dyn_cast<llvm::DefInit>(parameterType)) {
    const auto *desc = param->getDef()->getValue("summary");
    if (llvm::StringInit *ci = dyn_cast<llvm::StringInit>(desc->getValue()))
      return ci->getValue();
  }
  return Optional<StringRef>();
}

StringRef AttrOrTypeParameter::getSyntax() const {
  auto *parameterType = def->getArg(index);
  if (auto *stringType = dyn_cast<llvm::StringInit>(parameterType))
    return stringType->getValue();
  if (auto *param = dyn_cast<llvm::DefInit>(parameterType)) {
    const auto *syntax = param->getDef()->getValue("syntax");
    if (syntax && isa<llvm::StringInit>(syntax->getValue()))
      return cast<llvm::StringInit>(syntax->getValue())->getValue();
    return getCppType();
  }
  llvm::PrintFatalError("Parameters DAG arguments must be either strings or "
                        "defs which inherit from AttrOrTypeParameter");
}

const llvm::Init *AttrOrTypeParameter::getDef() const {
  return def->getArg(index);
}

//===----------------------------------------------------------------------===//
// AttributeSelfTypeParameter
//===----------------------------------------------------------------------===//

bool AttributeSelfTypeParameter::classof(const AttrOrTypeParameter *param) {
  const llvm::Init *paramDef = param->getDef();
  if (auto *paramDefInit = dyn_cast<llvm::DefInit>(paramDef))
    return paramDefInit->getDef()->isSubClassOf("AttributeSelfTypeParameter");
  return false;
}
