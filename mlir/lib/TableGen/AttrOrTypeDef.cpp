//===- AttrOrTypeDef.cpp - AttrOrTypeDef wrapper classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
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
  }

  // Populate the traits.
  if (auto *traitList = def->getValueAsListInit("traits")) {
    SmallPtrSet<const llvm::Init *, 32> traitSet;
    traits.reserve(traitSet.size());
    for (auto *traitInit : *traitList)
      if (traitSet.insert(traitInit).second)
        traits.push_back(Trait::create(traitInit));
  }

  // Populate the parameters.
  if (auto *parametersDag = def->getValueAsDag("parameters")) {
    for (unsigned i = 0, e = parametersDag->getNumArgs(); i < e; ++i)
      parameters.push_back(AttrOrTypeParameter(parametersDag, i));
  }

  // Verify the use of the mnemonic field.
  bool hasCppFormat = hasCustomAssemblyFormat();
  bool hasDeclarativeFormat = getAssemblyFormat().has_value();
  if (getMnemonic()) {
    if (hasCppFormat && hasDeclarativeFormat) {
      PrintFatalError(getLoc(), "cannot specify both 'assemblyFormat' "
                                "and 'hasCustomAssemblyFormat'");
    }
    if (!parameters.empty() && !hasCppFormat && !hasDeclarativeFormat) {
      PrintFatalError(getLoc(),
                      "must specify either 'assemblyFormat' or "
                      "'hasCustomAssemblyFormat' when 'mnemonic' is set");
    }
  } else if (hasCppFormat || hasDeclarativeFormat) {
    PrintFatalError(getLoc(),
                    "'assemblyFormat' or 'hasCustomAssemblyFormat' can only be "
                    "used when 'mnemonic' is set");
  }
  // Assembly format parser requires builders with the same prototype
  // as the default-builders.
  // TODO: attempt to detect when a custom builder matches the prototype.
  if (hasDeclarativeFormat && skipDefaultBuilders()) {
    PrintWarning(getLoc(),
                 "using 'assemblyFormat' with 'skipDefaultBuilders=1' may "
                 "result in C++ compilation errors");
  }
  // Assembly format printer requires accessors to be generated.
  if (hasDeclarativeFormat && !genAccessors()) {
    PrintFatalError(getLoc(),
                    "'assemblyFormat' requires 'genAccessors' to be true");
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

unsigned AttrOrTypeDef::getNumParameters() const {
  auto *parametersDag = def->getValueAsDag("parameters");
  return parametersDag ? parametersDag->getNumArgs() : 0;
}

Optional<StringRef> AttrOrTypeDef::getMnemonic() const {
  return def->getValueAsOptionalString("mnemonic");
}

bool AttrOrTypeDef::hasCustomAssemblyFormat() const {
  return def->getValueAsBit("hasCustomAssemblyFormat");
}

Optional<StringRef> AttrOrTypeDef::getAssemblyFormat() const {
  return def->getValueAsOptionalString("assemblyFormat");
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

ArrayRef<SMLoc> AttrOrTypeDef::getLoc() const { return def->getLoc(); }

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

template <typename InitT>
auto AttrOrTypeParameter::getDefValue(StringRef name) const {
  Optional<decltype(std::declval<InitT>().getValue())> result;
  if (auto *param = dyn_cast<llvm::DefInit>(getDef()))
    if (auto *init = param->getDef()->getValue(name))
      if (auto *value = dyn_cast_or_null<InitT>(init->getValue()))
        result = value->getValue();
  return result;
}

bool AttrOrTypeParameter::isAnonymous() const {
  return !def->getArgName(index);
}

StringRef AttrOrTypeParameter::getName() const {
  return def->getArgName(index)->getValue();
}

std::string AttrOrTypeParameter::getAccessorName() const {
  return "get" +
         llvm::convertToCamelFromSnakeCase(getName(), /*capitalizeFirst=*/true);
}

Optional<StringRef> AttrOrTypeParameter::getAllocator() const {
  return getDefValue<llvm::StringInit>("allocator");
}

StringRef AttrOrTypeParameter::getComparator() const {
  return getDefValue<llvm::StringInit>("comparator").value_or("$_lhs == $_rhs");
}

StringRef AttrOrTypeParameter::getCppType() const {
  if (auto *stringType = dyn_cast<llvm::StringInit>(getDef()))
    return stringType->getValue();
  return getDefValue<llvm::StringInit>("cppType").getValue();
}

StringRef AttrOrTypeParameter::getCppAccessorType() const {
  return getDefValue<llvm::StringInit>("cppAccessorType")
      .value_or(getCppType());
}

StringRef AttrOrTypeParameter::getCppStorageType() const {
  return getDefValue<llvm::StringInit>("cppStorageType").value_or(getCppType());
}

StringRef AttrOrTypeParameter::getConvertFromStorage() const {
  return getDefValue<llvm::StringInit>("convertFromStorage").value_or("$_self");
}

Optional<StringRef> AttrOrTypeParameter::getParser() const {
  return getDefValue<llvm::StringInit>("parser");
}

Optional<StringRef> AttrOrTypeParameter::getPrinter() const {
  return getDefValue<llvm::StringInit>("printer");
}

Optional<StringRef> AttrOrTypeParameter::getSummary() const {
  return getDefValue<llvm::StringInit>("summary");
}

StringRef AttrOrTypeParameter::getSyntax() const {
  if (auto *stringType = dyn_cast<llvm::StringInit>(getDef()))
    return stringType->getValue();
  return getDefValue<llvm::StringInit>("syntax").value_or(getCppType());
}

bool AttrOrTypeParameter::isOptional() const {
  // Parameters with default values are automatically optional.
  return getDefValue<llvm::BitInit>("isOptional").value_or(false) ||
         getDefaultValue();
}

Optional<StringRef> AttrOrTypeParameter::getDefaultValue() const {
  return getDefValue<llvm::StringInit>("defaultValue");
}

llvm::Init *AttrOrTypeParameter::getDef() const { return def->getArg(index); }

//===----------------------------------------------------------------------===//
// AttributeSelfTypeParameter
//===----------------------------------------------------------------------===//

bool AttributeSelfTypeParameter::classof(const AttrOrTypeParameter *param) {
  llvm::Init *paramDef = param->getDef();
  if (auto *paramDefInit = dyn_cast<llvm::DefInit>(paramDef))
    return paramDefInit->getDef()->isSubClassOf("AttributeSelfTypeParameter");
  return false;
}
