//===- AttrOrTypeDef.cpp - AttrOrTypeDef wrapper classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

std::optional<StringRef> AttrOrTypeBuilder::getReturnType() const {
  std::optional<StringRef> type = def->getValueAsOptionalString("returnType");
  return type && !type->empty() ? type : std::nullopt;
}

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
    llvm::unique_function<void(llvm::ListInit *)> processTraitList =
        [&](llvm::ListInit *traitList) {
          for (auto *traitInit : *traitList) {
            if (!traitSet.insert(traitInit).second)
              continue;

            // If this is an interface, add any bases to the trait list.
            auto *traitDef = cast<llvm::DefInit>(traitInit)->getDef();
            if (traitDef->isSubClassOf("Interface")) {
              if (auto *bases = traitDef->getValueAsListInit("baseInterfaces"))
                processTraitList(bases);
            }

            traits.push_back(Trait::create(traitInit));
          }
        };
    processTraitList(traitList);
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
  // Assembly format printer requires accessors to be generated.
  if (hasDeclarativeFormat && !genAccessors()) {
    PrintFatalError(getLoc(),
                    "'assemblyFormat' requires 'genAccessors' to be true");
  }
  // TODO: Ensure that a suitable builder prototype can be generated:
  // https://llvm.org/PR56415
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

std::optional<StringRef> AttrOrTypeDef::getMnemonic() const {
  return def->getValueAsOptionalString("mnemonic");
}

bool AttrOrTypeDef::hasCustomAssemblyFormat() const {
  return def->getValueAsBit("hasCustomAssemblyFormat");
}

std::optional<StringRef> AttrOrTypeDef::getAssemblyFormat() const {
  return def->getValueAsOptionalString("assemblyFormat");
}

bool AttrOrTypeDef::genAccessors() const {
  return def->getValueAsBit("genAccessors");
}

bool AttrOrTypeDef::genVerifyDecl() const {
  return def->getValueAsBit("genVerifyDecl");
}

bool AttrOrTypeDef::genVerifyInvariantsImpl() const {
  return any_of(parameters, [](const AttrOrTypeParameter &p) {
    return p.getConstraint() != std::nullopt;
  });
}

std::optional<StringRef> AttrOrTypeDef::getExtraDecls() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? std::optional<StringRef>() : value;
}

std::optional<StringRef> AttrOrTypeDef::getExtraDefs() const {
  auto value = def->getValueAsString("extraClassDefinition");
  return value.empty() ? std::optional<StringRef>() : value;
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

std::optional<StringRef> AttrDef::getTypeBuilder() const {
  return def->getValueAsOptionalString("typeBuilder");
}

bool AttrDef::classof(const AttrOrTypeDef *def) {
  return def->getDef()->isSubClassOf("AttrDef");
}

StringRef AttrDef::getAttrName() const {
  return def->getValueAsString("attrName");
}

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

bool TypeDef::classof(const AttrOrTypeDef *def) {
  return def->getDef()->isSubClassOf("TypeDef");
}

StringRef TypeDef::getTypeName() const {
  return def->getValueAsString("typeName");
}

//===----------------------------------------------------------------------===//
// AttrOrTypeParameter
//===----------------------------------------------------------------------===//

template <typename InitT>
auto AttrOrTypeParameter::getDefValue(StringRef name) const {
  std::optional<decltype(std::declval<InitT>().getValue())> result;
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

std::optional<StringRef> AttrOrTypeParameter::getAllocator() const {
  return getDefValue<llvm::StringInit>("allocator");
}

StringRef AttrOrTypeParameter::getComparator() const {
  return getDefValue<llvm::StringInit>("comparator").value_or("$_lhs == $_rhs");
}

StringRef AttrOrTypeParameter::getCppType() const {
  if (auto *stringType = dyn_cast<llvm::StringInit>(getDef()))
    return stringType->getValue();
  auto cppType = getDefValue<llvm::StringInit>("cppType");
  if (cppType)
    return *cppType;
  if (auto *init = dyn_cast<llvm::DefInit>(getDef()))
    llvm::PrintFatalError(
        init->getDef()->getLoc(),
        Twine("Missing `cppType` field in Attribute/Type parameter: ") +
            init->getAsString());
  llvm::report_fatal_error(
      Twine("Missing `cppType` field in Attribute/Type parameter: ") +
          getDef()->getAsString(),
      /*gen_crash_diag=*/false);
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

std::optional<StringRef> AttrOrTypeParameter::getParser() const {
  return getDefValue<llvm::StringInit>("parser");
}

std::optional<StringRef> AttrOrTypeParameter::getPrinter() const {
  return getDefValue<llvm::StringInit>("printer");
}

std::optional<StringRef> AttrOrTypeParameter::getSummary() const {
  return getDefValue<llvm::StringInit>("summary");
}

StringRef AttrOrTypeParameter::getSyntax() const {
  if (auto *stringType = dyn_cast<llvm::StringInit>(getDef()))
    return stringType->getValue();
  return getDefValue<llvm::StringInit>("syntax").value_or(getCppType());
}

bool AttrOrTypeParameter::isOptional() const {
  return getDefaultValue().has_value();
}

std::optional<StringRef> AttrOrTypeParameter::getDefaultValue() const {
  std::optional<StringRef> result =
      getDefValue<llvm::StringInit>("defaultValue");
  return result && !result->empty() ? result : std::nullopt;
}

llvm::Init *AttrOrTypeParameter::getDef() const { return def->getArg(index); }

std::optional<Constraint> AttrOrTypeParameter::getConstraint() const {
  if (auto *param = dyn_cast<llvm::DefInit>(getDef()))
    if (param->getDef()->isSubClassOf("Constraint"))
      return Constraint(param->getDef());
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// AttributeSelfTypeParameter
//===----------------------------------------------------------------------===//

bool AttributeSelfTypeParameter::classof(const AttrOrTypeParameter *param) {
  llvm::Init *paramDef = param->getDef();
  if (auto *paramDefInit = dyn_cast<llvm::DefInit>(paramDef))
    return paramDefInit->getDef()->isSubClassOf("AttributeSelfTypeParameter");
  return false;
}
