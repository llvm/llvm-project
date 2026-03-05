//===- AttrOrTypeCAPIGen.cpp - MLIR Attribute and Type CAPI generation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AttrOrTypeFormatGen.h"
#include "CppGenUtilities.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-attr-or-type-capi-gen"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::Record;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory attrOrTypeCAPIDefGenCat(
    "Options for -gen-attr-capi-* and -gen-typedef-capi-*");
static llvm::cl::opt<std::string>
    capiDialect("attr-or-type-capi-dialect",
                llvm::cl::desc("Generate C APIs for this dialect"),
                llvm::cl::cat(attrOrTypeCAPIDefGenCat),
                llvm::cl::CommaSeparated);
static llvm::cl::opt<std::string> capiNamespacePrefix(
    "attr-or-type-capi-namespace-prefix",
    llvm::cl::desc("Generate C APIs with this namespace prefix"),
    llvm::cl::cat(attrOrTypeCAPIDefGenCat));

static std::string makeIdentifier(StringRef str) {
  if (!str.empty() && llvm::isDigit(static_cast<unsigned char>(str.front()))) {
    std::string newStr = std::string("_") + str.str();
    return newStr;
  }
  return str.str();
}

static std::string withCapitalFirstLetter(std::string name) {
  name[0] = static_cast<std::string::value_type>(
      std::toupper(static_cast<unsigned char>(name[0])));
  return name;
}

static std::string namespacePrefix() {
  static const std::string prefix = [] {
    if (!capiNamespacePrefix.empty())
      return capiNamespacePrefix.getValue();
    return withCapitalFirstLetter(capiDialect.getValue());
  }();
  return prefix;
}

static void emitAttrTypeHeader(StringRef name, raw_ostream &os) {
  const char *const header = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//

)";
  os << formatv(header, name);
}

namespace {
struct CAPIDefGenerator : DefGenerator {
  CAPIDefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os,
                   const StringRef &defType, const StringRef &valueType,
                   bool isAttrGenerator)
      : DefGenerator(defs, os, defType, valueType, isAttrGenerator) {}

  bool emitDecls(StringRef selectedDialect) override;
};
} // namespace

static bool isUnsupportedParam(const AttrOrTypeParameter &param) {
  if (auto *defInit = dyn_cast<llvm::DefInit>(param.getDef())) {
    const Record *rec = defInit->getDef();
    if (rec->isSubClassOf("ArrayRefParameter"))
      return true;
    if (rec->isSubClassOf("OptionalArrayRefParameter"))
      return true;
  }
  return false;
}

static std::string mapParamTypeToCAPI(const AttrOrTypeParameter &param) {
  StringRef cppType = param.getCppType();
  if (const llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(param.getDef())) {
    const Record *rec = defInit->getDef();
    if (rec->isSubClassOf("EnumParameter")) {
      std::string type = "mlir";
      type += namespacePrefix();
      type += rec->getValueAsString("underlyingEnumName");
      return type;
    }
    if (rec->isSubClassOf("StringRefParameter"))
      return "MlirStringRef";
  }
  if (cppType == "Type")
    return "MlirType";
  if (cppType == "Attribute" || cppType == "::mlir::DictionaryAttr" ||
      cppType == "::mlir::ArrayAttr" || cppType == "DictionaryAttr" ||
      cppType == "ArrayAttr")
    return "MlirAttribute";
  return cppType.str();
}

static SmallVector<MethodParameter>
getGettorParams(ArrayRef<AttrOrTypeParameter> params,
                std::initializer_list<MethodParameter> prefix) {
  SmallVector<MethodParameter> builderParams;
  builderParams.append(prefix.begin(), prefix.end());
  for (auto &param : params) {
    builderParams.emplace_back(mapParamTypeToCAPI(param), param.getName());
  }
  return builderParams;
}

static void emitGettorDecl(StringRef name, ArrayRef<AttrOrTypeParameter> params,
                           raw_ostream &os, bool isAttrGenerator) {
  os << "MLIR_CAPI_EXPORTED ";
  if (isAttrGenerator)
    os << "MlirAttribute ";
  else
    os << "MlirType ";
  os << "mlir" + namespacePrefix() << name << "Get(";
  SmallVector<MethodParameter> params_ =
      getGettorParams(params, {{"MlirContext", "context"}});
  for (auto [i, param] : llvm::enumerate(params_)) {
    os << param.getType() << " " << param.getName()
       << (i < params_.size() - 1 ? ", " : "");
  }
  os << ");\n";
}

static void emitAccessorDecls(StringRef name,
                              ArrayRef<AttrOrTypeParameter> params,
                              raw_ostream &os, bool isAttrGenerator) {
  for (AttrOrTypeParameter param : params) {
    if (isUnsupportedParam(param))
      continue;
    std::string paramName = param.getName().str();
    os << "MLIR_CAPI_EXPORTED ";
    os << mapParamTypeToCAPI(param) << " mlir" << namespacePrefix() << name
       << "Get" << withCapitalFirstLetter(param.getName().str());
    if (isAttrGenerator)
      os << "(MlirAttribute attr);";
    else
      os << "(MlirType type);";
    os << "\n";
  }
}

static void emitTypeIDDecl(StringRef name, raw_ostream &os) {
  os << "MLIR_CAPI_EXPORTED MlirTypeID mlir" << namespacePrefix() << name
     << "GetTypeID();\n";
}

static void emitIsADecl(StringRef name, raw_ostream &os, bool isAttrGenerator) {
  os << "MLIR_CAPI_EXPORTED bool mlir";
  if (isAttrGenerator)
    os << "Attribute";
  else
    os << "Type";
  os << "IsA" << namespacePrefix() << name;
  if (isAttrGenerator)
    os << "(MlirAttribute attr);";
  else
    os << "(MlirType type);";
  os << "\n";
}

static bool emitEnumDecls(ArrayRef<const Record *> records, raw_ostream &os) {
  {
    llvm::IfDefEmitter scope(os, "GET_ENUM_CAPI_DECLS");
    for (const auto *rec : records) {
      EnumInfo enumInfo(*rec);
      os << "#define GET_" + enumInfo.getEnumClassName().upper() +
                "_ENUM_CAPI_DECL\n";
    }
  }

  for (const auto *rec : records) {
    EnumInfo enumInfo(*rec);
    llvm::IfDefEmitter scope(os, "GET_" + enumInfo.getEnumClassName().upper() +
                                     "_ENUM_CAPI_DECL");
    os << "// " << enumInfo.getSummary() << "\n";
    os << "enum mlir" << namespacePrefix() << enumInfo.getEnumClassName();

    if (!enumInfo.getUnderlyingType().empty())
      os << " : " << enumInfo.getUnderlyingType();
    os << " {\n";

    for (const EnumCase &enumerant : enumInfo.getAllCases()) {
      auto symbol = makeIdentifier(enumerant.getSymbol());
      auto value = enumerant.getValue();
      if (value >= 0)
        os << formatv("  {0} = {1},\n", symbol, value);
      else
        os << formatv("  {0},\n", symbol);
    }
    os << "};\n";
  }

  os << "\n";

  return false;
}

static bool emitEnumAttrDecls(ArrayRef<const Record *> records, raw_ostream &os,
                              StringRef selectedDialect) {
  {
    llvm::IfDefEmitter scope(os, "GET_ENUM_ATTR_CAPI_DECLS");
    for (const auto *rec : records) {
      AttrOrTypeDef attr(&*rec);
      StringRef dialect = attr.getDialect().getName();
      if (dialect != selectedDialect)
        continue;
      EnumInfo enumInfo(*attr.getDef()->getValueAsDef("enum"));
      StringRef name = enumInfo.getEnumClassName();
      os << "#define GET_" + namespacePrefix() + "_" + name.upper() +
                "_ENUM_ATTR_CAPI_DECL\n";
    }
  }

  for (const Record *rec : records) {
    AttrOrTypeDef attr(&*rec);
    StringRef dialect = attr.getDialect().getName();
    if (dialect != selectedDialect)
      continue;

    EnumInfo enumInfo(*attr.getDef()->getValueAsDef("enum"));
    llvm::IfDefEmitter scope(os, "GET_" + namespacePrefix() + "_" +
                                     enumInfo.getEnumClassName().upper() +
                                     "_ENUM_ATTR_CAPI_DECL");

    os << "MLIR_CAPI_EXPORTED MlirAttribute mlir" << namespacePrefix()
       << enumInfo.getEnumClassName() << "AttrGet(MlirContext context, mlir"
       << namespacePrefix() << enumInfo.getEnumClassName() << " value);\n";

    std::string name = enumInfo.getEnumClassName().str() + "Attr";
    emitTypeIDDecl(name, os);
    emitIsADecl(name, os, /*isAttrGenerator*/ true);

    os << "MLIR_CAPI_EXPORTED mlir" << namespacePrefix()
       << enumInfo.getEnumClassName() << " mlir";
    os << namespacePrefix() << name << "GetValue(MlirAttribute attr);\n";
  }

  os << "\n";

  return false;
}

bool CAPIDefGenerator::emitDecls(StringRef selectedDialect) {
  emitSourceFileHeader((defType + "Def C API Def Declarations").str(), os);

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;

  {
    llvm::IfDefEmitter scope(os, "GET_" + defType.upper() + "_CAPI_DECLS");
    for (const AttrOrTypeDef &def : defs) {
      StringRef name = def.getCppClassName();
      os << "#define GET_" + name.upper() + "_" + defType.upper() +
                "_CAPI_DECL\n";
    }
  }

  for (const AttrOrTypeDef &def : defs) {
    StringRef name = def.getCppClassName();
    llvm::IfDefEmitter scope(os, "GET_" + name.upper() + "_" + defType.upper() +
                                     "_CAPI_DECL");

    ArrayRef<AttrOrTypeParameter> params = def.getParameters();
    emitAttrTypeHeader(name, os);
    if (!llvm::any_of(params, isUnsupportedParam))
      emitGettorDecl(name, params, os, isAttrGenerator);
    emitTypeIDDecl(name, os);
    emitIsADecl(name, os, isAttrGenerator);
    if (def.genAccessors() && !params.empty())
      emitAccessorDecls(name, params, os, isAttrGenerator);
  }

  os << "\n";

  return false;
}

namespace {
/// A specialized generator for AttrDefs.
struct CAPIAttrDefGenerator : public CAPIDefGenerator {
  CAPIAttrDefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os)
      : CAPIDefGenerator(defs, os, "Attr", "Attribute",
                         /*isAttrGenerator=*/true) {}
};
/// A specialized generator for TypeDefs.
struct CAPITypeDefGenerator : public CAPIDefGenerator {
  CAPITypeDefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os)
      : CAPIDefGenerator(defs, os, "Type", "Type",
                         /*isAttrGenerator=*/false) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

static mlir::GenRegistration genEnumDecls(
    "gen-enum-capi-decls", "Generate Enum C API declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      emitSourceFileHeader("Enum C API Declarations", os);
      emitEnumDecls(records.getAllDerivedDefinitionsIfDefined("EnumInfo"), os);
      emitEnumAttrDecls(records.getAllDerivedDefinitionsIfDefined("EnumAttr"),
                        os, capiDialect);
      return false;
    });

static mlir::GenRegistration genAttrDecls(
    "gen-attrdef-capi-decls", "Generate AttrDef C API declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      CAPIAttrDefGenerator generator(
          records.getAllDerivedDefinitionsIfDefined("AttrDef"), os);
      return generator.emitDecls(capiDialect);
    });

// static mlir::GenRegistration
//     genAttrDefs("gen-attrdef-capi-defs", "Generate AttrDef C API
//     definitions",
//                 [](const RecordKeeper &records, raw_ostream &os) {
//                   CAPIAttrDefGenerator generator(records, os);
//                   return generator.emitDefs(attrDialect);
//                 });

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

static mlir::GenRegistration genTypeDecls(
    "gen-typedef-capi-decls", "Generate TypeDef C API declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      CAPITypeDefGenerator generator(
          records.getAllDerivedDefinitionsIfDefined("TypeDef"), os);
      return generator.emitDecls(capiDialect);
    });

// static mlir::GenRegistration
//     genTypeDefs("gen-typedef-capi-defs", "Generate TypeDef C API
//     definitions",
//                 [](const RecordKeeper &records, raw_ostream &os) {
//                   CAPITypeDefGenerator generator(records, os);
//                   return generator.emitDefs(capiDialect);
//                 });
