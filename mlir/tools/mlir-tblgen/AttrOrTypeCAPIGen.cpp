//===- AttrOrTypeCAPIGen.cpp - MLIR Attribute and Type CAPI generation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include "AttrOrTypeFormatGen.h"
#include "CppGenUtilities.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
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

/* static std::string namespacePrefix() {
  static const std::string prefix = [] {
    if (!capiNamespacePrefix.empty())
      return capiNamespacePrefix.getValue();
    return withCapitalFirstLetter(capiDialect.getValue());
  }();
  return prefix;
} */

static void emitAttrTypeHeader(StringRef name, raw_ostream &os) {
  const char *const header = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//

)";
  os << formatv(header, name);
}

namespace {

static const bool EMIT_DECLS = true;
static const bool EMIT_DEFS = false;
struct CAPIDefGenerator : DefGenerator {
  CAPIDefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os,
                   const StringRef &defType, const StringRef &valueType,
                   bool isAttrGenerator)
      : DefGenerator(defs, os, defType, valueType, isAttrGenerator) {}

  bool emitDecls(StringRef selectedDialect) override {
    return emitDeclsOrDefs(selectedDialect, EMIT_DECLS);
  }
  bool emitDefs(StringRef selectedDialect) override {
    return emitDeclsOrDefs(selectedDialect, EMIT_DEFS);
  }

  bool emitDeclsOrDefs(StringRef selectedDialect, bool isDeclGenerator);

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

/* static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
} */

// Transforms a C++ namespace string (e.g. "::ns1::ns2") into a camel-case
// identifier prefix (e.g. "ns1Ns2") suitable for use in C API names.
static std::string cppNamespaceToPrefix(StringRef ns) {
  std::string result;
  SmallVector<StringRef, 4> parts;
  ns.split(parts, "::", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (auto [i, part] : llvm::enumerate(parts)) {
    std::string s = part.str();
    if (i > 0 && !s.empty())
      s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    result += s;
  }
  return result;
}

static std::string cppNamespaceToUpper(StringRef ns) {
  std::string prefix = cppNamespaceToPrefix(ns);
  std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::toupper);
  return prefix;
}

// Inverse of cppNamespaceToPrefix: converts a lower-camel-case identifier like
// "classDialectSomeMoreWords" into "class::dialect::SomeMoreWords", treating
// the first `numNsParts` camel-case words as (lowercased) namespace segments.
/* static std::string camelCaseToCppNamespace(StringRef name,
                                           unsigned numNsParts = 2) {
  std::string result;
  unsigned partsSeen = 0;
  size_t wordStart = 0;

  for (size_t i = 1; i <= name.size() && partsSeen < numNsParts; ++i) {
    if (i == name.size() ||
        llvm::isUpper(static_cast<unsigned char>(name[i]))) {
      result += name.slice(wordStart, i).lower() + "::";
      wordStart = i;
      ++partsSeen;
    }
  }

  result += name.drop_front(wordStart).str();
  return result;
} */

struct AttrOrTypeOrBuilderParam;

static bool isEnumParam(const AttrOrTypeOrBuilderParam &param);
struct AttrOrTypeOrBuilderParam 
{
  // Only one of these must be non-null at a type
  private: 
    StringRef name;
    bool is_builder_param;
    const void *ptr;

  public:
    AttrOrTypeOrBuilderParam(const AttrOrTypeParameter &param)
      : name(param.getName()), is_builder_param(false), ptr(&param)  {}
    AttrOrTypeOrBuilderParam(const AttrOrTypeParameter &param, StringRef name)
      : name(name), is_builder_param(false), ptr(&param)  {}
    AttrOrTypeOrBuilderParam(const Builder::Parameter &param, StringRef name)
      : name(name), is_builder_param(true), ptr(&param) {}

    StringRef getName() const {
      if (is_builder_param) {
        return ((const Builder::Parameter *)ptr)->getName().value();
      } else {
        return ((const AttrOrTypeParameter*)ptr)->getName();
      }
    }


    
    std::string getCAPIType() const;

    StringRef getCppType() const {
      if (is_builder_param) {
        return ((const Builder::Parameter *)ptr)->getCppType();
      } else {
        return ((const AttrOrTypeParameter*)ptr)->getCppType();
      }
    }

    const llvm::Init * getDef() const {
      if (is_builder_param) {
        return ((const Builder::Parameter *)ptr)->getDef();
      } else {
        return ((const AttrOrTypeParameter*)ptr)->getDef();
      }
    }


    bool isEnumParam() const {
      return ::isEnumParam(*this);
    }

};

// BuilderParams are just strings, so no way to tell if they are an enum or not. So when they
// are seen elsewhere we register them here so when used in alt builders the types are here.
static llvm::StringSet<llvm::MallocAllocator> enum_types;

static bool isEnumParam(const AttrOrTypeOrBuilderParam &param) {
  if (const llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(param.getDef())) {
    const Record *rec = defInit->getDef();
    if (rec->isSubClassOf("EnumParameter")) {
      enum_types.insert(param.getCppType());
      return true;
    } else {
      return false;
    }
  } else {
    return enum_types.contains(param.getCppType());
  }
} 


/**
 * @brief Map an AttrOrTypeOrBuilderParam to its CAPI type
 * 
 * @param param 
 * @param return_type True if this param represents a return value, for example as the return value of an attribute getter.
 * @return std::string 
 */
static std::string mapParamTypeToCAPI(const AttrOrTypeOrBuilderParam *param, bool return_type) {
  if (const llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(param->getDef())) {
    const Record *rec = defInit->getDef();
    if (rec->isSubClassOf("EnumParameter")) { // || rec->isSubClassOf("EnumInfo")) {
      std::string type = "";
      type += cppNamespaceToPrefix(param->getCppType());
      // type += rec->getValueAsString("underlyingEnumName");
      return type;
    }
    if (rec->isSubClassOf("StringRefParameter"))
      return "MlirStringRef";
  }
  auto cppType = param->getCppType();
  if (cppType == "Type" || cppType == "::mlir::Type" || cppType == "mlir::Type")
    return "MlirType";
  if (cppType == "Attribute" || cppType == "::mlir::Attribute" || cppType.ends_with("Attr")) {
    if (return_type && cppType == "::mlir::StringAttr") {
      // For some reason these map to MlirIdentifier when wrapped instead of MlirAttribute
      return "MlirIdentifier";
    } else {
      return "MlirAttribute";
    }
  }
  if (cppType == "::llvm::StringRef" || cppType == "::mlir::StringRef") {
    return "MlirStringRef";
  }
  if (cppType.starts_with("::")) {
    auto debug = llvm::StringRef(cppNamespaceToPrefix(cppType));
    if (debug.starts_with("mlirStringRef")) {
      llvm::PrintFatalNote(formatv(" from {0}", cppType));
    }
    return cppNamespaceToPrefix(cppType);
  }
  return cppType.str();
}

std::string AttrOrTypeOrBuilderParam::getCAPIType() const {
    return mapParamTypeToCAPI(this, false);
}

/* static bool paramIsEnum(const AttrOrTypeParameter &param) {
    if (const llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(param.getDef())) {
      const Record *rec = defInit->getDef();
      if (rec->isSubClassOf("EnumParameter")) {
        return true;
      }
    }
    return false;
} */

static SmallVector<AttrOrTypeOrBuilderParam>
getGettorParams(ArrayRef<AttrOrTypeParameter> const params) {
  SmallVector<AttrOrTypeOrBuilderParam> builderParams;
  for (auto &param : params) {
    builderParams.emplace_back(param, param.getName());
  }
  return builderParams;
}

static SmallVector<AttrOrTypeOrBuilderParam>
getGettorParams(ArrayRef<mlir::tblgen::Builder::Parameter> params) {
  SmallVector<AttrOrTypeOrBuilderParam> builderParams;
  for (auto &param : params) {
    builderParams.emplace_back(param, param.getName().value());
  }
  return builderParams;
}

static llvm::StringRef getDefCppType(const AttrOrTypeDef &def) {
    const llvm::Record *rec = def.getDef();
    const llvm::RecordVal *name_val = rec->getValue("cppType");
    const llvm::Init *name_init = name_val->getValue();
    return llvm::cast<llvm::StringInit>(name_init)->getValue();
}

static llvm::StringRef getDefCppType(const EnumInfo &def) {
    const llvm::Record rec = def.getDef();
    const llvm::RecordVal *name_val = rec.getValue("cppType");
    const llvm::Init *name_init = name_val->getValue();
    return llvm::cast<llvm::StringInit>(name_init)->getValue();
}


static void emitGettorDeclOrDef(const AttrOrTypeDef &def, 
                                ArrayRef<AttrOrTypeOrBuilderParam> params, bool hasInferredContextParameter,
                                raw_ostream &os, bool isAttrGenerator, bool isDeclGenerator, unsigned altIndex) {
  // Output *Get function signature
  os << "MLIR_CAPI_EXPORTED ";
  if (isAttrGenerator)
    os << "MlirAttribute ";
  else
    os << "MlirType ";
  os << llvm::StringRef(cppNamespaceToPrefix(def.getDialect().getCppNamespace())) << def.getCppClassName() << "Get";
  if (altIndex == 0) {
    os << "(";
  } else {
    os << "Alt" << altIndex << "(";
  }

  os << "MlirContext context";
  if (!params.empty() > 0) {
    os << ",";
  }
  for (auto [i, param] : llvm::enumerate(params)) {
    os << param.getCAPIType() << " " << param.getName()
       << (i < (params.size() - 1) ? ", " : "");
  }
  if (isDeclGenerator) {
    os << ");\n";
  } else {
    // Output *get function definition
    os << ") {\n";
    /* if (body.has_value()) {
      os << body.value() << "\n";
    } else { */
      os << "\treturn ";
      // Don't wrap enums, they are scalar types
      // if (!def.getDef()->isSubClassOf("EnumAttr")) {
        os << "wrap(";
      // }
      os << getDefCppType(def) << "::get(";
      if (!hasInferredContextParameter) {
        os << "unwrap(context)";
        if (params.size() > 0) {
          os << ", ";
        } 
      }
      for (auto [i, param] : llvm::enumerate(params)) {
        // If this is an enum, just use C++ static cast
        if (param.isEnumParam()) {
          os << "(" << formatv("/* Line {0} */", __LINE__) << param.getCppType() << ")" << param.getName();
        } else {
          if (param.getCppType().contains(':')) {
            // std::string debug_str;
            // if (const llvm::DefInit *defInit = dyn_cast<llvm::DefInit>(param.getDef())) {
            //   defInit;
            //   debug_str = formatv("/* def = {0} */", (void *)param.getDef()).str();
            // } else {
            //   debug_str = formatv("/* not a def {0} */", (void *)param.getDef()).str();
            // }
            os << /* debug_str << */ "llvm::cast<" << param.getCppType() << ">(";
          } 
          // Is this a wrapped type? If so unwrap it, otherwised don't
          if (StringRef(param.getCAPIType()).starts_with("Mlir")) {
            os << "(unwrap(" << param.getName() << "))";
          } else {
            os << "(" << param.getName() << ")";
          }
          if (param.getCppType().contains(':')) {
            os << ")"; // close the cast
          }
        }
        os << (i < params.size() - 1 ? ", " : "");
      }
      // if (!def.getDef()->isSubClassOf("EnumAttr")) {
        // Close the wrap
        os << ")";
      // }
      os << ");\n";
    // }
    os << "}\n";
  }
}

static void emitAccessorDeclsOrDefs(const AttrOrTypeDef &def,
                              ArrayRef<AttrOrTypeParameter> params,
                              raw_ostream &os, bool isAttrGenerator, bool isDeclGenerator) {

  for (AttrOrTypeParameter param : params) {
    if (isUnsupportedParam(param)) {
      os << "// Skipping accessor for unsupported parameter " << param.getName() << "\n";
      continue;
    }
    std::string paramName = param.getName().str();
    os << "MLIR_CAPI_EXPORTED ";
    const AttrOrTypeOrBuilderParam aotob_param = AttrOrTypeOrBuilderParam(param);
    std::string return_type_string = mapParamTypeToCAPI(&aotob_param,true);
    StringRef return_type = StringRef(return_type_string);
    os << return_type << " " << cppNamespaceToPrefix(def.getDialect().getCppNamespace()) << def.getCppClassName()
       << "Get" << withCapitalFirstLetter(param.getName().str());
    if (isAttrGenerator)
      os << "(MlirAttribute attr";
    else
      os << "(MlirType type";
    if (isDeclGenerator) {
      os << ");\n";
     } else {
      bool wrapReturn = !isEnumParam(param) && !return_type.starts_with("uint") && !return_type.starts_with("int");
      os << ") {\n";
      if (wrapReturn) {
        os << "\treturn wrap(";
      } else {
        // Enums are enums not objects
        os << "\treturn (" << mapParamTypeToCAPI(&aotob_param,false) << ")";
      }
      os << "llvm::cast<" << getDefCppType(def) << ">(unwrap(attr)).get" << withCapitalFirstLetter(param.getName().str()) << "()";
      if (wrapReturn) {
        // close the wrap
        os << ")";
      }
      os << ";\n";
      os << "}\n";
     }
  }
}

static void emitTypeIDDeclOrDef(const AttrOrTypeDef &def, raw_ostream &os, bool isDeclGenerator) {
  os << "MLIR_CAPI_EXPORTED MlirTypeID " << cppNamespaceToPrefix(def.getDialect().getCppNamespace()) << def.getCppClassName()
     << "GetTypeID(";
     if (isDeclGenerator) {
      os << ");\n";
     } else {
      os << ") {\n";
      os << "\treturn wrap(" << getDefCppType(def) << "::getTypeID());\n";
      os << "}\n";
     }
}

static void emitTypeIDDeclOrDef(const EnumInfo &enumInfo, raw_ostream &os, bool isDeclGenerator) {
  os << "MLIR_CAPI_EXPORTED MlirTypeID " << cppNamespaceToPrefix(enumInfo.getCppNamespace()) << enumInfo.getEnumClassName()
     << "GetTypeID(";
     if (isDeclGenerator) {
      os << ");\n";
     } else {
      os << ") {\n";
      os << "\treturn wrap(" << getDefCppType(enumInfo) << "::getTypeID());\n";
      os << "}\n";
     }
}

static void emitIsADeclOrDef(const AttrOrTypeDef &def, raw_ostream &os,
                              bool isAttrGenerator, bool isDeclGenerator) {
  os << "MLIR_CAPI_EXPORTED bool mlir"; // JEG: Perhaps this one is correct?
  if (isAttrGenerator)
    os << "Attribute";
  else
    os << "Type";
  os << "IsA" << cppNamespaceToPrefix(def.getDialect().getCppNamespace()) << def.getCppClassName();
  if (isAttrGenerator)
    os << "(MlirAttribute attr";
  else
    os << "(MlirType type";

  if (isDeclGenerator) {
    os << ");\n";
  } else {

    os << ") {\n";
    os << "\treturn llvm::isa<" << getDefCppType(def)
                              << ">(unwrap(" << (isAttrGenerator? "attr" : "type") << "));\n";
    os << "}\n";
  }
}

static void emitIsADeclOrDef(const EnumInfo &enumInfo, raw_ostream &os,
                              bool isDeclGenerator) {
  std::string name = enumInfo.getEnumClassName().str() + "Attr";
  os << "MLIR_CAPI_EXPORTED bool mlirAttributeIsA" << cppNamespaceToPrefix(enumInfo.getCppNamespace()) << name;
  os << "(MlirAttribute attr";
  if (isDeclGenerator) {
    os << ");\n";
  } else {
    os << ") {\n";
    os << "\treturn llvm::isa<" << getDefCppType(enumInfo) << ">(unwrap(attr));\n";
    os << "}\n";
  }
}

static bool emitEnumDecls(ArrayRef<const Record *> records, raw_ostream &os) {

  for (const auto *rec : records) {
    EnumInfo enumInfo(*rec);

    // cppNamespace cannot being empty when generating the CAPI disambiguated enums, so 
    // error out if it is missing
    std::string enum_class_name = "";
    enum_class_name += enumInfo.getCppNamespace();
    if (enum_class_name == "") {
      PrintFatalNote(formatv("enumInfo for {0} is missing a cppNamespace value", 
                      enumInfo.getEnumClassName()));
    }
    enum_class_name += enumInfo.getEnumClassName();

    llvm::IfNDefGuardEmitter scope(os, "NO_" + cppNamespaceToUpper(enum_class_name) +
                                     "_ENUM_CAPI_DECL");
    os << "// " << enumInfo.getSummary() << "\n";
    // Move the enum name creation into mapParamTypeToCAPI, or dup there
    os << "enum " << cppNamespaceToPrefix(enum_class_name);

    if (!enumInfo.getUnderlyingType().empty())
      os << " : " << enumInfo.getUnderlyingType();
    os << " {\n";

    auto prefix = formatv("{0}_", cppNamespaceToPrefix(enum_class_name));
    for (const EnumCase &enumerant : enumInfo.getAllCases()) {
      auto symbol = makeIdentifier(enumerant.getSymbol());
      auto value = enumerant.getValue();
      if (value >= 0)
        os << formatv("  {0}{1} = {2},\n", prefix, symbol, value);
      else
        os << formatv("  {0}{1},\n", prefix, symbol);
    }
    os << "};\n";
    // Add convenience typedef
    os << formatv("typedef enum {0} {0}; \n", cppNamespaceToPrefix(enum_class_name));
  }

  os << "\n";

  return false;
}

static bool emitEnumAttrDecls(ArrayRef<const Record *> records, raw_ostream &os,
                              StringRef selectedDialect) {
  for (const Record *rec : records) {
    AttrOrTypeDef attr(&*rec);
    StringRef dialect = attr.getDialect().getName();
    if (dialect != selectedDialect)
      continue;

    EnumInfo enumInfo(*attr.getDef()->getValueAsDef("enum"));
    llvm::IfNDefGuardEmitter scope(os, "NO_" + cppNamespaceToUpper(enumInfo.getCppNamespace()) + "_" +
                                     enumInfo.getEnumClassName().upper() +
                                     "_ENUM_ATTR_CAPI_DECL");

    os << "MLIR_CAPI_EXPORTED MlirAttribute " << cppNamespaceToPrefix(enumInfo.getCppNamespace())
       << enumInfo.getEnumClassName() << "AttrGet(MlirContext context, "
       << cppNamespaceToPrefix(enumInfo.getCppNamespace()) << enumInfo.getEnumClassName() << " value);\n";

    std::string name = enumInfo.getEnumClassName().str() + "Attr";
    emitTypeIDDeclOrDef(enumInfo, os, EMIT_DECLS);
    emitIsADeclOrDef(enumInfo, os, EMIT_DECLS);

    os << "MLIR_CAPI_EXPORTED " << cppNamespaceToPrefix(enumInfo.getCppNamespace())
       << enumInfo.getEnumClassName() << " ";
    os << cppNamespaceToPrefix(enumInfo.getCppNamespace()) << name << "GetValue(MlirAttribute attr);\n";
  }

  os << "\n";
  return false;
}

bool CAPIDefGenerator::emitDeclsOrDefs(StringRef selectedDialect, bool isDeclGenerator) {
  emitSourceFileHeader((defType + "Def C API Def Declarations").str(), os);

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;

  for (const AttrOrTypeDef &def : defs) {
    // StringRef name = cppNamespaceToPrefix(def.getCppClassName());
    StringRef name = def.getCppClassName();
    llvm::IfNDefGuardEmitter scope(os, "NO_" + name.upper() + "_CAPI_DECL");

    ArrayRef<AttrOrTypeParameter> params = def.getParameters();
    emitAttrTypeHeader(name, os);
    bool has_unsupported_params = llvm::any_of(params, isUnsupportedParam);
    if (!def.skipDefaultBuilders() && !has_unsupported_params) {
      emitGettorDeclOrDef(def, getGettorParams(params), false, os, isAttrGenerator, isDeclGenerator, 0);
    } else if (has_unsupported_params) {
      os << formatv("// Skipping gettor {0} for {1}, it has an unsupported parameter type \n", 
                    0, name);
    }
      
    unsigned altNum = 1;
    for (const AttrOrTypeBuilder &builder : def.getBuilders()) {
      // If the builder doesn't have a body
      // if (!builder.getBody().has_value()) {
        os << "// " << getDefCppType(def).str() << " Alt " << altNum << "\n";
        emitGettorDeclOrDef(def, getGettorParams(builder.getParameters()), builder.hasInferredContextParameter(),
                            os, isAttrGenerator, isDeclGenerator, altNum);
      /* } else {
        os << "// Has Body " << getDefCppType(def).str() << " Alt " << altNum << "\n";        
        // Do substitutions into body
        FmtContext fmtctx;
        fmtctx.addSubst("_get", getDefCppType(def).str() + "::get");
        if (!builder.hasInferredContextParameter()) {
          fmtctx.addSubst("_ctxt", "unwrap(context)");
        }
        std::string body_str = tgfmt(builder.getBody().value(), &fmtctx);
        std::optional<StringRef> body_str_ref = body_str;
        emitGettorDeclOrDef(def, getGettorParams(builder.getParameters()), body_str_ref,
                            os, isAttrGenerator, isDeclGenerator, altNum);
      } */
      altNum++;
    }
    emitTypeIDDeclOrDef(def, os, isDeclGenerator);
    emitIsADeclOrDef(def, os, isAttrGenerator, isDeclGenerator);
    if (def.genAccessors() && !params.empty())
      emitAccessorDeclsOrDefs(def, params, os, isAttrGenerator, isDeclGenerator);
  }
  os << "\n";

  return false;
}

/* bool CAPIDefGenerator::emitDefs(StringRef selectedDialect) {
  emitSourceFileHeader((defType + "Def C API Defs").str(), os);

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;

  for (const AttrOrTypeDef &def : defs) {
    // StringRef name = cppNamespaceToPrefix(def.getCppClassName());
    StringRef name = def.getCppClassName();
    llvm::IfNDefGuardEmitter scope(os, "NO_" + name.upper() + "_CAPI_DECL");
    ArrayRef<AttrOrTypeParameter> params = def.getParameters();
    if (!llvm::any_of(params, isUnsupportedParam)) {
        emitGettorDeclOrDef(def, params, os, isAttrGenerator, EMIT_DEFS);
    }
    emitTypeIDDeclOrDef(def, os, EMIT_DEFS);
    emitIsADeclOrDef(def, os, isAttrGenerator, EMIT_DEFS);
    emitAccessorDeclsOrDefs(def, params, os, isAttrGenerator, EMIT_DEFS);
  }

  return false;
}*/

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

static mlir::GenRegistration genAttrDefs(
  "gen-attrdef-capi-defs", "Generate AttrDef C API definitions",
  [](const RecordKeeper &records, raw_ostream &os) {
      CAPIAttrDefGenerator generator(records.getAllDerivedDefinitionsIfDefined("AttrDef"), os);
      return generator.emitDefs(capiDialect);
});

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
