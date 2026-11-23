//===- DialectGen.cpp - MLIR dialect definitions generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DialectGen uses the description of dialects to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "CppGenUtilities.h"
#include "DialectGenUtilities.h"
#include "PartitionHelpers.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-opdefgen"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
static llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

static void populateDiscardableAttributes(
    Dialect &dialect, const llvm::DagInit *discardableAttrDag,
    SmallVector<std::pair<std::string, std::string>> &discardableAttributes) {
  for (int i : llvm::seq<int>(0, discardableAttrDag->getNumArgs())) {
    const llvm::Init *arg = discardableAttrDag->getArg(i);

    StringRef givenName = discardableAttrDag->getArgNameStr(i);
    if (givenName.empty())
      PrintFatalError(dialect.getDef()->getLoc(),
                      "discardable attributes must be named");
    discardableAttributes.push_back(
        {givenName.str(), arg->getAsUnquotedString()});
  }
}

std::optional<Dialect>
tblgen::findDialectToGenerate(ArrayRef<Dialect> dialects) {
  if (dialects.empty()) {
    llvm::errs() << "no dialect was found\n";
    return std::nullopt;
  }

  // Select the dialect to gen for.
  if (dialects.size() == 1 && selectedDialect.getNumOccurrences() == 0)
    return dialects.front();

  if (selectedDialect.getNumOccurrences() == 0) {
    llvm::errs() << "when more than 1 dialect is present, one must be selected "
                    "via '-dialect'\n";
    return std::nullopt;
  }

  const auto *dialectIt = llvm::find_if(dialects, [](const Dialect &dialect) {
    return dialect.getName() == selectedDialect;
  });
  if (dialectIt == dialects.end()) {
    llvm::errs() << "selected dialect with '-dialect' does not exist\n";
    return std::nullopt;
  }
  return *dialectIt;
}

/// Returns the name of `kind` as used in the registration hook names.
static StringRef getRegistrationKindName(RegistrationKind kind) {
  switch (kind) {
  case RegistrationKind::Operation:
    return "Operation";
  case RegistrationKind::Attribute:
    return "Attribute";
  case RegistrationKind::Type:
    return "Type";
  }
  llvm_unreachable("unknown registration kind");
}

/// Returns the class whose `insert` method registers a definition of `kind`.
static StringRef getRegistrar(RegistrationKind kind) {
  switch (kind) {
  case RegistrationKind::Operation:
    return "::mlir::RegisteredOperationName";
  case RegistrationKind::Attribute:
    return "::mlir::AbstractAttribute";
  case RegistrationKind::Type:
    return "::mlir::AbstractType";
  }
  llvm_unreachable("unknown registration kind");
}

std::string tblgen::getRegistrationHookName(const Dialect &dialect,
                                            RegistrationKind kind,
                                            StringRef registrationName,
                                            StringRef part) {
  return llvm::formatv("register{0}{1}s{2}",
                       registrationName.empty() ? dialect.getCppClassName()
                                                : registrationName.str(),
                       getRegistrationKindName(kind),
                       getPartitionFunctionSuffix(part))
      .str();
}

void tblgen::emitRegistrationHookDecls(raw_ostream &os, const Dialect &dialect,
                                       RegistrationKind kind,
                                       StringRef registrationName,
                                       ArrayRef<std::string> parts,
                                       StringRef partsDescription) {
  std::string dialectClass = dialect.getCppClassName();
  std::string kindName = getRegistrationKindName(kind).lower();
  os << "class " << dialectClass << ";\n";
  if (parts.empty())
    os << llvm::formatv(
        "/// Registers the {0}s defined in this file with `dialect`.\n",
        kindName);
  else
    os << llvm::formatv("/// Registers the {0}s of all {1} with `dialect`.\n",
                        kindName, partsDescription);
  auto emitDecl = [&](StringRef part) {
    os << llvm::formatv(
        "void {0}({1} *dialect);\n",
        getRegistrationHookName(dialect, kind, registrationName, part),
        dialectClass);
  };
  emitDecl("");
  for (StringRef part : parts)
    emitDecl(part);
}

void tblgen::emitRegistrationHookDefs(
    raw_ostream &os, const Dialect &dialect, RegistrationKind kind,
    StringRef registrationNameSuffix, StringRef part,
    ArrayRef<std::string> parts, ArrayRef<std::string> classNamesToRegister) {
  std::string dialectClass = dialect.getCppClassName();
  const char *const hookDef = "void {0}({1} *dialect) {{\n";
  if (!parts.empty() && parts.front() == part) {
    os << llvm::formatv(
        hookDef, getRegistrationHookName(dialect, kind, registrationNameSuffix),
        dialectClass);
    for (StringRef otherPart : parts)
      os << "  "
         << getRegistrationHookName(dialect, kind, registrationNameSuffix,
                                    otherPart)
         << "(dialect);\n";
    os << "}\n";
  }

  // Register each class with a separate call instead of a variadic one like
  // `addOperations<...>()`, whose single instantiation over all classes takes
  // noticeably longer to compile.
  os << llvm::formatv(
      hookDef,
      getRegistrationHookName(dialect, kind, registrationNameSuffix, part),
      dialectClass);
  for (StringRef className : classNamesToRegister)
    os << "  " << getRegistrar(kind) << "::insert<" << className
       << ">(*dialect);\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// GEN: Dialect declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a dialect class declaration.
///
/// {0}: The name of the dialect class.
/// {1}: The dialect namespace.
/// {2}: The dialect parent class.
/// {3}: The declaration of initialize(), if the dialect has an initializer.
static const char *const dialectDeclBeginStr = R"(
class {0} : public ::mlir::{2} {
  explicit {0}(::mlir::MLIRContext *context);
{3}
  friend class ::mlir::MLIRContext;
public:
  ~{0}() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("{1}");
  }
)";

/// Registration for a single dependent dialect: to be inserted in the ctor
/// above for each dependent dialect.
const char *const dialectRegistrationTemplate =
    "getContext()->loadDialect<{0}>();";

/// The code block for the attribute parser/printer hooks.
static const char *const attrParserDecl = R"(
  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;
)";

/// The code block for the type parser/printer hooks.
static const char *const typeParserDecl = R"(
  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
)";

/// The code block for the canonicalization pattern registration hook.
static const char *const canonicalizerDecl = R"(
  /// Register canonicalization patterns.
  void getCanonicalizationPatterns(
      ::mlir::RewritePatternSet &results) const override;
)";

/// The code block for the constant materializer hook.
static const char *const constantMaterializerDecl = R"(
  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;
)";

/// The code block for the operation attribute verifier hook.
static const char *const opAttrVerifierDecl = R"(
    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::llvm::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the region argument attribute verifier hook.
static const char *const regionArgAttrVerifierDecl = R"(
    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region argument.
    ::llvm::LogicalResult verifyRegionArgAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned argIndex,
        ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the region result attribute verifier hook.
static const char *const regionResultAttrVerifierDecl = R"(
    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region result.
    ::llvm::LogicalResult verifyRegionResultAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned resultIndex,
        ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the op interface fallback hook.
static const char *const operationInterfaceFallbackDecl = R"(
    /// Provides a hook for op interface.
    void *getRegisteredInterfaceForOp(mlir::TypeID interfaceID,
                                      mlir::OperationName opName) override;
)";

/// The code block for the discardable attribute helper.
static const char *const discardableAttrHelperDecl = R"(
    /// Helper to manage the discardable attribute `{1}`.
    class {0}AttrHelper {{
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {{
        return "{4}.{1}";
      }
      constexpr ::mlir::StringAttr getName() const {{
        return name;
      }

      explicit {0}AttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {{}

     {2} getAttr(::mlir::Operation *op) const {{
       return op->getDiscardableAttrOfType<{2}>(name);
     }
     void setAttr(::mlir::Operation *op, {2} val) const {{
       op->setDiscardableAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) const {{
       return op->hasDiscardableAttrOfType<{2}>(name);
     }
     void removeAttr(::mlir::Operation *op) const {{
       assert(op->hasDiscardableAttrOfType<{2}>(name));
       op->removeDiscardableAttr(name);
     }
   };
   {0}AttrHelper get{0}AttrHelper() {
     return {3}AttrName;
   }
 private:
   {0}AttrHelper {3}AttrName;
 public:
)";

/// Generate the declaration for the given dialect class.
static void emitDialectDecl(Dialect &dialect, raw_ostream &os) {
  // Emit all nested namespaces.
  {
    DialectNamespaceEmitter nsEmitter(os, dialect);

    // Emit the start of the decl.
    std::string cppName = dialect.getCppClassName();
    StringRef superClassName =
        dialect.isExtensible() ? "ExtensibleDialect" : "Dialect";

    tblgen::emitSummaryAndDescComments(os, dialect.getSummary(),
                                       dialect.getDescription(),
                                       /*terminateCmment=*/false);
    os << llvm::formatv(
        dialectDeclBeginStr, cppName, dialect.getName(), superClassName,
        dialect.hasInitializer() ? "\n  void initialize();" : "");

    // If the dialect requested the default attribute printer and parser, emit
    // the declarations for the hooks.
    if (dialect.useDefaultAttributePrinterParser())
      os << attrParserDecl;
    // If the dialect requested the default type printer and parser, emit the
    // delcarations for the hooks.
    if (dialect.useDefaultTypePrinterParser())
      os << typeParserDecl;

    // Add the decls for the various features of the dialect.
    if (dialect.hasCanonicalizer())
      os << canonicalizerDecl;
    if (dialect.hasConstantMaterializer())
      os << constantMaterializerDecl;
    if (dialect.hasOperationAttrVerify())
      os << opAttrVerifierDecl;
    if (dialect.hasRegionArgAttrVerify())
      os << regionArgAttrVerifierDecl;
    if (dialect.hasRegionResultAttrVerify())
      os << regionResultAttrVerifierDecl;
    if (dialect.hasOperationInterfaceFallback())
      os << operationInterfaceFallbackDecl;

    const llvm::DagInit *discardableAttrDag =
        dialect.getDiscardableAttributes();
    SmallVector<std::pair<std::string, std::string>> discardableAttributes;
    populateDiscardableAttributes(dialect, discardableAttrDag,
                                  discardableAttributes);

    for (const auto &attrPair : discardableAttributes) {
      std::string camelNameUpper = llvm::convertToCamelFromSnakeCase(
          attrPair.first, /*capitalizeFirst=*/true);
      std::string camelName = llvm::convertToCamelFromSnakeCase(
          attrPair.first, /*capitalizeFirst=*/false);
      os << llvm::formatv(discardableAttrHelperDecl, camelNameUpper,
                          attrPair.first, attrPair.second, camelName,
                          dialect.getName());
    }

    if (std::optional<StringRef> extraDecl = dialect.getExtraClassDeclaration())
      os << *extraDecl;

    // End the dialect decl.
    os << "};\n";
  }
  if (!dialect.getCppNamespace().empty())
    os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << dialect.getCppClassName() << ")\n";
}

/// Emit the `#include` directives for the headers declaring the C++ constructs
/// used by the declaration of `dialect`.
static void emitDialectDeclIncludes(const Dialect &dialect, raw_ostream &os) {
  SmallVector<StringRef> includes;
  // `::mlir::Dialect` or `::mlir::ExtensibleDialect`.
  includes.push_back(dialect.isExtensible() ? "mlir/IR/ExtensibleDialect.h"
                                            : "mlir/IR/Dialect.h");
  if (dialect.getDiscardableAttributes()->getNumArgs() != 0) {
    // `::mlir::StringAttr` and `::mlir::Operation`, used by the discardable
    // attribute helpers.
    includes.push_back("mlir/IR/BuiltinAttributes.h");
    includes.push_back("mlir/IR/Operation.h");
  }
  tblgen::emitIncludes(os, includes);
}

static bool emitDialectDecls(const RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Dialect Declarations", os, records);

  auto dialectDefs = records.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect = findDialectToGenerate(dialects);
  if (!dialect)
    return true;
  emitDialectDeclIncludes(*dialect, os);
  emitDialectDecl(*dialect, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Dialect definitions
//===----------------------------------------------------------------------===//

/// The code block to generate a dialect constructor definition.
///
/// {0}: The name of the dialect class.
/// {1}: Initialization code that is emitted in the ctor body, such as
///      dependent dialect registration, the registration of the attributes,
///      types and operations generated by ODS and the call to initialize().
/// {2}: The dialect parent class.
/// {3}: Extra members to initialize
static const char *const dialectConstructorStr = R"(
{0}::{0}(::mlir::MLIRContext *context)
    : ::mlir::{2}(getDialectNamespace(), context, ::mlir::TypeID::get<{0}>())
    {3}
     {{
  {1}
}
)";

/// The code block to generate a default destructor definition.
///
/// {0}: The name of the dialect class.
static const char *const dialectDestructorStr = R"(
{0}::~{0}() = default;

)";

static void emitDialectDef(Dialect &dialect, const RecordKeeper &records,
                           raw_ostream &os) {
  std::string cppClassName = dialect.getCppClassName();

  // Emit the TypeID explicit specializations to have a single symbol def.
  if (!dialect.getCppNamespace().empty())
    os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << cppClassName << ")\n";

  // Emit all nested namespaces.
  DialectNamespaceEmitter nsEmitter(os, dialect);

  // Build the initialization code: the registration of the dependent dialects
  // and of the attributes, types and operations generated by ODS, whose
  // registration hooks are defined alongside the generated definitions.
  SmallVector<std::string> hooks;
  if (dialect.useDefaultAttributeRegistration())
    hooks.push_back(
        getRegistrationHookName(dialect, RegistrationKind::Attribute));
  if (dialect.useDefaultTypeRegistration())
    hooks.push_back(getRegistrationHookName(dialect, RegistrationKind::Type));
  if (dialect.useDefaultOperationRegistration())
    hooks.push_back(
        getRegistrationHookName(dialect, RegistrationKind::Operation));
  for (const std::string &hook : hooks)
    os << llvm::formatv("void {0}({1} *dialect);\n", hook, cppClassName);

  std::string initCode;
  llvm::raw_string_ostream initOs(initCode);
  llvm::ListSeparator sep("\n  ");
  for (StringRef dependentDialect : dialect.getDependentDialects())
    initOs << sep
           << llvm::formatv(dialectRegistrationTemplate, dependentDialect);
  for (const std::string &hook : hooks)
    initOs << sep << hook << "(this);";
  if (dialect.hasInitializer())
    initOs << sep << "initialize();";

  // Emit the constructor and destructor.
  StringRef superClassName =
      dialect.isExtensible() ? "ExtensibleDialect" : "Dialect";

  const llvm::DagInit *discardableAttrDag = dialect.getDiscardableAttributes();
  SmallVector<std::pair<std::string, std::string>> discardableAttributes;
  populateDiscardableAttributes(dialect, discardableAttrDag,
                                discardableAttributes);
  std::string discardableAttributesInit;
  for (const auto &attrPair : discardableAttributes) {
    std::string camelName = llvm::convertToCamelFromSnakeCase(
        attrPair.first, /*capitalizeFirst=*/false);
    llvm::raw_string_ostream os(discardableAttributesInit);
    os << ", " << camelName << "AttrName(context)";
  }

  os << llvm::formatv(dialectConstructorStr, cppClassName, initCode,
                      superClassName, discardableAttributesInit);
  if (!dialect.hasNonDefaultDestructor())
    os << llvm::formatv(dialectDestructorStr, cppClassName);
}

static bool emitDialectDefs(const RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Dialect Definitions", os, records);

  auto dialectDefs = records.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect = findDialectToGenerate(dialects);
  if (!dialect)
    return true;
  // `::mlir::MLIRContext::loadDialect`, used to load the dependent dialects.
  if (!dialect->getDependentDialects().empty())
    tblgen::emitIncludes(os, StringRef("mlir/IR/MLIRContext.h"));
  emitDialectDef(*dialect, records, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Dialect registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return emitDialectDecls(records, os);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitDialectDefs(records, os);
                   });
