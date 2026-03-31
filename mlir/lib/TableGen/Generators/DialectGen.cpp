//===- DialectGen.cpp - MLIR dialect C++ definitions generator ------------===//
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

#include "mlir/TableGen/Generators/DialectGen.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Generators/CppGenUtilities.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::Record;
using llvm::RecordKeeper;

//===----------------------------------------------------------------------===//
// GEN: Dialect declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a dialect class declaration.
///
/// {0}: The name of the dialect class.
/// {1}: The dialect namespace.
/// {2}: The dialect parent class.
static const char *const dialectDeclBeginStr = R"(
class {0} : public ::mlir::{2} {
  explicit {0}(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~{0}() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("{1}");
  }
)";

/// Registration for a single dependent dialect: to be inserted in the ctor
/// above for each dependent dialect.
static const char *const dialectRegistrationTemplate =
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
       return op->getAttrOfType<{2}>(name);
     }
     void setAttr(::mlir::Operation *op, {2} val) const {{
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) const {{
       return op->hasAttrOfType<{2}>(name);
     }
     void removeAttr(::mlir::Operation *op) const {{
       assert(op->hasAttrOfType<{2}>(name));
       op->removeAttr(name);
     }
   };
   {0}AttrHelper get{0}AttrHelper() {
     return {3}AttrName;
   }
 private:
   {0}AttrHelper {3}AttrName;
 public:
)";

/// The code block to generate a dialect constructor definition.
///
/// {0}: The name of the dialect class.
/// {1}: Initialization code emitted in the ctor body before initialize().
/// {2}: The dialect parent class.
/// {3}: Extra members to initialize.
static const char *const dialectConstructorStr = R"(
{0}::{0}(::mlir::MLIRContext *context)
    : ::mlir::{2}(getDialectNamespace(), context, ::mlir::TypeID::get<{0}>())
    {3}
     {{
  {1}
  initialize();
}
)";

/// The code block to generate a default destructor definition.
///
/// {0}: The name of the dialect class.
static const char *const dialectDestructorStr = R"(
{0}::~{0}() = default;

)";

void mlir::tblgen::populateDiscardableAttributes(
    Dialect &dialect, const llvm::DagInit *discardableAttrDag,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>>
        &discardableAttributes) {
  for (int i : llvm::seq<int>(0, discardableAttrDag->getNumArgs())) {
    const llvm::Init *arg = discardableAttrDag->getArg(i);

    llvm::StringRef givenName = discardableAttrDag->getArgNameStr(i);
    if (givenName.empty())
      llvm::PrintFatalError(dialect.getDef()->getLoc(),
                            "discardable attributes must be named");
    discardableAttributes.push_back(
        {givenName.str(), arg->getAsUnquotedString()});
  }
}

std::optional<Dialect>
mlir::tblgen::findDialectToGenerate(llvm::ArrayRef<Dialect> dialects,
                                    llvm::StringRef selectedDialect) {
  if (dialects.empty()) {
    llvm::errs() << "no dialect was found\n";
    return std::nullopt;
  }

  // If there is exactly one dialect and none was explicitly selected, use it.
  if (dialects.size() == 1 && selectedDialect.empty())
    return dialects.front();

  if (selectedDialect.empty()) {
    llvm::errs() << "when more than 1 dialect is present, one must be selected "
                    "via '-dialect'\n";
    return std::nullopt;
  }

  const auto *dialectIt = llvm::find_if(dialects, [&](const Dialect &dialect) {
    return dialect.getName() == selectedDialect;
  });
  if (dialectIt == dialects.end()) {
    llvm::errs() << "selected dialect with '-dialect' does not exist\n";
    return std::nullopt;
  }
  return *dialectIt;
}

void mlir::tblgen::emitDialectDecl(Dialect &dialect, llvm::raw_ostream &os) {
  // Emit all nested namespaces.
  {
    DialectNamespaceEmitter nsEmitter(os, dialect);

    // Emit the start of the decl.
    std::string cppName = dialect.getCppClassName();
    llvm::StringRef superClassName =
        dialect.isExtensible() ? "ExtensibleDialect" : "Dialect";

    tblgen::emitSummaryAndDescComments(os, dialect.getSummary(),
                                       dialect.getDescription(),
                                       /*terminateComment=*/false);
    os << llvm::formatv(dialectDeclBeginStr, cppName, dialect.getName(),
                        superClassName);

    // If the dialect requested the default attribute printer and parser, emit
    // the declarations for the hooks.
    if (dialect.useDefaultAttributePrinterParser())
      os << attrParserDecl;
    // If the dialect requested the default type printer and parser, emit the
    // declarations for the hooks.
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
    llvm::SmallVector<std::pair<std::string, std::string>>
        discardableAttributes;
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

    if (std::optional<llvm::StringRef> extraDecl =
            dialect.getExtraClassDeclaration())
      os << *extraDecl;

    // End the dialect decl.
    os << "};\n";
  }
  if (!dialect.getCppNamespace().empty())
    os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << dialect.getCppClassName() << ")\n";
}

bool mlir::tblgen::emitDialectDecls(const RecordKeeper &records,
                                    llvm::StringRef selectedDialect,
                                    llvm::raw_ostream &os) {
  emitSourceFileHeader("Dialect Declarations", os, records);

  auto dialectDefs = records.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  llvm::SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect =
      findDialectToGenerate(dialects, selectedDialect);
  if (!dialect)
    return true;
  emitDialectDecl(*dialect, os);
  return false;
}

void mlir::tblgen::emitDialectDef(Dialect &dialect, const RecordKeeper &records,
                                  llvm::raw_ostream &os) {
  std::string cppClassName = dialect.getCppClassName();

  // Emit the TypeID explicit specializations to have a single symbol def.
  if (!dialect.getCppNamespace().empty())
    os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << cppClassName << ")\n";

  // Emit all nested namespaces.
  DialectNamespaceEmitter nsEmitter(os, dialect);

  // Build the list of dependent dialects.
  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    llvm::interleave(
        dialect.getDependentDialects(), dialectsOs,
        [&](llvm::StringRef dependentDialect) {
          dialectsOs << llvm::formatv(dialectRegistrationTemplate,
                                      dependentDialect);
        },
        "\n  ");
  }

  // Emit the constructor and destructor.
  llvm::StringRef superClassName =
      dialect.isExtensible() ? "ExtensibleDialect" : "Dialect";

  const llvm::DagInit *discardableAttrDag = dialect.getDiscardableAttributes();
  llvm::SmallVector<std::pair<std::string, std::string>> discardableAttributes;
  populateDiscardableAttributes(dialect, discardableAttrDag,
                                discardableAttributes);
  std::string discardableAttributesInit;
  for (const auto &attrPair : discardableAttributes) {
    std::string camelName = llvm::convertToCamelFromSnakeCase(
        attrPair.first, /*capitalizeFirst=*/false);
    llvm::raw_string_ostream initOs(discardableAttributesInit);
    initOs << ", " << camelName << "AttrName(context)";
  }

  os << llvm::formatv(dialectConstructorStr, cppClassName,
                      dependentDialectRegistrations, superClassName,
                      discardableAttributesInit);
  if (!dialect.hasNonDefaultDestructor())
    os << llvm::formatv(dialectDestructorStr, cppClassName);
}

bool mlir::tblgen::emitDialectDefs(const RecordKeeper &records,
                                   llvm::StringRef selectedDialect,
                                   llvm::raw_ostream &os) {
  emitSourceFileHeader("Dialect Definitions", os, records);

  auto dialectDefs = records.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  llvm::SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect =
      findDialectToGenerate(dialects, selectedDialect);
  if (!dialect)
    return true;
  emitDialectDef(*dialect, records, os);
  return false;
}
