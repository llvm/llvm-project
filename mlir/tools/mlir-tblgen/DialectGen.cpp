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

#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/Optional.h"
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

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

/// Utility iterator used for filtering records for a specific dialect.
namespace {
using DialectFilterIterator =
    llvm::filter_iterator<ArrayRef<llvm::Record *>::iterator,
                          std::function<bool(const llvm::Record *)>>;
} // namespace

/// Given a set of records for a T, filter the ones that correspond to
/// the given dialect.
template <typename T>
static iterator_range<DialectFilterIterator>
filterForDialect(ArrayRef<llvm::Record *> records, Dialect &dialect) {
  auto filterFn = [&](const llvm::Record *record) {
    return T(record).getDialect() == dialect;
  };
  return {DialectFilterIterator(records.begin(), records.end(), filterFn),
          DialectFilterIterator(records.end(), records.end(), filterFn)};
}

static Optional<Dialect>
findSelectedDialect(ArrayRef<const llvm::Record *> dialectDefs) {
  // Select the dialect to gen for.
  if (dialectDefs.size() == 1 && selectedDialect.getNumOccurrences() == 0) {
    return Dialect(dialectDefs.front());
  }

  if (selectedDialect.getNumOccurrences() == 0) {
    llvm::errs() << "when more than 1 dialect is present, one must be selected "
                    "via '-dialect'\n";
    return llvm::None;
  }

  const auto *dialectIt =
      llvm::find_if(dialectDefs, [](const llvm::Record *def) {
        return Dialect(def).getName() == selectedDialect;
      });
  if (dialectIt == dialectDefs.end()) {
    llvm::errs() << "selected dialect with '-dialect' does not exist\n";
    return llvm::None;
  }
  return Dialect(*dialectIt);
}

//===----------------------------------------------------------------------===//
// GEN: Dialect declarations
//===----------------------------------------------------------------------===//

/// The code block for the start of a dialect class declaration.
///
/// {0}: The name of the dialect class.
/// {1}: The dialect namespace.
/// {2}: initialization code that is emitted in the ctor body before calling
/// initialize()
static const char *const dialectDeclBeginStr = R"(
class {0} : public ::mlir::Dialect {
  explicit {0}(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<{0}>()) {{
    {2}
    initialize();
  }

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
const char *const dialectRegistrationTemplate = R"(
    getContext()->getOrLoadDialect<{0}>();
)";

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
    ::mlir::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the region argument attribute verifier hook.
static const char *const regionArgAttrVerifierDecl = R"(
    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region argument.
    ::mlir::LogicalResult verifyRegionArgAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned argIndex,
        ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the region result attribute verifier hook.
static const char *const regionResultAttrVerifierDecl = R"(
    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region result.
    ::mlir::LogicalResult verifyRegionResultAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned resultIndex,
        ::mlir::NamedAttribute attribute) override;
)";

/// The code block for the op interface fallback hook.
static const char *const operationInterfaceFallbackDecl = R"(
    /// Provides a hook for op interface.
    void *getRegisteredInterfaceForOp(mlir::TypeID interfaceID,
                                      mlir::OperationName opName) override;
)";

/// Generate the declaration for the given dialect class.
static void emitDialectDecl(Dialect &dialect,
                            iterator_range<DialectFilterIterator> dialectAttrs,
                            iterator_range<DialectFilterIterator> dialectTypes,
                            raw_ostream &os) {
  /// Build the list of dependent dialects
  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    for (StringRef dependentDialect : dialect.getDependentDialects())
      dialectsOs << llvm::formatv(dialectRegistrationTemplate,
                                  dependentDialect);
  }

  // Emit all nested namespaces.
  {
    NamespaceEmitter nsEmitter(os, dialect);

    // Emit the start of the decl.
    std::string cppName = dialect.getCppClassName();
    os << llvm::formatv(dialectDeclBeginStr, cppName, dialect.getName(),
                        dependentDialectRegistrations);

    // Check for any attributes/types registered to this dialect.  If there are,
    // add the hooks for parsing/printing.
    if (!dialectAttrs.empty() || dialect.useDefaultAttributePrinterParser())
      os << attrParserDecl;
    if (!dialectTypes.empty() || dialect.useDefaultTypePrinterParser())
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
    if (llvm::Optional<StringRef> extraDecl =
            dialect.getExtraClassDeclaration())
      os << *extraDecl;

    // End the dialect decl.
    os << "};\n";
  }
  if (!dialect.getCppNamespace().empty())
    os << "DECLARE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << dialect.getCppClassName() << ")\n";
}

static bool emitDialectDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("Dialect Declarations", os);

  auto dialectDefs = recordKeeper.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  Optional<Dialect> dialect = findSelectedDialect(dialectDefs);
  if (!dialect)
    return true;
  auto attrDefs = recordKeeper.getAllDerivedDefinitions("DialectAttr");
  auto typeDefs = recordKeeper.getAllDerivedDefinitions("DialectType");
  emitDialectDecl(*dialect, filterForDialect<Attribute>(attrDefs, *dialect),
                  filterForDialect<Type>(typeDefs, *dialect), os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Dialect definitions
//===----------------------------------------------------------------------===//

/// The code block to generate a default desturctor definition.
///
/// {0}: The name of the dialect class.
static const char *const dialectDestructorStr = R"(
{0}::~{0}() = default;

)";

static void emitDialectDef(Dialect &dialect, raw_ostream &os) {
  // Emit the TypeID explicit specializations to have a single symbol def.
  if (!dialect.getCppNamespace().empty())
    os << "DEFINE_EXPLICIT_TYPE_ID(" << dialect.getCppNamespace()
       << "::" << dialect.getCppClassName() << ")\n";

  // Emit all nested namespaces.
  NamespaceEmitter nsEmitter(os, dialect);

  if (!dialect.hasNonDefaultDestructor())
    os << llvm::formatv(dialectDestructorStr, dialect.getCppClassName());
}

static bool emitDialectDefs(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os) {
  emitSourceFileHeader("Dialect Definitions", os);

  auto dialectDefs = recordKeeper.getAllDerivedDefinitions("Dialect");
  if (dialectDefs.empty())
    return false;

  Optional<Dialect> dialect = findSelectedDialect(dialectDefs);
  if (!dialect)
    return true;
  emitDialectDef(*dialect, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Dialect registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitDialectDecls(records, os);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const llvm::RecordKeeper &records, raw_ostream &os) {
                     return emitDialectDefs(records, os);
                   });
