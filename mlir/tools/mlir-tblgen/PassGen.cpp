//===- Pass.cpp - MLIR pass registration generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PassGen uses the description of passes to generate base classes for passes
// and command line registration.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory passGenCat("Options for -gen-pass-decls");
static llvm::cl::opt<std::string>
    groupName("name", llvm::cl::desc("The name of this group of passes"),
              llvm::cl::cat(passGenCat));

/// Extract the list of passes from the TableGen records.
static std::vector<Pass> getPasses(const RecordKeeper &recordKeeper) {
  std::vector<Pass> passes;

  for (const auto *def : recordKeeper.getAllDerivedDefinitions("PassBase"))
    passes.emplace_back(def);

  return passes;
}

const char *const passHeader = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//
)";

//===----------------------------------------------------------------------===//
// GEN: Pass registration generation
//===----------------------------------------------------------------------===//

/// The code snippet used to generate a pass registration.
///
/// {0}: The def name of the pass record.
/// {1}: The pass constructor call.
const char *const passRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//

inline void register{0}() {{
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {{
    return {1};
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void register{0}Pass() {{
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {{
    return {1};
  });
}
)";

/// The code snippet used to generate a function to register all passes in a
/// group.
///
/// {0}: The name of the pass group.
const char *const passGroupRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//

inline void register{0}Passes() {{
)";

/// Emits the definition of the struct to be used to control the pass options.
static void emitPassOptionsStruct(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  ArrayRef<PassOption> options = pass.getOptions();

  // Emit the struct only if the pass has at least one option.
  if (options.empty())
    return;

  os << formatv("struct {0}Options {{\n", passName);

  for (const PassOption &opt : options) {
    std::string type = opt.getType().str();

    if (opt.isListOption())
      type = "::llvm::SmallVector<" + type + ">";

    os.indent(2) << formatv("{0} {1}", type, opt.getCppVariableName());

    if (std::optional<StringRef> defaultVal = opt.getDefaultValue())
      os << " = " << defaultVal;

    os << ";\n";
  }

  os << "};\n";
}

static std::string getPassDeclVarName(const Pass &pass) {
  return "GEN_PASS_DECL_" + pass.getDef()->getName().upper();
}

/// Emit the code to be included in the public header of the pass.
static void emitPassDecls(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  std::string enableVarName = getPassDeclVarName(pass);

  os << "#ifdef " << enableVarName << "\n";
  emitPassOptionsStruct(pass, os);

  if (StringRef constructor = pass.getConstructor(); constructor.empty()) {
    // Default constructor declaration.
    os << "std::unique_ptr<::mlir::Pass> create" << passName << "();\n";

    // Declaration of the constructor with options.
    if (ArrayRef<PassOption> options = pass.getOptions(); !options.empty())
      os << formatv("std::unique_ptr<::mlir::Pass> create{0}("
                    "{0}Options options);\n",
                    passName);
  }

  os << "#undef " << enableVarName << "\n";
  os << "#endif // " << enableVarName << "\n";
}

/// Emit the code for registering each of the given passes with the global
/// PassRegistry.
static void emitRegistrations(llvm::ArrayRef<Pass> passes, raw_ostream &os) {
  os << "#ifdef GEN_PASS_REGISTRATION\n";

  for (const Pass &pass : passes) {
    std::string constructorCall;
    if (StringRef constructor = pass.getConstructor(); !constructor.empty())
      constructorCall = constructor.str();
    else
      constructorCall = formatv("create{0}()", pass.getDef()->getName()).str();

    os << formatv(passRegistrationCode, pass.getDef()->getName(),
                  constructorCall);
  }

  os << formatv(passGroupRegistrationCode, groupName);

  for (const Pass &pass : passes)
    os << "  register" << pass.getDef()->getName() << "();\n";

  os << "}\n";
  os << "#undef GEN_PASS_REGISTRATION\n";
  os << "#endif // GEN_PASS_REGISTRATION\n";
}

//===----------------------------------------------------------------------===//
// GEN: Pass base class generation
//===----------------------------------------------------------------------===//

/// The code snippet used to generate the start of a pass base class.
///
/// {0}: The def name of the pass record.
/// {1}: The base class for the pass.
/// {2): The command line argument for the pass.
/// {3}: The summary for the pass.
/// {4}: The dependent dialects registration.
const char *const baseClassBegin = R"(
template <typename DerivedT>
class {0}Base : public {1} {
public:
  using Base = {0}Base;

  {0}Base() : {1}(::mlir::TypeID::get<DerivedT>()) {{}
  {0}Base(const {0}Base &other) : {1}(other) {{}
  {0}Base& operator=(const {0}Base &) = delete;
  {0}Base({0}Base &&) = delete;
  {0}Base& operator=({0}Base &&) = delete;
  ~{0}Base() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("{2}");
  }
  ::llvm::StringRef getArgument() const override { return "{2}"; }

  ::llvm::StringRef getDescription() const override { return "{3}"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("{0}");
  }
  ::llvm::StringRef getName() const override { return "{0}"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {{
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {{
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    {4}
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID({0}Base<DerivedT>)

)";

/// Registration for a single dependent dialect, to be inserted for each
/// dependent dialect in the `getDependentDialects` above.
const char *const dialectRegistrationTemplate = "registry.insert<{0}>();";

const char *const friendDefaultConstructorDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}();
} // namespace impl
)";

const char *const friendDefaultConstructorWithOptionsDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}({0}Options options);
} // namespace impl
)";

const char *const friendDefaultConstructorDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}() {{
    return std::make_unique<DerivedT>();
  }
)";

const char *const friendDefaultConstructorWithOptionsDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}({0}Options options) {{
    return std::make_unique<DerivedT>(std::move(options));
  }
)";

const char *const defaultConstructorDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}() {{
  return impl::create{0}();
}
)";

const char *const defaultConstructorWithOptionsDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}({0}Options options) {{
  return impl::create{0}(std::move(options));
}
)";

/// Emit the declarations for each of the pass options.
static void emitPassOptionDecls(const Pass &pass, raw_ostream &os) {
  for (const PassOption &opt : pass.getOptions()) {
    os.indent(2) << "::mlir::Pass::"
                 << (opt.isListOption() ? "ListOption" : "Option");

    os << formatv(R"(<{0}> {1}{{*this, "{2}", ::llvm::cl::desc("{3}"))",
                  opt.getType(), opt.getCppVariableName(), opt.getArgument(),
                  opt.getDescription());
    if (std::optional<StringRef> defaultVal = opt.getDefaultValue())
      os << ", ::llvm::cl::init(" << defaultVal << ")";
    if (std::optional<StringRef> additionalFlags = opt.getAdditionalFlags())
      os << ", " << *additionalFlags;
    os << "};\n";
  }
}

/// Emit the declarations for each of the pass statistics.
static void emitPassStatisticDecls(const Pass &pass, raw_ostream &os) {
  for (const PassStatistic &stat : pass.getStatistics()) {
    os << formatv("  ::mlir::Pass::Statistic {0}{{this, \"{1}\", \"{2}\"};\n",
                  stat.getCppVariableName(), stat.getName(),
                  stat.getDescription());
  }
}

/// Emit the code to be used in the implementation of the pass.
static void emitPassDefs(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  std::string enableVarName = "GEN_PASS_DEF_" + passName.upper();
  bool emitDefaultConstructors = pass.getConstructor().empty();
  bool emitDefaultConstructorWithOptions = !pass.getOptions().empty();

  os << "#ifdef " << enableVarName << "\n";

  if (emitDefaultConstructors) {
    os << formatv(friendDefaultConstructorDeclTemplate, passName);

    if (emitDefaultConstructorWithOptions)
      os << formatv(friendDefaultConstructorWithOptionsDeclTemplate, passName);
  }

  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    llvm::interleave(
        pass.getDependentDialects(), dialectsOs,
        [&](StringRef dependentDialect) {
          dialectsOs << formatv(dialectRegistrationTemplate, dependentDialect);
        },
        "\n    ");
  }

  os << "namespace impl {\n";
  os << formatv(baseClassBegin, passName, pass.getBaseClass(),
                pass.getArgument(), pass.getSummary(),
                dependentDialectRegistrations);

  if (ArrayRef<PassOption> options = pass.getOptions(); !options.empty()) {
    os.indent(2) << formatv("{0}Base({0}Options options) : {0}Base() {{\n",
                            passName);

    for (const PassOption &opt : pass.getOptions())
      os.indent(4) << formatv("{0} = std::move(options.{0});\n",
                              opt.getCppVariableName());

    os.indent(2) << "}\n";
  }

  // Protected content
  os << "protected:\n";
  emitPassOptionDecls(pass, os);
  emitPassStatisticDecls(pass, os);

  // Private content
  os << "private:\n";

  if (emitDefaultConstructors) {
    os << formatv(friendDefaultConstructorDefTemplate, passName);

    if (!pass.getOptions().empty())
      os << formatv(friendDefaultConstructorWithOptionsDefTemplate, passName);
  }

  os << "};\n";
  os << "} // namespace impl\n";

  if (emitDefaultConstructors) {
    os << formatv(defaultConstructorDefTemplate, passName);

    if (emitDefaultConstructorWithOptions)
      os << formatv(defaultConstructorWithOptionsDefTemplate, passName);
  }

  os << "#undef " << enableVarName << "\n";
  os << "#endif // " << enableVarName << "\n";
}

static void emitPass(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  os << formatv(passHeader, passName);

  emitPassDecls(pass, os);
  emitPassDefs(pass, os);
}

// TODO: Drop old pass declarations.
// The old pass base class is being kept until all the passes have switched to
// the new decls/defs design.
const char *const oldPassDeclBegin = R"(
template <typename DerivedT>
class {0}Base : public {1} {
public:
  using Base = {0}Base;

  {0}Base() : {1}(::mlir::TypeID::get<DerivedT>()) {{}
  {0}Base(const {0}Base &other) : {1}(other) {{}
  {0}Base& operator=(const {0}Base &) = delete;
  {0}Base({0}Base &&) = delete;
  {0}Base& operator=({0}Base &&) = delete;
  ~{0}Base() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("{2}");
  }
  ::llvm::StringRef getArgument() const override { return "{2}"; }

  ::llvm::StringRef getDescription() const override { return "{3}"; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("{0}");
  }
  ::llvm::StringRef getName() const override { return "{0}"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {{
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {{
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Register the dialects that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    {4}
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID({0}Base<DerivedT>)

protected:
)";

// TODO: Drop old pass declarations.
/// Emit a backward-compatible declaration of the pass base class.
static void emitOldPassDecl(const Pass &pass, raw_ostream &os) {
  StringRef defName = pass.getDef()->getName();
  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    llvm::interleave(
        pass.getDependentDialects(), dialectsOs,
        [&](StringRef dependentDialect) {
          dialectsOs << formatv(dialectRegistrationTemplate, dependentDialect);
        },
        "\n    ");
  }
  os << formatv(oldPassDeclBegin, defName, pass.getBaseClass(),
                pass.getArgument(), pass.getSummary(),
                dependentDialectRegistrations);
  emitPassOptionDecls(pass, os);
  emitPassStatisticDecls(pass, os);
  os << "};\n";
}

static void emitPasses(const RecordKeeper &recordKeeper, raw_ostream &os) {
  std::vector<Pass> passes = getPasses(recordKeeper);
  os << "/* Autogenerated by mlir-tblgen; don't manually edit */\n";

  os << "\n";
  os << "#ifdef GEN_PASS_DECL\n";
  os << "// Generate declarations for all passes.\n";
  for (const Pass &pass : passes)
    os << "#define " << getPassDeclVarName(pass) << "\n";
  os << "#undef GEN_PASS_DECL\n";
  os << "#endif // GEN_PASS_DECL\n";

  for (const Pass &pass : passes)
    emitPass(pass, os);

  emitRegistrations(passes, os);

  // TODO: Drop old pass declarations.
  // Emit the old code until all the passes have switched to the new design.
  os << "// Deprecated. Please use the new per-pass macros.\n";
  os << "#ifdef GEN_PASS_CLASSES\n";
  for (const Pass &pass : passes)
    emitOldPassDecl(pass, os);
  os << "#undef GEN_PASS_CLASSES\n";
  os << "#endif // GEN_PASS_CLASSES\n";
}

static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitPasses(records, os);
                   return false;
                 });
