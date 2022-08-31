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

static llvm::cl::OptionCategory passGenCat("Options for -gen-pass-decls");
static llvm::cl::opt<std::string>
    groupName("name", llvm::cl::desc("The name of this group of passes"),
              llvm::cl::cat(passGenCat));

/// Extract the list of passes from the TableGen records.
static std::vector<Pass> getPasses(const llvm::RecordKeeper &recordKeeper) {
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

  os << llvm::formatv("struct {0}Options {{\n", passName);

  for (const PassOption &opt : options) {
    std::string type = opt.getType().str();

    if (opt.isListOption())
      type = "::llvm::ArrayRef<" + type + ">";

    os.indent(2) << llvm::formatv("{0} {1}", type, opt.getCppVariableName());

    if (Optional<StringRef> defaultVal = opt.getDefaultValue())
      os << " = " << defaultVal;

    os << ";\n";
  }

  os << "};\n";
}

/// Emit the code to be included in the public header of the pass.
static void emitPassDecls(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  std::string enableVarName = "GEN_PASS_DECL_" + passName.upper();

  os << "#ifdef " << enableVarName << "\n";
  emitPassOptionsStruct(pass, os);

  if (StringRef constructor = pass.getConstructor(); constructor.empty()) {
    // Default constructor declaration.
    os << "std::unique_ptr<::mlir::Pass> create" << passName << "();\n";

    // Declaration of the constructor with options.
    if (ArrayRef<PassOption> options = pass.getOptions(); !options.empty())
      os << llvm::formatv("std::unique_ptr<::mlir::Pass> create{0}(const "
                          "{0}Options &options);\n",
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
      constructorCall =
          llvm::formatv("create{0}()", pass.getDef()->getName()).str();

    os << llvm::formatv(passRegistrationCode, pass.getDef()->getName(),
                        constructorCall);
  }

  os << llvm::formatv(passGroupRegistrationCode, groupName);

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
/// {3}: The dependent dialects registration.
const char *const baseClassBegin = R"(
template <typename DerivedT>
class {0}Base : public {1} {
public:
  using Base = {0}Base;

  {0}Base() : {1}(::mlir::TypeID::get<DerivedT>()) {{}
  {0}Base(const {0}Base &other) : {1}(other) {{}

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
const char *const dialectRegistrationTemplate = R"(
  registry.insert<{0}>();
)";

const char *const friendDefaultConstructorDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}();
} // namespace impl
)";

const char *const friendDefaultConstructorWithOptionsDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}(const {0}Options &options);
} // namespace impl
)";

const char *const friendDefaultConstructorDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}() {{
    return std::make_unique<DerivedT>();
  }
)";

const char *const friendDefaultConstructorWithOptionsDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}(const {0}Options &options) {{
    return std::make_unique<DerivedT>(options);
  }
)";

const char *const defaultConstructorDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}() {{
  return impl::create{0}();
}
)";

const char *const defaultConstructorWithOptionsDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}(const {0}Options &options) {{
  return impl::create{0}(options);
}
)";

/// Emit the declarations for each of the pass options.
static void emitPassOptionDecls(const Pass &pass, raw_ostream &os) {
  for (const PassOption &opt : pass.getOptions()) {
    os.indent(2) << "::mlir::Pass::"
                 << (opt.isListOption() ? "ListOption" : "Option");

    os << llvm::formatv(R"(<{0}> {1}{{*this, "{2}", ::llvm::cl::desc("{3}"))",
                        opt.getType(), opt.getCppVariableName(),
                        opt.getArgument(), opt.getDescription());
    if (Optional<StringRef> defaultVal = opt.getDefaultValue())
      os << ", ::llvm::cl::init(" << defaultVal << ")";
    if (Optional<StringRef> additionalFlags = opt.getAdditionalFlags())
      os << ", " << *additionalFlags;
    os << "};\n";
  }
}

/// Emit the declarations for each of the pass statistics.
static void emitPassStatisticDecls(const Pass &pass, raw_ostream &os) {
  for (const PassStatistic &stat : pass.getStatistics()) {
    os << llvm::formatv(
        "  ::mlir::Pass::Statistic {0}{{this, \"{1}\", \"{2}\"};\n",
        stat.getCppVariableName(), stat.getName(), stat.getDescription());
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
    os << llvm::formatv(friendDefaultConstructorDeclTemplate, passName);

    if (emitDefaultConstructorWithOptions)
      os << llvm::formatv(friendDefaultConstructorWithOptionsDeclTemplate,
                          passName);
  }

  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    for (StringRef dependentDialect : pass.getDependentDialects())
      dialectsOs << llvm::formatv(dialectRegistrationTemplate,
                                  dependentDialect);
  }

  os << "namespace impl {\n";
  os << llvm::formatv(baseClassBegin, passName, pass.getBaseClass(),
                      pass.getArgument(), pass.getSummary(),
                      dependentDialectRegistrations);

  if (ArrayRef<PassOption> options = pass.getOptions(); !options.empty()) {
    os.indent(2) << llvm::formatv(
        "{0}Base(const {0}Options &options) : {0}Base() {{\n", passName);

    for (const PassOption &opt : pass.getOptions())
      os.indent(4) << llvm::formatv("{0} = options.{0};\n",
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
    os << llvm::formatv(friendDefaultConstructorDefTemplate, passName);

    if (!pass.getOptions().empty())
      os << llvm::formatv(friendDefaultConstructorWithOptionsDefTemplate,
                          passName);
  }

  os << "};\n";
  os << "} // namespace impl\n";

  if (emitDefaultConstructors) {
    os << llvm::formatv(defaultConstructorDefTemplate, passName);

    if (emitDefaultConstructorWithOptions)
      os << llvm::formatv(defaultConstructorWithOptionsDefTemplate, passName);
  }

  os << "#undef " << enableVarName << "\n";
  os << "#endif // " << enableVarName << "\n";
}

static void emitPass(const Pass &pass, raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  os << llvm::formatv(passHeader, passName);

  emitPassDecls(pass, os);
  emitPassDefs(pass, os);
}

static void emitPasses(const llvm::RecordKeeper &recordKeeper,
                       raw_ostream &os) {
  std::vector<Pass> passes = getPasses(recordKeeper);
  os << "/* Autogenerated by mlir-tblgen; don't manually edit */\n";

  for (const Pass &pass : passes)
    emitPass(pass, os);

  emitRegistrations(passes, os);
}

static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   emitPasses(records, os);
                   return false;
                 });
