//===- PassGen.cpp - MLIR pass C++ generation utilities -------------------===//
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

#include "mlir/TableGen/Generators/PassGen.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

static const char *const passHeader = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//
)";

/// The code snippet used to generate a pass registration.
///
/// {0}: The def name of the pass record.
/// {1}: The pass constructor call.
static const char *const passRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//
#ifdef {1}

inline void register{0}() {{
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {{
    return {2};
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void register{0}Pass() {{
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {{
    return {2};
  });
}

#undef {1}
#endif // {1}
)";

/// The code snippet used to generate a function to register all passes in a
/// group.
///
/// {0}: The name of the pass group.
static const char *const passGroupRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//

inline void register{0}Passes() {{
)";

/// The code snippet used to generate the start of a pass base class.
///
/// {0}: The def name of the pass record.
/// {1}: The base class for the pass.
/// {2}: The command line argument for the pass.
/// {3}: The summary for the pass.
/// {4}: The dependent dialects registration.
static const char *const baseClassBegin = R"(
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

  ::llvm::StringRef getDescription() const override { return R"PD({3})PD"; }

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

static const char *const dialectRegistrationTemplate =
    "registry.insert<{0}>();";

static const char *const friendDefaultConstructorDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}();
} // namespace impl
)";

static const char *const friendDefaultConstructorWithOptionsDeclTemplate = R"(
namespace impl {{
  std::unique_ptr<::mlir::Pass> create{0}({0}Options options);
} // namespace impl
)";

static const char *const friendDefaultConstructorDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}() {{
    return std::make_unique<DerivedT>();
  }
)";

static const char *const friendDefaultConstructorWithOptionsDefTemplate = R"(
  friend std::unique_ptr<::mlir::Pass> create{0}({0}Options options) {{
    return std::make_unique<DerivedT>(std::move(options));
  }
)";

static const char *const defaultConstructorDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}() {{
  return impl::create{0}();
}
)";

static const char *const defaultConstructorWithOptionsDefTemplate = R"(
std::unique_ptr<::mlir::Pass> create{0}({0}Options options) {{
  return impl::create{0}(std::move(options));
}
)";

static std::string getPassDeclVarName(const Pass &pass) {
  return "GEN_PASS_DECL_" + pass.getDef()->getName().upper();
}

static std::string getPassRegistrationVarName(const Pass &pass) {
  return "GEN_PASS_REGISTRATION_" + pass.getDef()->getName().upper();
}

std::vector<Pass> mlir::tblgen::getPasses(const RecordKeeper &records) {
  std::vector<Pass> passes;
  for (const auto *def : records.getAllDerivedDefinitions("PassBase"))
    passes.emplace_back(def);
  return passes;
}

void mlir::tblgen::emitPassOptionsStruct(const Pass &pass,
                                         llvm::raw_ostream &os) {
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

void mlir::tblgen::emitPassDecls(const Pass &pass, llvm::raw_ostream &os) {
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

void mlir::tblgen::emitPassOptionDecls(const Pass &pass,
                                       llvm::raw_ostream &os) {
  for (const PassOption &opt : pass.getOptions()) {
    os.indent(2) << "::mlir::Pass::"
                 << (opt.isListOption() ? "ListOption" : "Option");

    os << formatv(R"(<{0}> {1}{{*this, "{2}", ::llvm::cl::desc(R"PO({3})PO"))",
                  opt.getType(), opt.getCppVariableName(), opt.getArgument(),
                  opt.getDescription().trim());
    if (std::optional<StringRef> defaultVal = opt.getDefaultValue())
      os << ", ::llvm::cl::init(" << defaultVal << ")";
    if (std::optional<StringRef> additionalFlags = opt.getAdditionalFlags())
      os << ", " << *additionalFlags;
    os << "};\n";
  }
}

void mlir::tblgen::emitPassStatisticDecls(const Pass &pass,
                                          llvm::raw_ostream &os) {
  for (const PassStatistic &stat : pass.getStatistics()) {
    os << formatv(
        "  ::mlir::Pass::Statistic {0}{{this, \"{1}\", R\"PS({2})PS\"};\n",
        stat.getCppVariableName(), stat.getName(),
        stat.getDescription().trim());
  }
}

void mlir::tblgen::emitPassDefs(const Pass &pass, llvm::raw_ostream &os) {
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
                pass.getArgument(), pass.getSummary().trim(),
                dependentDialectRegistrations);

  if (ArrayRef<PassOption> options = pass.getOptions(); !options.empty()) {
    os.indent(2) << formatv("{0}Base({0}Options options) : {0}Base() {{\n",
                            passName);

    for (const PassOption &opt : pass.getOptions())
      os.indent(4) << formatv("{0} = std::move(options.{0});\n",
                              opt.getCppVariableName());

    os.indent(2) << "}\n";
  }

  // Protected content.
  os << "protected:\n";
  emitPassOptionDecls(pass, os);
  emitPassStatisticDecls(pass, os);

  // Private content.
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

void mlir::tblgen::emitPass(const Pass &pass, llvm::raw_ostream &os) {
  StringRef passName = pass.getDef()->getName();
  os << formatv(passHeader, passName);

  emitPassDecls(pass, os);
  emitPassDefs(pass, os);
}

void mlir::tblgen::emitRegistrations(llvm::ArrayRef<Pass> passes,
                                     llvm::StringRef groupName,
                                     llvm::raw_ostream &os) {
  os << "#ifdef GEN_PASS_REGISTRATION\n";
  os << "// Generate registrations for all passes.\n";
  for (const Pass &pass : passes)
    os << "#define " << getPassRegistrationVarName(pass) << "\n";
  os << "#endif // GEN_PASS_REGISTRATION\n";

  for (const Pass &pass : passes) {
    std::string passName = pass.getDef()->getName().str();
    std::string passEnableVarName = getPassRegistrationVarName(pass);

    std::string constructorCall;
    if (StringRef constructor = pass.getConstructor(); !constructor.empty())
      constructorCall = constructor.str();
    else
      constructorCall = formatv("create{0}()", passName).str();
    os << formatv(passRegistrationCode, passName, passEnableVarName,
                  constructorCall);
  }

  os << "#ifdef GEN_PASS_REGISTRATION\n";
  os << formatv(passGroupRegistrationCode, groupName);

  for (const Pass &pass : passes)
    os << "  register" << pass.getDef()->getName() << "();\n";

  os << "}\n";
  os << "#undef GEN_PASS_REGISTRATION\n";
  os << "#endif // GEN_PASS_REGISTRATION\n";
}

void mlir::tblgen::emitPasses(const RecordKeeper &records,
                              llvm::StringRef groupName,
                              llvm::raw_ostream &os) {
  std::vector<Pass> passes = getPasses(records);
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

  emitRegistrations(passes, groupName, os);

  // TODO: Remove warning, kept in to make error understandable.
  // Emit the old code until all the passes have switched to the new design.
  os << "#ifdef GEN_PASS_CLASSES\n";
  os << "#error \"GEN_PASS_CLASSES is deprecated; use per-pass macros\"\n";
  os << "#undef GEN_PASS_CLASSES\n";
  os << "#endif // GEN_PASS_CLASSES\n";
}
