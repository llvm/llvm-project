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

//===----------------------------------------------------------------------===//
// GEN: Pass base class generation
//===----------------------------------------------------------------------===//

/// The code snippet used to generate the start of a pass base class.
///
/// {0}: The def name of the pass record.
/// {1}: The base class for the pass.
/// {2): The command line argument for the pass.
/// {3}: The dependent dialects registration.
const char *const passDeclBegin = R"(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//

template <typename DerivedT>
class {0}Base : public {1} {
public:
  using Base = {0}Base;

  {0}Base() : {1}(::mlir::TypeID::get<DerivedT>()) {{}
  {0}Base(const {0}Base &) : {1}(::mlir::TypeID::get<DerivedT>()) {{}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("{2}");
  }
  ::llvm::StringRef getArgument() const override { return "{2}"; }

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
    {3}
  }

protected:
)";

/// Registration for a single dependent dialect, to be inserted for each
/// dependent dialect in the `getDependentDialects` above.
const char *const dialectRegistrationTemplate = R"(
  registry.insert<{0}>();
)";

/// Emit the declarations for each of the pass options.
static void emitPassOptionDecls(const Pass &pass, raw_ostream &os) {
  for (const PassOption &opt : pass.getOptions()) {
    os.indent(2) << "::mlir::Pass::"
                 << (opt.isListOption() ? "ListOption" : "Option");

    os << llvm::formatv("<{0}> {1}{{*this, \"{2}\", ::llvm::cl::desc(\"{3}\")",
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

static void emitPassDecl(const Pass &pass, raw_ostream &os) {
  StringRef defName = pass.getDef()->getName();
  std::string dependentDialectRegistrations;
  {
    llvm::raw_string_ostream dialectsOs(dependentDialectRegistrations);
    for (StringRef dependentDialect : pass.getDependentDialects())
      dialectsOs << llvm::formatv(dialectRegistrationTemplate,
                                  dependentDialect);
  }
  os << llvm::formatv(passDeclBegin, defName, pass.getBaseClass(),
                      pass.getArgument(), dependentDialectRegistrations);
  emitPassOptionDecls(pass, os);
  emitPassStatisticDecls(pass, os);
  os << "};\n";
}

/// Emit the code for registering each of the given passes with the global
/// PassRegistry.
static void emitPassDecls(ArrayRef<Pass> passes, raw_ostream &os) {
  os << "#ifdef GEN_PASS_CLASSES\n";
  for (const Pass &pass : passes)
    emitPassDecl(pass, os);
  os << "#undef GEN_PASS_CLASSES\n";
  os << "#endif // GEN_PASS_CLASSES\n";
}

//===----------------------------------------------------------------------===//
// GEN: Pass registration generation
//===----------------------------------------------------------------------===//

/// The code snippet used to generate the start of a pass base class.
///
/// {0}: The def name of the pass record.
/// {1}: The argument of the pass.
/// {2): The summary of the pass.
/// {3}: The code for constructing the pass.
const char *const passRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//

inline void register{0}Pass() {{
  ::mlir::registerPass("{1}", "{2}", []() -> std::unique_ptr<::mlir::Pass> {{
    return {3};
  });
}
)";

/// {0}: The name of the pass group.
const char *const passGroupRegistrationCode = R"(
//===----------------------------------------------------------------------===//
// {0} Registration
//===----------------------------------------------------------------------===//

inline void register{0}Passes() {{
)";

/// Emit the code for registering each of the given passes with the global
/// PassRegistry.
static void emitRegistration(ArrayRef<Pass> passes, raw_ostream &os) {
  os << "#ifdef GEN_PASS_REGISTRATION\n";
  for (const Pass &pass : passes) {
    os << llvm::formatv(passRegistrationCode, pass.getDef()->getName(),
                        pass.getArgument(), pass.getSummary(),
                        pass.getConstructor());
  }

  os << llvm::formatv(passGroupRegistrationCode, groupName);
  for (const Pass &pass : passes)
    os << "  register" << pass.getDef()->getName() << "Pass();\n";
  os << "}\n";
  os << "#undef GEN_PASS_REGISTRATION\n";
  os << "#endif // GEN_PASS_REGISTRATION\n";
}

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

static void emitDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit */\n";
  std::vector<Pass> passes;
  for (const auto *def : recordKeeper.getAllDerivedDefinitions("PassBase"))
    passes.push_back(Pass(def));

  emitPassDecls(passes, os);
  emitRegistration(passes, os);
}

static mlir::GenRegistration
    genRegister("gen-pass-decls", "Generate operation documentation",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  emitDecls(records, os);
                  return false;
                });
