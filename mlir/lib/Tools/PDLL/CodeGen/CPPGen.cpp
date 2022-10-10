//===- CPPGen.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files contains a PDLL generator that outputs C++ code that defines PDLL
// patterns as individual C++ PDLPatternModules for direct use in native code,
// and also defines any native constraints whose bodies were defined in PDLL.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// CodeGen
//===----------------------------------------------------------------------===//

namespace {
class CodeGen {
public:
  CodeGen(raw_ostream &os) : os(os) {}

  /// Generate C++ code for the given PDL pattern module.
  void generate(const ast::Module &astModule, ModuleOp module);

private:
  void generate(pdl::PatternOp pattern, StringRef patternName,
                StringSet<> &nativeFunctions);

  /// Generate C++ code for all user defined constraints and rewrites with
  /// native code.
  void generateConstraintAndRewrites(const ast::Module &astModule,
                                     ModuleOp module,
                                     StringSet<> &nativeFunctions);
  void generate(const ast::UserConstraintDecl *decl,
                StringSet<> &nativeFunctions);
  void generate(const ast::UserRewriteDecl *decl, StringSet<> &nativeFunctions);
  void generateConstraintOrRewrite(const ast::CallableDecl *decl,
                                   bool isConstraint,
                                   StringSet<> &nativeFunctions);

  /// Return the native name for the type of the given type.
  StringRef getNativeTypeName(ast::Type type);

  /// Return the native name for the type of the given variable decl.
  StringRef getNativeTypeName(ast::VariableDecl *decl);

  /// The stream to output to.
  raw_ostream &os;
};
} // namespace

void CodeGen::generate(const ast::Module &astModule, ModuleOp module) {
  SetVector<std::string, SmallVector<std::string>, StringSet<>> patternNames;
  StringSet<> nativeFunctions;

  // Generate code for any native functions within the module.
  generateConstraintAndRewrites(astModule, module, nativeFunctions);

  os << "namespace {\n";
  std::string basePatternName = "GeneratedPDLLPattern";
  int patternIndex = 0;
  for (pdl::PatternOp pattern : module.getOps<pdl::PatternOp>()) {
    // If the pattern has a name, use that. Otherwise, generate a unique name.
    if (Optional<StringRef> patternName = pattern.getSymName()) {
      patternNames.insert(patternName->str());
    } else {
      std::string name;
      do {
        name = (basePatternName + Twine(patternIndex++)).str();
      } while (!patternNames.insert(name));
    }

    generate(pattern, patternNames.back(), nativeFunctions);
  }
  os << "} // end namespace\n\n";

  // Emit function to add the generated matchers to the pattern list.
  os << "static void LLVM_ATTRIBUTE_UNUSED populateGeneratedPDLLPatterns("
        "::mlir::RewritePatternSet &patterns) {\n";
  for (const auto &name : patternNames)
    os << "  patterns.add<" << name << ">(patterns.getContext());\n";
  os << "}\n";
}

void CodeGen::generate(pdl::PatternOp pattern, StringRef patternName,
                       StringSet<> &nativeFunctions) {
  const char *patternClassStartStr = R"(
struct {0} : ::mlir::PDLPatternModule {{
  {0}(::mlir::MLIRContext *context)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
)";
  os << llvm::formatv(patternClassStartStr, patternName);

  os << "R\"mlir(";
  pattern->print(os, OpPrintingFlags().enableDebugInfo());
  os << "\n    )mlir\", context)) {\n";

  // Register any native functions used within the pattern.
  StringSet<> registeredNativeFunctions;
  auto checkRegisterNativeFn = [&](StringRef fnName, StringRef fnType) {
    if (!nativeFunctions.count(fnName) ||
        !registeredNativeFunctions.insert(fnName).second)
      return;
    os << "    register" << fnType << "Function(\"" << fnName << "\", "
       << fnName << "PDLFn);\n";
  };
  pattern.walk([&](Operation *op) {
    if (auto constraintOp = dyn_cast<pdl::ApplyNativeConstraintOp>(op))
      checkRegisterNativeFn(constraintOp.getName(), "Constraint");
    else if (auto rewriteOp = dyn_cast<pdl::ApplyNativeRewriteOp>(op))
      checkRegisterNativeFn(rewriteOp.getName(), "Rewrite");
  });
  os << "  }\n};\n\n";
}

void CodeGen::generateConstraintAndRewrites(const ast::Module &astModule,
                                            ModuleOp module,
                                            StringSet<> &nativeFunctions) {
  // First check to see which constraints and rewrites are actually referenced
  // in the module.
  StringSet<> usedFns;
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<pdl::ApplyNativeConstraintOp, pdl::ApplyNativeRewriteOp>(
            [&](auto op) { usedFns.insert(op.getName()); });
  });

  for (const ast::Decl *decl : astModule.getChildren()) {
    TypeSwitch<const ast::Decl *>(decl)
        .Case<ast::UserConstraintDecl, ast::UserRewriteDecl>(
            [&](const auto *decl) {
              // We only generate code for inline native decls that have been
              // referenced.
              if (decl->getCodeBlock() &&
                  usedFns.contains(decl->getName().getName()))
                this->generate(decl, nativeFunctions);
            });
  }
}

void CodeGen::generate(const ast::UserConstraintDecl *decl,
                       StringSet<> &nativeFunctions) {
  return generateConstraintOrRewrite(cast<ast::CallableDecl>(decl),
                                     /*isConstraint=*/true, nativeFunctions);
}

void CodeGen::generate(const ast::UserRewriteDecl *decl,
                       StringSet<> &nativeFunctions) {
  return generateConstraintOrRewrite(cast<ast::CallableDecl>(decl),
                                     /*isConstraint=*/false, nativeFunctions);
}

StringRef CodeGen::getNativeTypeName(ast::Type type) {
  return llvm::TypeSwitch<ast::Type, StringRef>(type)
      .Case([&](ast::AttributeType) { return "::mlir::Attribute"; })
      .Case([&](ast::OperationType opType) -> StringRef {
        // Use the derived Op class when available.
        if (const auto *odsOp = opType.getODSOperation())
          return odsOp->getNativeClassName();
        return "::mlir::Operation *";
      })
      .Case([&](ast::TypeType) { return "::mlir::Type"; })
      .Case([&](ast::ValueType) { return "::mlir::Value"; })
      .Case([&](ast::TypeRangeType) { return "::mlir::TypeRange"; })
      .Case([&](ast::ValueRangeType) { return "::mlir::ValueRange"; });
}

StringRef CodeGen::getNativeTypeName(ast::VariableDecl *decl) {
  // Try to extract a type name from the variable's constraints.
  for (ast::ConstraintRef &cst : decl->getConstraints()) {
    if (auto *userCst = dyn_cast<ast::UserConstraintDecl>(cst.constraint)) {
      if (Optional<StringRef> name = userCst->getNativeInputType(0))
        return *name;
      return getNativeTypeName(userCst->getInputs()[0]);
    }
  }

  // Otherwise, use the type of the variable.
  return getNativeTypeName(decl->getType());
}

void CodeGen::generateConstraintOrRewrite(const ast::CallableDecl *decl,
                                          bool isConstraint,
                                          StringSet<> &nativeFunctions) {
  StringRef name = decl->getName()->getName();
  nativeFunctions.insert(name);

  os << "static ";

  // TODO: Work out a proper modeling for "optionality".

  // Emit the result type.
  // If this is a constraint, we always return a LogicalResult.
  // TODO: This will need to change if we allow Constraints to return values as
  // well.
  if (isConstraint) {
    os << "::mlir::LogicalResult";
  } else {
    // Otherwise, generate a type based on the results of the callable.
    // If the callable has explicit results, use those to build the result.
    // Otherwise, use the type of the callable.
    ArrayRef<ast::VariableDecl *> results = decl->getResults();
    if (results.empty()) {
      os << "void";
    } else if (results.size() == 1) {
      os << getNativeTypeName(results[0]);
    } else {
      os << "std::tuple<";
      llvm::interleaveComma(results, os, [&](ast::VariableDecl *result) {
        os << getNativeTypeName(result);
      });
      os << ">";
    }
  }

  os << " " << name << "PDLFn(::mlir::PatternRewriter &rewriter";
  if (!decl->getInputs().empty()) {
    os << ", ";
    llvm::interleaveComma(decl->getInputs(), os, [&](ast::VariableDecl *input) {
      os << getNativeTypeName(input) << " " << input->getName().getName();
    });
  }
  os << ") {\n";
  os << "  " << decl->getCodeBlock()->trim() << "\n}\n\n";
}

//===----------------------------------------------------------------------===//
// CPPGen
//===----------------------------------------------------------------------===//

void mlir::pdll::codegenPDLLToCPP(const ast::Module &astModule, ModuleOp module,
                                  raw_ostream &os) {
  CodeGen codegen(os);
  codegen.generate(astModule, module);
}
