

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include <iostream>
#include <memory>

namespace mlir {
namespace obs {

class CodeGenConsumer : public clang::ASTConsumer {
public:
  CodeGenConsumer() {}
  void HandleTranslationUnit(clang::ASTContext &context) override;
};

class CodeGenFrontendAction : public clang::ASTFrontendAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, clang::StringRef) override {
    return std::make_unique<CodeGenConsumer>();
  }
};

class CodeGenFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
public:
  CodeGenFrontendActionFactory() {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<CodeGenFrontendAction>();
  }
};

} // namespace obs
} // namespace mlir