

#include "CodeGenAction.h"
#include "CodeGen.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclGroup.h>

#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <ostream>

using namespace clang;

namespace mlir {
namespace obs {

void CodeGenConsumer::HandleTranslationUnit(ASTContext &context) {
  llvm::outs() << "Enter HandleTranslationUnit\n";
  MLIRContext codegenContext;
  MLIRGenImpl mlirGen(codegenContext);
  mlirGen.TraverseDecl(context.getTranslationUnitDecl());
}

} // namespace obs
} // namespace mlir