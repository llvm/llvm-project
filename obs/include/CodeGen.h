
#ifndef _OBS_CODEGEN_H
#define _OBS_CODEGEN_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

namespace mlir {
namespace obs {

class MLIRGenImpl : public clang::RecursiveASTVisitor<MLIRGenImpl> {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  bool VisitFunctionDecl(clang::FunctionDecl *funcDecl);

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
};

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          clang::TranslationUnitDecl &decl);

} // namespace obs
} // namespace mlir

#endif