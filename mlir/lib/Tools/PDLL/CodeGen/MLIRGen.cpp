//===- MLIRGen.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/AST/Types.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::pdll;

//===----------------------------------------------------------------------===//
// CodeGen
//===----------------------------------------------------------------------===//

namespace {
class CodeGen {
public:
  CodeGen(MLIRContext *mlirContext, const ast::Context &context,
          const llvm::SourceMgr &sourceMgr)
      : builder(mlirContext), odsContext(context.getODSContext()),
        sourceMgr(sourceMgr) {
    // Make sure that the PDL dialect is loaded.
    mlirContext->loadDialect<pdl::PDLDialect>();
  }

  OwningOpRef<ModuleOp> generate(const ast::Module &module);

private:
  /// Generate an MLIR location from the given source location.
  Location genLoc(llvm::SMLoc loc);
  Location genLoc(llvm::SMRange loc) { return genLoc(loc.Start); }

  /// Generate an MLIR type from the given source type.
  Type genType(ast::Type type);

  /// Generate MLIR for the given AST node.
  void gen(const ast::Node *node);

  //===--------------------------------------------------------------------===//
  // Statements
  //===--------------------------------------------------------------------===//

  void genImpl(const ast::CompoundStmt *stmt);
  void genImpl(const ast::EraseStmt *stmt);
  void genImpl(const ast::LetStmt *stmt);
  void genImpl(const ast::ReplaceStmt *stmt);
  void genImpl(const ast::RewriteStmt *stmt);
  void genImpl(const ast::ReturnStmt *stmt);

  //===--------------------------------------------------------------------===//
  // Decls
  //===--------------------------------------------------------------------===//

  void genImpl(const ast::UserConstraintDecl *decl);
  void genImpl(const ast::UserRewriteDecl *decl);
  void genImpl(const ast::PatternDecl *decl);

  /// Generate the set of MLIR values defined for the given variable decl, and
  /// apply any attached constraints.
  SmallVector<Value> genVar(const ast::VariableDecl *varDecl);

  /// Generate the value for a variable that does not have an initializer
  /// expression, i.e. create the PDL value based on the type/constraints of the
  /// variable.
  Value genNonInitializerVar(const ast::VariableDecl *varDecl, Location loc);

  /// Apply the constraints of the given variable to `values`, which correspond
  /// to the MLIR values of the variable.
  void applyVarConstraints(const ast::VariableDecl *varDecl, ValueRange values);

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  Value genSingleExpr(const ast::Expr *expr);
  SmallVector<Value> genExpr(const ast::Expr *expr);
  Value genExprImpl(const ast::AttributeExpr *expr);
  SmallVector<Value> genExprImpl(const ast::CallExpr *expr);
  SmallVector<Value> genExprImpl(const ast::DeclRefExpr *expr);
  Value genExprImpl(const ast::MemberAccessExpr *expr);
  Value genExprImpl(const ast::OperationExpr *expr);
  Value genExprImpl(const ast::RangeExpr *expr);
  SmallVector<Value> genExprImpl(const ast::TupleExpr *expr);
  Value genExprImpl(const ast::TypeExpr *expr);

  SmallVector<Value> genConstraintCall(const ast::UserConstraintDecl *decl,
                                       Location loc, ValueRange inputs);
  SmallVector<Value> genRewriteCall(const ast::UserRewriteDecl *decl,
                                    Location loc, ValueRange inputs);
  template <typename PDLOpT, typename T>
  SmallVector<Value> genConstraintOrRewriteCall(const T *decl, Location loc,
                                                ValueRange inputs);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The MLIR builder used for building the resultant IR.
  OpBuilder builder;

  /// A map from variable declarations to the MLIR equivalent.
  using VariableMapTy =
      llvm::ScopedHashTable<const ast::VariableDecl *, SmallVector<Value>>;
  VariableMapTy variables;

  /// A reference to the ODS context.
  const ods::Context &odsContext;

  /// The source manager of the PDLL ast.
  const llvm::SourceMgr &sourceMgr;
};
} // namespace

OwningOpRef<ModuleOp> CodeGen::generate(const ast::Module &module) {
  OwningOpRef<ModuleOp> mlirModule =
      builder.create<ModuleOp>(genLoc(module.getLoc()));
  builder.setInsertionPointToStart(mlirModule->getBody());

  // Generate code for each of the decls within the module.
  for (const ast::Decl *decl : module.getChildren())
    gen(decl);

  return mlirModule;
}

Location CodeGen::genLoc(llvm::SMLoc loc) {
  unsigned fileID = sourceMgr.FindBufferContainingLoc(loc);

  // TODO: Fix performance issues in SourceMgr::getLineAndColumn so that we can
  //       use it here.
  auto &bufferInfo = sourceMgr.getBufferInfo(fileID);
  unsigned lineNo = bufferInfo.getLineNumber(loc.getPointer());
  unsigned column =
      (loc.getPointer() - bufferInfo.getPointerForLineNumber(lineNo)) + 1;
  auto *buffer = sourceMgr.getMemoryBuffer(fileID);

  return FileLineColLoc::get(builder.getContext(),
                             buffer->getBufferIdentifier(), lineNo, column);
}

Type CodeGen::genType(ast::Type type) {
  return TypeSwitch<ast::Type, Type>(type)
      .Case([&](ast::AttributeType astType) -> Type {
        return builder.getType<pdl::AttributeType>();
      })
      .Case([&](ast::OperationType astType) -> Type {
        return builder.getType<pdl::OperationType>();
      })
      .Case([&](ast::TypeType astType) -> Type {
        return builder.getType<pdl::TypeType>();
      })
      .Case([&](ast::ValueType astType) -> Type {
        return builder.getType<pdl::ValueType>();
      })
      .Case([&](ast::RangeType astType) -> Type {
        return pdl::RangeType::get(genType(astType.getElementType()));
      });
}

void CodeGen::gen(const ast::Node *node) {
  TypeSwitch<const ast::Node *>(node)
      .Case<const ast::CompoundStmt, const ast::EraseStmt, const ast::LetStmt,
            const ast::ReplaceStmt, const ast::RewriteStmt,
            const ast::ReturnStmt, const ast::UserConstraintDecl,
            const ast::UserRewriteDecl, const ast::PatternDecl>(
          [&](auto derivedNode) { this->genImpl(derivedNode); })
      .Case([&](const ast::Expr *expr) { genExpr(expr); });
}

//===----------------------------------------------------------------------===//
// CodeGen: Statements
//===----------------------------------------------------------------------===//

void CodeGen::genImpl(const ast::CompoundStmt *stmt) {
  VariableMapTy::ScopeTy varScope(variables);
  for (const ast::Stmt *childStmt : stmt->getChildren())
    gen(childStmt);
}

/// If the given builder is nested under a PDL PatternOp, build a rewrite
/// operation and update the builder to nest under it. This is necessary for
/// PDLL operation rewrite statements that are directly nested within a Pattern.
static void checkAndNestUnderRewriteOp(OpBuilder &builder, Value rootExpr,
                                       Location loc) {
  if (isa<pdl::PatternOp>(builder.getInsertionBlock()->getParentOp())) {
    pdl::RewriteOp rewrite =
        builder.create<pdl::RewriteOp>(loc, rootExpr, /*name=*/StringAttr(),
                                       /*externalArgs=*/ValueRange());
    builder.createBlock(&rewrite.getBodyRegion());
  }
}

void CodeGen::genImpl(const ast::EraseStmt *stmt) {
  OpBuilder::InsertionGuard insertGuard(builder);
  Value rootExpr = genSingleExpr(stmt->getRootOpExpr());
  Location loc = genLoc(stmt->getLoc());

  // Make sure we are nested in a RewriteOp.
  OpBuilder::InsertionGuard guard(builder);
  checkAndNestUnderRewriteOp(builder, rootExpr, loc);
  builder.create<pdl::EraseOp>(loc, rootExpr);
}

void CodeGen::genImpl(const ast::LetStmt *stmt) { genVar(stmt->getVarDecl()); }

void CodeGen::genImpl(const ast::ReplaceStmt *stmt) {
  OpBuilder::InsertionGuard insertGuard(builder);
  Value rootExpr = genSingleExpr(stmt->getRootOpExpr());
  Location loc = genLoc(stmt->getLoc());

  // Make sure we are nested in a RewriteOp.
  OpBuilder::InsertionGuard guard(builder);
  checkAndNestUnderRewriteOp(builder, rootExpr, loc);

  SmallVector<Value> replValues;
  for (ast::Expr *replExpr : stmt->getReplExprs())
    replValues.push_back(genSingleExpr(replExpr));

  // Check to see if the statement has a replacement operation, or a range of
  // replacement values.
  bool usesReplOperation =
      replValues.size() == 1 &&
      isa<pdl::OperationType>(replValues.front().getType());
  builder.create<pdl::ReplaceOp>(
      loc, rootExpr, usesReplOperation ? replValues[0] : Value(),
      usesReplOperation ? ValueRange() : ValueRange(replValues));
}

void CodeGen::genImpl(const ast::RewriteStmt *stmt) {
  OpBuilder::InsertionGuard insertGuard(builder);
  Value rootExpr = genSingleExpr(stmt->getRootOpExpr());

  // Make sure we are nested in a RewriteOp.
  OpBuilder::InsertionGuard guard(builder);
  checkAndNestUnderRewriteOp(builder, rootExpr, genLoc(stmt->getLoc()));
  gen(stmt->getRewriteBody());
}

void CodeGen::genImpl(const ast::ReturnStmt *stmt) {
  // ReturnStmt generation is handled by the respective constraint or rewrite
  // parent node.
}

//===----------------------------------------------------------------------===//
// CodeGen: Decls
//===----------------------------------------------------------------------===//

void CodeGen::genImpl(const ast::UserConstraintDecl *decl) {
  // All PDLL constraints get inlined when called, and the main native
  // constraint declarations doesn't require any MLIR to be generated, only uses
  // of it do.
}

void CodeGen::genImpl(const ast::UserRewriteDecl *decl) {
  // All PDLL rewrites get inlined when called, and the main native
  // rewrite declarations doesn't require any MLIR to be generated, only uses
  // of it do.
}

void CodeGen::genImpl(const ast::PatternDecl *decl) {
  const ast::Name *name = decl->getName();

  // FIXME: Properly model HasBoundedRecursion in PDL so that we don't drop it
  // here.
  pdl::PatternOp pattern = builder.create<pdl::PatternOp>(
      genLoc(decl->getLoc()), decl->getBenefit(),
      name ? std::optional<StringRef>(name->getName())
           : std::optional<StringRef>());

  OpBuilder::InsertionGuard savedInsertPoint(builder);
  builder.setInsertionPointToStart(pattern.getBody());
  gen(decl->getBody());
}

SmallVector<Value> CodeGen::genVar(const ast::VariableDecl *varDecl) {
  auto it = variables.begin(varDecl);
  if (it != variables.end())
    return *it;

  // If the variable has an initial value, use that as the base value.
  // Otherwise, generate a value using the constraint list.
  SmallVector<Value> values;
  if (const ast::Expr *initExpr = varDecl->getInitExpr())
    values = genExpr(initExpr);
  else
    values.push_back(genNonInitializerVar(varDecl, genLoc(varDecl->getLoc())));

  // Apply the constraints of the values of the variable.
  applyVarConstraints(varDecl, values);

  variables.insert(varDecl, values);
  return values;
}

Value CodeGen::genNonInitializerVar(const ast::VariableDecl *varDecl,
                                    Location loc) {
  // A functor used to generate expressions nested
  auto getTypeConstraint = [&]() -> Value {
    for (const ast::ConstraintRef &constraint : varDecl->getConstraints()) {
      Value typeValue =
          TypeSwitch<const ast::Node *, Value>(constraint.constraint)
              .Case<ast::AttrConstraintDecl, ast::ValueConstraintDecl,
                    ast::ValueRangeConstraintDecl>(
                  [&, this](auto *cst) -> Value {
                    if (auto *typeConstraintExpr = cst->getTypeExpr())
                      return this->genSingleExpr(typeConstraintExpr);
                    return Value();
                  })
              .Default(Value());
      if (typeValue)
        return typeValue;
    }
    return Value();
  };

  // Generate a value based on the type of the variable.
  ast::Type type = varDecl->getType();
  Type mlirType = genType(type);
  if (type.isa<ast::ValueType>())
    return builder.create<pdl::OperandOp>(loc, mlirType, getTypeConstraint());
  if (type.isa<ast::TypeType>())
    return builder.create<pdl::TypeOp>(loc, mlirType, /*type=*/TypeAttr());
  if (type.isa<ast::AttributeType>())
    return builder.create<pdl::AttributeOp>(loc, getTypeConstraint());
  if (ast::OperationType opType = type.dyn_cast<ast::OperationType>()) {
    Value operands = builder.create<pdl::OperandsOp>(
        loc, pdl::RangeType::get(builder.getType<pdl::ValueType>()),
        /*type=*/Value());
    Value results = builder.create<pdl::TypesOp>(
        loc, pdl::RangeType::get(builder.getType<pdl::TypeType>()),
        /*types=*/ArrayAttr());
    return builder.create<pdl::OperationOp>(
        loc, opType.getName(), operands, std::nullopt, ValueRange(), results);
  }

  if (ast::RangeType rangeTy = type.dyn_cast<ast::RangeType>()) {
    ast::Type eleTy = rangeTy.getElementType();
    if (eleTy.isa<ast::ValueType>())
      return builder.create<pdl::OperandsOp>(loc, mlirType,
                                             getTypeConstraint());
    if (eleTy.isa<ast::TypeType>())
      return builder.create<pdl::TypesOp>(loc, mlirType, /*types=*/ArrayAttr());
  }

  llvm_unreachable("invalid non-initialized variable type");
}

void CodeGen::applyVarConstraints(const ast::VariableDecl *varDecl,
                                  ValueRange values) {
  // Generate calls to any user constraints that were attached via the
  // constraint list.
  for (const ast::ConstraintRef &ref : varDecl->getConstraints())
    if (const auto *userCst = dyn_cast<ast::UserConstraintDecl>(ref.constraint))
      genConstraintCall(userCst, genLoc(ref.referenceLoc), values);
}

//===----------------------------------------------------------------------===//
// CodeGen: Expressions
//===----------------------------------------------------------------------===//

Value CodeGen::genSingleExpr(const ast::Expr *expr) {
  return TypeSwitch<const ast::Expr *, Value>(expr)
      .Case<const ast::AttributeExpr, const ast::MemberAccessExpr,
            const ast::OperationExpr, const ast::RangeExpr,
            const ast::TypeExpr>(
          [&](auto derivedNode) { return this->genExprImpl(derivedNode); })
      .Case<const ast::CallExpr, const ast::DeclRefExpr, const ast::TupleExpr>(
          [&](auto derivedNode) {
            SmallVector<Value> results = this->genExprImpl(derivedNode);
            assert(results.size() == 1 && "expected single expression result");
            return results[0];
          });
}

SmallVector<Value> CodeGen::genExpr(const ast::Expr *expr) {
  return TypeSwitch<const ast::Expr *, SmallVector<Value>>(expr)
      .Case<const ast::CallExpr, const ast::DeclRefExpr, const ast::TupleExpr>(
          [&](auto derivedNode) { return this->genExprImpl(derivedNode); })
      .Default([&](const ast::Expr *expr) -> SmallVector<Value> {
        return {genSingleExpr(expr)};
      });
}

Value CodeGen::genExprImpl(const ast::AttributeExpr *expr) {
  Attribute attr = parseAttribute(expr->getValue(), builder.getContext());
  assert(attr && "invalid MLIR attribute data");
  return builder.create<pdl::AttributeOp>(genLoc(expr->getLoc()), attr);
}

SmallVector<Value> CodeGen::genExprImpl(const ast::CallExpr *expr) {
  Location loc = genLoc(expr->getLoc());
  SmallVector<Value> arguments;
  for (const ast::Expr *arg : expr->getArguments())
    arguments.push_back(genSingleExpr(arg));

  // Resolve the callable expression of this call.
  auto *callableExpr = dyn_cast<ast::DeclRefExpr>(expr->getCallableExpr());
  assert(callableExpr && "unhandled CallExpr callable");

  // Generate the PDL based on the type of callable.
  const ast::Decl *callable = callableExpr->getDecl();
  if (const auto *decl = dyn_cast<ast::UserConstraintDecl>(callable))
    return genConstraintCall(decl, loc, arguments);
  if (const auto *decl = dyn_cast<ast::UserRewriteDecl>(callable))
    return genRewriteCall(decl, loc, arguments);
  llvm_unreachable("unhandled CallExpr callable");
}

SmallVector<Value> CodeGen::genExprImpl(const ast::DeclRefExpr *expr) {
  if (const auto *varDecl = dyn_cast<ast::VariableDecl>(expr->getDecl()))
    return genVar(varDecl);
  llvm_unreachable("unknown decl reference expression");
}

Value CodeGen::genExprImpl(const ast::MemberAccessExpr *expr) {
  Location loc = genLoc(expr->getLoc());
  StringRef name = expr->getMemberName();
  SmallVector<Value> parentExprs = genExpr(expr->getParentExpr());
  ast::Type parentType = expr->getParentExpr()->getType();

  // Handle operation based member access.
  if (ast::OperationType opType = parentType.dyn_cast<ast::OperationType>()) {
    if (isa<ast::AllResultsMemberAccessExpr>(expr)) {
      Type mlirType = genType(expr->getType());
      if (isa<pdl::ValueType>(mlirType))
        return builder.create<pdl::ResultOp>(loc, mlirType, parentExprs[0],
                                             builder.getI32IntegerAttr(0));
      return builder.create<pdl::ResultsOp>(loc, mlirType, parentExprs[0]);
    }

    const ods::Operation *odsOp = opType.getODSOperation();
    if (!odsOp) {
      assert(llvm::isDigit(name[0]) &&
             "unregistered op only allows numeric indexing");
      unsigned resultIndex;
      name.getAsInteger(/*Radix=*/10, resultIndex);
      IntegerAttr index = builder.getI32IntegerAttr(resultIndex);
      return builder.create<pdl::ResultOp>(loc, genType(expr->getType()),
                                           parentExprs[0], index);
    }

    // Find the result with the member name or by index.
    ArrayRef<ods::OperandOrResult> results = odsOp->getResults();
    unsigned resultIndex = results.size();
    if (llvm::isDigit(name[0])) {
      name.getAsInteger(/*Radix=*/10, resultIndex);
    } else {
      auto findFn = [&](const ods::OperandOrResult &result) {
        return result.getName() == name;
      };
      resultIndex = llvm::find_if(results, findFn) - results.begin();
    }
    assert(resultIndex < results.size() && "invalid result index");

    // Generate the result access.
    IntegerAttr index = builder.getI32IntegerAttr(resultIndex);
    return builder.create<pdl::ResultsOp>(loc, genType(expr->getType()),
                                          parentExprs[0], index);
  }

  // Handle tuple based member access.
  if (auto tupleType = parentType.dyn_cast<ast::TupleType>()) {
    auto elementNames = tupleType.getElementNames();

    // The index is either a numeric index, or a name.
    unsigned index = 0;
    if (llvm::isDigit(name[0]))
      name.getAsInteger(/*Radix=*/10, index);
    else
      index = llvm::find(elementNames, name) - elementNames.begin();

    assert(index < parentExprs.size() && "invalid result index");
    return parentExprs[index];
  }

  llvm_unreachable("unhandled member access expression");
}

Value CodeGen::genExprImpl(const ast::OperationExpr *expr) {
  Location loc = genLoc(expr->getLoc());
  std::optional<StringRef> opName = expr->getName();

  // Operands.
  SmallVector<Value> operands;
  for (const ast::Expr *operand : expr->getOperands())
    operands.push_back(genSingleExpr(operand));

  // Attributes.
  SmallVector<StringRef> attrNames;
  SmallVector<Value> attrValues;
  for (const ast::NamedAttributeDecl *attr : expr->getAttributes()) {
    attrNames.push_back(attr->getName().getName());
    attrValues.push_back(genSingleExpr(attr->getValue()));
  }

  // Results.
  SmallVector<Value> results;
  for (const ast::Expr *result : expr->getResultTypes())
    results.push_back(genSingleExpr(result));

  return builder.create<pdl::OperationOp>(loc, opName, operands, attrNames,
                                          attrValues, results);
}

Value CodeGen::genExprImpl(const ast::RangeExpr *expr) {
  SmallVector<Value> elements;
  for (const ast::Expr *element : expr->getElements())
    llvm::append_range(elements, genExpr(element));

  return builder.create<pdl::RangeOp>(genLoc(expr->getLoc()),
                                      genType(expr->getType()), elements);
}

SmallVector<Value> CodeGen::genExprImpl(const ast::TupleExpr *expr) {
  SmallVector<Value> elements;
  for (const ast::Expr *element : expr->getElements())
    elements.push_back(genSingleExpr(element));
  return elements;
}

Value CodeGen::genExprImpl(const ast::TypeExpr *expr) {
  Type type = parseType(expr->getValue(), builder.getContext());
  assert(type && "invalid MLIR type data");
  return builder.create<pdl::TypeOp>(genLoc(expr->getLoc()),
                                     builder.getType<pdl::TypeType>(),
                                     TypeAttr::get(type));
}

SmallVector<Value>
CodeGen::genConstraintCall(const ast::UserConstraintDecl *decl, Location loc,
                           ValueRange inputs) {
  // Apply any constraints defined on the arguments to the input values.
  for (auto it : llvm::zip(decl->getInputs(), inputs))
    applyVarConstraints(std::get<0>(it), std::get<1>(it));

  // Generate the constraint call.
  SmallVector<Value> results =
      genConstraintOrRewriteCall<pdl::ApplyNativeConstraintOp>(decl, loc,
                                                               inputs);

  // Apply any constraints defined on the results of the constraint.
  for (auto it : llvm::zip(decl->getResults(), results))
    applyVarConstraints(std::get<0>(it), std::get<1>(it));
  return results;
}

SmallVector<Value> CodeGen::genRewriteCall(const ast::UserRewriteDecl *decl,
                                           Location loc, ValueRange inputs) {
  return genConstraintOrRewriteCall<pdl::ApplyNativeRewriteOp>(decl, loc,
                                                               inputs);
}

template <typename PDLOpT, typename T>
SmallVector<Value> CodeGen::genConstraintOrRewriteCall(const T *decl,
                                                       Location loc,
                                                       ValueRange inputs) {
  const ast::CompoundStmt *cstBody = decl->getBody();

  // If the decl doesn't have a statement body, it is a native decl.
  if (!cstBody) {
    ast::Type declResultType = decl->getResultType();
    SmallVector<Type> resultTypes;
    if (ast::TupleType tupleType = declResultType.dyn_cast<ast::TupleType>()) {
      for (ast::Type type : tupleType.getElementTypes())
        resultTypes.push_back(genType(type));
    } else {
      resultTypes.push_back(genType(declResultType));
    }
    Operation *pdlOp = builder.create<PDLOpT>(
        loc, resultTypes, decl->getName().getName(), inputs);
    return pdlOp->getResults();
  }

  // Otherwise, this is a PDLL decl.
  VariableMapTy::ScopeTy varScope(variables);

  // Map the inputs of the call to the decl arguments.
  // Note: This is only valid because we do not support recursion, meaning
  // we don't need to worry about conflicting mappings here.
  for (auto it : llvm::zip(inputs, decl->getInputs()))
    variables.insert(std::get<1>(it), {std::get<0>(it)});

  // Visit the body of the call as normal.
  gen(cstBody);

  // If the decl has no results, there is nothing to do.
  if (cstBody->getChildren().empty())
    return SmallVector<Value>();
  auto *returnStmt = dyn_cast<ast::ReturnStmt>(cstBody->getChildren().back());
  if (!returnStmt)
    return SmallVector<Value>();

  // Otherwise, grab the results from the return statement.
  return genExpr(returnStmt->getResultExpr());
}

//===----------------------------------------------------------------------===//
// MLIRGen
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> mlir::pdll::codegenPDLLToMLIR(
    MLIRContext *mlirContext, const ast::Context &context,
    const llvm::SourceMgr &sourceMgr, const ast::Module &module) {
  CodeGen codegen(mlirContext, context, sourceMgr);
  OwningOpRef<ModuleOp> mlirModule = codegen.generate(module);
  if (failed(verify(*mlirModule)))
    return nullptr;
  return mlirModule;
}
