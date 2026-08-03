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
  void genImpl(const ast::TakeRegionStmt *stmt);
  void genImpl(const ast::MoveBlockStmt *stmt);

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

  /// Coerce a value to the target PDL type by inserting structural navigation
  /// ops. For example, coercing !pdl.operation to !pdl.region inserts a
  /// pdl::RegionOp (region_of at 0), and to !pdl.block inserts region_of +
  /// block_of.
  Value coerceToType(Value val, Type targetType, Location loc);

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  Value genSingleExpr(const ast::Expr *expr);
  SmallVector<Value> genExpr(const ast::Expr *expr);
  Value genExprImpl(const ast::AttributeExpr *expr);
  Value genExprImpl(const ast::BlockExpr *expr);
  SmallVector<Value> genExprImpl(const ast::CallExpr *expr);
  SmallVector<Value> genExprImpl(const ast::DeclRefExpr *expr);
  Value genExprImpl(const ast::MemberAccessExpr *expr);
  Value genExprImpl(const ast::OperationExpr *expr);
  Value genExprImpl(const ast::RangeExpr *expr);
  Value genExprImpl(const ast::RegionExpr *expr);
  SmallVector<Value> genExprImpl(const ast::TupleExpr *expr);
  Value genExprImpl(const ast::TypeExpr *expr);

  SmallVector<Value> genConstraintCall(const ast::UserConstraintDecl *decl,
                                       Location loc, ValueRange inputs,
                                       bool isNegated = false);
  SmallVector<Value> genRewriteCall(const ast::UserRewriteDecl *decl,
                                    Location loc, ValueRange inputs);
  template <typename PDLOpT, typename T>
  SmallVector<Value> genConstraintOrRewriteCall(const T *decl, Location loc,
                                                ValueRange inputs,
                                                bool isNegated = false);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The MLIR builder used for building the resultant IR.
  OpBuilder builder;

  /// A map from variable declarations to the MLIR equivalent.
  using VariableMapTy =
      llvm::ScopedHashTable<const ast::VariableDecl *, SmallVector<Value>>;
  VariableMapTy variables;

  /// A map from BlockExpr AST nodes to the pdl::BlockOp values created during
  /// OperationExpr processing. This enables cross-context block references:
  /// when a named block (^entry) is matched, the real pdl::BlockOp value is
  /// stored here so that genExprImpl(BlockExpr) can return it instead of a
  /// placeholder UnrealizedConversionCastOp.
  llvm::DenseMap<const ast::BlockExpr *, Value> blockExprValues;

  /// A reference to the ODS context.
  const ods::Context &odsContext;

  /// The source manager of the PDLL ast.
  const llvm::SourceMgr &sourceMgr;
};
} // namespace

OwningOpRef<ModuleOp> CodeGen::generate(const ast::Module &module) {
  OwningOpRef<ModuleOp> mlirModule =
      ModuleOp::create(builder, genLoc(module.getLoc()));
  builder.setInsertionPointToStart(mlirModule->getBody());

  // Generate code for each of the decls within the module.
  for (const ast::Decl *decl : module.getChildren())
    gen(decl);

  return mlirModule;
}

Location CodeGen::genLoc(llvm::SMLoc loc) {
  unsigned fileID = sourceMgr.FindBufferContainingLoc(loc);

  auto [lineNo, column] = sourceMgr.getLineAndColumn(loc);
  auto *buffer = sourceMgr.getMemoryBuffer(fileID);

  return FileLineColLoc::get(builder.getContext(),
                             buffer->getBufferIdentifier(), lineNo, column);
}

Type CodeGen::genType(ast::Type type) {
  return TypeSwitch<ast::Type, Type>(type)
      .Case([&](ast::AttributeType astType) -> Type {
        return builder.getType<pdl::AttributeType>();
      })
      .Case([&](ast::BlockType astType) -> Type {
        return builder.getType<pdl::BlockType>();
      })
      .Case([&](ast::OperationType astType) -> Type {
        return builder.getType<pdl::OperationType>();
      })
      .Case([&](ast::RegionType astType) -> Type {
        return builder.getType<pdl::RegionType>();
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
            const ast::ReturnStmt, const ast::TakeRegionStmt,
            const ast::MoveBlockStmt, const ast::UserConstraintDecl,
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
        pdl::RewriteOp::create(builder, loc, rootExpr, /*name=*/StringAttr(),
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
  pdl::EraseOp::create(builder, loc, rootExpr);
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
  pdl::ReplaceOp::create(
      builder, loc, rootExpr, usesReplOperation ? replValues[0] : Value(),
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

Value CodeGen::coerceToType(Value val, Type targetType, Location loc) {
  if (val.getType() == targetType)
    return val;

  // Operation -> Region: insert pdl::RegionOp (region_of at index 0).
  if (isa<pdl::OperationType>(val.getType()) &&
      isa<pdl::RegionType>(targetType))
    return pdl::RegionOp::create(builder, loc, targetType, val,
                                 builder.getI32IntegerAttr(0));

  // Operation -> Block: insert region_of + block_of (both at index 0).
  if (isa<pdl::OperationType>(val.getType()) &&
      isa<pdl::BlockType>(targetType)) {
    auto regionType = builder.getType<pdl::RegionType>();
    Value region = pdl::RegionOp::create(builder, loc, regionType, val,
                                         builder.getI32IntegerAttr(0));
    return pdl::BlockOp::create(builder, loc, targetType, region,
                                builder.getI32IntegerAttr(0));
  }

  // Region -> Block: insert pdl::BlockOp (block_of at index 0).
  if (isa<pdl::RegionType>(val.getType()) && isa<pdl::BlockType>(targetType))
    return pdl::BlockOp::create(builder, loc, targetType, val,
                                builder.getI32IntegerAttr(0));

  return val;
}

void CodeGen::genImpl(const ast::TakeRegionStmt *stmt) {
  OpBuilder::InsertionGuard insertGuard(builder);
  Value regionExpr = genSingleExpr(stmt->getRegion());
  Value destOpExpr = genSingleExpr(stmt->getDestOp());
  Location loc = genLoc(stmt->getLoc());

  // Coerce operands to !pdl.region if they are !pdl.operation-typed.
  auto regionType = builder.getType<pdl::RegionType>();
  regionExpr = coerceToType(regionExpr, regionType, loc);
  destOpExpr = coerceToType(destOpExpr, regionType, loc);

  // Make sure we are nested in a RewriteOp.
  checkAndNestUnderRewriteOp(builder, destOpExpr, loc);

  pdl::TakeRegionOp::create(builder, loc, regionExpr, destOpExpr);
}

void CodeGen::genImpl(const ast::MoveBlockStmt *stmt) {
  OpBuilder::InsertionGuard insertGuard(builder);
  Value blockExpr = genSingleExpr(stmt->getBlock());
  Value destBlockExpr = genSingleExpr(stmt->getDestBlock());
  Location loc = genLoc(stmt->getLoc());

  // Coerce operands to !pdl.block if they are !pdl.operation or
  // !pdl.region-typed. Both source and dest are blocks (Block::moveBefore).
  auto blockType = builder.getType<pdl::BlockType>();
  blockExpr = coerceToType(blockExpr, blockType, loc);
  destBlockExpr = coerceToType(destBlockExpr, blockType, loc);

  // Make sure we are nested in a RewriteOp.
  checkAndNestUnderRewriteOp(builder, blockExpr, loc);

  pdl::MoveBlockOp::create(builder, loc, blockExpr, destBlockExpr);
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
  pdl::PatternOp pattern = pdl::PatternOp::create(
      builder, genLoc(decl->getLoc()), decl->getBenefit(),
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
  if (isa<ast::ValueType>(type))
    return pdl::OperandOp::create(builder, loc, mlirType, getTypeConstraint());
  if (isa<ast::TypeType>(type))
    return pdl::TypeOp::create(builder, loc, mlirType, /*type=*/TypeAttr());
  if (isa<ast::AttributeType>(type))
    return pdl::AttributeOp::create(builder, loc, getTypeConstraint());
  if (ast::OperationType opType = dyn_cast<ast::OperationType>(type)) {
    Value operands = pdl::OperandsOp::create(
        builder, loc, pdl::RangeType::get(builder.getType<pdl::ValueType>()),
        /*type=*/Value());
    Value results = pdl::TypesOp::create(
        builder, loc, pdl::RangeType::get(builder.getType<pdl::TypeType>()),
        /*types=*/ArrayAttr());
    return pdl::OperationOp::create(builder, loc, opType.getName(), operands,
                                    ArrayRef<StringRef>(), ValueRange(),
                                    results);
  }

  if (ast::RangeType rangeTy = dyn_cast<ast::RangeType>(type)) {
    ast::Type eleTy = rangeTy.getElementType();
    if (isa<ast::ValueType>(eleTy))
      return pdl::OperandsOp::create(builder, loc, mlirType,
                                     getTypeConstraint());
    if (isa<ast::TypeType>(eleTy))
      return pdl::TypesOp::create(builder, loc, mlirType,
                                  /*types=*/ArrayAttr());
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
      .Case<const ast::AttributeExpr, const ast::BlockExpr,
            const ast::MemberAccessExpr, const ast::OperationExpr,
            const ast::RangeExpr, const ast::RegionExpr, const ast::TypeExpr>(
          [&](auto derivedNode) { return this->genExprImpl(derivedNode); })
      .Case<const ast::CallExpr, const ast::DeclRefExpr, const ast::TupleExpr>(
          [&](auto derivedNode) {
            return llvm::getSingleElement(this->genExprImpl(derivedNode));
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
  return pdl::AttributeOp::create(builder, genLoc(expr->getLoc()), attr);
}

Value CodeGen::genExprImpl(const ast::BlockExpr *expr) {
  // If this BlockExpr was already processed by genExprImpl(OperationExpr),
  // return the real pdl::BlockOp value. This enables cross-context block
  // references (e.g., ^entry matched in match, referenced in rewrite).
  auto it = blockExprValues.find(expr);
  if (it != blockExprValues.end())
    return it->second;

  Location loc = genLoc(expr->getLoc());

  // Generate block argument variable declarations. Each block argument
  // becomes a pdl.Value variable that will be constrained to the corresponding
  // block argument index via a native constraint during structural navigation.
  for (const ast::VariableDecl *arg : expr->getArgs())
    genVar(arg);

  // Generate each child operation expression within this block.
  // Children are operation constraints/declarations that need to be created
  // first, establishing the structural nesting via PDL's declarative pattern.
  for (const ast::Expr *child : expr->getChildren())
    genSingleExpr(child);

  // Create a placeholder block-typed value. In the match context, the
  // enclosing RegionExpr and OperationExpr will establish the structural
  // navigation (pdl.region_of/pdl.block_of). The PDL-to-PDLInterp lowering
  // handles wiring block constraints from the predicate tree.
  return mlir::UnrealizedConversionCastOp::create(
             builder, loc, genType(expr->getType()), ValueRange{})
      .getResult(0);
}

Value CodeGen::genExprImpl(const ast::RegionExpr *expr) {
  Location loc = genLoc(expr->getLoc());

  // Generate each block expression within this region.
  SmallVector<Value> blockValues;
  for (const ast::Expr *block : expr->getBlocks())
    blockValues.push_back(genSingleExpr(block));

  // Create a placeholder region-typed value. Similar to BlockExpr, the
  // structural navigation is established by the PDL-to-PDLInterp lowering
  // which generates pdl_interp.get_region/get_block/check_* from the
  // predicate tree built during pattern analysis.
  return mlir::UnrealizedConversionCastOp::create(
             builder, loc, genType(expr->getType()), blockValues)
      .getResult(0);
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
    return genConstraintCall(decl, loc, arguments, expr->getIsNegated());
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
  if (ast::OperationType opType = dyn_cast<ast::OperationType>(parentType)) {
    if (isa<ast::AllResultsMemberAccessExpr>(expr)) {
      Type mlirType = genType(expr->getType());
      if (isa<pdl::ValueType>(mlirType))
        return pdl::ResultOp::create(builder, loc, mlirType, parentExprs[0],
                                     builder.getI32IntegerAttr(0));
      return pdl::ResultsOp::create(builder, loc, mlirType, parentExprs[0]);
    }

    const ods::Operation *odsOp = opType.getODSOperation();
    if (!odsOp) {
      assert(llvm::isDigit(name[0]) &&
             "unregistered op only allows numeric indexing");
      unsigned resultIndex;
      name.getAsInteger(/*Radix=*/10, resultIndex);
      IntegerAttr index = builder.getI32IntegerAttr(resultIndex);
      return pdl::ResultOp::create(builder, loc, genType(expr->getType()),
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
    return pdl::ResultsOp::create(builder, loc, genType(expr->getType()),
                                  parentExprs[0], index);
  }

  // Handle tuple based member access.
  if (auto tupleType = dyn_cast<ast::TupleType>(parentType)) {
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

  Value opVal = pdl::OperationOp::create(builder, loc, opName, operands,
                                         attrNames, attrValues, results);

  // Generate structural navigation for attached regions.
  // For each attached RegionExpr, generate pdl.region_of from the parent
  // operation and pdl.block_of from each region, connecting the structural
  // constraints.
  for (auto [regionIdx, regionExpr] : llvm::enumerate(expr->getRegions())) {
    const auto *rgnExpr = cast<ast::RegionExpr>(regionExpr);

    // Navigate to the region from the parent operation.
    auto regionType = builder.getType<pdl::RegionType>();
    auto regionIdxAttr = builder.getI32IntegerAttr(regionIdx);
    Value regionVal =
        pdl::RegionOp::create(builder, loc, regionType, opVal, regionIdxAttr);

    // Navigate to each block within the region.
    auto blockType = builder.getType<pdl::BlockType>();
    for (auto [blockIdx, blockExpr] : llvm::enumerate(rgnExpr->getBlocks())) {
      const auto *blkExpr = cast<ast::BlockExpr>(blockExpr);
      auto blockIdxAttr = builder.getI32IntegerAttr(blockIdx);
      Value blockVal = pdl::BlockOp::create(builder, loc, blockType, regionVal,
                                            blockIdxAttr);

      // Bind each block argument to its index within the block using
      // pdl.block_arg or pdl.block_args. Scalar args use pdl.block_arg,
      // while variadic args (ValueRange) use pdl.block_args for range
      // capture.
      for (auto [argIdx, argDecl] : llvm::enumerate(blkExpr->getArgs())) {
        Value blockArgVal;
        if (isa<ast::ValueRangeType>(argDecl->getType())) {
          // ValueRange arg — emit pdl.block_args with the starting index.
          auto rangeType =
              pdl::RangeType::get(builder.getType<pdl::ValueType>());
          auto argIdxAttr = builder.getI32IntegerAttr(argIdx);
          blockArgVal = pdl::BlockArgsOp::create(builder, loc, rangeType,
                                                 blockVal, argIdxAttr);
        } else {
          // Scalar Value arg — emit pdl.block_arg.
          auto valueType = builder.getType<pdl::ValueType>();
          auto argIdxAttr = builder.getI32IntegerAttr(argIdx);
          blockArgVal = pdl::BlockArgOp::create(builder, loc, valueType,
                                                blockVal, argIdxAttr);
        }
        // Register the block arg value as the variable for this decl.
        variables.insert(argDecl, {blockArgVal});
        // Apply any constraints declared on the block argument.
        applyVarConstraints(argDecl, {blockArgVal});
      }

      // Generate each child operation within this block and scope it to
      // the parent block via the parentBlock operand.
      for (const ast::Expr *child : blkExpr->getChildren()) {
        Value innerOp = genSingleExpr(child);
        // Set the parentBlock operand on the inner pdl.operation.
        if (auto opOp = innerOp.getDefiningOp<pdl::OperationOp>())
          opOp.getParentBlockMutable().assign(blockVal);
      }

      // Register this block value for the BlockExpr so that named block
      // variables (e.g., ^entry) can resolve to the real pdl::BlockOp
      // value when referenced in the rewrite context.
      blockExprValues[blkExpr] = blockVal;
    }
  }

  return opVal;
}

Value CodeGen::genExprImpl(const ast::RangeExpr *expr) {
  SmallVector<Value> elements;
  for (const ast::Expr *element : expr->getElements())
    llvm::append_range(elements, genExpr(element));

  return pdl::RangeOp::create(builder, genLoc(expr->getLoc()),
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
  return pdl::TypeOp::create(builder, genLoc(expr->getLoc()),
                             builder.getType<pdl::TypeType>(),
                             TypeAttr::get(type));
}

SmallVector<Value>
CodeGen::genConstraintCall(const ast::UserConstraintDecl *decl, Location loc,
                           ValueRange inputs, bool isNegated) {
  // Apply any constraints defined on the arguments to the input values.
  for (auto it : llvm::zip(decl->getInputs(), inputs))
    applyVarConstraints(std::get<0>(it), std::get<1>(it));

  // Generate the constraint call.
  SmallVector<Value> results =
      genConstraintOrRewriteCall<pdl::ApplyNativeConstraintOp>(
          decl, loc, inputs, isNegated);

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
SmallVector<Value>
CodeGen::genConstraintOrRewriteCall(const T *decl, Location loc,
                                    ValueRange inputs, bool isNegated) {
  const ast::CompoundStmt *cstBody = decl->getBody();

  // If the decl doesn't have a statement body, it is a native decl.
  if (!cstBody) {
    ast::Type declResultType = decl->getResultType();
    SmallVector<Type> resultTypes;
    if (ast::TupleType tupleType = dyn_cast<ast::TupleType>(declResultType)) {
      for (ast::Type type : tupleType.getElementTypes())
        resultTypes.push_back(genType(type));
    } else {
      resultTypes.push_back(genType(declResultType));
    }
    PDLOpT pdlOp = PDLOpT::create(builder, loc, resultTypes,
                                  decl->getName().getName(), inputs);
    if (isNegated && std::is_same_v<PDLOpT, pdl::ApplyNativeConstraintOp>)
      cast<pdl::ApplyNativeConstraintOp>(pdlOp).setIsNegated(true);
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
