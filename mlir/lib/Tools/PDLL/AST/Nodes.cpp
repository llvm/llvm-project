//===- Nodes.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::pdll::ast;

/// Copy a string reference into the context with a null terminator.
static StringRef copyStringWithNull(Context &ctx, StringRef str) {
  if (str.empty())
    return str;

  char *data = ctx.getAllocator().Allocate<char>(str.size() + 1);
  std::copy(str.begin(), str.end(), data);
  data[str.size()] = 0;
  return StringRef(data, str.size());
}

//===----------------------------------------------------------------------===//
// Name
//===----------------------------------------------------------------------===//

const Name &Name::create(Context &ctx, StringRef name, SMRange location) {
  return *new (ctx.getAllocator().Allocate<Name>())
      Name(copyStringWithNull(ctx, name), location);
}

//===----------------------------------------------------------------------===//
// DeclScope
//===----------------------------------------------------------------------===//

void DeclScope::add(Decl *decl) {
  const Name *name = decl->getName();
  assert(name && "expected a named decl");
  assert(!decls.count(name->getName()) && "decl with this name already exists");
  decls.try_emplace(name->getName(), decl);
}

Decl *DeclScope::lookup(StringRef name) {
  if (Decl *decl = decls.lookup(name))
    return decl;
  return parent ? parent->lookup(name) : nullptr;
}

//===----------------------------------------------------------------------===//
// CompoundStmt
//===----------------------------------------------------------------------===//

CompoundStmt *CompoundStmt::create(Context &ctx, SMRange loc,
                                   ArrayRef<Stmt *> children) {
  unsigned allocSize = CompoundStmt::totalSizeToAlloc<Stmt *>(children.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(CompoundStmt));

  CompoundStmt *stmt = new (rawData) CompoundStmt(loc, children.size());
  std::uninitialized_copy(children.begin(), children.end(),
                          stmt->getChildren().begin());
  return stmt;
}

//===----------------------------------------------------------------------===//
// LetStmt
//===----------------------------------------------------------------------===//

LetStmt *LetStmt::create(Context &ctx, SMRange loc,
                         VariableDecl *varDecl) {
  return new (ctx.getAllocator().Allocate<LetStmt>()) LetStmt(loc, varDecl);
}

//===----------------------------------------------------------------------===//
// OpRewriteStmt
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EraseStmt

EraseStmt *EraseStmt::create(Context &ctx, SMRange loc, Expr *rootOp) {
  return new (ctx.getAllocator().Allocate<EraseStmt>()) EraseStmt(loc, rootOp);
}

//===----------------------------------------------------------------------===//
// ReplaceStmt

ReplaceStmt *ReplaceStmt::create(Context &ctx, SMRange loc, Expr *rootOp,
                                 ArrayRef<Expr *> replExprs) {
  unsigned allocSize = ReplaceStmt::totalSizeToAlloc<Expr *>(replExprs.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(ReplaceStmt));

  ReplaceStmt *stmt = new (rawData) ReplaceStmt(loc, rootOp, replExprs.size());
  std::uninitialized_copy(replExprs.begin(), replExprs.end(),
                          stmt->getReplExprs().begin());
  return stmt;
}

//===----------------------------------------------------------------------===//
// RewriteStmt

RewriteStmt *RewriteStmt::create(Context &ctx, SMRange loc, Expr *rootOp,
                                 CompoundStmt *rewriteBody) {
  return new (ctx.getAllocator().Allocate<RewriteStmt>())
      RewriteStmt(loc, rootOp, rewriteBody);
}

//===----------------------------------------------------------------------===//
// AttributeExpr
//===----------------------------------------------------------------------===//

AttributeExpr *AttributeExpr::create(Context &ctx, SMRange loc,
                                     StringRef value) {
  return new (ctx.getAllocator().Allocate<AttributeExpr>())
      AttributeExpr(ctx, loc, copyStringWithNull(ctx, value));
}

//===----------------------------------------------------------------------===//
// DeclRefExpr
//===----------------------------------------------------------------------===//

DeclRefExpr *DeclRefExpr::create(Context &ctx, SMRange loc, Decl *decl,
                                 Type type) {
  return new (ctx.getAllocator().Allocate<DeclRefExpr>())
      DeclRefExpr(loc, decl, type);
}

//===----------------------------------------------------------------------===//
// MemberAccessExpr
//===----------------------------------------------------------------------===//

MemberAccessExpr *MemberAccessExpr::create(Context &ctx, SMRange loc,
                                           const Expr *parentExpr,
                                           StringRef memberName, Type type) {
  return new (ctx.getAllocator().Allocate<MemberAccessExpr>()) MemberAccessExpr(
      loc, parentExpr, memberName.copy(ctx.getAllocator()), type);
}

//===----------------------------------------------------------------------===//
// OperationExpr
//===----------------------------------------------------------------------===//

OperationExpr *OperationExpr::create(
    Context &ctx, SMRange loc, const OpNameDecl *name,
    ArrayRef<Expr *> operands, ArrayRef<Expr *> resultTypes,
    ArrayRef<NamedAttributeDecl *> attributes) {
  unsigned allocSize =
      OperationExpr::totalSizeToAlloc<Expr *, NamedAttributeDecl *>(
          operands.size() + resultTypes.size(), attributes.size());
  void *rawData =
      ctx.getAllocator().Allocate(allocSize, alignof(OperationExpr));

  Type resultType = OperationType::get(ctx, name->getName());
  OperationExpr *opExpr = new (rawData)
      OperationExpr(loc, resultType, name, operands.size(), resultTypes.size(),
                    attributes.size(), name->getLoc());
  std::uninitialized_copy(operands.begin(), operands.end(),
                          opExpr->getOperands().begin());
  std::uninitialized_copy(resultTypes.begin(), resultTypes.end(),
                          opExpr->getResultTypes().begin());
  std::uninitialized_copy(attributes.begin(), attributes.end(),
                          opExpr->getAttributes().begin());
  return opExpr;
}

Optional<StringRef> OperationExpr::getName() const {
  return getNameDecl()->getName();
}

//===----------------------------------------------------------------------===//
// TupleExpr
//===----------------------------------------------------------------------===//

TupleExpr *TupleExpr::create(Context &ctx, SMRange loc,
                             ArrayRef<Expr *> elements,
                             ArrayRef<StringRef> names) {
  unsigned allocSize = TupleExpr::totalSizeToAlloc<Expr *>(elements.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(TupleExpr));

  auto elementTypes = llvm::map_range(
      elements, [](const Expr *expr) { return expr->getType(); });
  TupleType type = TupleType::get(ctx, llvm::to_vector(elementTypes), names);

  TupleExpr *expr = new (rawData) TupleExpr(loc, type);
  std::uninitialized_copy(elements.begin(), elements.end(),
                          expr->getElements().begin());
  return expr;
}

//===----------------------------------------------------------------------===//
// TypeExpr
//===----------------------------------------------------------------------===//

TypeExpr *TypeExpr::create(Context &ctx, SMRange loc, StringRef value) {
  return new (ctx.getAllocator().Allocate<TypeExpr>())
      TypeExpr(ctx, loc, copyStringWithNull(ctx, value));
}

//===----------------------------------------------------------------------===//
// AttrConstraintDecl
//===----------------------------------------------------------------------===//

AttrConstraintDecl *AttrConstraintDecl::create(Context &ctx, SMRange loc,
                                               Expr *typeExpr) {
  return new (ctx.getAllocator().Allocate<AttrConstraintDecl>())
      AttrConstraintDecl(loc, typeExpr);
}

//===----------------------------------------------------------------------===//
// OpConstraintDecl
//===----------------------------------------------------------------------===//

OpConstraintDecl *OpConstraintDecl::create(Context &ctx, SMRange loc,
                                           const OpNameDecl *nameDecl) {
  if (!nameDecl)
    nameDecl = OpNameDecl::create(ctx, SMRange());

  return new (ctx.getAllocator().Allocate<OpConstraintDecl>())
      OpConstraintDecl(loc, nameDecl);
}

Optional<StringRef> OpConstraintDecl::getName() const {
  return getNameDecl()->getName();
}

//===----------------------------------------------------------------------===//
// TypeConstraintDecl
//===----------------------------------------------------------------------===//

TypeConstraintDecl *TypeConstraintDecl::create(Context &ctx,
                                               SMRange loc) {
  return new (ctx.getAllocator().Allocate<TypeConstraintDecl>())
      TypeConstraintDecl(loc);
}

//===----------------------------------------------------------------------===//
// TypeRangeConstraintDecl
//===----------------------------------------------------------------------===//

TypeRangeConstraintDecl *TypeRangeConstraintDecl::create(Context &ctx,
                                                         SMRange loc) {
  return new (ctx.getAllocator().Allocate<TypeRangeConstraintDecl>())
      TypeRangeConstraintDecl(loc);
}

//===----------------------------------------------------------------------===//
// ValueConstraintDecl
//===----------------------------------------------------------------------===//

ValueConstraintDecl *
ValueConstraintDecl::create(Context &ctx, SMRange loc, Expr *typeExpr) {
  return new (ctx.getAllocator().Allocate<ValueConstraintDecl>())
      ValueConstraintDecl(loc, typeExpr);
}

//===----------------------------------------------------------------------===//
// ValueRangeConstraintDecl
//===----------------------------------------------------------------------===//

ValueRangeConstraintDecl *ValueRangeConstraintDecl::create(Context &ctx,
                                                           SMRange loc,
                                                           Expr *typeExpr) {
  return new (ctx.getAllocator().Allocate<ValueRangeConstraintDecl>())
      ValueRangeConstraintDecl(loc, typeExpr);
}

//===----------------------------------------------------------------------===//
// NamedAttributeDecl
//===----------------------------------------------------------------------===//

NamedAttributeDecl *NamedAttributeDecl::create(Context &ctx, const Name &name,
                                               Expr *value) {
  return new (ctx.getAllocator().Allocate<NamedAttributeDecl>())
      NamedAttributeDecl(name, value);
}

//===----------------------------------------------------------------------===//
// OpNameDecl
//===----------------------------------------------------------------------===//

OpNameDecl *OpNameDecl::create(Context &ctx, const Name &name) {
  return new (ctx.getAllocator().Allocate<OpNameDecl>()) OpNameDecl(name);
}
OpNameDecl *OpNameDecl::create(Context &ctx, SMRange loc) {
  return new (ctx.getAllocator().Allocate<OpNameDecl>()) OpNameDecl(loc);
}

//===----------------------------------------------------------------------===//
// PatternDecl
//===----------------------------------------------------------------------===//

PatternDecl *PatternDecl::create(Context &ctx, SMRange loc,
                                 const Name *name, Optional<uint16_t> benefit,
                                 bool hasBoundedRecursion,
                                 const CompoundStmt *body) {
  return new (ctx.getAllocator().Allocate<PatternDecl>())
      PatternDecl(loc, name, benefit, hasBoundedRecursion, body);
}

//===----------------------------------------------------------------------===//
// VariableDecl
//===----------------------------------------------------------------------===//

VariableDecl *VariableDecl::create(Context &ctx, const Name &name, Type type,
                                   Expr *initExpr,
                                   ArrayRef<ConstraintRef> constraints) {
  unsigned allocSize =
      VariableDecl::totalSizeToAlloc<ConstraintRef>(constraints.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(VariableDecl));

  VariableDecl *varDecl =
      new (rawData) VariableDecl(name, type, initExpr, constraints.size());
  std::uninitialized_copy(constraints.begin(), constraints.end(),
                          varDecl->getConstraints().begin());
  return varDecl;
}

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

Module *Module::create(Context &ctx, SMLoc loc,
                       ArrayRef<Decl *> children) {
  unsigned allocSize = Module::totalSizeToAlloc<Decl *>(children.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(Module));

  Module *module = new (rawData) Module(loc, children.size());
  std::uninitialized_copy(children.begin(), children.end(),
                          module->getChildren().begin());
  return module;
}
