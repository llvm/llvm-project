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
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

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
// Node
//===----------------------------------------------------------------------===//

namespace {
class NodeVisitor {
public:
  explicit NodeVisitor(function_ref<void(const Node *)> visitFn)
      : visitFn(visitFn) {}

  void visit(const Node *node) {
    if (!node || !alreadyVisited.insert(node).second)
      return;

    visitFn(node);
    TypeSwitch<const Node *>(node)
        .Case<
            // Statements.
            const CompoundStmt, const EraseStmt, const LetStmt,
            const ReplaceStmt, const ReturnStmt, const RewriteStmt,

            // Expressions.
            const AttributeExpr, const CallExpr, const DeclRefExpr,
            const MemberAccessExpr, const OperationExpr, const RangeExpr,
            const TupleExpr, const TypeExpr,

            // Core Constraint Decls.
            const AttrConstraintDecl, const OpConstraintDecl,
            const TypeConstraintDecl, const TypeRangeConstraintDecl,
            const ValueConstraintDecl, const ValueRangeConstraintDecl,

            // Decls.
            const NamedAttributeDecl, const OpNameDecl, const PatternDecl,
            const UserConstraintDecl, const UserRewriteDecl, const VariableDecl,

            const Module>(
            [&](auto derivedNode) { this->visitImpl(derivedNode); })
        .Default([](const Node *) { llvm_unreachable("unknown AST node"); });
  }

private:
  void visitImpl(const CompoundStmt *stmt) {
    for (const Node *child : stmt->getChildren())
      visit(child);
  }
  void visitImpl(const EraseStmt *stmt) { visit(stmt->getRootOpExpr()); }
  void visitImpl(const LetStmt *stmt) { visit(stmt->getVarDecl()); }
  void visitImpl(const ReplaceStmt *stmt) {
    visit(stmt->getRootOpExpr());
    for (const Node *child : stmt->getReplExprs())
      visit(child);
  }
  void visitImpl(const ReturnStmt *stmt) { visit(stmt->getResultExpr()); }
  void visitImpl(const RewriteStmt *stmt) {
    visit(stmt->getRootOpExpr());
    visit(stmt->getRewriteBody());
  }

  void visitImpl(const AttributeExpr *expr) {}
  void visitImpl(const CallExpr *expr) {
    visit(expr->getCallableExpr());
    for (const Node *child : expr->getArguments())
      visit(child);
  }
  void visitImpl(const DeclRefExpr *expr) { visit(expr->getDecl()); }
  void visitImpl(const MemberAccessExpr *expr) { visit(expr->getParentExpr()); }
  void visitImpl(const OperationExpr *expr) {
    visit(expr->getNameDecl());
    for (const Node *child : expr->getOperands())
      visit(child);
    for (const Node *child : expr->getResultTypes())
      visit(child);
    for (const Node *child : expr->getAttributes())
      visit(child);
  }
  void visitImpl(const RangeExpr *expr) {
    for (const Node *child : expr->getElements())
      visit(child);
  }
  void visitImpl(const TupleExpr *expr) {
    for (const Node *child : expr->getElements())
      visit(child);
  }
  void visitImpl(const TypeExpr *expr) {}

  void visitImpl(const AttrConstraintDecl *decl) { visit(decl->getTypeExpr()); }
  void visitImpl(const OpConstraintDecl *decl) { visit(decl->getNameDecl()); }
  void visitImpl(const TypeConstraintDecl *decl) {}
  void visitImpl(const TypeRangeConstraintDecl *decl) {}
  void visitImpl(const ValueConstraintDecl *decl) {
    visit(decl->getTypeExpr());
  }
  void visitImpl(const ValueRangeConstraintDecl *decl) {
    visit(decl->getTypeExpr());
  }

  void visitImpl(const NamedAttributeDecl *decl) { visit(decl->getValue()); }
  void visitImpl(const OpNameDecl *decl) {}
  void visitImpl(const PatternDecl *decl) { visit(decl->getBody()); }
  void visitImpl(const UserConstraintDecl *decl) {
    for (const Node *child : decl->getInputs())
      visit(child);
    for (const Node *child : decl->getResults())
      visit(child);
    visit(decl->getBody());
  }
  void visitImpl(const UserRewriteDecl *decl) {
    for (const Node *child : decl->getInputs())
      visit(child);
    for (const Node *child : decl->getResults())
      visit(child);
    visit(decl->getBody());
  }
  void visitImpl(const VariableDecl *decl) {
    visit(decl->getInitExpr());
    for (const ConstraintRef &child : decl->getConstraints())
      visit(child.constraint);
  }

  void visitImpl(const Module *module) {
    for (const Node *child : module->getChildren())
      visit(child);
  }

  function_ref<void(const Node *)> visitFn;
  SmallPtrSet<const Node *, 16> alreadyVisited;
};
} // namespace

void Node::walk(function_ref<void(const Node *)> walkFn) const {
  return NodeVisitor(walkFn).visit(this);
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

LetStmt *LetStmt::create(Context &ctx, SMRange loc, VariableDecl *varDecl) {
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
// ReturnStmt
//===----------------------------------------------------------------------===//

ReturnStmt *ReturnStmt::create(Context &ctx, SMRange loc, Expr *resultExpr) {
  return new (ctx.getAllocator().Allocate<ReturnStmt>())
      ReturnStmt(loc, resultExpr);
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
// CallExpr
//===----------------------------------------------------------------------===//

CallExpr *CallExpr::create(Context &ctx, SMRange loc, Expr *callable,
                           ArrayRef<Expr *> arguments, Type resultType,
                           bool isNegated) {
  unsigned allocSize = CallExpr::totalSizeToAlloc<Expr *>(arguments.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(CallExpr));

  CallExpr *expr = new (rawData)
      CallExpr(loc, resultType, callable, arguments.size(), isNegated);
  std::uninitialized_copy(arguments.begin(), arguments.end(),
                          expr->getArguments().begin());
  return expr;
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

OperationExpr *
OperationExpr::create(Context &ctx, SMRange loc, const ods::Operation *odsOp,
                      const OpNameDecl *name, ArrayRef<Expr *> operands,
                      ArrayRef<Expr *> resultTypes,
                      ArrayRef<NamedAttributeDecl *> attributes) {
  unsigned allocSize =
      OperationExpr::totalSizeToAlloc<Expr *, NamedAttributeDecl *>(
          operands.size() + resultTypes.size(), attributes.size());
  void *rawData =
      ctx.getAllocator().Allocate(allocSize, alignof(OperationExpr));

  Type resultType = OperationType::get(ctx, name->getName(), odsOp);
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

std::optional<StringRef> OperationExpr::getName() const {
  return getNameDecl()->getName();
}

//===----------------------------------------------------------------------===//
// RangeExpr
//===----------------------------------------------------------------------===//

RangeExpr *RangeExpr::create(Context &ctx, SMRange loc,
                             ArrayRef<Expr *> elements, RangeType type) {
  unsigned allocSize = RangeExpr::totalSizeToAlloc<Expr *>(elements.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(TupleExpr));

  RangeExpr *expr = new (rawData) RangeExpr(loc, type, elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(),
                          expr->getElements().begin());
  return expr;
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
// Decl
//===----------------------------------------------------------------------===//

void Decl::setDocComment(Context &ctx, StringRef comment) {
  docComment = comment.copy(ctx.getAllocator());
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

std::optional<StringRef> OpConstraintDecl::getName() const {
  return getNameDecl()->getName();
}

//===----------------------------------------------------------------------===//
// TypeConstraintDecl
//===----------------------------------------------------------------------===//

TypeConstraintDecl *TypeConstraintDecl::create(Context &ctx, SMRange loc) {
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

ValueConstraintDecl *ValueConstraintDecl::create(Context &ctx, SMRange loc,
                                                 Expr *typeExpr) {
  return new (ctx.getAllocator().Allocate<ValueConstraintDecl>())
      ValueConstraintDecl(loc, typeExpr);
}

//===----------------------------------------------------------------------===//
// ValueRangeConstraintDecl
//===----------------------------------------------------------------------===//

ValueRangeConstraintDecl *
ValueRangeConstraintDecl::create(Context &ctx, SMRange loc, Expr *typeExpr) {
  return new (ctx.getAllocator().Allocate<ValueRangeConstraintDecl>())
      ValueRangeConstraintDecl(loc, typeExpr);
}

//===----------------------------------------------------------------------===//
// UserConstraintDecl
//===----------------------------------------------------------------------===//

std::optional<StringRef>
UserConstraintDecl::getNativeInputType(unsigned index) const {
  return hasNativeInputTypes ? getTrailingObjects<StringRef>()[index]
                             : std::optional<StringRef>();
}

UserConstraintDecl *UserConstraintDecl::createImpl(
    Context &ctx, const Name &name, ArrayRef<VariableDecl *> inputs,
    ArrayRef<StringRef> nativeInputTypes, ArrayRef<VariableDecl *> results,
    std::optional<StringRef> codeBlock, const CompoundStmt *body,
    Type resultType) {
  bool hasNativeInputTypes = !nativeInputTypes.empty();
  assert(!hasNativeInputTypes || nativeInputTypes.size() == inputs.size());

  unsigned allocSize =
      UserConstraintDecl::totalSizeToAlloc<VariableDecl *, StringRef>(
          inputs.size() + results.size(),
          hasNativeInputTypes ? inputs.size() : 0);
  void *rawData =
      ctx.getAllocator().Allocate(allocSize, alignof(UserConstraintDecl));
  if (codeBlock)
    codeBlock = codeBlock->copy(ctx.getAllocator());

  UserConstraintDecl *decl = new (rawData)
      UserConstraintDecl(name, inputs.size(), hasNativeInputTypes,
                         results.size(), codeBlock, body, resultType);
  std::uninitialized_copy(inputs.begin(), inputs.end(),
                          decl->getInputs().begin());
  std::uninitialized_copy(results.begin(), results.end(),
                          decl->getResults().begin());
  if (hasNativeInputTypes) {
    StringRef *nativeInputTypesPtr = decl->getTrailingObjects<StringRef>();
    for (unsigned i = 0, e = inputs.size(); i < e; ++i)
      nativeInputTypesPtr[i] = nativeInputTypes[i].copy(ctx.getAllocator());
  }

  return decl;
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

PatternDecl *PatternDecl::create(Context &ctx, SMRange loc, const Name *name,
                                 std::optional<uint16_t> benefit,
                                 bool hasBoundedRecursion,
                                 const CompoundStmt *body) {
  return new (ctx.getAllocator().Allocate<PatternDecl>())
      PatternDecl(loc, name, benefit, hasBoundedRecursion, body);
}

//===----------------------------------------------------------------------===//
// UserRewriteDecl
//===----------------------------------------------------------------------===//

UserRewriteDecl *UserRewriteDecl::createImpl(Context &ctx, const Name &name,
                                             ArrayRef<VariableDecl *> inputs,
                                             ArrayRef<VariableDecl *> results,
                                             std::optional<StringRef> codeBlock,
                                             const CompoundStmt *body,
                                             Type resultType) {
  unsigned allocSize = UserRewriteDecl::totalSizeToAlloc<VariableDecl *>(
      inputs.size() + results.size());
  void *rawData =
      ctx.getAllocator().Allocate(allocSize, alignof(UserRewriteDecl));
  if (codeBlock)
    codeBlock = codeBlock->copy(ctx.getAllocator());

  UserRewriteDecl *decl = new (rawData) UserRewriteDecl(
      name, inputs.size(), results.size(), codeBlock, body, resultType);
  std::uninitialized_copy(inputs.begin(), inputs.end(),
                          decl->getInputs().begin());
  std::uninitialized_copy(results.begin(), results.end(),
                          decl->getResults().begin());
  return decl;
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

Module *Module::create(Context &ctx, SMLoc loc, ArrayRef<Decl *> children) {
  unsigned allocSize = Module::totalSizeToAlloc<Decl *>(children.size());
  void *rawData = ctx.getAllocator().Allocate(allocSize, alignof(Module));

  Module *module = new (rawData) Module(loc, children.size());
  std::uninitialized_copy(children.begin(), children.end(),
                          module->getChildren().begin());
  return module;
}
