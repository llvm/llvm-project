//===- Nodes.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_AST_NODES_H_
#define MLIR_TOOLS_PDLL_AST_NODES_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Tools/PDLL/AST/Types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TrailingObjects.h"
#include <optional>

namespace mlir {
namespace pdll {
namespace ast {
class Context;
class Decl;
class Expr;
class NamedAttributeDecl;
class OpNameDecl;
class VariableDecl;

//===----------------------------------------------------------------------===//
// Name
//===----------------------------------------------------------------------===//

/// This class provides a convenient API for interacting with source names. It
/// contains a string name as well as the source location for that name.
struct Name {
  static const Name &create(Context &ctx, StringRef name, SMRange location);

  /// Return the raw string name.
  StringRef getName() const { return name; }

  /// Get the location of this name.
  SMRange getLoc() const { return location; }

private:
  Name() = delete;
  Name(const Name &) = delete;
  Name &operator=(const Name &) = delete;
  Name(StringRef name, SMRange location) : name(name), location(location) {}

  /// The string name of the decl.
  StringRef name;
  /// The location of the decl name.
  SMRange location;
};

//===----------------------------------------------------------------------===//
// DeclScope
//===----------------------------------------------------------------------===//

/// This class represents a scope for named AST decls. A scope determines the
/// visibility and lifetime of a named declaration.
class DeclScope {
public:
  /// Create a new scope with an optional parent scope.
  DeclScope(DeclScope *parent = nullptr) : parent(parent) {}

  /// Return the parent scope of this scope, or nullptr if there is no parent.
  DeclScope *getParentScope() { return parent; }
  const DeclScope *getParentScope() const { return parent; }

  /// Return all of the decls within this scope.
  auto getDecls() const { return llvm::make_second_range(decls); }

  /// Add a new decl to the scope.
  void add(Decl *decl);

  /// Lookup a decl with the given name starting from this scope. Returns
  /// nullptr if no decl could be found.
  Decl *lookup(StringRef name);
  template <typename T>
  T *lookup(StringRef name) {
    return dyn_cast_or_null<T>(lookup(name));
  }
  const Decl *lookup(StringRef name) const {
    return const_cast<DeclScope *>(this)->lookup(name);
  }
  template <typename T>
  const T *lookup(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

private:
  /// The parent scope, or null if this is a top-level scope.
  DeclScope *parent;
  /// The decls defined within this scope.
  llvm::StringMap<Decl *> decls;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

/// This class represents a base AST node. All AST nodes are derived from this
/// class, and it contains many of the base functionality for interacting with
/// nodes.
class Node {
public:
  /// This CRTP class provides several utilies when defining new AST nodes.
  template <typename T, typename BaseT>
  class NodeBase : public BaseT {
  public:
    using Base = NodeBase<T, BaseT>;

    /// Provide type casting support.
    static bool classof(const Node *node) {
      return node->getTypeID() == TypeID::get<T>();
    }

  protected:
    template <typename... Args>
    explicit NodeBase(SMRange loc, Args &&...args)
        : BaseT(TypeID::get<T>(), loc, std::forward<Args>(args)...) {}
  };

  /// Return the type identifier of this node.
  TypeID getTypeID() const { return typeID; }

  /// Return the location of this node.
  SMRange getLoc() const { return loc; }

  /// Print this node to the given stream.
  void print(raw_ostream &os) const;

  /// Walk all of the nodes including, and nested under, this node in pre-order.
  void walk(function_ref<void(const Node *)> walkFn) const;
  template <typename WalkFnT, typename ArgT = typename llvm::function_traits<
                                  WalkFnT>::template arg_t<0>>
  std::enable_if_t<!std::is_convertible<const Node *, ArgT>::value>
  walk(WalkFnT &&walkFn) const {
    walk([&](const Node *node) {
      if (const ArgT *derivedNode = dyn_cast<ArgT>(node))
        walkFn(derivedNode);
    });
  }

protected:
  Node(TypeID typeID, SMRange loc) : typeID(typeID), loc(loc) {}

private:
  /// A unique type identifier for this node.
  TypeID typeID;

  /// The location of this node.
  SMRange loc;
};

//===----------------------------------------------------------------------===//
// Stmt
//===----------------------------------------------------------------------===//

/// This class represents a base AST Statement node.
class Stmt : public Node {
public:
  using Node::Node;

  /// Provide type casting support.
  static bool classof(const Node *node);
};

//===----------------------------------------------------------------------===//
// CompoundStmt
//===----------------------------------------------------------------------===//

/// This statement represents a compound statement, which contains a collection
/// of other statements.
class CompoundStmt final : public Node::NodeBase<CompoundStmt, Stmt>,
                           private llvm::TrailingObjects<CompoundStmt, Stmt *> {
public:
  static CompoundStmt *create(Context &ctx, SMRange location,
                              ArrayRef<Stmt *> children);

  /// Return the children of this compound statement.
  MutableArrayRef<Stmt *> getChildren() {
    return {getTrailingObjects<Stmt *>(), numChildren};
  }
  ArrayRef<Stmt *> getChildren() const {
    return const_cast<CompoundStmt *>(this)->getChildren();
  }
  ArrayRef<Stmt *>::iterator begin() const { return getChildren().begin(); }
  ArrayRef<Stmt *>::iterator end() const { return getChildren().end(); }

private:
  CompoundStmt(SMRange location, unsigned numChildren)
      : Base(location), numChildren(numChildren) {}

  /// The number of held children statements.
  unsigned numChildren;

  // Allow access to various privates.
  friend class llvm::TrailingObjects<CompoundStmt, Stmt *>;
};

//===----------------------------------------------------------------------===//
// LetStmt
//===----------------------------------------------------------------------===//

/// This statement represents a `let` statement in PDLL. This statement is used
/// to define variables.
class LetStmt final : public Node::NodeBase<LetStmt, Stmt> {
public:
  static LetStmt *create(Context &ctx, SMRange loc, VariableDecl *varDecl);

  /// Return the variable defined by this statement.
  VariableDecl *getVarDecl() const { return varDecl; }

private:
  LetStmt(SMRange loc, VariableDecl *varDecl) : Base(loc), varDecl(varDecl) {}

  /// The variable defined by this statement.
  VariableDecl *varDecl;
};

//===----------------------------------------------------------------------===//
// OpRewriteStmt
//===----------------------------------------------------------------------===//

/// This class represents a base operation rewrite statement. Operation rewrite
/// statements perform a set of transformations on a given root operation.
class OpRewriteStmt : public Stmt {
public:
  /// Provide type casting support.
  static bool classof(const Node *node);

  /// Return the root operation of this rewrite.
  Expr *getRootOpExpr() const { return rootOp; }

protected:
  OpRewriteStmt(TypeID typeID, SMRange loc, Expr *rootOp)
      : Stmt(typeID, loc), rootOp(rootOp) {}

protected:
  /// The root operation being rewritten.
  Expr *rootOp;
};

//===----------------------------------------------------------------------===//
// EraseStmt

/// This statement represents the `erase` statement in PDLL. This statement
/// erases the given root operation, corresponding roughly to the
/// PatternRewriter::eraseOp API.
class EraseStmt final : public Node::NodeBase<EraseStmt, OpRewriteStmt> {
public:
  static EraseStmt *create(Context &ctx, SMRange loc, Expr *rootOp);

private:
  EraseStmt(SMRange loc, Expr *rootOp) : Base(loc, rootOp) {}
};

//===----------------------------------------------------------------------===//
// ReplaceStmt

/// This statement represents the `replace` statement in PDLL. This statement
/// replace the given root operation with a set of values, corresponding roughly
/// to the PatternRewriter::replaceOp API.
class ReplaceStmt final : public Node::NodeBase<ReplaceStmt, OpRewriteStmt>,
                          private llvm::TrailingObjects<ReplaceStmt, Expr *> {
public:
  static ReplaceStmt *create(Context &ctx, SMRange loc, Expr *rootOp,
                             ArrayRef<Expr *> replExprs);

  /// Return the replacement values of this statement.
  MutableArrayRef<Expr *> getReplExprs() {
    return {getTrailingObjects<Expr *>(), numReplExprs};
  }
  ArrayRef<Expr *> getReplExprs() const {
    return const_cast<ReplaceStmt *>(this)->getReplExprs();
  }

private:
  ReplaceStmt(SMRange loc, Expr *rootOp, unsigned numReplExprs)
      : Base(loc, rootOp), numReplExprs(numReplExprs) {}

  /// The number of replacement values within this statement.
  unsigned numReplExprs;

  /// TrailingObject utilities.
  friend class llvm::TrailingObjects<ReplaceStmt, Expr *>;
};

//===----------------------------------------------------------------------===//
// RewriteStmt

/// This statement represents an operation rewrite that contains a block of
/// nested rewrite commands. This allows for building more complex operation
/// rewrites that span across multiple statements, which may be unconnected.
class RewriteStmt final : public Node::NodeBase<RewriteStmt, OpRewriteStmt> {
public:
  static RewriteStmt *create(Context &ctx, SMRange loc, Expr *rootOp,
                             CompoundStmt *rewriteBody);

  /// Return the compound rewrite body.
  CompoundStmt *getRewriteBody() const { return rewriteBody; }

private:
  RewriteStmt(SMRange loc, Expr *rootOp, CompoundStmt *rewriteBody)
      : Base(loc, rootOp), rewriteBody(rewriteBody) {}

  /// The body of nested rewriters within this statement.
  CompoundStmt *rewriteBody;
};

//===----------------------------------------------------------------------===//
// ReturnStmt
//===----------------------------------------------------------------------===//

/// This statement represents a return from a "callable" like decl, e.g. a
/// Constraint or a Rewrite.
class ReturnStmt final : public Node::NodeBase<ReturnStmt, Stmt> {
public:
  static ReturnStmt *create(Context &ctx, SMRange loc, Expr *resultExpr);

  /// Return the result expression of this statement.
  Expr *getResultExpr() { return resultExpr; }
  const Expr *getResultExpr() const { return resultExpr; }

  /// Set the result expression of this statement.
  void setResultExpr(Expr *expr) { resultExpr = expr; }

private:
  ReturnStmt(SMRange loc, Expr *resultExpr)
      : Base(loc), resultExpr(resultExpr) {}

  // The result expression of this statement.
  Expr *resultExpr;
};

//===----------------------------------------------------------------------===//
// Expr
//===----------------------------------------------------------------------===//

/// This class represents a base AST Expression node.
class Expr : public Stmt {
public:
  /// Return the type of this expression.
  Type getType() const { return type; }

  /// Provide type casting support.
  static bool classof(const Node *node);

protected:
  Expr(TypeID typeID, SMRange loc, Type type) : Stmt(typeID, loc), type(type) {}

private:
  /// The type of this expression.
  Type type;
};

//===----------------------------------------------------------------------===//
// AttributeExpr
//===----------------------------------------------------------------------===//

/// This expression represents a literal MLIR Attribute, and contains the
/// textual assembly format of that attribute.
class AttributeExpr : public Node::NodeBase<AttributeExpr, Expr> {
public:
  static AttributeExpr *create(Context &ctx, SMRange loc, StringRef value);

  /// Get the raw value of this expression. This is the textual assembly format
  /// of the MLIR Attribute.
  StringRef getValue() const { return value; }

private:
  AttributeExpr(Context &ctx, SMRange loc, StringRef value)
      : Base(loc, AttributeType::get(ctx)), value(value) {}

  /// The value referenced by this expression.
  StringRef value;
};

//===----------------------------------------------------------------------===//
// CallExpr
//===----------------------------------------------------------------------===//

/// This expression represents a call to a decl, such as a
/// UserConstraintDecl/UserRewriteDecl.
class CallExpr final : public Node::NodeBase<CallExpr, Expr>,
                       private llvm::TrailingObjects<CallExpr, Expr *> {
public:
  static CallExpr *create(Context &ctx, SMRange loc, Expr *callable,
                          ArrayRef<Expr *> arguments, Type resultType);

  /// Return the callable of this call.
  Expr *getCallableExpr() const { return callable; }

  /// Return the arguments of this call.
  MutableArrayRef<Expr *> getArguments() {
    return {getTrailingObjects<Expr *>(), numArgs};
  }
  ArrayRef<Expr *> getArguments() const {
    return const_cast<CallExpr *>(this)->getArguments();
  }

private:
  CallExpr(SMRange loc, Type type, Expr *callable, unsigned numArgs)
      : Base(loc, type), callable(callable), numArgs(numArgs) {}

  /// The callable of this call.
  Expr *callable;

  /// The number of arguments of the call.
  unsigned numArgs;

  /// TrailingObject utilities.
  friend llvm::TrailingObjects<CallExpr, Expr *>;
};

//===----------------------------------------------------------------------===//
// DeclRefExpr
//===----------------------------------------------------------------------===//

/// This expression represents a reference to a Decl node.
class DeclRefExpr : public Node::NodeBase<DeclRefExpr, Expr> {
public:
  static DeclRefExpr *create(Context &ctx, SMRange loc, Decl *decl, Type type);

  /// Get the decl referenced by this expression.
  Decl *getDecl() const { return decl; }

private:
  DeclRefExpr(SMRange loc, Decl *decl, Type type)
      : Base(loc, type), decl(decl) {}

  /// The decl referenced by this expression.
  Decl *decl;
};

//===----------------------------------------------------------------------===//
// MemberAccessExpr
//===----------------------------------------------------------------------===//

/// This expression represents a named member or field access of a given parent
/// expression.
class MemberAccessExpr : public Node::NodeBase<MemberAccessExpr, Expr> {
public:
  static MemberAccessExpr *create(Context &ctx, SMRange loc,
                                  const Expr *parentExpr, StringRef memberName,
                                  Type type);

  /// Get the parent expression of this access.
  const Expr *getParentExpr() const { return parentExpr; }

  /// Return the name of the member being accessed.
  StringRef getMemberName() const { return memberName; }

private:
  MemberAccessExpr(SMRange loc, const Expr *parentExpr, StringRef memberName,
                   Type type)
      : Base(loc, type), parentExpr(parentExpr), memberName(memberName) {}

  /// The parent expression of this access.
  const Expr *parentExpr;

  /// The name of the member being accessed from the parent.
  StringRef memberName;
};

//===----------------------------------------------------------------------===//
// AllResultsMemberAccessExpr

/// This class represents an instance of MemberAccessExpr that references all
/// results of an operation.
class AllResultsMemberAccessExpr : public MemberAccessExpr {
public:
  /// Return the member name used for the "all-results" access.
  static StringRef getMemberName() { return "$results"; }

  static AllResultsMemberAccessExpr *create(Context &ctx, SMRange loc,
                                            const Expr *parentExpr, Type type) {
    return cast<AllResultsMemberAccessExpr>(
        MemberAccessExpr::create(ctx, loc, parentExpr, getMemberName(), type));
  }

  /// Provide type casting support.
  static bool classof(const Node *node) {
    const MemberAccessExpr *memAccess = dyn_cast<MemberAccessExpr>(node);
    return memAccess && memAccess->getMemberName() == getMemberName();
  }
};

//===----------------------------------------------------------------------===//
// OperationExpr
//===----------------------------------------------------------------------===//

/// This expression represents the structural form of an MLIR Operation. It
/// represents either an input operation to match, or an operation to create
/// within a rewrite.
class OperationExpr final
    : public Node::NodeBase<OperationExpr, Expr>,
      private llvm::TrailingObjects<OperationExpr, Expr *,
                                    NamedAttributeDecl *> {
public:
  static OperationExpr *create(Context &ctx, SMRange loc,
                               const ods::Operation *odsOp,
                               const OpNameDecl *nameDecl,
                               ArrayRef<Expr *> operands,
                               ArrayRef<Expr *> resultTypes,
                               ArrayRef<NamedAttributeDecl *> attributes);

  /// Return the name of the operation, or std::nullopt if there isn't one.
  std::optional<StringRef> getName() const;

  /// Return the declaration of the operation name.
  const OpNameDecl *getNameDecl() const { return nameDecl; }

  /// Return the location of the name of the operation expression, or an invalid
  /// location if there isn't a name.
  SMRange getNameLoc() const { return nameLoc; }

  /// Return the operands of this operation.
  MutableArrayRef<Expr *> getOperands() {
    return {getTrailingObjects<Expr *>(), numOperands};
  }
  ArrayRef<Expr *> getOperands() const {
    return const_cast<OperationExpr *>(this)->getOperands();
  }

  /// Return the result types of this operation.
  MutableArrayRef<Expr *> getResultTypes() {
    return {getTrailingObjects<Expr *>() + numOperands, numResultTypes};
  }
  MutableArrayRef<Expr *> getResultTypes() const {
    return const_cast<OperationExpr *>(this)->getResultTypes();
  }

  /// Return the attributes of this operation.
  MutableArrayRef<NamedAttributeDecl *> getAttributes() {
    return {getTrailingObjects<NamedAttributeDecl *>(), numAttributes};
  }
  MutableArrayRef<NamedAttributeDecl *> getAttributes() const {
    return const_cast<OperationExpr *>(this)->getAttributes();
  }

private:
  OperationExpr(SMRange loc, Type type, const OpNameDecl *nameDecl,
                unsigned numOperands, unsigned numResultTypes,
                unsigned numAttributes, SMRange nameLoc)
      : Base(loc, type), nameDecl(nameDecl), numOperands(numOperands),
        numResultTypes(numResultTypes), numAttributes(numAttributes),
        nameLoc(nameLoc) {}

  /// The name decl of this expression.
  const OpNameDecl *nameDecl;

  /// The number of operands, result types, and attributes of the operation.
  unsigned numOperands, numResultTypes, numAttributes;

  /// The location of the operation name in the expression if it has a name.
  SMRange nameLoc;

  /// TrailingObject utilities.
  friend llvm::TrailingObjects<OperationExpr, Expr *, NamedAttributeDecl *>;
  size_t numTrailingObjects(OverloadToken<Expr *>) const {
    return numOperands + numResultTypes;
  }
};

//===----------------------------------------------------------------------===//
// RangeExpr
//===----------------------------------------------------------------------===//

/// This expression builds a range from a set of element values (which may be
/// ranges themselves).
class RangeExpr final : public Node::NodeBase<RangeExpr, Expr>,
                        private llvm::TrailingObjects<RangeExpr, Expr *> {
public:
  static RangeExpr *create(Context &ctx, SMRange loc, ArrayRef<Expr *> elements,
                           RangeType type);

  /// Return the element expressions of this range.
  MutableArrayRef<Expr *> getElements() {
    return {getTrailingObjects<Expr *>(), numElements};
  }
  ArrayRef<Expr *> getElements() const {
    return const_cast<RangeExpr *>(this)->getElements();
  }

  /// Return the range result type of this expression.
  RangeType getType() const { return Base::getType().cast<RangeType>(); }

private:
  RangeExpr(SMRange loc, RangeType type, unsigned numElements)
      : Base(loc, type), numElements(numElements) {}

  /// The number of element values for this range.
  unsigned numElements;

  /// TrailingObject utilities.
  friend class llvm::TrailingObjects<RangeExpr, Expr *>;
};

//===----------------------------------------------------------------------===//
// TupleExpr
//===----------------------------------------------------------------------===//

/// This expression builds a tuple from a set of element values.
class TupleExpr final : public Node::NodeBase<TupleExpr, Expr>,
                        private llvm::TrailingObjects<TupleExpr, Expr *> {
public:
  static TupleExpr *create(Context &ctx, SMRange loc, ArrayRef<Expr *> elements,
                           ArrayRef<StringRef> elementNames);

  /// Return the element expressions of this tuple.
  MutableArrayRef<Expr *> getElements() {
    return {getTrailingObjects<Expr *>(), getType().size()};
  }
  ArrayRef<Expr *> getElements() const {
    return const_cast<TupleExpr *>(this)->getElements();
  }

  /// Return the tuple result type of this expression.
  TupleType getType() const { return Base::getType().cast<TupleType>(); }

private:
  TupleExpr(SMRange loc, TupleType type) : Base(loc, type) {}

  /// TrailingObject utilities.
  friend class llvm::TrailingObjects<TupleExpr, Expr *>;
};

//===----------------------------------------------------------------------===//
// TypeExpr
//===----------------------------------------------------------------------===//

/// This expression represents a literal MLIR Type, and contains the textual
/// assembly format of that type.
class TypeExpr : public Node::NodeBase<TypeExpr, Expr> {
public:
  static TypeExpr *create(Context &ctx, SMRange loc, StringRef value);

  /// Get the raw value of this expression. This is the textual assembly format
  /// of the MLIR Type.
  StringRef getValue() const { return value; }

private:
  TypeExpr(Context &ctx, SMRange loc, StringRef value)
      : Base(loc, TypeType::get(ctx)), value(value) {}

  /// The value referenced by this expression.
  StringRef value;
};

//===----------------------------------------------------------------------===//
// Decl
//===----------------------------------------------------------------------===//

/// This class represents the base Decl node.
class Decl : public Node {
public:
  /// Return the name of the decl, or nullptr if it doesn't have one.
  const Name *getName() const { return name; }

  /// Provide type casting support.
  static bool classof(const Node *node);

  /// Set the documentation comment for this decl.
  void setDocComment(Context &ctx, StringRef comment);

  /// Return the documentation comment attached to this decl if it has been set.
  /// Otherwise, returns std::nullopt.
  std::optional<StringRef> getDocComment() const { return docComment; }

protected:
  Decl(TypeID typeID, SMRange loc, const Name *name = nullptr)
      : Node(typeID, loc), name(name) {}

private:
  /// The name of the decl. This is optional for some decls, such as
  /// PatternDecl.
  const Name *name;

  /// The documentation comment attached to this decl. Defaults to std::nullopt
  /// if the comment is unset/unknown.
  std::optional<StringRef> docComment;
};

//===----------------------------------------------------------------------===//
// ConstraintDecl
//===----------------------------------------------------------------------===//

/// This class represents the base of all AST Constraint decls. Constraints
/// apply matcher conditions to, and define the type of PDLL variables.
class ConstraintDecl : public Decl {
public:
  /// Provide type casting support.
  static bool classof(const Node *node);

protected:
  ConstraintDecl(TypeID typeID, SMRange loc, const Name *name = nullptr)
      : Decl(typeID, loc, name) {}
};

/// This class represents a reference to a constraint, and contains a constraint
/// and the location of the reference.
struct ConstraintRef {
  ConstraintRef(const ConstraintDecl *constraint, SMRange refLoc)
      : constraint(constraint), referenceLoc(refLoc) {}
  explicit ConstraintRef(const ConstraintDecl *constraint)
      : ConstraintRef(constraint, constraint->getLoc()) {}

  const ConstraintDecl *constraint;
  SMRange referenceLoc;
};

//===----------------------------------------------------------------------===//
// CoreConstraintDecl
//===----------------------------------------------------------------------===//

/// This class represents the base of all "core" constraints. Core constraints
/// are those that generally represent a concrete IR construct, such as
/// `Type`s or `Value`s.
class CoreConstraintDecl : public ConstraintDecl {
public:
  /// Provide type casting support.
  static bool classof(const Node *node);

protected:
  CoreConstraintDecl(TypeID typeID, SMRange loc, const Name *name = nullptr)
      : ConstraintDecl(typeID, loc, name) {}
};

//===----------------------------------------------------------------------===//
// AttrConstraintDecl

/// The class represents an Attribute constraint, and constrains a variable to
/// be an Attribute.
class AttrConstraintDecl
    : public Node::NodeBase<AttrConstraintDecl, CoreConstraintDecl> {
public:
  static AttrConstraintDecl *create(Context &ctx, SMRange loc,
                                    Expr *typeExpr = nullptr);

  /// Return the optional type the attribute is constrained to.
  Expr *getTypeExpr() { return typeExpr; }
  const Expr *getTypeExpr() const { return typeExpr; }

protected:
  AttrConstraintDecl(SMRange loc, Expr *typeExpr)
      : Base(loc), typeExpr(typeExpr) {}

  /// An optional type that the attribute is constrained to.
  Expr *typeExpr;
};

//===----------------------------------------------------------------------===//
// OpConstraintDecl

/// The class represents an Operation constraint, and constrains a variable to
/// be an Operation.
class OpConstraintDecl
    : public Node::NodeBase<OpConstraintDecl, CoreConstraintDecl> {
public:
  static OpConstraintDecl *create(Context &ctx, SMRange loc,
                                  const OpNameDecl *nameDecl = nullptr);

  /// Return the name of the operation, or std::nullopt if there isn't one.
  std::optional<StringRef> getName() const;

  /// Return the declaration of the operation name.
  const OpNameDecl *getNameDecl() const { return nameDecl; }

protected:
  explicit OpConstraintDecl(SMRange loc, const OpNameDecl *nameDecl)
      : Base(loc), nameDecl(nameDecl) {}

  /// The operation name of this constraint.
  const OpNameDecl *nameDecl;
};

//===----------------------------------------------------------------------===//
// TypeConstraintDecl

/// The class represents a Type constraint, and constrains a variable to be a
/// Type.
class TypeConstraintDecl
    : public Node::NodeBase<TypeConstraintDecl, CoreConstraintDecl> {
public:
  static TypeConstraintDecl *create(Context &ctx, SMRange loc);

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// TypeRangeConstraintDecl

/// The class represents a TypeRange constraint, and constrains a variable to be
/// a TypeRange.
class TypeRangeConstraintDecl
    : public Node::NodeBase<TypeRangeConstraintDecl, CoreConstraintDecl> {
public:
  static TypeRangeConstraintDecl *create(Context &ctx, SMRange loc);

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// ValueConstraintDecl

/// The class represents a Value constraint, and constrains a variable to be a
/// Value.
class ValueConstraintDecl
    : public Node::NodeBase<ValueConstraintDecl, CoreConstraintDecl> {
public:
  static ValueConstraintDecl *create(Context &ctx, SMRange loc, Expr *typeExpr);

  /// Return the optional type the value is constrained to.
  Expr *getTypeExpr() { return typeExpr; }
  const Expr *getTypeExpr() const { return typeExpr; }

protected:
  ValueConstraintDecl(SMRange loc, Expr *typeExpr)
      : Base(loc), typeExpr(typeExpr) {}

  /// An optional type that the value is constrained to.
  Expr *typeExpr;
};

//===----------------------------------------------------------------------===//
// ValueRangeConstraintDecl

/// The class represents a ValueRange constraint, and constrains a variable to
/// be a ValueRange.
class ValueRangeConstraintDecl
    : public Node::NodeBase<ValueRangeConstraintDecl, CoreConstraintDecl> {
public:
  static ValueRangeConstraintDecl *create(Context &ctx, SMRange loc,
                                          Expr *typeExpr = nullptr);

  /// Return the optional type the value range is constrained to.
  Expr *getTypeExpr() { return typeExpr; }
  const Expr *getTypeExpr() const { return typeExpr; }

protected:
  ValueRangeConstraintDecl(SMRange loc, Expr *typeExpr)
      : Base(loc), typeExpr(typeExpr) {}

  /// An optional type that the value range is constrained to.
  Expr *typeExpr;
};

//===----------------------------------------------------------------------===//
// UserConstraintDecl
//===----------------------------------------------------------------------===//

/// This decl represents a user defined constraint. This is either:
///   * an imported native constraint
///     - Similar to an external function declaration. This is a native
///       constraint defined externally, and imported into PDLL via a
///       declaration.
///   * a native constraint defined in PDLL
///     - This is a native constraint, i.e. a constraint whose implementation is
///       defined in C++(or potentially some other non-PDLL language). The
///       implementation of this constraint is specified as a string code block
///       in PDLL.
///   * a PDLL constraint
///     - This is a constraint which is defined using only PDLL constructs.
class UserConstraintDecl final
    : public Node::NodeBase<UserConstraintDecl, ConstraintDecl>,
      llvm::TrailingObjects<UserConstraintDecl, VariableDecl *, StringRef> {
public:
  /// Create a native constraint with the given optional code block.
  static UserConstraintDecl *
  createNative(Context &ctx, const Name &name, ArrayRef<VariableDecl *> inputs,
               ArrayRef<VariableDecl *> results,
               std::optional<StringRef> codeBlock, Type resultType,
               ArrayRef<StringRef> nativeInputTypes = {}) {
    return createImpl(ctx, name, inputs, nativeInputTypes, results, codeBlock,
                      /*body=*/nullptr, resultType);
  }

  /// Create a PDLL constraint with the given body.
  static UserConstraintDecl *createPDLL(Context &ctx, const Name &name,
                                        ArrayRef<VariableDecl *> inputs,
                                        ArrayRef<VariableDecl *> results,
                                        const CompoundStmt *body,
                                        Type resultType) {
    return createImpl(ctx, name, inputs, /*nativeInputTypes=*/std::nullopt,
                      results, /*codeBlock=*/std::nullopt, body, resultType);
  }

  /// Return the name of the constraint.
  const Name &getName() const { return *Decl::getName(); }

  /// Return the input arguments of this constraint.
  MutableArrayRef<VariableDecl *> getInputs() {
    return {getTrailingObjects<VariableDecl *>(), numInputs};
  }
  ArrayRef<VariableDecl *> getInputs() const {
    return const_cast<UserConstraintDecl *>(this)->getInputs();
  }

  /// Return the explicit native type to use for the given input. Returns
  /// std::nullopt if no explicit type was set.
  std::optional<StringRef> getNativeInputType(unsigned index) const;

  /// Return the explicit results of the constraint declaration. May be empty,
  /// even if the constraint has results (e.g. in the case of inferred results).
  MutableArrayRef<VariableDecl *> getResults() {
    return {getTrailingObjects<VariableDecl *>() + numInputs, numResults};
  }
  ArrayRef<VariableDecl *> getResults() const {
    return const_cast<UserConstraintDecl *>(this)->getResults();
  }

  /// Return the optional code block of this constraint, if this is a native
  /// constraint with a provided implementation.
  std::optional<StringRef> getCodeBlock() const { return codeBlock; }

  /// Return the body of this constraint if this constraint is a PDLL
  /// constraint, otherwise returns nullptr.
  const CompoundStmt *getBody() const { return constraintBody; }

  /// Return the result type of this constraint.
  Type getResultType() const { return resultType; }

  /// Returns true if this constraint is external.
  bool isExternal() const { return !constraintBody && !codeBlock; }

private:
  /// Create either a PDLL constraint or a native constraint with the given
  /// components.
  static UserConstraintDecl *createImpl(Context &ctx, const Name &name,
                                        ArrayRef<VariableDecl *> inputs,
                                        ArrayRef<StringRef> nativeInputTypes,
                                        ArrayRef<VariableDecl *> results,
                                        std::optional<StringRef> codeBlock,
                                        const CompoundStmt *body,
                                        Type resultType);

  UserConstraintDecl(const Name &name, unsigned numInputs,
                     bool hasNativeInputTypes, unsigned numResults,
                     std::optional<StringRef> codeBlock,
                     const CompoundStmt *body, Type resultType)
      : Base(name.getLoc(), &name), numInputs(numInputs),
        numResults(numResults), codeBlock(codeBlock), constraintBody(body),
        resultType(resultType), hasNativeInputTypes(hasNativeInputTypes) {}

  /// The number of inputs to this constraint.
  unsigned numInputs;

  /// The number of explicit results to this constraint.
  unsigned numResults;

  /// The optional code block of this constraint.
  std::optional<StringRef> codeBlock;

  /// The optional body of this constraint.
  const CompoundStmt *constraintBody;

  /// The result type of the constraint.
  Type resultType;

  /// Flag indicating if this constraint has explicit native input types.
  bool hasNativeInputTypes;

  /// Allow access to various internals.
  friend llvm::TrailingObjects<UserConstraintDecl, VariableDecl *, StringRef>;
  size_t numTrailingObjects(OverloadToken<VariableDecl *>) const {
    return numInputs + numResults;
  }
};

//===----------------------------------------------------------------------===//
// NamedAttributeDecl
//===----------------------------------------------------------------------===//

/// This Decl represents a NamedAttribute, and contains a string name and
/// attribute value.
class NamedAttributeDecl : public Node::NodeBase<NamedAttributeDecl, Decl> {
public:
  static NamedAttributeDecl *create(Context &ctx, const Name &name,
                                    Expr *value);

  /// Return the name of the attribute.
  const Name &getName() const { return *Decl::getName(); }

  /// Return value of the attribute.
  Expr *getValue() const { return value; }

private:
  NamedAttributeDecl(const Name &name, Expr *value)
      : Base(name.getLoc(), &name), value(value) {}

  /// The value of the attribute.
  Expr *value;
};

//===----------------------------------------------------------------------===//
// OpNameDecl
//===----------------------------------------------------------------------===//

/// This Decl represents an OperationName.
class OpNameDecl : public Node::NodeBase<OpNameDecl, Decl> {
public:
  static OpNameDecl *create(Context &ctx, const Name &name);
  static OpNameDecl *create(Context &ctx, SMRange loc);

  /// Return the name of this operation, or none if the name is unknown.
  std::optional<StringRef> getName() const {
    const Name *name = Decl::getName();
    return name ? std::optional<StringRef>(name->getName()) : std::nullopt;
  }

private:
  explicit OpNameDecl(const Name &name) : Base(name.getLoc(), &name) {}
  explicit OpNameDecl(SMRange loc) : Base(loc) {}
};

//===----------------------------------------------------------------------===//
// PatternDecl
//===----------------------------------------------------------------------===//

/// This Decl represents a single Pattern.
class PatternDecl : public Node::NodeBase<PatternDecl, Decl> {
public:
  static PatternDecl *create(Context &ctx, SMRange location, const Name *name,
                             std::optional<uint16_t> benefit,
                             bool hasBoundedRecursion,
                             const CompoundStmt *body);

  /// Return the benefit of this pattern if specified, or std::nullopt.
  std::optional<uint16_t> getBenefit() const { return benefit; }

  /// Return if this pattern has bounded rewrite recursion.
  bool hasBoundedRewriteRecursion() const { return hasBoundedRecursion; }

  /// Return the body of this pattern.
  const CompoundStmt *getBody() const { return patternBody; }

  /// Return the root rewrite statement of this pattern.
  const OpRewriteStmt *getRootRewriteStmt() const {
    return cast<OpRewriteStmt>(patternBody->getChildren().back());
  }

private:
  PatternDecl(SMRange loc, const Name *name, std::optional<uint16_t> benefit,
              bool hasBoundedRecursion, const CompoundStmt *body)
      : Base(loc, name), benefit(benefit),
        hasBoundedRecursion(hasBoundedRecursion), patternBody(body) {}

  /// The benefit of the pattern if it was explicitly specified, None otherwise.
  std::optional<uint16_t> benefit;

  /// If the pattern has properly bounded rewrite recursion or not.
  bool hasBoundedRecursion;

  /// The compound statement representing the body of the pattern.
  const CompoundStmt *patternBody;
};

//===----------------------------------------------------------------------===//
// UserRewriteDecl
//===----------------------------------------------------------------------===//

/// This decl represents a user defined rewrite. This is either:
///   * an imported native rewrite
///     - Similar to an external function declaration. This is a native
///       rewrite defined externally, and imported into PDLL via a declaration.
///   * a native rewrite defined in PDLL
///     - This is a native rewrite, i.e. a rewrite whose implementation is
///       defined in C++(or potentially some other non-PDLL language). The
///       implementation of this rewrite is specified as a string code block
///       in PDLL.
///   * a PDLL rewrite
///     - This is a rewrite which is defined using only PDLL constructs.
class UserRewriteDecl final
    : public Node::NodeBase<UserRewriteDecl, Decl>,
      llvm::TrailingObjects<UserRewriteDecl, VariableDecl *> {
public:
  /// Create a native rewrite with the given optional code block.
  static UserRewriteDecl *createNative(Context &ctx, const Name &name,
                                       ArrayRef<VariableDecl *> inputs,
                                       ArrayRef<VariableDecl *> results,
                                       std::optional<StringRef> codeBlock,
                                       Type resultType) {
    return createImpl(ctx, name, inputs, results, codeBlock, /*body=*/nullptr,
                      resultType);
  }

  /// Create a PDLL rewrite with the given body.
  static UserRewriteDecl *createPDLL(Context &ctx, const Name &name,
                                     ArrayRef<VariableDecl *> inputs,
                                     ArrayRef<VariableDecl *> results,
                                     const CompoundStmt *body,
                                     Type resultType) {
    return createImpl(ctx, name, inputs, results, /*codeBlock=*/std::nullopt,
                      body, resultType);
  }

  /// Return the name of the rewrite.
  const Name &getName() const { return *Decl::getName(); }

  /// Return the input arguments of this rewrite.
  MutableArrayRef<VariableDecl *> getInputs() {
    return {getTrailingObjects<VariableDecl *>(), numInputs};
  }
  ArrayRef<VariableDecl *> getInputs() const {
    return const_cast<UserRewriteDecl *>(this)->getInputs();
  }

  /// Return the explicit results of the rewrite declaration. May be empty,
  /// even if the rewrite has results (e.g. in the case of inferred results).
  MutableArrayRef<VariableDecl *> getResults() {
    return {getTrailingObjects<VariableDecl *>() + numInputs, numResults};
  }
  ArrayRef<VariableDecl *> getResults() const {
    return const_cast<UserRewriteDecl *>(this)->getResults();
  }

  /// Return the optional code block of this rewrite, if this is a native
  /// rewrite with a provided implementation.
  std::optional<StringRef> getCodeBlock() const { return codeBlock; }

  /// Return the body of this rewrite if this rewrite is a PDLL rewrite,
  /// otherwise returns nullptr.
  const CompoundStmt *getBody() const { return rewriteBody; }

  /// Return the result type of this rewrite.
  Type getResultType() const { return resultType; }

  /// Returns true if this rewrite is external.
  bool isExternal() const { return !rewriteBody && !codeBlock; }

private:
  /// Create either a PDLL rewrite or a native rewrite with the given
  /// components.
  static UserRewriteDecl *createImpl(Context &ctx, const Name &name,
                                     ArrayRef<VariableDecl *> inputs,
                                     ArrayRef<VariableDecl *> results,
                                     std::optional<StringRef> codeBlock,
                                     const CompoundStmt *body, Type resultType);

  UserRewriteDecl(const Name &name, unsigned numInputs, unsigned numResults,
                  std::optional<StringRef> codeBlock, const CompoundStmt *body,
                  Type resultType)
      : Base(name.getLoc(), &name), numInputs(numInputs),
        numResults(numResults), codeBlock(codeBlock), rewriteBody(body),
        resultType(resultType) {}

  /// The number of inputs to this rewrite.
  unsigned numInputs;

  /// The number of explicit results to this rewrite.
  unsigned numResults;

  /// The optional code block of this rewrite.
  std::optional<StringRef> codeBlock;

  /// The optional body of this rewrite.
  const CompoundStmt *rewriteBody;

  /// The result type of the rewrite.
  Type resultType;

  /// Allow access to various internals.
  friend llvm::TrailingObjects<UserRewriteDecl, VariableDecl *>;
};

//===----------------------------------------------------------------------===//
// CallableDecl
//===----------------------------------------------------------------------===//

/// This decl represents a shared interface for all callable decls.
class CallableDecl : public Decl {
public:
  /// Return the callable type of this decl.
  StringRef getCallableType() const {
    if (isa<UserConstraintDecl>(this))
      return "constraint";
    assert(isa<UserRewriteDecl>(this) && "unknown callable type");
    return "rewrite";
  }

  /// Return the inputs of this decl.
  ArrayRef<VariableDecl *> getInputs() const {
    if (const auto *cst = dyn_cast<UserConstraintDecl>(this))
      return cst->getInputs();
    return cast<UserRewriteDecl>(this)->getInputs();
  }

  /// Return the result type of this decl.
  Type getResultType() const {
    if (const auto *cst = dyn_cast<UserConstraintDecl>(this))
      return cst->getResultType();
    return cast<UserRewriteDecl>(this)->getResultType();
  }

  /// Return the explicit results of the declaration. Note that these may be
  /// empty, even if the callable has results (e.g. in the case of inferred
  /// results).
  ArrayRef<VariableDecl *> getResults() const {
    if (const auto *cst = dyn_cast<UserConstraintDecl>(this))
      return cst->getResults();
    return cast<UserRewriteDecl>(this)->getResults();
  }

  /// Return the optional code block of this callable, if this is a native
  /// callable with a provided implementation.
  std::optional<StringRef> getCodeBlock() const {
    if (const auto *cst = dyn_cast<UserConstraintDecl>(this))
      return cst->getCodeBlock();
    return cast<UserRewriteDecl>(this)->getCodeBlock();
  }

  /// Support LLVM type casting facilities.
  static bool classof(const Node *decl) {
    return isa<UserConstraintDecl, UserRewriteDecl>(decl);
  }
};

//===----------------------------------------------------------------------===//
// VariableDecl
//===----------------------------------------------------------------------===//

/// This Decl represents the definition of a PDLL variable.
class VariableDecl final
    : public Node::NodeBase<VariableDecl, Decl>,
      private llvm::TrailingObjects<VariableDecl, ConstraintRef> {
public:
  static VariableDecl *create(Context &ctx, const Name &name, Type type,
                              Expr *initExpr,
                              ArrayRef<ConstraintRef> constraints);

  /// Return the constraints of this variable.
  MutableArrayRef<ConstraintRef> getConstraints() {
    return {getTrailingObjects<ConstraintRef>(), numConstraints};
  }
  ArrayRef<ConstraintRef> getConstraints() const {
    return const_cast<VariableDecl *>(this)->getConstraints();
  }

  /// Return the initializer expression of this statement, or nullptr if there
  /// was no initializer.
  Expr *getInitExpr() const { return initExpr; }

  /// Return the name of the decl.
  const Name &getName() const { return *Decl::getName(); }

  /// Return the type of the decl.
  Type getType() const { return type; }

private:
  VariableDecl(const Name &name, Type type, Expr *initExpr,
               unsigned numConstraints)
      : Base(name.getLoc(), &name), type(type), initExpr(initExpr),
        numConstraints(numConstraints) {}

  /// The type of the variable.
  Type type;

  /// The optional initializer expression of this statement.
  Expr *initExpr;

  /// The number of constraints attached to this variable.
  unsigned numConstraints;

  /// Allow access to various internals.
  friend llvm::TrailingObjects<VariableDecl, ConstraintRef>;
};

//===----------------------------------------------------------------------===//
// Module
//===----------------------------------------------------------------------===//

/// This class represents a top-level AST module.
class Module final : public Node::NodeBase<Module, Node>,
                     private llvm::TrailingObjects<Module, Decl *> {
public:
  static Module *create(Context &ctx, SMLoc loc, ArrayRef<Decl *> children);

  /// Return the children of this module.
  MutableArrayRef<Decl *> getChildren() {
    return {getTrailingObjects<Decl *>(), numChildren};
  }
  ArrayRef<Decl *> getChildren() const {
    return const_cast<Module *>(this)->getChildren();
  }

private:
  Module(SMLoc loc, unsigned numChildren)
      : Base(SMRange{loc, loc}), numChildren(numChildren) {}

  /// The number of decls held by this module.
  unsigned numChildren;

  /// Allow access to various internals.
  friend llvm::TrailingObjects<Module, Decl *>;
};

//===----------------------------------------------------------------------===//
// Defered Method Definitions
//===----------------------------------------------------------------------===//

inline bool Decl::classof(const Node *node) {
  return isa<ConstraintDecl, NamedAttributeDecl, OpNameDecl, PatternDecl,
             UserRewriteDecl, VariableDecl>(node);
}

inline bool ConstraintDecl::classof(const Node *node) {
  return isa<CoreConstraintDecl, UserConstraintDecl>(node);
}

inline bool CoreConstraintDecl::classof(const Node *node) {
  return isa<AttrConstraintDecl, OpConstraintDecl, TypeConstraintDecl,
             TypeRangeConstraintDecl, ValueConstraintDecl,
             ValueRangeConstraintDecl>(node);
}

inline bool Expr::classof(const Node *node) {
  return isa<AttributeExpr, CallExpr, DeclRefExpr, MemberAccessExpr,
             OperationExpr, RangeExpr, TupleExpr, TypeExpr>(node);
}

inline bool OpRewriteStmt::classof(const Node *node) {
  return isa<EraseStmt, ReplaceStmt, RewriteStmt>(node);
}

inline bool Stmt::classof(const Node *node) {
  return isa<CompoundStmt, LetStmt, OpRewriteStmt, Expr>(node);
}

} // namespace ast
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_AST_NODES_H_
