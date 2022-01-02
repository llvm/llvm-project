//===- NodePrinter.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;
using namespace mlir::pdll::ast;

//===----------------------------------------------------------------------===//
// NodePrinter
//===----------------------------------------------------------------------===//

namespace {
class NodePrinter {
public:
  NodePrinter(raw_ostream &os) : os(os) {}

  /// Print the given type to the stream.
  void print(Type type);

  /// Print the given node to the stream.
  void print(const Node *node);

private:
  /// Print a range containing children of a node.
  template <typename RangeT,
            std::enable_if_t<!std::is_convertible<RangeT, const Node *>::value>
                * = nullptr>
  void printChildren(RangeT &&range) {
    if (llvm::empty(range))
      return;

    // Print the first N-1 elements with a prefix of "|-".
    auto it = std::begin(range);
    for (unsigned i = 0, e = llvm::size(range) - 1; i < e; ++i, ++it)
      print(*it);

    // Print the last element.
    elementIndentStack.back() = true;
    print(*it);
  }
  template <typename RangeT, typename... OthersT,
            std::enable_if_t<std::is_convertible<RangeT, const Node *>::value>
                * = nullptr>
  void printChildren(RangeT &&range, OthersT &&...others) {
    printChildren(ArrayRef<const Node *>({range, others...}));
  }
  /// Print a range containing children of a node, nesting the children under
  /// the given label.
  template <typename RangeT>
  void printChildren(StringRef label, RangeT &&range) {
    if (llvm::empty(range))
      return;
    elementIndentStack.reserve(elementIndentStack.size() + 1);
    llvm::SaveAndRestore<bool> lastElement(elementIndentStack.back(), true);

    printIndent();
    os << label << "`\n";
    elementIndentStack.push_back(/*isLastElt*/ false);
    printChildren(std::forward<RangeT>(range));
    elementIndentStack.pop_back();
  }

  /// Print the given derived node to the stream.
  void printImpl(const CompoundStmt *stmt);
  void printImpl(const EraseStmt *stmt);
  void printImpl(const LetStmt *stmt);
  void printImpl(const ReplaceStmt *stmt);
  void printImpl(const ReturnStmt *stmt);
  void printImpl(const RewriteStmt *stmt);

  void printImpl(const AttributeExpr *expr);
  void printImpl(const CallExpr *expr);
  void printImpl(const DeclRefExpr *expr);
  void printImpl(const MemberAccessExpr *expr);
  void printImpl(const OperationExpr *expr);
  void printImpl(const TupleExpr *expr);
  void printImpl(const TypeExpr *expr);

  void printImpl(const AttrConstraintDecl *decl);
  void printImpl(const OpConstraintDecl *decl);
  void printImpl(const TypeConstraintDecl *decl);
  void printImpl(const TypeRangeConstraintDecl *decl);
  void printImpl(const UserConstraintDecl *decl);
  void printImpl(const ValueConstraintDecl *decl);
  void printImpl(const ValueRangeConstraintDecl *decl);
  void printImpl(const NamedAttributeDecl *decl);
  void printImpl(const OpNameDecl *decl);
  void printImpl(const PatternDecl *decl);
  void printImpl(const UserRewriteDecl *decl);
  void printImpl(const VariableDecl *decl);
  void printImpl(const Module *module);

  /// Print the current indent stack.
  void printIndent() {
    if (elementIndentStack.empty())
      return;

    for (bool isLastElt : llvm::makeArrayRef(elementIndentStack).drop_back())
      os << (isLastElt ? "  " : " |");
    os << (elementIndentStack.back() ? " `" : " |");
  }

  /// The raw output stream.
  raw_ostream &os;

  /// A stack of indents and a flag indicating if the current element being
  /// printed at that indent is the last element.
  SmallVector<bool> elementIndentStack;
};
} // namespace

void NodePrinter::print(Type type) {
  // Protect against invalid inputs.
  if (!type) {
    os << "Type<NULL>";
    return;
  }

  TypeSwitch<Type>(type)
      .Case([&](AttributeType) { os << "Attr"; })
      .Case([&](ConstraintType) { os << "Constraint"; })
      .Case([&](OperationType type) {
        os << "Op";
        if (Optional<StringRef> name = type.getName())
          os << "<" << *name << ">";
      })
      .Case([&](RangeType type) {
        print(type.getElementType());
        os << "Range";
      })
      .Case([&](RewriteType) { os << "Rewrite"; })
      .Case([&](TupleType type) {
        os << "Tuple<";
        llvm::interleaveComma(
            llvm::zip(type.getElementNames(), type.getElementTypes()), os,
            [&](auto it) {
              if (!std::get<0>(it).empty())
                os << std::get<0>(it) << ": ";
              this->print(std::get<1>(it));
            });
        os << ">";
      })
      .Case([&](TypeType) { os << "Type"; })
      .Case([&](ValueType) { os << "Value"; })
      .Default([](Type) { llvm_unreachable("unknown AST type"); });
}

void NodePrinter::print(const Node *node) {
  printIndent();
  os << "-";

  elementIndentStack.push_back(/*isLastElt*/ false);
  TypeSwitch<const Node *>(node)
      .Case<
          // Statements.
          const CompoundStmt, const EraseStmt, const LetStmt, const ReplaceStmt,
          const ReturnStmt, const RewriteStmt,

          // Expressions.
          const AttributeExpr, const CallExpr, const DeclRefExpr,
          const MemberAccessExpr, const OperationExpr, const TupleExpr,
          const TypeExpr,

          // Decls.
          const AttrConstraintDecl, const OpConstraintDecl,
          const TypeConstraintDecl, const TypeRangeConstraintDecl,
          const UserConstraintDecl, const ValueConstraintDecl,
          const ValueRangeConstraintDecl, const NamedAttributeDecl,
          const OpNameDecl, const PatternDecl, const UserRewriteDecl,
          const VariableDecl,

          const Module>([&](auto derivedNode) { this->printImpl(derivedNode); })
      .Default([](const Node *) { llvm_unreachable("unknown AST node"); });
  elementIndentStack.pop_back();
}

void NodePrinter::printImpl(const CompoundStmt *stmt) {
  os << "CompoundStmt " << stmt << "\n";
  printChildren(stmt->getChildren());
}

void NodePrinter::printImpl(const EraseStmt *stmt) {
  os << "EraseStmt " << stmt << "\n";
  printChildren(stmt->getRootOpExpr());
}

void NodePrinter::printImpl(const LetStmt *stmt) {
  os << "LetStmt " << stmt << "\n";
  printChildren(stmt->getVarDecl());
}

void NodePrinter::printImpl(const ReplaceStmt *stmt) {
  os << "ReplaceStmt " << stmt << "\n";
  printChildren(stmt->getRootOpExpr());
  printChildren("ReplValues", stmt->getReplExprs());
}

void NodePrinter::printImpl(const ReturnStmt *stmt) {
  os << "ReturnStmt " << stmt << "\n";
  printChildren(stmt->getResultExpr());
}

void NodePrinter::printImpl(const RewriteStmt *stmt) {
  os << "RewriteStmt " << stmt << "\n";
  printChildren(stmt->getRootOpExpr(), stmt->getRewriteBody());
}

void NodePrinter::printImpl(const AttributeExpr *expr) {
  os << "AttributeExpr " << expr << " Value<\"" << expr->getValue() << "\">\n";
}

void NodePrinter::printImpl(const CallExpr *expr) {
  os << "CallExpr " << expr << " Type<";
  print(expr->getType());
  os << ">\n";
  printChildren(expr->getCallableExpr());
  printChildren("Arguments", expr->getArguments());
}

void NodePrinter::printImpl(const DeclRefExpr *expr) {
  os << "DeclRefExpr " << expr << " Type<";
  print(expr->getType());
  os << ">\n";
  printChildren(expr->getDecl());
}

void NodePrinter::printImpl(const MemberAccessExpr *expr) {
  os << "MemberAccessExpr " << expr << " Member<" << expr->getMemberName()
     << "> Type<";
  print(expr->getType());
  os << ">\n";
  printChildren(expr->getParentExpr());
}

void NodePrinter::printImpl(const OperationExpr *expr) {
  os << "OperationExpr " << expr << " Type<";
  print(expr->getType());
  os << ">\n";

  printChildren(expr->getNameDecl());
  printChildren("Operands", expr->getOperands());
  printChildren("Result Types", expr->getResultTypes());
  printChildren("Attributes", expr->getAttributes());
}

void NodePrinter::printImpl(const TupleExpr *expr) {
  os << "TupleExpr " << expr << " Type<";
  print(expr->getType());
  os << ">\n";

  printChildren(expr->getElements());
}

void NodePrinter::printImpl(const TypeExpr *expr) {
  os << "TypeExpr " << expr << " Value<\"" << expr->getValue() << "\">\n";
}

void NodePrinter::printImpl(const AttrConstraintDecl *decl) {
  os << "AttrConstraintDecl " << decl << "\n";
  if (const auto *typeExpr = decl->getTypeExpr())
    printChildren(typeExpr);
}

void NodePrinter::printImpl(const OpConstraintDecl *decl) {
  os << "OpConstraintDecl " << decl << "\n";
  printChildren(decl->getNameDecl());
}

void NodePrinter::printImpl(const TypeConstraintDecl *decl) {
  os << "TypeConstraintDecl " << decl << "\n";
}

void NodePrinter::printImpl(const TypeRangeConstraintDecl *decl) {
  os << "TypeRangeConstraintDecl " << decl << "\n";
}

void NodePrinter::printImpl(const UserConstraintDecl *decl) {
  os << "UserConstraintDecl " << decl << " Name<" << decl->getName().getName()
     << "> ResultType<" << decl->getResultType() << ">";
  if (Optional<StringRef> codeBlock = decl->getCodeBlock()) {
    os << " Code<";
    llvm::printEscapedString(*codeBlock, os);
    os << ">";
  }
  os << "\n";
  printChildren("Inputs", decl->getInputs());
  printChildren("Results", decl->getResults());
  if (const CompoundStmt *body = decl->getBody())
    printChildren(body);
}

void NodePrinter::printImpl(const ValueConstraintDecl *decl) {
  os << "ValueConstraintDecl " << decl << "\n";
  if (const auto *typeExpr = decl->getTypeExpr())
    printChildren(typeExpr);
}

void NodePrinter::printImpl(const ValueRangeConstraintDecl *decl) {
  os << "ValueRangeConstraintDecl " << decl << "\n";
  if (const auto *typeExpr = decl->getTypeExpr())
    printChildren(typeExpr);
}

void NodePrinter::printImpl(const NamedAttributeDecl *decl) {
  os << "NamedAttributeDecl " << decl << " Name<" << decl->getName().getName()
     << ">\n";
  printChildren(decl->getValue());
}

void NodePrinter::printImpl(const OpNameDecl *decl) {
  os << "OpNameDecl " << decl;
  if (Optional<StringRef> name = decl->getName())
    os << " Name<" << name << ">";
  os << "\n";
}

void NodePrinter::printImpl(const PatternDecl *decl) {
  os << "PatternDecl " << decl;
  if (const Name *name = decl->getName())
    os << " Name<" << name->getName() << ">";
  if (Optional<uint16_t> benefit = decl->getBenefit())
    os << " Benefit<" << *benefit << ">";
  if (decl->hasBoundedRewriteRecursion())
    os << " Recursion";

  os << "\n";
  printChildren(decl->getBody());
}

void NodePrinter::printImpl(const UserRewriteDecl *decl) {
  os << "UserRewriteDecl " << decl << " Name<" << decl->getName().getName()
     << "> ResultType<" << decl->getResultType() << ">";
  if (Optional<StringRef> codeBlock = decl->getCodeBlock()) {
    os << " Code<";
    llvm::printEscapedString(*codeBlock, os);
    os << ">";
  }
  os << "\n";
  printChildren("Inputs", decl->getInputs());
  printChildren("Results", decl->getResults());
  if (const CompoundStmt *body = decl->getBody())
    printChildren(body);
}

void NodePrinter::printImpl(const VariableDecl *decl) {
  os << "VariableDecl " << decl << " Name<" << decl->getName().getName()
     << "> Type<";
  print(decl->getType());
  os << ">\n";
  if (Expr *initExpr = decl->getInitExpr())
    printChildren(initExpr);

  auto constraints =
      llvm::map_range(decl->getConstraints(),
                      [](const ConstraintRef &ref) { return ref.constraint; });
  printChildren("Constraints", constraints);
}

void NodePrinter::printImpl(const Module *module) {
  os << "Module " << module << "\n";
  printChildren(module->getChildren());
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

void Node::print(raw_ostream &os) const { NodePrinter(os).print(this); }

void Type::print(raw_ostream &os) const { NodePrinter(os).print(*this); }
