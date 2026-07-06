//===-- DILAST.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILAST.h"
#include "llvm/Support/ErrorHandling.h"

namespace lldb_private::dil {

BinaryOpKind GetBinaryOpKindFromToken(Token::Kind token_kind) {
  switch (token_kind) {
  case Token::equal:
    return BinaryOpKind::Assign;
  case Token::minus:
    return BinaryOpKind::Sub;
  case Token::minusequal:
    return BinaryOpKind::SubAssign;
  case Token::plus:
    return BinaryOpKind::Add;
  case Token::plusequal:
    return BinaryOpKind::AddAssign;
  case Token::star:
    return BinaryOpKind::Mul;
  case Token::slash:
    return BinaryOpKind::Div;
  case Token::percent:
    return BinaryOpKind::Rem;
  case Token::lessless:
    return BinaryOpKind::Shl;
  case Token::greatergreater:
    return BinaryOpKind::Shr;
  default:
    break;
  }
  llvm_unreachable("Unknown binary operator kind.");
}

llvm::Expected<lldb::ValueObjectSP> ErrorNode::Accept(Visitor *v) const {
  llvm_unreachable("Attempting to Visit a DIL ErrorNode.");
}

llvm::Expected<lldb::ValueObjectSP> IdentifierNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP> MemberOfNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP> UnaryOpNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP> BinaryOpNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP>
ArraySubscriptNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP>
BitFieldExtractionNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP>
IntegerLiteralNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP> FloatLiteralNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP>
BooleanLiteralNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

llvm::Expected<lldb::ValueObjectSP> CastNode::Accept(Visitor *v) const {
  return v->Visit(*this);
}

} // namespace lldb_private::dil
