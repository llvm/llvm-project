//===-- PostfixExpression.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements support for postfix expressions found in several symbol
//  file formats, and their conversion to DWARF.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/PostfixExpression.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb_private;
using namespace lldb_private::postfix;

static llvm::Optional<BinaryOpNode::OpType>
GetBinaryOpType(llvm::StringRef token) {
  if (token.size() != 1)
    return llvm::None;
  switch (token[0]) {
  case '@':
    return BinaryOpNode::Align;
  case '-':
    return BinaryOpNode::Minus;
  case '+':
    return BinaryOpNode::Plus;
  }
  return llvm::None;
}

static llvm::Optional<UnaryOpNode::OpType>
GetUnaryOpType(llvm::StringRef token) {
  if (token == "^")
    return UnaryOpNode::Deref;
  return llvm::None;
}

Node *postfix::Parse(llvm::StringRef expr, llvm::BumpPtrAllocator &alloc) {
  llvm::SmallVector<Node *, 4> stack;

  llvm::StringRef token;
  while (std::tie(token, expr) = getToken(expr), !token.empty()) {
    if (auto op_type = GetBinaryOpType(token)) {
      // token is binary operator
      if (stack.size() < 2)
        return nullptr;

      Node *right = stack.pop_back_val();
      Node *left = stack.pop_back_val();
      stack.push_back(MakeNode<BinaryOpNode>(alloc, *op_type, *left, *right));
      continue;
    }

    if (auto op_type = GetUnaryOpType(token)) {
      // token is unary operator
      if (stack.empty())
        return nullptr;

      Node *operand = stack.pop_back_val();
      stack.push_back(MakeNode<UnaryOpNode>(alloc, *op_type, *operand));
      continue;
    }

    uint32_t value;
    if (to_integer(token, value, 10)) {
      // token is integer literal
      stack.push_back(MakeNode<IntegerNode>(alloc, value));
      continue;
    }

    stack.push_back(MakeNode<SymbolNode>(alloc, token));
  }

  if (stack.size() != 1)
    return nullptr;

  return stack.back();
}
