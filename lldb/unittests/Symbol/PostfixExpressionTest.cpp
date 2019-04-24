//===-- PostfixExpressionTest.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/PostfixExpression.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::postfix;

static std::string ToString(BinaryOpNode::OpType type) {
  switch (type) {
  case BinaryOpNode::Align:
    return "@";
  case BinaryOpNode::Minus:
    return "-";
  case BinaryOpNode::Plus:
    return "+";
  }
  llvm_unreachable("Fully covered switch!");
}

static std::string ToString(UnaryOpNode::OpType type) {
  switch (type) {
  case UnaryOpNode::Deref:
    return "^";
  }
  llvm_unreachable("Fully covered switch!");
}

struct ASTPrinter : public Visitor<std::string> {
protected:
  std::string Visit(BinaryOpNode &binary, Node *&) override {
    return llvm::formatv("{0}({1}, {2})", ToString(binary.GetOpType()),
                         Dispatch(binary.Left()), Dispatch(binary.Right()));
  }

  std::string Visit(IntegerNode &integer, Node *&) override {
    return llvm::formatv("int({0})", integer.GetValue());
  }

  std::string Visit(RegisterNode &reg, Node *&) override {
    return llvm::formatv("reg({0})", reg.GetRegNum());
  }

  std::string Visit(SymbolNode &symbol, Node *&) override {
    return symbol.GetName();
  }

  std::string Visit(UnaryOpNode &unary, Node *&) override {
    return llvm::formatv("{0}({1})", ToString(unary.GetOpType()),
                         Dispatch(unary.Operand()));
  }

public:
  static std::string Print(Node *node) {
    if (node)
      return ASTPrinter().Dispatch(node);
    return "nullptr";
  }
};

static std::string ParseAndStringify(llvm::StringRef expr) {
  llvm::BumpPtrAllocator alloc;
  return ASTPrinter::Print(Parse(expr, alloc));
}

TEST(PostfixExpression, Parse) {
  EXPECT_EQ("int(47)", ParseAndStringify("47"));
  EXPECT_EQ("$foo", ParseAndStringify("$foo"));
  EXPECT_EQ("+(int(1), int(2))", ParseAndStringify("1 2 +"));
  EXPECT_EQ("-(int(1), int(2))", ParseAndStringify("1 2 -"));
  EXPECT_EQ("@(int(1), int(2))", ParseAndStringify("1 2 @"));
  EXPECT_EQ("+(int(1), +(int(2), int(3)))", ParseAndStringify("1 2 3 + +"));
  EXPECT_EQ("+(+(int(1), int(2)), int(3))", ParseAndStringify("1 2 + 3 +"));
  EXPECT_EQ("^(int(1))", ParseAndStringify("1 ^"));
  EXPECT_EQ("^(^(int(1)))", ParseAndStringify("1 ^ ^"));
  EXPECT_EQ("^(+(int(1), ^(int(2))))", ParseAndStringify("1 2 ^ + ^"));
  EXPECT_EQ("-($foo, int(47))", ParseAndStringify("$foo 47 -"));

  EXPECT_EQ("nullptr", ParseAndStringify("+"));
  EXPECT_EQ("nullptr", ParseAndStringify("^"));
  EXPECT_EQ("nullptr", ParseAndStringify("1 +"));
  EXPECT_EQ("nullptr", ParseAndStringify("1 2 ^"));
  EXPECT_EQ("nullptr", ParseAndStringify("1 2 3 +"));
  EXPECT_EQ("nullptr", ParseAndStringify("^ 1"));
  EXPECT_EQ("nullptr", ParseAndStringify("+ 1 2"));
  EXPECT_EQ("nullptr", ParseAndStringify("1 + 2"));
  EXPECT_EQ("nullptr", ParseAndStringify("1 2"));
  EXPECT_EQ("nullptr", ParseAndStringify(""));
}
