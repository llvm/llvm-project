//===-- PostfixExpressionTest.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/PostfixExpression.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"
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

  std::string Visit(InitialValueNode &, Node *&) override { return "InitialValue"; }

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
  EXPECT_EQ("+(int(47), int(-42))", ParseAndStringify("47 -42 +"));

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

static std::string ParseAndGenerateDWARF(llvm::StringRef expr) {
  llvm::BumpPtrAllocator alloc;
  Node *ast = Parse(expr, alloc);
  if (!ast)
    return "Parse failed.";
  if (!ResolveSymbols(ast, [&](SymbolNode &symbol) -> Node * {
        if (symbol.GetName() == "INIT")
          return MakeNode<InitialValueNode>(alloc);

        uint32_t num;
        if (to_integer(symbol.GetName().drop_front(), num))
          return MakeNode<RegisterNode>(alloc, num);
        return nullptr;
      })) {
    return "Resolution failed.";
  }

  const size_t addr_size = 4;
  StreamString dwarf(Stream::eBinary, addr_size, lldb::eByteOrderLittle);
  ToDWARF(*ast, dwarf);

  // print dwarf expression to comparable textual representation
  DataExtractor extractor(dwarf.GetData(), dwarf.GetSize(),
                          lldb::eByteOrderLittle, addr_size);

  StreamString result;
  if (!DWARFExpression::PrintDWARFExpression(result, extractor, addr_size,
                                             /*dwarf_ref_size*/ 4,
                                             /*location_expression*/ false)) {
    return "DWARF printing failed.";
  }

  return result.GetString();
}

TEST(PostfixExpression, ToDWARF) {
  EXPECT_EQ("DW_OP_consts +0", ParseAndGenerateDWARF("0"));

  EXPECT_EQ("DW_OP_breg1 +0", ParseAndGenerateDWARF("R1"));

  EXPECT_EQ("DW_OP_bregx 65 0", ParseAndGenerateDWARF("R65"));

  EXPECT_EQ("DW_OP_pick 0x00", ParseAndGenerateDWARF("INIT"));

  EXPECT_EQ("DW_OP_pick 0x00, DW_OP_pick 0x01, DW_OP_plus ",
            ParseAndGenerateDWARF("INIT INIT +"));

  EXPECT_EQ("DW_OP_breg1 +0, DW_OP_pick 0x01, DW_OP_plus ",
            ParseAndGenerateDWARF("R1 INIT +"));

  EXPECT_EQ("DW_OP_consts +1, DW_OP_pick 0x01, DW_OP_deref , DW_OP_plus ",
            ParseAndGenerateDWARF("1 INIT ^ +"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_consts +5, DW_OP_plus ",
            ParseAndGenerateDWARF("4 5 +"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_consts +5, DW_OP_minus ",
            ParseAndGenerateDWARF("4 5 -"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_deref ", ParseAndGenerateDWARF("4 ^"));

  EXPECT_EQ("DW_OP_breg6 +0, DW_OP_consts +128, DW_OP_lit1 "
            ", DW_OP_minus , DW_OP_not , DW_OP_and ",
            ParseAndGenerateDWARF("R6 128 @"));
}
