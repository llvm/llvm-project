//===-- PostfixExpressionTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/PostfixExpression.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
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
    return std::string(
        llvm::formatv("{0}({1}, {2})", ToString(binary.GetOpType()),
                      Dispatch(binary.Left()), Dispatch(binary.Right())));
  }

  std::string Visit(InitialValueNode &, Node *&) override { return "InitialValue"; }

  std::string Visit(IntegerNode &integer, Node *&) override {
    return std::string(llvm::formatv("int({0})", integer.GetValue()));
  }

  std::string Visit(RegisterNode &reg, Node *&) override {
    return std::string(llvm::formatv("reg({0})", reg.GetRegNum()));
  }

  std::string Visit(SymbolNode &symbol, Node *&) override {
    return std::string(symbol.GetName());
  }

  std::string Visit(UnaryOpNode &unary, Node *&) override {
    return std::string(llvm::formatv("{0}({1})", ToString(unary.GetOpType()),
                                     Dispatch(unary.Operand())));
  }

public:
  static std::string Print(Node *node) {
    if (node)
      return ASTPrinter().Dispatch(node);
    return "nullptr";
  }
};

static std::string ParseOneAndStringify(llvm::StringRef expr) {
  llvm::BumpPtrAllocator alloc;
  return ASTPrinter::Print(ParseOneExpression(expr, alloc));
}

TEST(PostfixExpression, ParseOneExpression) {
  EXPECT_EQ("int(47)", ParseOneAndStringify("47"));
  EXPECT_EQ("$foo", ParseOneAndStringify("$foo"));
  EXPECT_EQ("+(int(1), int(2))", ParseOneAndStringify("1 2 +"));
  EXPECT_EQ("-(int(1), int(2))", ParseOneAndStringify("1 2 -"));
  EXPECT_EQ("@(int(1), int(2))", ParseOneAndStringify("1 2 @"));
  EXPECT_EQ("+(int(1), +(int(2), int(3)))", ParseOneAndStringify("1 2 3 + +"));
  EXPECT_EQ("+(+(int(1), int(2)), int(3))", ParseOneAndStringify("1 2 + 3 +"));
  EXPECT_EQ("^(int(1))", ParseOneAndStringify("1 ^"));
  EXPECT_EQ("^(^(int(1)))", ParseOneAndStringify("1 ^ ^"));
  EXPECT_EQ("^(+(int(1), ^(int(2))))", ParseOneAndStringify("1 2 ^ + ^"));
  EXPECT_EQ("-($foo, int(47))", ParseOneAndStringify("$foo 47 -"));
  EXPECT_EQ("+(int(47), int(-42))", ParseOneAndStringify("47 -42 +"));

  EXPECT_EQ("nullptr", ParseOneAndStringify("+"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("^"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("1 +"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("1 2 ^"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("1 2 3 +"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("^ 1"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("+ 1 2"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("1 + 2"));
  EXPECT_EQ("nullptr", ParseOneAndStringify("1 2"));
  EXPECT_EQ("nullptr", ParseOneAndStringify(""));
}

static std::vector<std::pair<std::string, std::string>>
ParseFPOAndStringify(llvm::StringRef prog) {
  llvm::BumpPtrAllocator alloc;
  std::vector<std::pair<llvm::StringRef, Node *>> parsed =
      ParseFPOProgram(prog, alloc);
  std::vector<std::pair<std::string, std::string>> result;
  for (const auto &p : parsed)
    result.emplace_back(p.first.str(), ASTPrinter::Print(p.second));
  return result;
}

TEST(PostfixExpression, ParseFPOProgram) {
  EXPECT_THAT(ParseFPOAndStringify("a 1 ="),
              testing::ElementsAre(std::make_pair("a", "int(1)")));
  EXPECT_THAT(ParseFPOAndStringify("a 1 = b 2 3 + ="),
              testing::ElementsAre(std::make_pair("a", "int(1)"),
                                   std::make_pair("b", "+(int(2), int(3))")));

  EXPECT_THAT(ParseFPOAndStringify(""), testing::IsEmpty());
  EXPECT_THAT(ParseFPOAndStringify("="), testing::IsEmpty());
  EXPECT_THAT(ParseFPOAndStringify("a 1"), testing::IsEmpty());
  EXPECT_THAT(ParseFPOAndStringify("a 1 = ="), testing::IsEmpty());
  EXPECT_THAT(ParseFPOAndStringify("a 1 + ="), testing::IsEmpty());
  EXPECT_THAT(ParseFPOAndStringify("= a 1 ="), testing::IsEmpty());
}

static std::string ParseAndGenerateDWARF(llvm::StringRef expr) {
  llvm::BumpPtrAllocator alloc;
  Node *ast = ParseOneExpression(expr, alloc);
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
  llvm::DataExtractor extractor(dwarf.GetString(), /*IsLittleEndian=*/true,
                                addr_size);

  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::DWARFExpression(extractor, addr_size, llvm::dwarf::DWARF32)
      .print(os, llvm::DIDumpOptions(), nullptr);
  return result;
}

TEST(PostfixExpression, ToDWARF) {
  EXPECT_EQ("DW_OP_consts +0", ParseAndGenerateDWARF("0"));

  EXPECT_EQ("DW_OP_breg1 +0", ParseAndGenerateDWARF("R1"));

  EXPECT_EQ("DW_OP_bregx 0x41 +0", ParseAndGenerateDWARF("R65"));

  EXPECT_EQ("DW_OP_pick 0x0", ParseAndGenerateDWARF("INIT"));

  EXPECT_EQ("DW_OP_pick 0x0, DW_OP_pick 0x1, DW_OP_plus",
            ParseAndGenerateDWARF("INIT INIT +"));

  EXPECT_EQ("DW_OP_breg1 +0, DW_OP_pick 0x1, DW_OP_plus",
            ParseAndGenerateDWARF("R1 INIT +"));

  EXPECT_EQ("DW_OP_consts +1, DW_OP_pick 0x1, DW_OP_deref, DW_OP_plus",
            ParseAndGenerateDWARF("1 INIT ^ +"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_consts +5, DW_OP_plus",
            ParseAndGenerateDWARF("4 5 +"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_consts +5, DW_OP_minus",
            ParseAndGenerateDWARF("4 5 -"));

  EXPECT_EQ("DW_OP_consts +4, DW_OP_deref", ParseAndGenerateDWARF("4 ^"));

  EXPECT_EQ("DW_OP_breg6 +0, DW_OP_consts +128, DW_OP_lit1, DW_OP_minus, "
            "DW_OP_not, DW_OP_and",
            ParseAndGenerateDWARF("R6 128 @"));
}
