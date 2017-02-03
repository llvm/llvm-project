//===-- GoParserTest.cpp ------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread>
// when
// exceptions are disabled.
#include <eh.h>
#endif

#include <sstream>

#include "gtest/gtest.h"

#include "Plugins/ExpressionParser/Go/GoParser.h"
#include "lldb/Utility/Error.h"

using namespace lldb_private;

namespace {
struct ASTPrinter {
  ASTPrinter(GoASTNode *n) { (*this)(n); }

  void operator()(GoASTNode *n) {
    if (n == nullptr) {
      m_stream << "nil ";
      return;
    }
    m_stream << "(" << n->GetKindName() << " ";
    n->WalkChildren(*this);
    if (auto *nn = llvm::dyn_cast<GoASTAssignStmt>(n))
      m_stream << nn->GetDefine() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTBasicLit>(n))
      m_stream << nn->GetValue().m_value.str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTBinaryExpr>(n))
      m_stream << GoLexer::LookupToken(nn->GetOp()).str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTIdent>(n))
      m_stream << nn->GetName().m_value.str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTBranchStmt>(n))
      m_stream << GoLexer::LookupToken(nn->GetTok()).str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTCallExpr>(n))
      m_stream << (nn->GetEllipsis() ? "..." : "") << " ";
    if (auto *nn = llvm::dyn_cast<GoASTChanType>(n))
      m_stream << nn->GetDir() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTGenDecl>(n))
      m_stream << GoLexer::LookupToken(nn->GetTok()).str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTIncDecStmt>(n))
      m_stream << GoLexer::LookupToken(nn->GetTok()).str() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTRangeStmt>(n))
      m_stream << nn->GetDefine() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTSliceExpr>(n))
      m_stream << nn->GetSlice3() << " ";
    if (auto *nn = llvm::dyn_cast<GoASTUnaryExpr>(n))
      m_stream << GoLexer::LookupToken(nn->GetOp()).str() << " ";
    m_stream << ") ";
  }

  const std::string str() const { return m_stream.str(); }
  std::stringstream m_stream;
};

testing::AssertionResult CheckStatement(const char *_s, const char *c_expr,
                                        const char *sexpr, const char *code) {
  GoParser parser(code);
  std::unique_ptr<GoASTStmt> stmt(parser.Statement());
  if (parser.Failed() || !stmt) {
    Error err;
    parser.GetError(err);
    return testing::AssertionFailure() << "Error parsing " << c_expr << "\n\t"
                                       << err.AsCString();
  }
  std::string actual_sexpr = ASTPrinter(stmt.get()).str();
  if (actual_sexpr == sexpr)
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << "Parsing: " << c_expr
                                     << "\nExpected: " << sexpr
                                     << "\nGot:      " << actual_sexpr;
}
} // namespace

#define EXPECT_PARSE(s, c) EXPECT_PRED_FORMAT2(CheckStatement, s, c)

TEST(GoParserTest, ParseBasicLiterals) {
  EXPECT_PARSE("(ExprStmt (BasicLit 0 ) ) ", "0");
  EXPECT_PARSE("(ExprStmt (BasicLit 42 ) ) ", "42");
  EXPECT_PARSE("(ExprStmt (BasicLit 0600 ) ) ", "0600");
  EXPECT_PARSE("(ExprStmt (BasicLit 0xBadFace ) ) ", "0xBadFace");
  EXPECT_PARSE(
      "(ExprStmt (BasicLit 170141183460469231731687303715884105727 ) ) ",
      "170141183460469231731687303715884105727");

  EXPECT_PARSE("(ExprStmt (BasicLit 0. ) ) ", "0.");
  EXPECT_PARSE("(ExprStmt (BasicLit 72.40 ) ) ", "72.40");
  EXPECT_PARSE("(ExprStmt (BasicLit 072.40 ) ) ", "072.40");
  EXPECT_PARSE("(ExprStmt (BasicLit 2.71828 ) ) ", "2.71828");
  EXPECT_PARSE("(ExprStmt (BasicLit 1.e+0 ) ) ", "1.e+0");
  EXPECT_PARSE("(ExprStmt (BasicLit 6.67428e-11 ) ) ", "6.67428e-11");
  EXPECT_PARSE("(ExprStmt (BasicLit 1E6 ) ) ", "1E6");
  EXPECT_PARSE("(ExprStmt (BasicLit .12345E+6 ) ) ", ".12345E+6");

  EXPECT_PARSE("(ExprStmt (BasicLit 0i ) ) ", "0i");
  EXPECT_PARSE("(ExprStmt (BasicLit 011i ) ) ", "011i");
  EXPECT_PARSE("(ExprStmt (BasicLit 0.i ) ) ", "0.i");
  EXPECT_PARSE("(ExprStmt (BasicLit 2.71828i ) ) ", "2.71828i");
  EXPECT_PARSE("(ExprStmt (BasicLit 6.67428e-11i ) ) ", "6.67428e-11i");
  EXPECT_PARSE("(ExprStmt (BasicLit 1E6i ) ) ", "1E6i");
  EXPECT_PARSE("(ExprStmt (BasicLit .12345E+6i ) ) ", ".12345E+6i");

  EXPECT_PARSE("(ExprStmt (BasicLit 'a' ) ) ", "'a'");
  EXPECT_PARSE("(ExprStmt (BasicLit '本' ) ) ", "'本'");
  EXPECT_PARSE("(ExprStmt (BasicLit \"abc\" ) ) ", "\"abc\"");
  EXPECT_PARSE("(ExprStmt (BasicLit `abc` ) ) ", "`abc`");
  EXPECT_PARSE("(ExprStmt (BasicLit `ab\nc` ) ) ", "`ab\nc`");
}

TEST(GoParserTest, ParseOperand) {
  EXPECT_PARSE("(ExprStmt (Ident a ) ) ", "a");
  EXPECT_PARSE("(ExprStmt (Ident _x9 ) ) ", "_x9");
  EXPECT_PARSE("(ExprStmt (Ident ThisVariableIsExported ) ) ",
               "ThisVariableIsExported");
  EXPECT_PARSE("(ExprStmt (Ident αβ ) ) ", "αβ");

  EXPECT_PARSE("(ExprStmt (SelectorExpr (Ident math ) (Ident Sin ) ) ) ",
               "math.Sin");
}

TEST(GoParserTest, ParseCompositeLiterals) {
  EXPECT_PARSE("(ExprStmt (CompositeLit (Ident Point3D ) ) ) ", "Point3D{}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (Ident Line ) (Ident origin ) "
               "(CompositeLit (Ident Point3D ) (KeyValueExpr "
               "(Ident y ) (UnaryExpr (BasicLit 4 ) - ) ) (KeyValueExpr (Ident "
               "z ) (BasicLit 12.3 ) ) ) ) ) ",
               "Line{origin, Point3D{y: -4, z: 12.3}}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType (BasicLit 10 ) (Ident "
               "string ) ) ) ) ",
               "[10]string{}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType (BasicLit 6 ) (Ident int ) "
               ") (BasicLit 1 ) (BasicLit 2 ) "
               "(BasicLit 3 ) (BasicLit 5 ) ) ) ",
               "[6]int {1, 2, 3, 5}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType nil (Ident int ) ) "
               "(BasicLit 2 ) (BasicLit 3 ) (BasicLit 5 ) "
               "(BasicLit 7 ) (BasicLit 9 ) (BasicLit 2147483647 ) ) ) ",
               "[]int{2, 3, 5, 7, 9, 2147483647}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType (BasicLit 128 ) (Ident bool "
               ") ) (KeyValueExpr (BasicLit 'a' ) "
               "(Ident true ) ) (KeyValueExpr (BasicLit 'e' ) (Ident true ) ) "
               "(KeyValueExpr (BasicLit 'i' ) (Ident "
               "true ) ) (KeyValueExpr (BasicLit 'o' ) (Ident true ) ) "
               "(KeyValueExpr (BasicLit 'u' ) (Ident true ) ) "
               "(KeyValueExpr (BasicLit 'y' ) (Ident true ) ) ) ) ",
               "[128]bool{'a': true, 'e': true, 'i': true, 'o': true, 'u': "
               "true, 'y': true}");
  EXPECT_PARSE(
      "(ExprStmt (CompositeLit (ArrayType (BasicLit 10 ) (Ident float32 ) ) "
      "(UnaryExpr (BasicLit 1 ) - ) "
      "(KeyValueExpr (BasicLit 4 ) (UnaryExpr (BasicLit 0.1 ) - ) ) (UnaryExpr "
      "(BasicLit 0.1 ) - ) "
      "(KeyValueExpr (BasicLit 9 ) (UnaryExpr (BasicLit 1 ) - ) ) ) ) ",
      "[10]float32{-1, 4: -0.1, -0.1, 9: -1}");
}

TEST(GoParserTest, ParseEllipsisArray) {
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType (Ellipsis nil ) (Ident "
               "string ) ) (BasicLit `Sat` ) (BasicLit `Sun` ) ) ) ",
               "[...]string {`Sat`, `Sun`}");
  EXPECT_PARSE("(ExprStmt (CompositeLit (ArrayType (Ellipsis nil ) (Ident "
               "Point ) ) (CompositeLit nil (BasicLit 1.5 "
               ") (UnaryExpr (BasicLit 3.5 ) - ) ) (CompositeLit nil (BasicLit "
               "0 ) (BasicLit 0 ) ) ) ) ",
               "[...]Point{{1.5, -3.5}, {0, 0}}");
}

TEST(GoParserTest, ParseMap) {
  EXPECT_PARSE("(ExprStmt (CompositeLit (MapType (Ident string ) (Ident "
               "float32 ) ) (KeyValueExpr (BasicLit `C0` ) "
               "(BasicLit 16.35 ) ) (KeyValueExpr (BasicLit `D0` ) (BasicLit "
               "18.35 ) ) ) ) ",
               "map[string]float32{`C0`: 16.35, `D0`: 18.35, }");
}

TEST(GoParserTest, UnaryExpr) {
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) + ) ) ", "+x");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) - ) ) ", "-x");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) ! ) ) ", "!x");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) ^ ) ) ", "^x");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) & ) ) ", "&x");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (Ident x ) <- ) ) ", "<-x");
  EXPECT_PARSE("(ExprStmt (StarExpr (Ident x ) ) ) ", "*x");
}

TEST(GoParserTest, BinaryExpr) {
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) || ) ) ", "a || b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) && ) ) ", "a && b");

  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) == ) ) ", "a == b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) != ) ) ", "a != b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) < ) ) ", "a < b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) <= ) ) ", "a <= b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) > ) ) ", "a > b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) >= ) ) ", "a >= b");

  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) + ) ) ", "a + b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) - ) ) ", "a - b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) | ) ) ", "a | b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) ^ ) ) ", "a ^ b");

  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) * ) ) ", "a * b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) / ) ) ", "a / b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) % ) ) ", "a % b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) << ) ) ", "a << b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) >> ) ) ", "a >> b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) & ) ) ", "a & b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (Ident b ) &^ ) ) ", "a &^ b");

  EXPECT_PARSE("(ExprStmt (BinaryExpr (BasicLit 23 ) (BinaryExpr (BasicLit 3 ) "
               "(IndexExpr (Ident x ) (Ident i ) ) * ) + ) ) ",
               "23 + 3*x[i]");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident a ) (UnaryExpr (UnaryExpr (Ident "
               "a ) + ) + ) + ) ) ",
               "a + + + a");
  EXPECT_PARSE(
      "(ExprStmt (BinaryExpr (UnaryExpr (Ident a ) ^ ) (Ident b ) >> ) ) ",
      "^a >> b");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (CallExpr (Ident f )  ) (CallExpr (Ident "
               "g )  ) || ) ) ",
               "f() || g()");
  EXPECT_PARSE(
      "(ExprStmt (BinaryExpr (BinaryExpr (Ident x ) (BinaryExpr (Ident y ) "
      "(BasicLit 1 ) + ) == ) "
      "(BinaryExpr (UnaryExpr (Ident chanPtr ) <- ) (BasicLit 0 ) > ) && ) ) ",
      "x == y+1 && <-chanPtr > 0");
}

TEST(GoParserTest, PrimaryExpr) {
  EXPECT_PARSE(
      "(ExprStmt (BinaryExpr (Ident x ) (CallExpr (Ident f )  ) <= ) ) ",
      "x <= f()");
  EXPECT_PARSE("(ExprStmt (BinaryExpr (Ident s ) (BasicLit `.txt` ) + ) ) ",
               "(s + `.txt`)");
  EXPECT_PARSE(
      "(ExprStmt (CallExpr (Ident f ) (BasicLit 3.1415 ) (Ident true )  ) ) ",
      "f(3.1415, true)");
  EXPECT_PARSE(
      "(ExprStmt (CallExpr (Ident f ) (BasicLit 3.1415 ) (Ident a ) ... ) ) ",
      "f(3.1415, a...)");
  EXPECT_PARSE("(ExprStmt (IndexExpr (Ident m ) (BasicLit '1' ) ) ) ",
               "m['1']");
  EXPECT_PARSE("(ExprStmt (SliceExpr (Ident s ) (Ident i ) (BinaryExpr (Ident "
               "j ) (BasicLit 1 ) + ) nil 0 ) ) ",
               "s[i : j + 1]");
  EXPECT_PARSE("(ExprStmt (SelectorExpr (Ident obj ) (Ident color ) ) ) ",
               "obj.color");
  EXPECT_PARSE("(ExprStmt (CallExpr (SelectorExpr (IndexExpr (SelectorExpr "
               "(Ident f ) (Ident p ) ) (Ident i ) ) "
               "(Ident x ) )  ) ) ",
               "f.p[i].x()");
}

TEST(GoParserTest, Conversions) {
  EXPECT_PARSE(
      "(ExprStmt (StarExpr (CallExpr (Ident Point ) (Ident p )  ) ) ) ",
      "*Point(p)");
  EXPECT_PARSE(
      "(ExprStmt (CallExpr (StarExpr (Ident Point ) ) (Ident p )  ) ) ",
      "(*Point)(p)");
  EXPECT_PARSE("(ExprStmt (UnaryExpr (CallExpr (ChanType (Ident int ) 0 ) "
               "(Ident c )  ) <- ) ) ",
               "<-chan int(c)");
  EXPECT_PARSE("(ExprStmt (TypeAssertExpr (Ident y ) (SelectorExpr (Ident io ) "
               "(Ident Reader ) ) ) ) ",
               "y.(io.Reader)");
}
