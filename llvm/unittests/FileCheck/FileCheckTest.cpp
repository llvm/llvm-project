//===- llvm/unittest/FileCheck/FileCheckTest.cpp - FileCheck tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FileCheck/FileCheck.h"
#include "../lib/FileCheck/FileCheckImpl.h"
#include "llvm/Support/Regex.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <tuple>
#include <unordered_set>

using namespace llvm;

namespace {

class FileCheckTest : public ::testing::Test {};

static StringRef bufferize(SourceMgr &SM, StringRef Str) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(Str, "TestBuffer");
  StringRef StrBufferRef = Buffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  return StrBufferRef;
}

static std::string toString(const std::unordered_set<std::string> &Set) {
  bool First = true;
  std::string Str;
  for (StringRef S : Set) {
    Str += Twine(First ? "{" + S : ", " + S).str();
    First = false;
  }
  Str += '}';
  return Str;
}

template <typename ErrorT>
static void expectSameErrors(std::unordered_set<std::string> ExpectedMsgs,
                             Error Err) {
  auto AnyErrorMsgMatch = [&ExpectedMsgs](std::string &&ErrorMsg) -> bool {
    for (auto ExpectedMsgItr = ExpectedMsgs.begin(),
              ExpectedMsgEnd = ExpectedMsgs.end();
         ExpectedMsgItr != ExpectedMsgEnd; ++ExpectedMsgItr) {
      if (ErrorMsg.find(*ExpectedMsgItr) != std::string::npos) {
        ExpectedMsgs.erase(ExpectedMsgItr);
        return true;
      }
    }
    return false;
  };

  Error RemainingErrors = std::move(Err);
  do {
    RemainingErrors =
        handleErrors(std::move(RemainingErrors), [&](const ErrorT &E) {
          EXPECT_TRUE(AnyErrorMsgMatch(E.message()))
              << "Unexpected error message:" << std::endl
              << E.message();
        });
  } while (RemainingErrors && !ExpectedMsgs.empty());
  EXPECT_THAT_ERROR(std::move(RemainingErrors), Succeeded());
  EXPECT_TRUE(ExpectedMsgs.empty())
      << "Error message(s) not found:" << std::endl
      << toString(ExpectedMsgs);
}

template <typename ErrorT>
static void expectError(StringRef ExpectedMsg, Error Err) {
  expectSameErrors<ErrorT>({ExpectedMsg.str()}, std::move(Err));
}

static void expectDiagnosticError(StringRef ExpectedMsg, Error Err) {
  expectError<ErrorDiagnostic>(ExpectedMsg, std::move(Err));
}

constexpr uint64_t MaxUint64 = std::numeric_limits<uint64_t>::max();
constexpr int64_t MaxInt64 = std::numeric_limits<int64_t>::max();
constexpr int64_t MinInt64 = std::numeric_limits<int64_t>::min();
constexpr uint64_t AbsoluteMinInt64 =
    static_cast<uint64_t>(-(MinInt64 + 1)) + 1;
constexpr uint64_t AbsoluteMaxInt64 = static_cast<uint64_t>(MaxInt64);
// Use 128 bitwidth for literals to keep one bit for sign for uint64_t
// literals.
constexpr unsigned LiteralsBitWidth = 128;

struct ExpressionFormatParameterisedFixture
    : public ::testing::TestWithParam<
          std::tuple<ExpressionFormat::Kind, unsigned, bool>> {
  bool AlternateForm;
  unsigned Precision;
  bool Signed;
  bool AllowHex;
  bool AllowUpperHex;
  ExpressionFormat Format;
  Regex WildcardRegex;

  StringRef TenStr;
  StringRef FifteenStr;
  std::string MaxUint64Str;
  std::string MaxInt64Str;
  std::string MinInt64Str;
  StringRef FirstInvalidCharDigits;
  StringRef AcceptedHexOnlyDigits;
  StringRef RefusedHexOnlyDigits;

  SourceMgr SM;

  void SetUp() override {
    ExpressionFormat::Kind Kind;
    std::tie(Kind, Precision, AlternateForm) = GetParam();
    AllowHex = Kind == ExpressionFormat::Kind::HexLower ||
               Kind == ExpressionFormat::Kind::HexUpper;
    AllowUpperHex = Kind == ExpressionFormat::Kind::HexUpper;
    Signed = Kind == ExpressionFormat::Kind::Signed;
    Format = ExpressionFormat(Kind, Precision, AlternateForm);

    if (!AllowHex) {
      MaxUint64Str = std::to_string(MaxUint64);
      MaxInt64Str = std::to_string(MaxInt64);
      MinInt64Str = std::to_string(MinInt64);
      TenStr = "10";
      FifteenStr = "15";
      FirstInvalidCharDigits = "aA";
      AcceptedHexOnlyDigits = RefusedHexOnlyDigits = "N/A";
      return;
    }

    MaxUint64Str = AllowUpperHex ? "FFFFFFFFFFFFFFFF" : "ffffffffffffffff";
    MaxInt64Str = AllowUpperHex ? "7FFFFFFFFFFFFFFF" : "7fffffffffffffff";
    TenStr = AllowUpperHex ? "A" : "a";
    FifteenStr = AllowUpperHex ? "F" : "f";
    AcceptedHexOnlyDigits = AllowUpperHex ? "ABCDEF" : "abcdef";
    RefusedHexOnlyDigits = AllowUpperHex ? "abcdef" : "ABCDEF";
    MinInt64Str = "N/A";
    FirstInvalidCharDigits = "gG";
  }

  void checkWildcardRegexMatch(StringRef Input,
                               unsigned TrailExtendTo = 0) const {
    ASSERT_TRUE(TrailExtendTo == 0 || AllowHex);
    SmallVector<StringRef, 4> Matches;
    std::string ExtendedInput = Input.str();
    size_t PrefixSize = AlternateForm ? 2 : 0;
    if (TrailExtendTo > Input.size() - PrefixSize) {
      size_t ExtensionSize = PrefixSize + TrailExtendTo - Input.size();
      ExtendedInput.append(ExtensionSize, Input[PrefixSize]);
    }
    ASSERT_TRUE(WildcardRegex.match(ExtendedInput, &Matches))
        << "Wildcard regex does not match " << ExtendedInput;
    EXPECT_EQ(Matches[0], ExtendedInput);
  }

  void checkWildcardRegexMatchFailure(StringRef Input) const {
    EXPECT_FALSE(WildcardRegex.match(Input));
  }

  std::string addBasePrefix(StringRef Num) const {
    StringRef Prefix = AlternateForm ? "0x" : "";
    return (Twine(Prefix) + Twine(Num)).str();
  }

  void checkPerCharWildcardRegexMatchFailure(StringRef Chars) const {
    for (auto C : Chars) {
      std::string Str = addBasePrefix(StringRef(&C, 1));
      EXPECT_FALSE(WildcardRegex.match(Str));
    }
  }

  std::string padWithLeadingZeros(StringRef NumStr) const {
    bool Negative = NumStr.startswith("-");
    if (NumStr.size() - unsigned(Negative) >= Precision)
      return NumStr.str();

    std::string PaddedStr;
    if (Negative) {
      PaddedStr = "-";
      NumStr = NumStr.drop_front();
    }
    PaddedStr.append(Precision - NumStr.size(), '0');
    PaddedStr.append(NumStr.str());
    return PaddedStr;
  }

  template <class T> void checkMatchingString(T Val, StringRef ExpectedStr) {
    APInt Value(LiteralsBitWidth, Val, std::is_signed_v<T>);
    Expected<std::string> MatchingString = Format.getMatchingString(Value);
    ASSERT_THAT_EXPECTED(MatchingString, Succeeded())
        << "No matching string for " << Val;
    EXPECT_EQ(*MatchingString, ExpectedStr);
  }

  template <class T> void checkMatchingStringFailure(T Val) {
    APInt Value(LiteralsBitWidth, Val, std::is_signed_v<T>);
    Expected<std::string> MatchingString = Format.getMatchingString(Value);
    // Error message tested in ExpressionFormat unit tests.
    EXPECT_THAT_EXPECTED(MatchingString, Failed());
  }

  template <class T>
  void checkValueFromStringRepr(StringRef Str, T ExpectedVal) {
    StringRef BufferizedStr = bufferize(SM, Str);
    APInt ResultValue = Format.valueFromStringRepr(BufferizedStr, SM);
    ASSERT_EQ(ResultValue.isNegative(), ExpectedVal < 0)
        << "Value for " << Str << " is not " << ExpectedVal;
    if (ResultValue.isNegative())
      EXPECT_EQ(ResultValue.getSExtValue(), static_cast<int64_t>(ExpectedVal));
    else
      EXPECT_EQ(ResultValue.getZExtValue(), static_cast<uint64_t>(ExpectedVal));
  }
};

TEST_P(ExpressionFormatParameterisedFixture, FormatGetWildcardRegex) {
  // Wildcard regex is valid.
  Expected<std::string> WildcardPattern = Format.getWildcardRegex();
  ASSERT_THAT_EXPECTED(WildcardPattern, Succeeded());
  WildcardRegex = Regex((Twine("^") + *WildcardPattern + "$").str());
  ASSERT_TRUE(WildcardRegex.isValid());

  // Does not match empty string.
  checkWildcardRegexMatchFailure("");

  // Matches all decimal digits, matches several of them and match 0x prefix
  // if and only if AlternateForm is true.
  StringRef LongNumber = "12345678901234567890";
  StringRef PrefixedLongNumber = "0x12345678901234567890";
  if (AlternateForm) {
    checkWildcardRegexMatch(PrefixedLongNumber);
    checkWildcardRegexMatchFailure(LongNumber);
  } else {
    checkWildcardRegexMatch(LongNumber);
    checkWildcardRegexMatchFailure(PrefixedLongNumber);
  }

  // Matches negative digits.
  LongNumber = "-12345678901234567890";
  if (Signed)
    checkWildcardRegexMatch(LongNumber);
  else
    checkWildcardRegexMatchFailure(LongNumber);

  // Check non digits or digits with wrong casing are not matched.
  std::string LongNumberStr;
  if (AllowHex) {
    LongNumberStr = addBasePrefix(AcceptedHexOnlyDigits);
    checkWildcardRegexMatch(LongNumberStr, 16);
    checkPerCharWildcardRegexMatchFailure(RefusedHexOnlyDigits);
  }
  checkPerCharWildcardRegexMatchFailure(FirstInvalidCharDigits);

  // Check leading zeros are only accepted if number of digits is less than the
  // precision.
  LongNumber = "01234567890123456789";
  if (Precision) {
    LongNumberStr = addBasePrefix(LongNumber.take_front(Precision));
    checkWildcardRegexMatch(LongNumberStr);
    LongNumberStr = addBasePrefix(LongNumber.take_front(Precision - 1));
    checkWildcardRegexMatchFailure(LongNumberStr);
    if (Precision < LongNumber.size()) {
      LongNumberStr = addBasePrefix(LongNumber.take_front(Precision + 1));
      checkWildcardRegexMatchFailure(LongNumberStr);
    }
  } else {
    LongNumberStr = addBasePrefix(LongNumber);
    checkWildcardRegexMatch(LongNumberStr);
  }
}

TEST_P(ExpressionFormatParameterisedFixture, FormatGetMatchingString) {
  checkMatchingString(0, addBasePrefix(padWithLeadingZeros("0")));
  checkMatchingString(9, addBasePrefix(padWithLeadingZeros("9")));

  if (Signed) {
    checkMatchingString(-5, padWithLeadingZeros("-5"));
    checkMatchingString(MaxUint64, padWithLeadingZeros(MaxUint64Str));
    checkMatchingString(MaxInt64, padWithLeadingZeros(MaxInt64Str));
    checkMatchingString(MinInt64, padWithLeadingZeros(MinInt64Str));
  } else {
    checkMatchingStringFailure(-5);
    checkMatchingString(MaxUint64,
                        addBasePrefix(padWithLeadingZeros(MaxUint64Str)));
    checkMatchingString(MaxInt64,
                        addBasePrefix(padWithLeadingZeros(MaxInt64Str)));
    checkMatchingStringFailure(MinInt64);
  }

  checkMatchingString(10, addBasePrefix(padWithLeadingZeros(TenStr)));
  checkMatchingString(15, addBasePrefix(padWithLeadingZeros(FifteenStr)));
}

TEST_P(ExpressionFormatParameterisedFixture, FormatValueFromStringRepr) {
  checkValueFromStringRepr(addBasePrefix("0"), 0);
  checkValueFromStringRepr(addBasePrefix("9"), 9);

  if (Signed) {
    checkValueFromStringRepr("-5", -5);
    checkValueFromStringRepr(MaxUint64Str, MaxUint64);
  } else {
    checkValueFromStringRepr(addBasePrefix(MaxUint64Str), MaxUint64);
  }

  checkValueFromStringRepr(addBasePrefix(TenStr), 10);
  checkValueFromStringRepr(addBasePrefix(FifteenStr), 15);

  // Wrong casing is not tested because valueFromStringRepr() relies on
  // StringRef's getAsInteger() which does not allow to restrict casing.

  // Likewise, wrong letter digit for hex value is not tested because it is
  // only caught by an assert in FileCheck due to getWildcardRegex()
  // guaranteeing only valid letter digits are used.
}

TEST_P(ExpressionFormatParameterisedFixture, FormatBoolOperator) {
  EXPECT_TRUE(bool(Format));
}

INSTANTIATE_TEST_SUITE_P(
    AllowedExplicitExpressionFormat, ExpressionFormatParameterisedFixture,
    ::testing::Values(
        std::make_tuple(ExpressionFormat::Kind::Unsigned, 0, false),
        std::make_tuple(ExpressionFormat::Kind::Signed, 0, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 0, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 0, true),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 0, false),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 0, true),

        std::make_tuple(ExpressionFormat::Kind::Unsigned, 1, false),
        std::make_tuple(ExpressionFormat::Kind::Signed, 1, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 1, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 1, true),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 1, false),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 1, true),

        std::make_tuple(ExpressionFormat::Kind::Unsigned, 16, false),
        std::make_tuple(ExpressionFormat::Kind::Signed, 16, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 16, false),
        std::make_tuple(ExpressionFormat::Kind::HexLower, 16, true),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 16, false),
        std::make_tuple(ExpressionFormat::Kind::HexUpper, 16, true),

        std::make_tuple(ExpressionFormat::Kind::Unsigned, 20, false),
        std::make_tuple(ExpressionFormat::Kind::Signed, 20, false)));

TEST_F(FileCheckTest, NoFormatProperties) {
  ExpressionFormat NoFormat(ExpressionFormat::Kind::NoFormat);
  expectError<StringError>("trying to match value with invalid format",
                           NoFormat.getWildcardRegex().takeError());
  expectError<StringError>(
      "trying to match value with invalid format",
      NoFormat.getMatchingString(APInt(64, 18u)).takeError());
  EXPECT_FALSE(bool(NoFormat));
}

TEST_F(FileCheckTest, FormatEqualityOperators) {
  ExpressionFormat UnsignedFormat(ExpressionFormat::Kind::Unsigned);
  ExpressionFormat UnsignedFormat2(ExpressionFormat::Kind::Unsigned);
  EXPECT_TRUE(UnsignedFormat == UnsignedFormat2);
  EXPECT_FALSE(UnsignedFormat != UnsignedFormat2);

  ExpressionFormat HexLowerFormat(ExpressionFormat::Kind::HexLower);
  EXPECT_FALSE(UnsignedFormat == HexLowerFormat);
  EXPECT_TRUE(UnsignedFormat != HexLowerFormat);

  ExpressionFormat NoFormat(ExpressionFormat::Kind::NoFormat);
  ExpressionFormat NoFormat2(ExpressionFormat::Kind::NoFormat);
  EXPECT_FALSE(NoFormat == NoFormat2);
  EXPECT_TRUE(NoFormat != NoFormat2);
}

TEST_F(FileCheckTest, FormatKindEqualityOperators) {
  ExpressionFormat UnsignedFormat(ExpressionFormat::Kind::Unsigned);
  EXPECT_TRUE(UnsignedFormat == ExpressionFormat::Kind::Unsigned);
  EXPECT_FALSE(UnsignedFormat != ExpressionFormat::Kind::Unsigned);
  EXPECT_FALSE(UnsignedFormat == ExpressionFormat::Kind::HexLower);
  EXPECT_TRUE(UnsignedFormat != ExpressionFormat::Kind::HexLower);
  ExpressionFormat NoFormat(ExpressionFormat::Kind::NoFormat);
  EXPECT_TRUE(NoFormat == ExpressionFormat::Kind::NoFormat);
  EXPECT_FALSE(NoFormat != ExpressionFormat::Kind::NoFormat);
}

static void expectOperationValueResult(binop_eval_t Operation, APInt LeftValue,
                                       APInt RightValue, APInt ExpectedValue) {
  bool Overflow;
  Expected<APInt> OperationResult = Operation(LeftValue, RightValue, Overflow);
  ASSERT_THAT_EXPECTED(OperationResult, Succeeded());
  EXPECT_EQ(*OperationResult, ExpectedValue);
}

template <class T1, class T2, class TR>
static void expectOperationValueResult(binop_eval_t Operation, T1 LeftValue,
                                       T2 RightValue, TR ResultValue) {
  APInt LeftVal(LiteralsBitWidth, LeftValue, std::is_signed_v<T1>);
  APInt RightVal(LiteralsBitWidth, RightValue, std::is_signed_v<T2>);
  APInt ResultVal;
  if constexpr (std::is_integral_v<TR>)
    ResultVal = APInt(LiteralsBitWidth, ResultValue, std::is_signed_v<TR>);
  else
    ResultVal = APInt(LiteralsBitWidth, ResultValue, /*Radix=*/10);
  expectOperationValueResult(Operation, LeftVal, RightVal, ResultVal);
}

template <class T1, class T2>
static void expectOperationValueResult(binop_eval_t Operation, T1 LeftValue,
                                       T2 RightValue) {
  bool Overflow;
  APInt LeftVal(LiteralsBitWidth, LeftValue, std::is_signed_v<T1>);
  APInt RightVal(LiteralsBitWidth, RightValue, std::is_signed_v<T2>);
  expectError<OverflowError>(
      "overflow error", Operation(LeftVal, RightVal, Overflow).takeError());
}

TEST_F(FileCheckTest, ExpressionValueAddition) {
  // Test both negative values.
  expectOperationValueResult(exprAdd, -10, -10, -20);

  // Test both negative values with underflow.
  expectOperationValueResult(exprAdd, MinInt64, -1, "-9223372036854775809");
  expectOperationValueResult(exprAdd, MinInt64, MinInt64,
                             "-18446744073709551616");

  // Test negative and positive value.
  expectOperationValueResult(exprAdd, -10, 10, 0);
  expectOperationValueResult(exprAdd, -10, 11, 1);
  expectOperationValueResult(exprAdd, -11, 10, -1);

  // Test positive and negative value.
  expectOperationValueResult(exprAdd, 10, -10, 0);
  expectOperationValueResult(exprAdd, 10, -11, -1);
  expectOperationValueResult(exprAdd, 11, -10, 1);

  // Test both positive values.
  expectOperationValueResult(exprAdd, 10, 10, 20);

  // Test both positive values with result not representable as uint64_t.
  expectOperationValueResult(exprAdd, MaxUint64, 1, "18446744073709551616");
  expectOperationValueResult(exprAdd, MaxUint64, MaxUint64,
                             "36893488147419103230");
}

TEST_F(FileCheckTest, ExpressionValueSubtraction) {
  // Test negative value and value bigger than int64_t max.
  expectOperationValueResult(exprSub, -10, MaxUint64, "-18446744073709551625");

  // Test negative and positive value with result not representable as int64_t.
  expectOperationValueResult(exprSub, MinInt64, 1, "-9223372036854775809");

  // Test negative and positive value.
  expectOperationValueResult(exprSub, -10, 10, -20);

  // Test both negative values.
  expectOperationValueResult(exprSub, -10, -10, 0);
  expectOperationValueResult(exprSub, -11, -10, -1);
  expectOperationValueResult(exprSub, -10, -11, 1);

  // Test positive and negative values.
  expectOperationValueResult(exprSub, 10, -10, 20);

  // Test both positive values with result positive.
  expectOperationValueResult(exprSub, 10, 5, 5);

  // Test both positive values with result not representable as int64_t.
  expectOperationValueResult(exprSub, 0, MaxUint64, "-18446744073709551615");
  expectOperationValueResult(exprSub, 0,
                             static_cast<uint64_t>(-(MinInt64 + 10)) + 11,
                             "-9223372036854775809");

  // Test both positive values with result < -(max int64_t)
  expectOperationValueResult(exprSub, 10, static_cast<uint64_t>(MaxInt64) + 11,
                             -MaxInt64 - 1);

  // Test both positive values with 0 > result > -(max int64_t)
  expectOperationValueResult(exprSub, 10, 11, -1);
}

TEST_F(FileCheckTest, ExpressionValueMultiplication) {
  // Test mixed signed values.
  expectOperationValueResult(exprMul, -3, 10, -30);
  expectOperationValueResult(exprMul, 2, -17, -34);
  expectOperationValueResult(exprMul, 0, MinInt64, 0);
  expectOperationValueResult(exprMul, MinInt64, 1, MinInt64);
  expectOperationValueResult(exprMul, 1, MinInt64, MinInt64);
  expectOperationValueResult(exprMul, MaxInt64, -1, -MaxInt64);
  expectOperationValueResult(exprMul, -1, MaxInt64, -MaxInt64);

  // Test both negative values.
  expectOperationValueResult(exprMul, -3, -10, 30);
  expectOperationValueResult(exprMul, -2, -17, 34);
  expectOperationValueResult(exprMul, MinInt64, -1, AbsoluteMinInt64);

  // Test both positive values.
  expectOperationValueResult(exprMul, 3, 10, 30);
  expectOperationValueResult(exprMul, 2, 17, 34);
  expectOperationValueResult(exprMul, 0, MaxUint64, 0);

  // Test negative results not representable as int64_t.
  expectOperationValueResult(exprMul, -10, MaxInt64, "-92233720368547758070");
  expectOperationValueResult(exprMul, MaxInt64, -10, "-92233720368547758070");
  expectOperationValueResult(exprMul, 10, MinInt64, "-92233720368547758080");
  expectOperationValueResult(exprMul, MinInt64, 10, "-92233720368547758080");
  expectOperationValueResult(exprMul, -1, MaxUint64, "-18446744073709551615");
  expectOperationValueResult(exprMul, MaxUint64, -1, "-18446744073709551615");
  expectOperationValueResult(exprMul, -1, AbsoluteMaxInt64 + 2,
                             "-9223372036854775809");
  expectOperationValueResult(exprMul, AbsoluteMaxInt64 + 2, -1,
                             "-9223372036854775809");

  // Test positive results not representable as uint64_t.
  expectOperationValueResult(exprMul, 10, MaxUint64, "184467440737095516150");
  expectOperationValueResult(exprMul, MaxUint64, 10, "184467440737095516150");
  expectOperationValueResult(exprMul, MinInt64, -10, "92233720368547758080");
  expectOperationValueResult(exprMul, -10, MinInt64, "92233720368547758080");
}

TEST_F(FileCheckTest, ExpressionValueDivision) {
  // Test mixed signed values.
  expectOperationValueResult(exprDiv, -30, 10, -3);
  expectOperationValueResult(exprDiv, 34, -17, -2);
  expectOperationValueResult(exprDiv, 0, -10, 0);
  expectOperationValueResult(exprDiv, MinInt64, 1, MinInt64);
  expectOperationValueResult(exprDiv, MaxInt64, -1, -MaxInt64);
  expectOperationValueResult(exprDiv, -MaxInt64, 1, -MaxInt64);

  // Test both negative values.
  expectOperationValueResult(exprDiv, -30, -10, 3);
  expectOperationValueResult(exprDiv, -34, -17, 2);

  // Test both positive values.
  expectOperationValueResult(exprDiv, 30, 10, 3);
  expectOperationValueResult(exprDiv, 34, 17, 2);
  expectOperationValueResult(exprDiv, 0, 10, 0);

  // Test divide by zero.
  expectOperationValueResult(exprDiv, -10, 0);
  expectOperationValueResult(exprDiv, 10, 0);
  expectOperationValueResult(exprDiv, 0, 0);

  // Test negative result not representable as int64_t.
  expectOperationValueResult(exprDiv, MaxUint64, -1, "-18446744073709551615");
  expectOperationValueResult(exprDiv, AbsoluteMaxInt64 + 2, -1,
                             "-9223372036854775809");
}

TEST_F(FileCheckTest, Literal) {
  SourceMgr SM;

  // Eval returns the literal's value.
  ExpressionLiteral Ten(bufferize(SM, "10"), APInt(64, 10u));
  Expected<APInt> Value = Ten.eval();
  ASSERT_THAT_EXPECTED(Value, Succeeded());
  EXPECT_EQ(10, Value->getSExtValue());
  Expected<ExpressionFormat> ImplicitFormat = Ten.getImplicitFormat(SM);
  ASSERT_THAT_EXPECTED(ImplicitFormat, Succeeded());
  EXPECT_EQ(*ImplicitFormat, ExpressionFormat::Kind::NoFormat);

  // Min value can be correctly represented.
  ExpressionLiteral Min(bufferize(SM, std::to_string(MinInt64)),
                        APInt(64, MinInt64, /*IsSigned=*/true));
  Value = Min.eval();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(MinInt64, Value->getSExtValue());

  // Max value can be correctly represented.
  ExpressionLiteral Max(bufferize(SM, std::to_string(MaxUint64)),
                        APInt(64, MaxUint64));
  Value = Max.eval();
  ASSERT_THAT_EXPECTED(Value, Succeeded());
  EXPECT_EQ(MaxUint64, Value->getZExtValue());
}

TEST_F(FileCheckTest, Expression) {
  SourceMgr SM;

  std::unique_ptr<ExpressionLiteral> Ten =
      std::make_unique<ExpressionLiteral>(bufferize(SM, "10"), APInt(64, 10u));
  ExpressionLiteral *TenPtr = Ten.get();
  Expression Expr(std::move(Ten),
                  ExpressionFormat(ExpressionFormat::Kind::HexLower));
  EXPECT_EQ(Expr.getAST(), TenPtr);
  EXPECT_EQ(Expr.getFormat(), ExpressionFormat::Kind::HexLower);
}

static void
expectUndefErrors(std::unordered_set<std::string> ExpectedUndefVarNames,
                  Error Err) {
  EXPECT_THAT_ERROR(handleErrors(std::move(Err),
                                 [&](const UndefVarError &E) {
                                   EXPECT_EQ(ExpectedUndefVarNames.erase(
                                                 std::string(E.getVarName())),
                                             1U);
                                 }),
                    Succeeded());
  EXPECT_TRUE(ExpectedUndefVarNames.empty()) << toString(ExpectedUndefVarNames);
}

TEST_F(FileCheckTest, NumericVariable) {
  SourceMgr SM;

  // Undefined variable: getValue and eval fail, error returned by eval holds
  // the name of the undefined variable.
  NumericVariable FooVar("FOO",
                         ExpressionFormat(ExpressionFormat::Kind::Unsigned), 1);
  EXPECT_EQ("FOO", FooVar.getName());
  EXPECT_EQ(FooVar.getImplicitFormat(), ExpressionFormat::Kind::Unsigned);
  NumericVariableUse FooVarUse("FOO", &FooVar);
  Expected<ExpressionFormat> ImplicitFormat = FooVarUse.getImplicitFormat(SM);
  ASSERT_THAT_EXPECTED(ImplicitFormat, Succeeded());
  EXPECT_EQ(*ImplicitFormat, ExpressionFormat::Kind::Unsigned);
  EXPECT_FALSE(FooVar.getValue());
  Expected<APInt> EvalResult = FooVarUse.eval();
  expectUndefErrors({"FOO"}, EvalResult.takeError());

  // Defined variable without string: only getValue and eval return value set.
  FooVar.setValue(APInt(64, 42u));
  std::optional<APInt> Value = FooVar.getValue();
  ASSERT_TRUE(Value);
  EXPECT_EQ(42, Value->getSExtValue());
  EXPECT_FALSE(FooVar.getStringValue());
  EvalResult = FooVarUse.eval();
  ASSERT_THAT_EXPECTED(EvalResult, Succeeded());
  EXPECT_EQ(42, EvalResult->getSExtValue());

  // Defined variable with string: getValue, eval, and getStringValue return
  // value set.
  StringRef StringValue = "925";
  FooVar.setValue(APInt(64, 925u), StringValue);
  Value = FooVar.getValue();
  ASSERT_TRUE(Value);
  EXPECT_EQ(925, Value->getSExtValue());
  // getStringValue should return the same memory not just the same characters.
  EXPECT_EQ(StringValue.begin(), FooVar.getStringValue()->begin());
  EXPECT_EQ(StringValue.end(), FooVar.getStringValue()->end());
  EvalResult = FooVarUse.eval();
  ASSERT_THAT_EXPECTED(EvalResult, Succeeded());
  EXPECT_EQ(925, EvalResult->getSExtValue());
  EXPECT_EQ(925, EvalResult->getSExtValue());

  // Clearing variable: getValue and eval fail. Error returned by eval holds
  // the name of the cleared variable.
  FooVar.clearValue();
  EXPECT_FALSE(FooVar.getValue());
  EXPECT_FALSE(FooVar.getStringValue());
  EvalResult = FooVarUse.eval();
  expectUndefErrors({"FOO"}, EvalResult.takeError());
}

TEST_F(FileCheckTest, Binop) {
  SourceMgr SM;

  StringRef ExprStr = bufferize(SM, "FOO+BAR");
  StringRef FooStr = ExprStr.take_front(3);
  NumericVariable FooVar(FooStr,
                         ExpressionFormat(ExpressionFormat::Kind::Unsigned), 1);
  FooVar.setValue(APInt(64, 42u));
  std::unique_ptr<NumericVariableUse> FooVarUse =
      std::make_unique<NumericVariableUse>(FooStr, &FooVar);
  StringRef BarStr = ExprStr.take_back(3);
  NumericVariable BarVar(BarStr,
                         ExpressionFormat(ExpressionFormat::Kind::Unsigned), 2);
  BarVar.setValue(APInt(64, 18u));
  std::unique_ptr<NumericVariableUse> BarVarUse =
      std::make_unique<NumericVariableUse>(BarStr, &BarVar);
  BinaryOperation Binop(ExprStr, exprAdd, std::move(FooVarUse),
                        std::move(BarVarUse));

  // Defined variables with same bitwidth and no overflow: eval returns right
  // value; implicit formas is as expected.
  Expected<APInt> Value = Binop.eval();
  ASSERT_THAT_EXPECTED(Value, Succeeded());
  EXPECT_EQ(60, Value->getSExtValue());
  Expected<ExpressionFormat> ImplicitFormat = Binop.getImplicitFormat(SM);
  ASSERT_THAT_EXPECTED(ImplicitFormat, Succeeded());
  EXPECT_EQ(*ImplicitFormat, ExpressionFormat::Kind::Unsigned);

  // Defined variables with different bitwidth and no overflow: eval succeeds
  // and return the right value.
  BarVar.setValue(APInt(32, 18u));
  Value = Binop.eval();
  ASSERT_THAT_EXPECTED(Value, Succeeded());
  EXPECT_EQ(60, Value->getSExtValue());

  // Defined variables with same bitwidth and wider result (i.e. overflow):
  // eval succeeds and return the right value in a wider APInt.
  BarVar.setValue(APInt(64, AbsoluteMaxInt64));
  Value = Binop.eval();
  ASSERT_THAT_EXPECTED(Value, Succeeded());
  EXPECT_EQ(128u, Value->getBitWidth());
  EXPECT_EQ(APInt(128, AbsoluteMaxInt64 + FooVar.getValue()->getZExtValue()),
            *Value);

  // 1 undefined variable: eval fails, error contains name of undefined
  // variable.
  FooVar.clearValue();
  Value = Binop.eval();
  expectUndefErrors({"FOO"}, Value.takeError());

  // 2 undefined variables: eval fails, error contains names of all undefined
  // variables.
  BarVar.clearValue();
  Value = Binop.eval();
  expectUndefErrors({"FOO", "BAR"}, Value.takeError());

  // Literal + Variable has format of variable.
  ExprStr = bufferize(SM, "FOO+18");
  FooStr = ExprStr.take_front(3);
  StringRef EighteenStr = ExprStr.take_back(2);
  FooVarUse = std::make_unique<NumericVariableUse>(FooStr, &FooVar);
  std::unique_ptr<ExpressionLiteral> Eighteen =
      std::make_unique<ExpressionLiteral>(EighteenStr, APInt(64, 18u));
  Binop = BinaryOperation(ExprStr, exprAdd, std::move(FooVarUse),
                          std::move(Eighteen));
  ImplicitFormat = Binop.getImplicitFormat(SM);
  ASSERT_THAT_EXPECTED(ImplicitFormat, Succeeded());
  EXPECT_EQ(*ImplicitFormat, ExpressionFormat::Kind::Unsigned);
  ExprStr = bufferize(SM, "18+FOO");
  FooStr = ExprStr.take_back(3);
  EighteenStr = ExprStr.take_front(2);
  FooVarUse = std::make_unique<NumericVariableUse>(FooStr, &FooVar);
  Eighteen = std::make_unique<ExpressionLiteral>(EighteenStr, APInt(64, 18u));
  Binop = BinaryOperation(ExprStr, exprAdd, std::move(Eighteen),
                          std::move(FooVarUse));
  ImplicitFormat = Binop.getImplicitFormat(SM);
  ASSERT_THAT_EXPECTED(ImplicitFormat, Succeeded());
  EXPECT_EQ(*ImplicitFormat, ExpressionFormat::Kind::Unsigned);

  // Variables with different implicit format conflict.
  ExprStr = bufferize(SM, "FOO+BAZ");
  FooStr = ExprStr.take_front(3);
  StringRef BazStr = ExprStr.take_back(3);
  NumericVariable BazVar(BazStr,
                         ExpressionFormat(ExpressionFormat::Kind::HexLower), 3);
  FooVarUse = std::make_unique<NumericVariableUse>(FooStr, &FooVar);
  std::unique_ptr<NumericVariableUse> BazVarUse =
      std::make_unique<NumericVariableUse>(BazStr, &BazVar);
  Binop = BinaryOperation(ExprStr, exprAdd, std::move(FooVarUse),
                          std::move(BazVarUse));
  ImplicitFormat = Binop.getImplicitFormat(SM);
  expectDiagnosticError(
      "implicit format conflict between 'FOO' (%u) and 'BAZ' (%x), "
      "need an explicit format specifier",
      ImplicitFormat.takeError());

  // All variable conflicts are reported.
  ExprStr = bufferize(SM, "(FOO+BAZ)+(FOO+QUUX)");
  StringRef Paren1ExprStr = ExprStr.substr(1, 7);
  FooStr = Paren1ExprStr.take_front(3);
  BazStr = Paren1ExprStr.take_back(3);
  StringRef Paren2ExprStr = ExprStr.substr(ExprStr.rfind('(') + 1, 8);
  StringRef FooStr2 = Paren2ExprStr.take_front(3);
  StringRef QuuxStr = Paren2ExprStr.take_back(4);
  FooVarUse = std::make_unique<NumericVariableUse>(FooStr, &FooVar);
  BazVarUse = std::make_unique<NumericVariableUse>(BazStr, &BazVar);
  std::unique_ptr<NumericVariableUse> FooVarUse2 =
      std::make_unique<NumericVariableUse>(FooStr2, &FooVar);
  NumericVariable QuuxVar(
      QuuxStr, ExpressionFormat(ExpressionFormat::Kind::HexLower), 4);
  std::unique_ptr<NumericVariableUse> QuuxVarUse =
      std::make_unique<NumericVariableUse>(QuuxStr, &QuuxVar);
  std::unique_ptr<BinaryOperation> Binop1 = std::make_unique<BinaryOperation>(
      ExprStr.take_front(9), exprAdd, std::move(FooVarUse),
      std::move(BazVarUse));
  std::unique_ptr<BinaryOperation> Binop2 = std::make_unique<BinaryOperation>(
      ExprStr.take_back(10), exprAdd, std::move(FooVarUse2),
      std::move(QuuxVarUse));
  std::unique_ptr<BinaryOperation> OuterBinop =
      std::make_unique<BinaryOperation>(ExprStr, exprAdd, std::move(Binop1),
                                        std::move(Binop2));
  ImplicitFormat = OuterBinop->getImplicitFormat(SM);
  expectSameErrors<ErrorDiagnostic>(
      {("implicit format conflict between 'FOO' (%u) and 'BAZ' (%x), need an "
       "explicit format specifier"),
       ("implicit format conflict between 'FOO' (%u) and 'QUUX' (%x), need an "
       "explicit format specifier")},
      ImplicitFormat.takeError());
}

TEST_F(FileCheckTest, ValidVarNameStart) {
  EXPECT_TRUE(Pattern::isValidVarNameStart('a'));
  EXPECT_TRUE(Pattern::isValidVarNameStart('G'));
  EXPECT_TRUE(Pattern::isValidVarNameStart('_'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('2'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('$'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('@'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('+'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('-'));
  EXPECT_FALSE(Pattern::isValidVarNameStart(':'));
}

TEST_F(FileCheckTest, ParseVar) {
  SourceMgr SM;
  StringRef OrigVarName = bufferize(SM, "GoodVar42");
  StringRef VarName = OrigVarName;
  Expected<Pattern::VariableProperties> ParsedVarResult =
      Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "$GoodGlobalVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "@GoodPseudoVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_TRUE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "42BadVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  expectDiagnosticError("invalid variable name", ParsedVarResult.takeError());

  VarName = bufferize(SM, "$@");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  expectDiagnosticError("invalid variable name", ParsedVarResult.takeError());

  VarName = OrigVarName = bufferize(SM, "B@dVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "B$dVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar+");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(VarName, "+");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar-");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(VarName, "-");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar:");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_THAT_EXPECTED(ParsedVarResult, Succeeded());
  EXPECT_EQ(VarName, ":");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);
}

static void expectNotFoundError(Error Err) {
  expectError<NotFoundError>("String not found in input", std::move(Err));
}

class PatternTester {
private:
  size_t LineNumber = 1;
  SourceMgr SM;
  FileCheckRequest Req;
  FileCheckPatternContext Context;
  Pattern P{Check::CheckPlain, &Context, LineNumber};

public:
  PatternTester() {
    std::vector<StringRef> GlobalDefines = {"#FOO=42", "BAR=BAZ", "#add=7"};
    // An ASSERT_FALSE would make more sense but cannot be used in a
    // constructor.
    EXPECT_THAT_ERROR(Context.defineCmdlineVariables(GlobalDefines, SM),
                      Succeeded());
    Context.createLineVariable();
    // Call parsePattern to have @LINE defined.
    P.parsePattern("N/A", "CHECK", SM, Req);
    // parsePattern does not expect to be called twice for the same line and
    // will set FixedStr and RegExStr incorrectly if it is. Therefore prepare
    // a pattern for a different line.
    initNextPattern();
  }

  void initNextPattern() {
    P = Pattern(Check::CheckPlain, &Context, ++LineNumber);
  }

  size_t getLineNumber() const { return LineNumber; }

  Expected<std::unique_ptr<Expression>>
  parseSubst(StringRef Expr, bool IsLegacyLineExpr = false) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    std::optional<NumericVariable *> DefinedNumericVariable;
    return P.parseNumericSubstitutionBlock(
        ExprBufferRef, DefinedNumericVariable, IsLegacyLineExpr, LineNumber,
        &Context, SM);
  }

  bool parsePattern(StringRef Pattern) {
    StringRef PatBufferRef = bufferize(SM, Pattern);
    return P.parsePattern(PatBufferRef, "CHECK", SM, Req);
  }

  Expected<size_t> match(StringRef Buffer) {
    StringRef BufferRef = bufferize(SM, Buffer);
    Pattern::MatchResult Res = P.match(BufferRef, SM);
    if (Res.TheError)
      return std::move(Res.TheError);
    return Res.TheMatch->Pos;
  }

  void printVariableDefs(FileCheckDiag::MatchType MatchTy,
                         std::vector<FileCheckDiag> &Diags) {
    P.printVariableDefs(SM, MatchTy, &Diags);
  }
};

TEST_F(FileCheckTest, ParseNumericSubstitutionBlock) {
  PatternTester Tester;

  // Variable definition.

  expectDiagnosticError("invalid variable name",
                        Tester.parseSubst("%VAR:").takeError());

  expectDiagnosticError("definition of pseudo numeric variable unsupported",
                        Tester.parseSubst("@LINE:").takeError());

  expectDiagnosticError("string variable with name 'BAR' already exists",
                        Tester.parseSubst("BAR:").takeError());

  expectDiagnosticError("unexpected characters after numeric variable name",
                        Tester.parseSubst("VAR GARBAGE:").takeError());

  // Change of format.
  expectDiagnosticError("format different from previous variable definition",
                        Tester.parseSubst("%X,FOO:").takeError());

  // Invalid format.
  expectDiagnosticError("invalid matching format specification in expression",
                        Tester.parseSubst("X,VAR1:").takeError());
  expectDiagnosticError("invalid format specifier in expression",
                        Tester.parseSubst("%F,VAR1:").takeError());
  expectDiagnosticError("invalid matching format specification in expression",
                        Tester.parseSubst("%X a,VAR1:").takeError());

  // Acceptable variable definition.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("VAR1:"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("  VAR2:"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("VAR3  :"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("VAR3:  "), Succeeded());

  // Acceptable variable definition with format specifier. Use parsePattern for
  // variables whose definition needs to be visible for later checks.
  EXPECT_FALSE(Tester.parsePattern("[[#%u, VAR_UNSIGNED:]]"));
  EXPECT_FALSE(Tester.parsePattern("[[#%x, VAR_LOWER_HEX:]]"));
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%X, VAR_UPPER_HEX:"), Succeeded());

  // Acceptable variable definition with precision specifier.
  EXPECT_FALSE(Tester.parsePattern("[[#%.8X, PADDED_ADDR:]]"));
  EXPECT_FALSE(Tester.parsePattern("[[#%.8, PADDED_NUM:]]"));

  // Acceptable variable definition in alternate form.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%#x, PREFIXED_ADDR:"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%#X, PREFIXED_ADDR:"), Succeeded());

  // Acceptable variable definition in alternate form.
  expectDiagnosticError("alternate form only supported for hex values",
                        Tester.parseSubst("%#u, PREFIXED_UNSI:").takeError());
  expectDiagnosticError("alternate form only supported for hex values",
                        Tester.parseSubst("%#d, PREFIXED_UNSI:").takeError());

  // Acceptable variable definition from a numeric expression.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOOBAR: FOO+1"), Succeeded());

  // Numeric expression. Switch to next line to make above valid definition
  // available in expressions.
  Tester.initNextPattern();

  // Invalid variable name.
  expectDiagnosticError("invalid matching constraint or operand format",
                        Tester.parseSubst("%VAR").takeError());

  expectDiagnosticError("invalid pseudo numeric variable '@FOO'",
                        Tester.parseSubst("@FOO").takeError());

  // parsePattern() is used here instead of parseSubst() for the variable to be
  // recorded in GlobalNumericVariableTable and thus appear defined to
  // parseNumericVariableUse(). Note that the same pattern object is used for
  // the parsePattern() and parseSubst() since no initNextPattern() is called,
  // thus appearing as being on the same line from the pattern's point of view.
  ASSERT_FALSE(Tester.parsePattern("[[#SAME_LINE_VAR:]]"));
  expectDiagnosticError("numeric variable 'SAME_LINE_VAR' defined earlier in "
                        "the same CHECK directive",
                        Tester.parseSubst("SAME_LINE_VAR").takeError());

  // Invalid use of variable defined on the same line from an expression not
  // using any variable defined on the same line.
  ASSERT_FALSE(Tester.parsePattern("[[#SAME_LINE_EXPR_VAR:@LINE+1]]"));
  expectDiagnosticError("numeric variable 'SAME_LINE_EXPR_VAR' defined earlier "
                        "in the same CHECK directive",
                        Tester.parseSubst("SAME_LINE_EXPR_VAR").takeError());

  // Valid use of undefined variable which creates the variable and record it
  // in GlobalNumericVariableTable.
  ASSERT_THAT_EXPECTED(Tester.parseSubst("UNDEF"), Succeeded());
  EXPECT_TRUE(Tester.parsePattern("[[UNDEF:.*]]"));

  // Invalid literal.
  expectDiagnosticError("unsupported operation 'U'",
                        Tester.parseSubst("42U").takeError());

  // Valid empty expression.
  EXPECT_THAT_EXPECTED(Tester.parseSubst(""), Succeeded());

  // Invalid equality matching constraint with empty expression.
  expectDiagnosticError("empty numeric expression should not have a constraint",
                        Tester.parseSubst("==").takeError());

  // Valid single operand expression.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("18"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst(std::to_string(MaxUint64)),
                       Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("0x12"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("-30"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst(std::to_string(MinInt64)),
                       Succeeded());

  // Valid optional matching constraint.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("==FOO"), Succeeded());

  // Invalid matching constraint.
  expectDiagnosticError("invalid matching constraint or operand format",
                        Tester.parseSubst("+=FOO").takeError());

  // Invalid format.
  expectDiagnosticError("invalid matching format specification in expression",
                        Tester.parseSubst("X,FOO:").takeError());
  expectDiagnosticError("invalid format specifier in expression",
                        Tester.parseSubst("%F,FOO").takeError());
  expectDiagnosticError("invalid matching format specification in expression",
                        Tester.parseSubst("%X a,FOO").takeError());

  // Valid expression with 2 or more operands.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO+3"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO+0xC"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO-3+FOO"), Succeeded());

  expectDiagnosticError("unsupported operation '/'",
                        Tester.parseSubst("@LINE/2").takeError());

  expectDiagnosticError("missing operand in expression",
                        Tester.parseSubst("@LINE+").takeError());

  // Errors in RHS operand are bubbled up by parseBinop() to
  // parseNumericSubstitutionBlock().
  expectDiagnosticError("invalid operand format",
                        Tester.parseSubst("@LINE+%VAR").takeError());

  // Invalid legacy @LINE expression with non literal rhs.
  expectDiagnosticError(
      "invalid operand format",
      Tester.parseSubst("@LINE+@LINE", /*IsLegacyNumExpr=*/true).takeError());

  // Invalid legacy @LINE expression made of a single literal.
  expectDiagnosticError(
      "invalid variable name",
      Tester.parseSubst("2", /*IsLegacyNumExpr=*/true).takeError());

  // Invalid hex literal in legacy @LINE expression.
  expectDiagnosticError(
      "unexpected characters at end of expression 'xC'",
      Tester.parseSubst("@LINE+0xC", /*LegacyLineExpr=*/true).takeError());

  // Valid expression with format specifier.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%u, FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%d, FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%x, FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%X, FOO"), Succeeded());

  // Valid expression with precision specifier.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%.8u, FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%.8, FOO"), Succeeded());

  // Valid legacy @LINE expression.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("@LINE+2", /*IsLegacyNumExpr=*/true),
                       Succeeded());

  // Invalid legacy @LINE expression with more than 2 operands.
  expectDiagnosticError(
      "unexpected characters at end of expression '+@LINE'",
      Tester.parseSubst("@LINE+2+@LINE", /*IsLegacyNumExpr=*/true).takeError());
  expectDiagnosticError(
      "unexpected characters at end of expression '+2'",
      Tester.parseSubst("@LINE+2+2", /*IsLegacyNumExpr=*/true).takeError());

  // Valid expression with several variables when their implicit formats do not
  // conflict.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO+VAR_UNSIGNED"), Succeeded());

  // Valid implicit format conflict in presence of explicit formats.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("%X,FOO+VAR_LOWER_HEX"), Succeeded());

  // Implicit format conflict.
  expectDiagnosticError(
      "implicit format conflict between 'FOO' (%u) and "
      "'VAR_LOWER_HEX' (%x), need an explicit format specifier",
      Tester.parseSubst("FOO+VAR_LOWER_HEX").takeError());

  // Simple parenthesized expressions:
  EXPECT_THAT_EXPECTED(Tester.parseSubst("(1)"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("(1+1)"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("(1)+1"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("((1)+1)"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("((1)+X)"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("((X)+Y)"), Succeeded());

  expectDiagnosticError("missing operand in expression",
                        Tester.parseSubst("(").takeError());
  expectDiagnosticError("missing ')' at end of nested expression",
                        Tester.parseSubst("(1").takeError());
  expectDiagnosticError("missing operand in expression",
                        Tester.parseSubst("(1+").takeError());
  expectDiagnosticError("missing ')' at end of nested expression",
                        Tester.parseSubst("(1+1").takeError());
  expectDiagnosticError("missing ')' at end of nested expression",
                        Tester.parseSubst("((1+2+3").takeError());
  expectDiagnosticError("missing ')' at end of nested expression",
                        Tester.parseSubst("((1+2)+3").takeError());

  // Test missing operation between operands:
  expectDiagnosticError("unsupported operation '('",
                        Tester.parseSubst("(1)(2)").takeError());
  expectDiagnosticError("unsupported operation '('",
                        Tester.parseSubst("2(X)").takeError());

  // Test more closing than opening parentheses. The diagnostic messages are
  // not ideal, but for now simply check that we reject invalid input.
  expectDiagnosticError("invalid matching constraint or operand format",
                        Tester.parseSubst(")").takeError());
  expectDiagnosticError("unsupported operation ')'",
                        Tester.parseSubst("1)").takeError());
  expectDiagnosticError("unsupported operation ')'",
                        Tester.parseSubst("(1+2))").takeError());
  expectDiagnosticError("unsupported operation ')'",
                        Tester.parseSubst("(2))").takeError());
  expectDiagnosticError("unsupported operation ')'",
                        Tester.parseSubst("(1))(").takeError());

  // Valid expression with function call.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add(FOO,3)"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add (FOO,3)"), Succeeded());
  // Valid expression with nested function call.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add(FOO, min(BAR,10))"), Succeeded());
  // Valid expression with function call taking expression as argument.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add(FOO, (BAR+10) + 3)"),
                       Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add(FOO, min (BAR,10) + 3)"),
                       Succeeded());
  // Valid expression with variable named the same as a function.
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add+FOO"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("FOO+add"), Succeeded());
  EXPECT_THAT_EXPECTED(Tester.parseSubst("add(add,add)+add"), Succeeded());

  // Malformed call syntax.
  expectDiagnosticError("missing ')' at end of call expression",
                        Tester.parseSubst("add(FOO,(BAR+7)").takeError());
  expectDiagnosticError("missing ')' at end of call expression",
                        Tester.parseSubst("add(FOO,min(BAR,7)").takeError());
  expectDiagnosticError("missing argument",
                        Tester.parseSubst("add(FOO,)").takeError());
  expectDiagnosticError("missing argument",
                        Tester.parseSubst("add(,FOO)").takeError());
  expectDiagnosticError("missing argument",
                        Tester.parseSubst("add(FOO,,3)").takeError());

  // Valid call, but to an unknown function.
  expectDiagnosticError("call to undefined function 'bogus_function'",
                        Tester.parseSubst("bogus_function(FOO,3)").takeError());
  expectDiagnosticError("call to undefined function '@add'",
                        Tester.parseSubst("@add(2,3)").takeError());
  expectDiagnosticError("call to undefined function '$add'",
                        Tester.parseSubst("$add(2,3)").takeError());
  expectDiagnosticError("call to undefined function 'FOO'",
                        Tester.parseSubst("FOO(2,3)").takeError());
  expectDiagnosticError("call to undefined function 'FOO'",
                        Tester.parseSubst("FOO (2,3)").takeError());

  // Valid call, but with incorrect argument count.
  expectDiagnosticError("function 'add' takes 2 arguments but 1 given",
                        Tester.parseSubst("add(FOO)").takeError());
  expectDiagnosticError("function 'add' takes 2 arguments but 3 given",
                        Tester.parseSubst("add(FOO,3,4)").takeError());

  // Valid call, but not part of a valid expression.
  expectDiagnosticError("unsupported operation 'a'",
                        Tester.parseSubst("2add(FOO,2)").takeError());
  expectDiagnosticError("unsupported operation 'a'",
                        Tester.parseSubst("FOO add(FOO,2)").takeError());
  expectDiagnosticError("unsupported operation 'a'",
                        Tester.parseSubst("add(FOO,2)add(FOO,2)").takeError());
}

TEST_F(FileCheckTest, ParsePattern) {
  PatternTester Tester;

  // Invalid space in string substitution.
  EXPECT_TRUE(Tester.parsePattern("[[ BAR]]"));

  // Invalid variable name in string substitution.
  EXPECT_TRUE(Tester.parsePattern("[[42INVALID]]"));

  // Invalid string variable definition.
  EXPECT_TRUE(Tester.parsePattern("[[@PAT:]]"));
  EXPECT_TRUE(Tester.parsePattern("[[PAT+2:]]"));

  // Collision with numeric variable.
  EXPECT_TRUE(Tester.parsePattern("[[FOO:]]"));

  // Invalid use of string variable.
  EXPECT_TRUE(Tester.parsePattern("[[FOO-BAR]]"));

  // Valid use of string variable.
  EXPECT_FALSE(Tester.parsePattern("[[BAR]]"));

  // Valid string variable definition.
  EXPECT_FALSE(Tester.parsePattern("[[PAT:[0-9]+]]"));

  // Invalid numeric substitution.
  EXPECT_TRUE(Tester.parsePattern("[[#42INVALID]]"));

  // Valid numeric substitution.
  EXPECT_FALSE(Tester.parsePattern("[[#FOO]]"));

  // Valid legacy @LINE expression.
  EXPECT_FALSE(Tester.parsePattern("[[@LINE+2]]"));

  // Invalid legacy @LINE expression with non decimal literal.
  EXPECT_TRUE(Tester.parsePattern("[[@LINE+0x3]]"));
}

TEST_F(FileCheckTest, Match) {
  PatternTester Tester;

  // Check a substitution error is diagnosed.
  ASSERT_FALSE(Tester.parsePattern("[[#%u, -1]]"));
  expectDiagnosticError(
      "unable to substitute variable or numeric expression: overflow error",
      Tester.match("").takeError());

  // Check matching an empty expression only matches a number.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#]]"));
  expectNotFoundError(Tester.match("FAIL").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());

  // Check matching a definition only matches a number with the right format.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR:]]"));
  expectNotFoundError(Tester.match("FAIL").takeError());
  expectNotFoundError(Tester.match("").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());
  Tester.initNextPattern();
  Tester.parsePattern("[[#%u,NUMVAR_UNSIGNED:]]");
  expectNotFoundError(Tester.match("C").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("20"), Succeeded());
  Tester.initNextPattern();
  Tester.parsePattern("[[#%x,NUMVAR_LOWER_HEX:]]");
  expectNotFoundError(Tester.match("g").takeError());
  expectNotFoundError(Tester.match("C").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("c"), Succeeded());
  Tester.initNextPattern();
  Tester.parsePattern("[[#%X,NUMVAR_UPPER_HEX:]]");
  expectNotFoundError(Tester.match("H").takeError());
  expectNotFoundError(Tester.match("b").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("B"), Succeeded());

  // Check matching expressions with no explicit format matches the values in
  // the right format.
  Tester.initNextPattern();
  Tester.parsePattern("[[#NUMVAR_UNSIGNED-5]]");
  expectNotFoundError(Tester.match("f").takeError());
  expectNotFoundError(Tester.match("F").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("15"), Succeeded());
  Tester.initNextPattern();
  Tester.parsePattern("[[#NUMVAR_LOWER_HEX+1]]");
  expectNotFoundError(Tester.match("13").takeError());
  expectNotFoundError(Tester.match("D").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("d"), Succeeded());
  Tester.initNextPattern();
  Tester.parsePattern("[[#NUMVAR_UPPER_HEX+1]]");
  expectNotFoundError(Tester.match("12").takeError());
  expectNotFoundError(Tester.match("c").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("C"), Succeeded());

  // Check matching an undefined variable returns a NotFound error.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("100"));
  expectNotFoundError(Tester.match("101").takeError());

  // Check matching the defined variable matches the correct number only.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR]]"));
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());

  // Check matching several substitutions does not match them independently.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR]] [[#NUMVAR+2]]"));
  expectNotFoundError(Tester.match("19 21").takeError());
  expectNotFoundError(Tester.match("18 21").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("18 20"), Succeeded());

  // Check matching a numeric expression using @LINE after a match failure uses
  // the correct value for @LINE.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#@LINE]]"));
  // Ok, @LINE matches the current line number.
  EXPECT_THAT_EXPECTED(Tester.match(std::to_string(Tester.getLineNumber())),
                       Succeeded());
  Tester.initNextPattern();
  // Match with substitution failure.
  ASSERT_FALSE(Tester.parsePattern("[[#UNKNOWN1+UNKNOWN2]]"));
  expectSameErrors<ErrorDiagnostic>(
      {"undefined variable: UNKNOWN1", "undefined variable: UNKNOWN2"},
      Tester.match("FOO").takeError());
  Tester.initNextPattern();
  // Check that @LINE matches the later (given the calls to initNextPattern())
  // line number.
  EXPECT_FALSE(Tester.parsePattern("[[#@LINE]]"));
  EXPECT_THAT_EXPECTED(Tester.match(std::to_string(Tester.getLineNumber())),
                       Succeeded());
}

TEST_F(FileCheckTest, MatchParen) {
  PatternTester Tester;
  // Check simple parenthesized expressions
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR:]]"));
  expectNotFoundError(Tester.match("FAIL").takeError());
  expectNotFoundError(Tester.match("").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());

  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR + (2 + 2)]]"));
  expectNotFoundError(Tester.match("21").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("22"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR + (2)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("20"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+(2)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("20"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+(NUMVAR)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("36"), Succeeded());

  // Check nested parenthesized expressions:
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+(2+(2))]]"));
  EXPECT_THAT_EXPECTED(Tester.match("22"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+(2+(NUMVAR))]]"));
  EXPECT_THAT_EXPECTED(Tester.match("38"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+((((NUMVAR))))]]"));
  EXPECT_THAT_EXPECTED(Tester.match("36"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR+((((NUMVAR)))-1)-1]]"));
  EXPECT_THAT_EXPECTED(Tester.match("34"), Succeeded());

  // Parentheses can also be the first character after the '#':
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#(NUMVAR)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#(NUMVAR+2)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("20"), Succeeded());
}

TEST_F(FileCheckTest, MatchBuiltinFunctions) {
  PatternTester Tester;
  // Esnure #NUMVAR has the expected value.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#NUMVAR:]]"));
  expectNotFoundError(Tester.match("FAIL").takeError());
  expectNotFoundError(Tester.match("").takeError());
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());

  // Check each builtin function generates the expected result.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#add(NUMVAR,13)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("31"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#div(NUMVAR,3)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("6"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#max(NUMVAR,5)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#max(NUMVAR,99)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("99"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#min(NUMVAR,5)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("5"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#min(NUMVAR,99)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("18"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#mul(NUMVAR,3)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("54"), Succeeded());
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#sub(NUMVAR,7)]]"));
  EXPECT_THAT_EXPECTED(Tester.match("11"), Succeeded());

  // Check nested function calls.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#add(min(7,2),max(4,10))]]"));
  EXPECT_THAT_EXPECTED(Tester.match("12"), Succeeded());

  // Check function call that uses a variable of the same name.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePattern("[[#add(add,add)+min (add,3)+add]]"));
  EXPECT_THAT_EXPECTED(Tester.match("24"), Succeeded());
}

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  EXPECT_THAT_ERROR(Context.defineCmdlineVariables({"FOO=BAR"}, SM),
                    Succeeded());

  // Substitution of an undefined string variable fails and error holds that
  // variable's name.
  StringSubstitution StringSubstitution(&Context, "VAR404", 42);
  Expected<std::string> SubstValue = StringSubstitution.getResult();
  expectUndefErrors({"VAR404"}, SubstValue.takeError());

  // Numeric substitution blocks constituted of defined numeric variables are
  // substituted for the variable's value.
  NumericVariable NVar("N", ExpressionFormat(ExpressionFormat::Kind::Unsigned),
                       1);
  NVar.setValue(APInt(64, 10u));
  auto NVarUse = std::make_unique<NumericVariableUse>("N", &NVar);
  auto ExpressionN = std::make_unique<Expression>(
      std::move(NVarUse), ExpressionFormat(ExpressionFormat::Kind::HexUpper));
  NumericSubstitution SubstitutionN(&Context, "N", std::move(ExpressionN),
                                    /*InsertIdx=*/30);
  SubstValue = SubstitutionN.getResult();
  ASSERT_THAT_EXPECTED(SubstValue, Succeeded());
  EXPECT_EQ("A", *SubstValue);

  // Substitution of an undefined numeric variable fails, error holds name of
  // undefined variable.
  NVar.clearValue();
  SubstValue = SubstitutionN.getResult();
  expectUndefErrors({"N"}, SubstValue.takeError());

  // Substitution of a defined string variable returns the right value.
  Pattern P(Check::CheckPlain, &Context, 1);
  StringSubstitution = llvm::StringSubstitution(&Context, "FOO", 42);
  SubstValue = StringSubstitution.getResult();
  ASSERT_THAT_EXPECTED(SubstValue, Succeeded());
  EXPECT_EQ("BAR", *SubstValue);
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt;
  SourceMgr SM;

  // No definition.
  EXPECT_THAT_ERROR(Cxt.defineCmdlineVariables({}, SM), Succeeded());

  // Missing equal sign.
  expectDiagnosticError("missing equal sign in global definition",
                        Cxt.defineCmdlineVariables({"LocalVar"}, SM));
  expectDiagnosticError("missing equal sign in global definition",
                        Cxt.defineCmdlineVariables({"#LocalNumVar"}, SM));

  // Empty variable name.
  expectDiagnosticError("empty variable name",
                        Cxt.defineCmdlineVariables({"=18"}, SM));
  expectDiagnosticError("empty variable name",
                        Cxt.defineCmdlineVariables({"#=18"}, SM));

  // Invalid variable name.
  expectDiagnosticError("invalid variable name",
                        Cxt.defineCmdlineVariables({"18LocalVar=18"}, SM));
  expectDiagnosticError("invalid variable name",
                        Cxt.defineCmdlineVariables({"#18LocalNumVar=18"}, SM));

  // Name conflict between pattern and numeric variable.
  expectDiagnosticError(
      "string variable with name 'LocalVar' already exists",
      Cxt.defineCmdlineVariables({"LocalVar=18", "#LocalVar=36"}, SM));
  Cxt = FileCheckPatternContext();
  expectDiagnosticError(
      "numeric variable with name 'LocalNumVar' already exists",
      Cxt.defineCmdlineVariables({"#LocalNumVar=18", "LocalNumVar=36"}, SM));
  Cxt = FileCheckPatternContext();

  // Invalid numeric value for numeric variable.
  expectUndefErrors({"x"}, Cxt.defineCmdlineVariables({"#LocalNumVar=x"}, SM));

  // Define local variables from command-line.
  std::vector<StringRef> GlobalDefines;
  // Clear local variables to remove dummy numeric variable x that
  // parseNumericSubstitutionBlock would have created and stored in
  // GlobalNumericVariableTable.
  Cxt.clearLocalVars();
  GlobalDefines.emplace_back("LocalVar=FOO");
  GlobalDefines.emplace_back("EmptyVar=");
  GlobalDefines.emplace_back("#LocalNumVar1=18");
  GlobalDefines.emplace_back("#%x,LocalNumVar2=LocalNumVar1+2");
  GlobalDefines.emplace_back("#LocalNumVar3=0xc");
  ASSERT_THAT_ERROR(Cxt.defineCmdlineVariables(GlobalDefines, SM), Succeeded());

  // Create @LINE pseudo numeric variable and check it is present by matching
  // it.
  size_t LineNumber = 1;
  Pattern P(Check::CheckPlain, &Cxt, LineNumber);
  FileCheckRequest Req;
  Cxt.createLineVariable();
  ASSERT_FALSE(P.parsePattern("[[@LINE]]", "CHECK", SM, Req));
  Pattern::MatchResult Res = P.match("1", SM);
  ASSERT_THAT_ERROR(std::move(Res.TheError), Succeeded());

#ifndef NDEBUG
  // Recreating @LINE pseudo numeric variable fails.
  EXPECT_DEATH(Cxt.createLineVariable(),
               "@LINE pseudo numeric variable already created");
#endif

  // Check defined variables are present and undefined ones are absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef LocalNumVar1Ref = bufferize(SM, "LocalNumVar1");
  StringRef LocalNumVar2Ref = bufferize(SM, "LocalNumVar2");
  StringRef LocalNumVar3Ref = bufferize(SM, "LocalNumVar3");
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  Expected<StringRef> LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  std::optional<NumericVariable *> DefinedNumericVariable;
  Expected<std::unique_ptr<Expression>> ExpressionPointer =
      P.parseNumericSubstitutionBlock(LocalNumVar1Ref, DefinedNumericVariable,
                                      /*IsLegacyLineExpr=*/false, LineNumber,
                                      &Cxt, SM);
  ASSERT_THAT_EXPECTED(LocalVar, Succeeded());
  EXPECT_EQ(*LocalVar, "FOO");
  Expected<StringRef> EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  Expected<StringRef> UnknownVar = Cxt.getPatternVarValue(UnknownVarStr);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  Expected<APInt> ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  ASSERT_THAT_EXPECTED(ExpressionVal, Succeeded());
  EXPECT_EQ(ExpressionVal->getSExtValue(), 18);
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar2Ref, DefinedNumericVariable,
      /*IsLegacyLineExpr=*/false, LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  ASSERT_THAT_EXPECTED(ExpressionVal, Succeeded());
  EXPECT_EQ(ExpressionVal->getSExtValue(), 20);
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar3Ref, DefinedNumericVariable,
      /*IsLegacyLineExpr=*/false, LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  ASSERT_THAT_EXPECTED(ExpressionVal, Succeeded());
  EXPECT_EQ(ExpressionVal->getSExtValue(), 12);
  ASSERT_THAT_EXPECTED(EmptyVar, Succeeded());
  EXPECT_EQ(*EmptyVar, "");
  expectUndefErrors({std::string(UnknownVarStr)}, UnknownVar.takeError());

  // Clear local variables and check they become absent.
  Cxt.clearLocalVars();
  LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  expectUndefErrors({std::string(LocalVarStr)}, LocalVar.takeError());
  // Check a numeric expression's evaluation fails if called after clearing of
  // local variables, if it was created before. This is important because local
  // variable clearing due to --enable-var-scope happens after numeric
  // expressions are linked to the numeric variables they use.
  expectUndefErrors({"LocalNumVar3"},
                    (*ExpressionPointer)->getAST()->eval().takeError());
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar1Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  expectUndefErrors({"LocalNumVar1"}, ExpressionVal.takeError());
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar2Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  expectUndefErrors({"LocalNumVar2"}, ExpressionVal.takeError());
  EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  expectUndefErrors({"EmptyVar"}, EmptyVar.takeError());
  // Clear again because parseNumericSubstitutionBlock would have created a
  // dummy variable and stored it in GlobalNumericVariableTable.
  Cxt.clearLocalVars();

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back("$GlobalVar=BAR");
  GlobalDefines.emplace_back("#$GlobalNumVar=36");
  ASSERT_THAT_ERROR(Cxt.defineCmdlineVariables(GlobalDefines, SM), Succeeded());
  StringRef GlobalVarStr = "$GlobalVar";
  StringRef GlobalNumVarRef = bufferize(SM, "$GlobalNumVar");
  Expected<StringRef> GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  ASSERT_THAT_EXPECTED(GlobalVar, Succeeded());
  EXPECT_EQ(*GlobalVar, "BAR");
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  ASSERT_THAT_EXPECTED(ExpressionVal, Succeeded());
  EXPECT_EQ(ExpressionVal->getSExtValue(), 36);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  EXPECT_THAT_EXPECTED(Cxt.getPatternVarValue(GlobalVarStr), Succeeded());
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionPointer = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_THAT_EXPECTED(ExpressionPointer, Succeeded());
  ExpressionVal = (*ExpressionPointer)->getAST()->eval();
  ASSERT_THAT_EXPECTED(ExpressionVal, Succeeded());
  EXPECT_EQ(ExpressionVal->getSExtValue(), 36);
}

TEST_F(FileCheckTest, CapturedVarDiags) {
  PatternTester Tester;
  ASSERT_FALSE(Tester.parsePattern("[[STRVAR:[a-z]+]] [[#NUMVAR:@LINE]]"));
  EXPECT_THAT_EXPECTED(Tester.match("foobar 2"), Succeeded());
  std::vector<FileCheckDiag> Diags;
  Tester.printVariableDefs(FileCheckDiag::MatchFoundAndExpected, Diags);
  EXPECT_EQ(Diags.size(), 2ul);
  for (const FileCheckDiag &Diag : Diags) {
    EXPECT_EQ(Diag.CheckTy, Check::CheckPlain);
    EXPECT_EQ(Diag.MatchTy, FileCheckDiag::MatchFoundAndExpected);
    EXPECT_EQ(Diag.InputStartLine, 1u);
    EXPECT_EQ(Diag.InputEndLine, 1u);
  }
  EXPECT_EQ(Diags[0].InputStartCol, 1u);
  EXPECT_EQ(Diags[0].InputEndCol, 7u);
  EXPECT_EQ(Diags[1].InputStartCol, 8u);
  EXPECT_EQ(Diags[1].InputEndCol, 9u);
  EXPECT_EQ(Diags[0].Note, "captured var \"STRVAR\"");
  EXPECT_EQ(Diags[1].Note, "captured var \"NUMVAR\"");
}
} // namespace
