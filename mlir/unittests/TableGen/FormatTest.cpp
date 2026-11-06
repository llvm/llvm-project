//===- FormatTest.cpp - TableGen Format Utility Tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "gmock/gmock.h"

using ::testing::StrEq;

using namespace mlir::tblgen;

static std::unique_ptr<FmtError>
takeFormatError(llvm::Expected<std::string> result) {
  EXPECT_FALSE(static_cast<bool>(result));
  if (result)
    return nullptr;
  std::unique_ptr<FmtError> error;
  llvm::handleAllErrors(
      result.takeError(),
      [&](std::unique_ptr<FmtError> e) { error = std::move(e); },
      [&](const llvm::ErrorInfoBase &e) {
        ADD_FAILURE() << "expected FmtError, got " << e.message();
      });
  return error;
}

TEST(FormatTest, EmptyFmtStr) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("", &ctx));
  EXPECT_TRUE(result.empty());
}

/// Allow extra unused positional parameters.
TEST(FormatTest, EmptyFmtStrExtraParams) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("", &ctx, "a", "b", "c"));
  EXPECT_TRUE(result.empty());
}

/// Allow unused placeholder substitution in context.
TEST(FormatTest, EmptyFmtStrPopulatedCtx) {
  FmtContext ctx;
  ctx.withBuilder("builder");
  std::string result = std::string(tgfmt("", &ctx));
  EXPECT_TRUE(result.empty());
}

TEST(FormatTest, LiteralFmtStr) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("void foo {}", &ctx));
  EXPECT_THAT(result, StrEq("void foo {}"));
}

/// Print single dollar literally.
TEST(FormatTest, AdjacentDollar) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$", &ctx));
  EXPECT_THAT(result, StrEq("$"));
}

/// Print dangling dollar literally.
TEST(FormatTest, DanglingDollar) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo bar baz$", &ctx));
  EXPECT_THAT(result, StrEq("foo bar baz$"));
}

/// Allow escape dollars with '$$'.
TEST(FormatTest, EscapeDollars) {
  FmtContext ctx;
  std::string result =
      std::string(tgfmt("$$ $$$$ $$$0 $$$_self", &ctx.withSelf("self"), "-0"));
  EXPECT_THAT(result, StrEq("$ $$ $-0 $self"));
}

TEST(FormatTest, PositionalFmtStr) {
  FmtContext ctx;
  std::string b = "b";
  int c = 42;
  char d = 'd';
  std::string result =
      std::string(tgfmt("$0 $1 $2 $3", &ctx, "a", b, c + 1, d));
  EXPECT_THAT(result, StrEq("a b 43 d"));
}

/// Output the placeholder if missing substitution.
TEST(FormatTest, PositionalFmtStrMissingParams) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$0 %1 $2", &ctx));
  EXPECT_THAT(result, StrEq("$0<no-subst-found> %1 $2<no-subst-found>"));
}

/// Allow flexible reference of positional parameters.
TEST(FormatTest, PositionalFmtStrFlexibleRef) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$2 $0 $2", &ctx, "a", "b", "c"));
  EXPECT_THAT(result, StrEq("c a c"));
}

TEST(FormatTest, PositionalFmtStrNoWhitespace) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo$0bar", &ctx, "-"));
  EXPECT_THAT(result, StrEq("foo-bar"));
}

TEST(FormatTest, PlaceHolderFmtStrWithSelf) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$_self", &ctx.withSelf("sss")));
  EXPECT_THAT(result, StrEq("sss"));
}

TEST(FormatTest, PlaceHolderFmtStrWithBuilder) {
  FmtContext ctx;

  std::string result = std::string(tgfmt("$_builder", &ctx.withBuilder("bbb")));
  EXPECT_THAT(result, StrEq("bbb"));
}

TEST(FormatTest, PlaceHolderMissingCtx) {
  std::string result = std::string(tgfmt("$_op", nullptr));
  EXPECT_THAT(result, StrEq("$_op<no-subst-found>"));
}

TEST(FormatTest, PlaceHolderMissingSubst) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("$_op", &ctx.withBuilder("builder")));
  EXPECT_THAT(result, StrEq("$_op<no-subst-found>"));
}

/// Test commonly used delimiters in C++.
TEST(FormatTest, PlaceHolderFmtStrDelimiter) {
  FmtContext ctx;
  ctx.addSubst("m", "");
  std::string result = std::string(tgfmt("$m{$m($m[$m]$m)$m}$m|", &ctx));
  EXPECT_THAT(result, StrEq("{([])}|"));
}

/// Test allowed characters in placeholder symbol.
TEST(FormatTest, CustomPlaceHolderFmtStrPlaceHolderChars) {
  FmtContext ctx;
  ctx.addSubst("m", "0 ");
  ctx.addSubst("m1", "1 ");
  ctx.addSubst("m2C", "2 ");
  ctx.addSubst("M_3", "3 ");
  std::string result = std::string(tgfmt("$m$m1$m2C$M_3", &ctx));
  EXPECT_THAT(result, StrEq("0 1 2 3 "));
}

TEST(FormatTest, CustomPlaceHolderFmtStrUnregisteredPlaceHolders) {
  FmtContext ctx;
  std::string result = std::string(tgfmt("foo($awesome, $param)", &ctx));
  EXPECT_THAT(result,
              StrEq("foo($awesome<no-subst-found>, $param<no-subst-found>)"));
}

TEST(FormatTest, MixedFmtStr) {
  FmtContext ctx;
  ctx.withBuilder("bbb");

  std::string result = std::string(tgfmt("$_builder.build($_self, {$0, $1})",
                                         &ctx.withSelf("sss"), "a", "b"));
  EXPECT_THAT(result, StrEq("bbb.build(sss, {a, b})"));
}

TEST(FormatTest, CheckedBuiltinRegistrationAndSnapshots) {
  FmtContext ctx({{"z", ""},
                  {"w", "w"},
                  {"_self", "self"},
                  {"a", "a"},
                  {"_builder", "builder"}});
  auto result = tgfmtChecked("$_builder($_self)$z", ctx);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(*result, "builder(self)");
  ctx.withSelf("first").addSubst("_self", "second");
  EXPECT_EQ(ctx.getSubstFor(FmtContext::PHKind::Self), "second");
  ctx.withSelf("third");
  EXPECT_EQ(ctx.getSubstFor("_self"), "third");
  auto bindings = ctx.getBindings();
  ASSERT_EQ(bindings.size(), 5u);
  EXPECT_EQ(bindings[0].name, "_builder");
  EXPECT_EQ(bindings[1].name, "_self");
  EXPECT_EQ(bindings[2].name, "a");
  EXPECT_EQ(bindings[3].name, "w");
  EXPECT_EQ(bindings[4].name, "z");
  EXPECT_EQ(bindings[4].replacement, "");
  ctx.withSelf("fourth");
  EXPECT_EQ(bindings[1].replacement, "third");
  EXPECT_EQ(*result, "builder(self)");
}

TEST(FormatTest, CheckedErrorsOwnInvocationData) {
  auto result = [] {
    FmtContext ctx({{"_self", "self"}, {"empty", ""}});
    std::string fmt = "$missing + $missing + $3";
    return tgfmtChecked(fmt, ctx, "only");
  }();
  auto error = takeFormatError(std::move(result));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getTemplate(), "$missing + $missing + $3");
  auto issues = error->getIssues();
  ASSERT_EQ(issues.size(), 3u);
  EXPECT_EQ(issues[0].kind, FmtIssueKind::MissingNamedBinding);
  EXPECT_EQ(issues[0].placeholder, "$missing");
  EXPECT_EQ(issues[0].offset, 0u);
  EXPECT_EQ(issues[1].offset, 11u);
  EXPECT_EQ(issues[2].offset, 22u);
  EXPECT_EQ(issues[2].kind, FmtIssueKind::PositionalOutOfRange);
  EXPECT_EQ(error->getNumPositionalArgs(), 1u);
  ASSERT_EQ(error->getBindings().size(), 2u);
  EXPECT_EQ(error->getBindings()[0].replacement, "self");
  EXPECT_EQ(error->getBindings()[1].replacement, "");
}

TEST(FormatTest, CheckedDecimalAndRangeBoundaries) {
  std::vector<std::string> values;
  for (unsigned i = 0; i != 11; ++i)
    values.push_back(std::to_string(i));
  auto result = tgfmtvChecked("$00 $08 $010 $9...", FmtContext{}, values);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(*result, "0 8 10 9, 10");
  EXPECT_EQ(tgfmtv("$08 $010 $10...", nullptr, values).str(), "8 10 10");
  auto single = tgfmtvChecked("$10...", FmtContext{}, values);
  ASSERT_TRUE(static_cast<bool>(single));
  EXPECT_EQ(*single, "10");
  auto error = takeFormatError(tgfmtChecked("$1...", FmtContext{}, "one"));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getIssues()[0].kind, FmtIssueKind::RangeStartOutOfRange);
  auto emptyRange = takeFormatError(tgfmtChecked("$0...", FmtContext{}));
  ASSERT_TRUE(emptyRange);
  EXPECT_EQ(emptyRange->getIssues()[0].kind,
            FmtIssueKind::RangeStartOutOfRange);
}

TEST(FormatTest, CheckedInvalidIndexAndLiteralSyntax) {
  llvm::StringRef huge =
      "$999999999999999999999999999999999999999999999999999...";
  auto error = takeFormatError(tgfmtChecked(huge, FmtContext{}));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getIssues()[0].kind, FmtIssueKind::InvalidPositionalIndex);
  EXPECT_EQ(error->getIssues()[0].placeholder, huge);
  EXPECT_EQ(std::string(tgfmt(huge, nullptr)),
            (huge + "<no-subst-found>").str());
  auto result = tgfmtChecked("$$ $- $? foo$0bar $0..2 $", FmtContext{}, "x");
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(*result, "$ $- $? fooxbar x..2 $");
}

TEST(FormatTest, CheckedInvalidIndexWithoutRange) {
  llvm::StringRef huge = "$99999999999999999999999999999999999999999999999999";
  auto error = takeFormatError(tgfmtChecked(huge, FmtContext{}));
  ASSERT_TRUE(error);
  ASSERT_EQ(error->getIssues().size(), 1u);
  EXPECT_EQ(error->getIssues()[0].kind, FmtIssueKind::InvalidPositionalIndex);
  EXPECT_EQ(error->getIssues()[0].placeholder, huge);
  EXPECT_EQ(error->getIssues()[0].offset, 0u);
}

TEST(FormatTest, CheckedMissingBuiltinAndNullLegacyContext) {
  auto missing = takeFormatError(tgfmtChecked("$_self", FmtContext{}));
  ASSERT_TRUE(missing);
  EXPECT_EQ(missing->getIssues()[0].kind, FmtIssueKind::MissingNamedBinding);
  EXPECT_EQ(missing->getIssues()[0].placeholder, "$_self");
  auto noContext = takeFormatError(tgfmt("$_self", nullptr).strChecked());
  ASSERT_TRUE(noContext);
  EXPECT_EQ(noContext->getIssues()[0].kind, FmtIssueKind::MissingNamedBinding);
  EXPECT_TRUE(noContext->getBindings().empty());
}

TEST(FormatTest, ErrorLogIncludesFailureAndAvailability) {
  auto result = tgfmtChecked("$_self $2", FmtContext{}, "only");
  ASSERT_FALSE(static_cast<bool>(result));
  std::string log = llvm::toString(addFmtOrigin(
      result.takeError(), FmtOrigin::describe("predicate").withUse({}, "op")));
  EXPECT_THAT(log, ::testing::HasSubstr("cannot substitute '$_self'"));
  EXPECT_THAT(log, ::testing::HasSubstr("template byte offset 0"));
  EXPECT_THAT(log, ::testing::HasSubstr("missing named binding"));
  EXPECT_THAT(log, ::testing::HasSubstr("'$2' at template byte offset 7"));
  EXPECT_THAT(log, ::testing::HasSubstr("positional argument out of range"));
  EXPECT_THAT(log, ::testing::HasSubstr("template: $_self $2"));
  EXPECT_THAT(log,
              ::testing::HasSubstr("available named substitutions: (none)"));
  EXPECT_THAT(log,
              ::testing::HasSubstr("positional substitutions supplied: 1"));
  EXPECT_THAT(log, ::testing::HasSubstr("while formatting predicate"));
  EXPECT_THAT(log, ::testing::HasSubstr("while expanding op"));
}

struct CountFormatCalls {
  unsigned *calls;
  void format(llvm::raw_ostream &os, llvm::StringRef) const {
    ++*calls;
    os << "value";
  }
};

TEST(FormatTest, CheckedValidationPrecedesRendering) {
  unsigned calls = 0;
  auto error = takeFormatError(
      tgfmtChecked("$0 $missing $2", FmtContext{}, CountFormatCalls{&calls}));
  ASSERT_TRUE(error);
  EXPECT_EQ(calls, 0u);
  EXPECT_EQ(error->getIssues().size(), 2u);
  auto result = tgfmtChecked("$0 $0", FmtContext{}, CountFormatCalls{&calls});
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(*result, "value value");
  EXPECT_EQ(calls, 2u);
}

TEST(FormatTest, CheckedReplacementTextIsLiteral) {
  FmtContext ctx({{"text", "$_missing<no-subst-found>"}});
  auto result = tgfmtChecked("$text $0", ctx, "$_also_missing");
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(*result, "$_missing<no-subst-found> $_also_missing");
}

TEST(FormatTest, OriginsAreOwnedAndKeepTypedErrors) {
  const char source[] = "definition and use";
  llvm::SMLoc loc = llvm::SMLoc::getFromPointer(source);
  llvm::SmallVector<llvm::SMLoc> locations{loc};
  std::string description = "predicate";
  FmtOrigin origin =
      FmtOrigin::at(locations, description).withUse(locations, "operation");
  locations.clear();
  description.clear();
  auto result = tgfmtChecked("$unknown", FmtContext{});
  ASSERT_FALSE(static_cast<bool>(result));
  auto error = addFmtOrigin(result.takeError(), origin);
  error = addFmtOrigin(std::move(error), FmtOrigin::describe("outer"));
  unsigned formattedErrors = 0, otherErrors = 0;
  error = llvm::joinErrors(std::move(error),
                           llvm::createStringError("unrelated error"));
  error = addFmtOrigin(std::move(error), FmtOrigin::describe("caller"));
  llvm::handleAllErrors(
      std::move(error),
      [&](const FmtError &e) {
        ++formattedErrors;
        ASSERT_EQ(e.getOrigins().size(), 3u);
        EXPECT_EQ(e.getOrigins()[0].getDefinition().description, "predicate");
        EXPECT_EQ(e.getOrigins()[0].getDefinition().locations[0], loc);
        EXPECT_EQ(e.getOrigins()[0].getUses()[0].description, "operation");
        EXPECT_EQ(e.getOrigins()[1].getDefinition().description, "outer");
        EXPECT_EQ(e.getOrigins()[2].getDefinition().description, "caller");
      },
      [&](const llvm::StringError &) { ++otherErrors; });
  EXPECT_EQ(formattedErrors, 1u);
  EXPECT_EQ(otherErrors, 1u);
  llvm::cantFail(addFmtOrigin(llvm::Error::success(), origin));
}

TEST(FormatTest, BindingLogsEscapeAndSort) {
  FmtContext ctx({{"z", "line\nbreak"}, {"_self", "a\"b\\c"}});
  std::string log;
  llvm::raw_string_ostream os(log);
  ctx.printBindings(os);
  EXPECT_EQ(log, "$_self = \"a\\\"b\\\\c\"\n$z = \"line\\nbreak\"\n");
  auto error = takeFormatError(tgfmtChecked("$missing", ctx));
  ASSERT_TRUE(error);
  ctx.withSelf("changed");
  std::string snapshotLog;
  llvm::raw_string_ostream snapshotStream(snapshotLog);
  error->printBindings(snapshotStream);
  EXPECT_EQ(snapshotLog, log);
}

TEST(FormatTest, LegacyRemainsLazy) {
  FmtContext ctx;
  ctx.withSelf("before");
  auto legacy = tgfmt("$_self", &ctx);
  auto eager = tgfmtChecked("$_self", ctx);
  ASSERT_TRUE(static_cast<bool>(eager));
  ctx.withSelf("after");
  EXPECT_EQ(legacy.str(), "after");
  EXPECT_EQ(*eager, "before");
}

TEST(FormatTest, DiagnosticsPreserveSourceContextsAndOwnMessages) {
  const char source[] = "definition multiclass use outer";
  llvm::SMLoc def = llvm::SMLoc::getFromPointer(source);
  llvm::SMLoc instantiation = llvm::SMLoc::getFromPointer(source + 11);
  llvm::SMLoc use = llvm::SMLoc::getFromPointer(source + 22);
  llvm::SMLoc outer = llvm::SMLoc::getFromPointer(source + 26);
  auto result =
      tgfmtChecked("$missing $3", FmtContext({{"_self", "self"}}), "one");
  ASSERT_FALSE(static_cast<bool>(result));
  llvm::Error error = addFmtOrigin(
      result.takeError(), FmtOrigin::at({def, instantiation}, "predicate")
                              .withUse({use}, "operation"));
  error =
      addFmtOrigin(std::move(error), FmtOrigin::at({outer}, "outer helper"));
  llvm::SmallVector<FmtDiagnostic> messages;
  llvm::handleAllErrors(std::move(error), [&](const FmtError &e) {
    messages = getFmtDiagnostics(e);
  });
  // The error is already destroyed; messages own text and location arrays.
  ASSERT_EQ(messages.size(), 8u);
  EXPECT_TRUE(messages[0].isError);
  ASSERT_EQ(messages[0].locations.size(), 2u);
  EXPECT_EQ(messages[0].locations[0], def);
  EXPECT_EQ(messages[0].locations[1], instantiation);
  EXPECT_THAT(messages[0].message,
              ::testing::HasSubstr("while formatting predicate"));
  EXPECT_THAT(messages[1].message,
              ::testing::HasSubstr("'$3' at template byte offset 9"));
  EXPECT_EQ(messages[2].message, "template: $missing $3");
  EXPECT_EQ(messages[3].message,
            "failed occurrence starts at template byte offset 0");
  EXPECT_EQ(messages[4].locations[0], use);
  EXPECT_EQ(messages[4].message, "while expanding operation");
  EXPECT_EQ(messages[5].locations[0], outer);
  EXPECT_EQ(messages[5].message, "while formatting outer helper");
  EXPECT_EQ(messages[6].message, "available named substitutions: $_self");
  EXPECT_EQ(messages[7].message, "positional substitutions supplied: 1 "
                                 "(available indices: $0 through $0)");
}

TEST(FormatTest, DiagnosticsFallBackToAnOuterLocation) {
  const char source[] = "outer";
  llvm::SMLoc outer = llvm::SMLoc::getFromPointer(source);
  auto result = tgfmtChecked("$_self", FmtContext{});
  ASSERT_FALSE(static_cast<bool>(result));
  auto error = addFmtOrigin(result.takeError(), FmtOrigin::describe("inner"));
  error = addFmtOrigin(std::move(error), FmtOrigin::at({outer}, "outer"));
  llvm::handleAllErrors(std::move(error), [&](const FmtError &e) {
    auto messages = getFmtDiagnostics(e);
    ASSERT_EQ(messages.front().locations.size(), 1u);
    EXPECT_EQ(messages.front().locations[0], outer);
    EXPECT_THAT(messages.front().message,
                ::testing::HasSubstr("formatting inner"));
    EXPECT_TRUE(llvm::any_of(messages, [](const FmtDiagnostic &message) {
      return message.message == "while formatting outer";
    }));
    EXPECT_EQ(messages.back().message,
              "no positional substitutions were supplied");
  });
}

TEST(FormatTest, FatalArrayFormattingSuccess) {
  std::string values[] = {"zero", "one"};
  FmtContext ctx({{"_self", "self"}});
  EXPECT_EQ(tgfmtvOrFatal(FmtOrigin::describe("array template"), "$1 $0 $_self",
                          ctx, values),
            "one zero self");
}

TEST(FormatTest, DiagnosticsWithoutOrigins) {
  auto error = takeFormatError(tgfmtChecked("$missing", FmtContext{}));
  ASSERT_TRUE(error);
  auto messages = getFmtDiagnostics(*error);
  EXPECT_THAT(messages.front().message,
              ::testing::HasSubstr("while formatting template"));
  EXPECT_TRUE(messages.front().locations.empty());
  EXPECT_EQ(messages[messages.size() - 2].message,
            "available named substitutions: (none)");
}

TEST(FormatTest, ArrayDiagnosticsForInvalidIndicesAndRanges) {
  auto error = takeFormatError(tgfmtvChecked(
      "$99999999999999999999999999999999999999999999 $2...", FmtContext{}, {}));
  ASSERT_TRUE(error);
  auto messages = getFmtDiagnostics(*error);
  EXPECT_THAT(messages[0].message,
              ::testing::HasSubstr("invalid positional index"));
  EXPECT_THAT(messages[1].message,
              ::testing::HasSubstr("positional range starts out of range"));
}

#if GTEST_HAS_DEATH_TEST
TEST(FormatTest, FatalFormattingRejectsSuccessErrors) {
  EXPECT_EXIT(
      reportFatalFmtError(llvm::Error::success(), FmtOrigin::describe("test")),
      ::testing::ExitedWithCode(1), "expected a non-success error");
}

TEST(FormatTest, FatalFormattingPrintsEveryErrorBeforeExit) {
  EXPECT_EXIT(
      {
        auto first = tgfmtChecked("$first", FmtContext{});
        auto second = tgfmtChecked("$second", FmtContext{});
        reportFatalFmtError(
            llvm::joinErrors(first.takeError(), second.takeError()),
            FmtOrigin::describe("test"));
      },
      ::testing::ExitedWithCode(1), "cannot substitute '\\$second'");
}

TEST(FormatTest, FatalFormattingPreservesUnrelatedErrors) {
  EXPECT_EXIT(
      {
        auto result = tgfmtChecked("$missing", FmtContext{});
        reportFatalFmtError(
            llvm::joinErrors(result.takeError(),
                             llvm::createStringError("unrelated failure")),
            FmtOrigin::describe("test"));
      },
      ::testing::ExitedWithCode(1), "unrelated failure");
}
#endif

TEST(FormatTest, ReferenceOverloadsReturnCheckedOwnedResults) {
  FmtContext ctx({{"name", "value"}});
  auto result = tgfmt("$name $0", ctx, 42);
  ASSERT_TRUE(static_cast<bool>(result));
  ctx.addSubst("name", "changed");
  EXPECT_EQ(*result, "value 42");
  std::string values[] = {"zero", "one"};
  auto arrayResult = tgfmtv("$1 $name", ctx, values);
  ASSERT_TRUE(static_cast<bool>(arrayResult));
  EXPECT_EQ(*arrayResult, "one changed");
  auto error = takeFormatError(tgfmt("$absent", ctx));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getIssues()[0].placeholder, "$absent");
  error = takeFormatError(tgfmtv("$2", ctx, values));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getIssues()[0].kind, FmtIssueKind::PositionalOutOfRange);
}

TEST(FormatTest, CheckedSummaryPreservesStreamingAndLiteralDollars) {
  FmtContext ctx({{"_self", "value"}});
  auto plain = buildErrorStreamingStringChecked("literal $text", ctx);
  ASSERT_TRUE(static_cast<bool>(plain));
  EXPECT_EQ(*plain, "literal $text");
  for (ErrorStreamType type :
       {ErrorStreamType::InString, ErrorStreamType::InsideOpError}) {
    auto result =
        buildErrorStreamingStringChecked("before {{$_self}} after", ctx, type);
    ASSERT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(*result, type == ErrorStreamType::InString
                           ? "before \" << value << \" after"
                           : "before \") << value << (\" after");
  }
}

TEST(FormatTest, SummaryErrorRetainsEscapedSegmentAndBindingSnapshot) {
  FmtContext ctx({{"_self", "value"}});
  auto error = takeFormatError(buildErrorStreamingStringChecked(
      R"(before {{"quoted" + \path + $missing}} after)", ctx));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->getTemplate(), R"(\"quoted\" + \\path + $missing)");
  EXPECT_EQ(error->getIssues()[0].offset,
            error->getTemplate().find("$missing"));
  ASSERT_EQ(error->getOrigins().size(), 1u);
  EXPECT_EQ(error->getOrigins()[0].getDefinition().description,
            "escaped summary expression");
  ctx.withSelf("changed");
  EXPECT_EQ(error->getBindings()[0].replacement, "value");
}
