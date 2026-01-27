//===- llvm/unittest/Support/MustacheTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test conforming to Mustache 1.4.2 spec found here:
// https://github.com/mustache/spec
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mustache.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::mustache;
using namespace llvm::json;

TEST(MustacheInterpolation, NoInterpolation) {
  // Mustache-free templates should render as-is.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello from {Mustache}!\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello from {Mustache}!\n", Out);
}

TEST(MustacheInterpolation, BasicInterpolation) {
  // Unadorned tags should interpolate content into the template.
  Value D = Object{{"subject", "World"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{subject}}!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheInterpolation, NoReinterpolation) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{{"template", "{{planet}}"}, {"planet", "Earth"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{template}}: {{planet}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("{{planet}}: Earth", Out);
}

TEST(MustacheInterpolation, HTMLEscaping) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{
      {"forbidden", "& \" < >"},
  };
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("These characters should be HTML escaped: {{forbidden}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("These characters should be HTML escaped: &amp; &quot; &lt; &gt;\n",
            Out);
}

TEST(MustacheInterpolation, Ampersand) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{
      {"forbidden", "& \" < >"},
  };
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("These characters should not be HTML escaped: {{&forbidden}}\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("These characters should not be HTML escaped: & \" < >\n", Out);
}

TEST(MustacheInterpolation, BasicIntegerInterpolation) {
  // Integers should interpolate seamlessly.
  Value D = Object{{"mph", 85}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{mph}} miles an hour!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheInterpolation, AmpersandIntegerInterpolation) {
  // Integers should interpolate seamlessly.
  Value D = Object{{"mph", 85}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{&mph}} miles an hour!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheInterpolation, BasicDecimalInterpolation) {
  // Decimals should interpolate seamlessly with proper significance.
  Value D = Object{{"power", 1.21}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{power}} jiggawatts!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1.21 jiggawatts!", Out);
}

TEST(MustacheInterpolation, BasicNullInterpolation) {
  // Nulls should interpolate as the empty string.
  Value D = Object{{"cannot", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("I ({{cannot}}) be seen!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, AmpersandNullInterpolation) {
  // Nulls should interpolate as the empty string.
  Value D = Object{{"cannot", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("I ({{&cannot}}) be seen!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, BasicContextMissInterpolation) {
  // Failed context lookups should default to empty strings.
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("I ({{cannot}}) be seen!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, DottedNamesBasicInterpolation) {
  // Dotted names should be considered a form of shorthand for sections.
  Value D = Object{{"person", Object{{"name", "Joe"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{person.name}} == {{#person}}{{name}}{{/person}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Joe == Joe", Out);
}

TEST(MustacheInterpolation, DottedNamesAmpersandInterpolation) {
  // Dotted names should be considered a form of shorthand for sections.
  Value D = Object{{"person", Object{{"name", "Joe"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{&person.name}} == {{#person}}{{&name}}{{/person}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Joe == Joe", Out);
}

TEST(MustacheInterpolation, DottedNamesArbitraryDepth) {
  // Dotted names should be functional to any level of nesting.
  Value D = Object{
      {"a",
       Object{{"b",
               Object{{"c",
                       Object{{"d",
                               Object{{"e", Object{{"name", "Phil"}}}}}}}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{a.b.c.d.e.name}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Phil", Out);
}

TEST(MustacheInterpolation, DottedNamesBrokenChains) {
  // Any falsey value prior to the last part of the name should yield ''.
  Value D = Object{{"a", Object{}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{a.b.c}} == ", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheInterpolation, DottedNamesBrokenChainResolution) {
  // Each part of a dotted name should resolve only against its parent.
  Value D =
      Object{{"a", Object{{"b", Object{}}}}, {"c", Object{{"name", "Jim"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{a.b.c.name}} == ", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheInterpolation, DottedNamesInitialResolution) {
  // The first part of a dotted name should resolve as any other name.
  Value D = Object{
      {"a",
       Object{
           {"b",
            Object{{"c",
                    Object{{"d", Object{{"e", Object{{"name", "Phil"}}}}}}}}}}},
      {"b",
       Object{{"c", Object{{"d", Object{{"e", Object{{"name", "Wrong"}}}}}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#a}}{{b.c.d.e.name}}{{/a}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Phil", Out);
}

TEST(MustacheInterpolation, DottedNamesContextPrecedence) {
  // Dotted names should be resolved against former resolutions.
  Value D =
      Object{{"a", Object{{"b", Object{}}}}, {"b", Object{{"c", "ERROR"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#a}}{{b.c}}{{/a}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, DottedNamesAreNotSingleKeys) {
  // Dotted names shall not be parsed as single, atomic keys
  Value D = Object{{"a.b", "c"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{a.b}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, DottedNamesNoMasking) {
  // Dotted Names in a given context are unavailable due to dot splitting
  Value D = Object{{"a.b", "c"}, {"a", Object{{"b", "d"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{a.b}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("d", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsBasicInterpolation) {
  // Unadorned tags should interpolate content into the template.
  Value D = "world";
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{.}}!\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, world!\n", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsAmersand) {
  // Basic interpolation should be HTML escaped.
  Value D = "& \" < >";
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("These characters should not be HTML escaped: {{&.}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("These characters should not be HTML escaped: & \" < >\n", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsInteger) {
  // Integers should interpolate seamlessly.
  Value D = 85;
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{.}} miles an hour!\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!\n", Out);
}

TEST(MustacheInterpolation, InterpolationSurroundingWhitespace) {
  // Interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| {{string}} |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheInterpolation, AmersandSurroundingWhitespace) {
  // Interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| {{&string}} |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheInterpolation, StandaloneInterpolationWithWhitespace) {
  // Standalone interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{string}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  ---\n", Out);
}

TEST(MustacheInterpolation, StandaloneAmpersandWithWhitespace) {
  // Standalone interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{&string}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  ---\n", Out);
}

TEST(MustacheInterpolation, InterpolationWithPadding) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{ string }}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheInterpolation, AmpersandWithPadding) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{& string }}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheInterpolation, InterpolationWithPaddingAndNewlines) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{ string \n\n\n }}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheSections, Truthy) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#boolean}}This should be rendered.{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheSections, Falsey) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#boolean}}This should not be rendered.{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, IsFalseyNull) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{#boolean}}World{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, ", Out);
}

TEST(MustacheInterpolation, IsFalseyArray) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", Array()}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{#boolean}}World{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, ", Out);
}

TEST(MustacheInterpolation, IsFalseyObject) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", Object{}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{#boolean}}World{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World", Out);
}

TEST(MustacheInterpolation, DoubleRendering) {
  // Mustache-free templates should render as-is.
  Value D1 = Object{{"subject", "World"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{subject}}!", Ctx);
  std::string Out1;
  raw_string_ostream OS1(Out1);
  T.render(D1, OS1);
  EXPECT_EQ("Hello, World!", Out1);
  std::string Out2;
  raw_string_ostream OS2(Out2);
  Value D2 = Object{{"subject", "New World"}};
  T.render(D2, OS2);
  EXPECT_EQ("Hello, New World!", Out2);
}

TEST(MustacheSections, NullIsFalsey) {
  Value D = Object{{"null", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#null}}This should not be rendered.{{/null}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheSections, Context) {
  Value D = Object{{"context", Object{{"name", "Joe"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#context}}Hi {{name}}.{{/context}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hi Joe.", Out);
}

TEST(MustacheSections, ParentContexts) {
  Value D = Object{{"a", "foo"},
                   {"b", "wrong"},
                   {"sec", Object{{"b", "bar"}}},
                   {"c", Object{{"d", "baz"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#sec}}{{a}}, {{b}}, {{c.d}}{{/sec}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("foo, bar, baz", Out);
}

TEST(MustacheSections, VariableTest) {
  Value D = Object{{"foo", "bar"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#foo}}{{.}} is {{foo}}{{/foo}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("bar is bar", Out);
}

TEST(MustacheSections, ListContexts) {
  Value D = Object{
      {"tops",
       Array{Object{
           {"tname", Object{{"upper", "A"}, {"lower", "a"}}},
           {"middles",
            Array{Object{{"mname", "1"},
                         {"bottoms", Array{Object{{"bname", "x"}},
                                           Object{{"bname", "y"}}}}}}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#tops}}"
             "{{#middles}}"
             "{{tname.lower}}{{mname}}."
             "{{#bottoms}}"
             "{{tname.upper}}{{mname}}{{bname}}."
             "{{/bottoms}}"
             "{{/middles}}"
             "{{/tops}}",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("a1.A1x.A1y.", Out);
}

TEST(MustacheSections, DeeplyNestedContexts) {
  Value D = Object{
      {"a", Object{{"one", 1}}},
      {"b", Object{{"two", 2}}},
      {"c", Object{{"three", 3}, {"d", Object{{"four", 4}, {"five", 5}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(
      "{{#a}}\n{{one}}\n{{#b}}\n{{one}}{{two}}{{one}}\n{{#c}}\n{{one}}{{two}}{{"
      "three}}{{two}}{{one}}\n{{#d}}\n{{one}}{{two}}{{three}}{{four}}{{three}}{"
      "{two}}{{one}}\n{{#five}}\n{{one}}{{two}}{{three}}{{four}}{{five}}{{four}"
      "}{{three}}{{two}}{{one}}\n{{one}}{{two}}{{three}}{{four}}{{.}}6{{.}}{{"
      "four}}{{three}}{{two}}{{one}}\n{{one}}{{two}}{{three}}{{four}}{{five}}{{"
      "four}}{{three}}{{two}}{{one}}\n{{/"
      "five}}\n{{one}}{{two}}{{three}}{{four}}{{three}}{{two}}{{one}}\n{{/"
      "d}}\n{{one}}{{two}}{{three}}{{two}}{{one}}\n{{/"
      "c}}\n{{one}}{{two}}{{one}}\n{{/b}}\n{{one}}\n{{/a}}\n",
      Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1\n121\n12321\n1234321\n123454321\n12345654321\n123454321\n1234321"
            "\n12321\n121\n1\n",
            Out);
}

TEST(MustacheSections, List) {
  Value D = Object{{"list", Array{Object{{"item", 1}}, Object{{"item", 2}},
                                  Object{{"item", 3}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}{{item}}{{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("123", Out);
}

TEST(MustacheSections, EmptyList) {
  Value D = Object{{"list", Array{}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}Yay lists!{{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheSections, Doubled) {
  Value D = Object{{"bool", true}, {"two", "second"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#bool}}\n* first\n{{/bool}}\n* "
             "{{two}}\n{{#bool}}\n* third\n{{/bool}}\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("* first\n* second\n* third\n", Out);
}

TEST(MustacheSections, NestedTruthy) {
  Value D = Object{{"bool", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| A {{#bool}}B {{#bool}}C{{/bool}} D{{/bool}} E |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A B C D E |", Out);
}

TEST(MustacheSections, NestedFalsey) {
  Value D = Object{{"bool", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| A {{#bool}}B {{#bool}}C{{/bool}} D{{/bool}} E |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A  E |", Out);
}

TEST(MustacheSections, ContextMisses) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[{{#missing}}Found key 'missing'!{{/missing}}]", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[]", Out);
}

TEST(MustacheSections, ImplicitIteratorString) {
  Value D = Object{{"list", Array{"a", "b", "c", "d", "e"}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{.}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(a)(b)(c)(d)(e)", Out);
}

TEST(MustacheSections, ImplicitIteratorInteger) {
  Value D = Object{{"list", Array{1, 2, 3, 4, 5}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{.}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(1)(2)(3)(4)(5)", Out);
}

TEST(MustacheSections, ImplicitIteratorArray) {
  Value D = Object{{"list", Array{Array{1, 2, 3}, Array{"a", "b", "c"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{#.}}{{.}}{{/.}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(123)(abc)", Out);
}

TEST(MustacheSections, ImplicitIteratorHTMLEscaping) {
  Value D = Object{{"list", Array{"&", "\"", "<", ">"}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{.}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(&amp;)(&quot;)(&lt;)(&gt;)", Out);
}

TEST(MustacheSections, ImplicitIteratorAmpersand) {
  Value D = Object{{"list", Array{"&", "\"", "<", ">"}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{&.}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(&)(\")(<)(>)", Out);
}

TEST(MustacheSections, ImplicitIteratorRootLevel) {
  Value D = Array{Object{{"value", "a"}}, Object{{"value", "b"}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#.}}({{value}}){{/.}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(a)(b)", Out);
}

TEST(MustacheSections, DottedNamesTruthy) {
  Value D = Object{{"a", Object{{"b", Object{{"c", true}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#a.b.c}}Here{{/a.b.c}} == Here", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Here == Here", Out);
}

TEST(MustacheSections, DottedNamesFalsey) {
  Value D = Object{{"a", Object{{"b", Object{{"c", false}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#a.b.c}}Here{{/a.b.c}} == ", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheSections, DottedNamesBrokenChains) {
  Value D = Object{{"a", Object{{}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#a.b.c}}Here{{/a.b.c}} == ", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheSections, SurroundingWhitespace) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" | {{#boolean}}\t|\t{{/boolean}} | \n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" | \t|\t | \n", Out);
}

TEST(MustacheSections, InternalWhitespace) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" | {{#boolean}} {{! Important Whitespace }}\n {{/boolean}} | \n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" |  \n  | \n", Out);
}

TEST(MustacheSections, IndentedInlineSections) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" {{#boolean}}YES{{/boolean}}\n {{#boolean}}GOOD{{/boolean}}\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" YES\n GOOD\n", Out);
}

TEST(MustacheSections, StandaloneLines) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| This Is\n{{#boolean}}\n|\n{{/boolean}}\n| A Line\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheSections, IndentedStandaloneLines) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| This Is\n  {{#boolean}}\n|\n  {{/boolean}}\n| A Line\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheSections, StandaloneLineEndings) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|\r\n{{#boolean}}\r\n{{/boolean}}\r\n|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheSections, StandaloneWithoutPreviousLine) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{#boolean}}\n#{{/boolean}}\n/", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("#\n/", Out);
}

TEST(MustacheSections, StandaloneWithoutNewline) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("#{{#boolean}}\n/\n  {{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("#\n/\n", Out);
}

TEST(MustacheSections, Padding) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{# boolean }}={{/ boolean }}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|=|", Out);
}

TEST(MustacheInvertedSections, Falsey) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^boolean}}This should be rendered.{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheInvertedSections, Truthy) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^boolean}}This should not be rendered.{{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, NullIsFalsey) {
  Value D = Object{{"null", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^null}}This should be rendered.{{/null}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheInvertedSections, Context) {
  Value D = Object{{"context", Object{{"name", "Joe"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^context}}Hi {{name}}.{{/context}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, List) {
  Value D = Object{
      {"list", Array{Object{{"n", 1}}, Object{{"n", 2}}, Object{{"n", 3}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^list}}{{n}}{{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, EmptyList) {
  Value D = Object{{"list", Array{}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^list}}Yay lists!{{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Yay lists!", Out);
}

TEST(MustacheInvertedSections, Doubled) {
  Value D = Object{{"bool", false}, {"two", "second"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^bool}}\n* first\n{{/bool}}\n* "
             "{{two}}\n{{^bool}}\n* third\n{{/bool}}\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("* first\n* second\n* third\n", Out);
}

TEST(MustacheInvertedSections, NestedFalsey) {
  Value D = Object{{"bool", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| A {{^bool}}B {{^bool}}C{{/bool}} D{{/bool}} E |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A B C D E |", Out);
}

TEST(MustacheInvertedSections, NestedTruthy) {
  Value D = Object{{"bool", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| A {{^bool}}B {{^bool}}C{{/bool}} D{{/bool}} E |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A  E |", Out);
}

TEST(MustacheInvertedSections, ContextMisses) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[{{^missing}}Cannot find key 'missing'!{{/missing}}]", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[Cannot find key 'missing'!]", Out);
}

TEST(MustacheInvertedSections, DottedNamesTruthy) {
  Value D = Object{{"a", Object{{"b", Object{{"c", true}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^a.b.c}}Not Here{{/a.b.c}} == ", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheInvertedSections, DottedNamesFalsey) {
  Value D = Object{{"a", Object{{"b", Object{{"c", false}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^a.b.c}}Not Here{{/a.b.c}} == Not Here", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Not Here == Not Here", Out);
}

TEST(MustacheInvertedSections, DottedNamesBrokenChains) {
  Value D = Object{{"a", Object{}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{^a.b.c}}Not Here{{/a.b.c}} == Not Here", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Not Here == Not Here", Out);
}

TEST(MustacheInvertedSections, SurroundingWhitespace) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" | {{^boolean}}\t|\t{{/boolean}} | \n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" | \t|\t | \n", Out);
}

TEST(MustacheInvertedSections, InternalWhitespace) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" | {{^boolean}} {{! Important Whitespace }}\n {{/boolean}} | \n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" |  \n  | \n", Out);
}

TEST(MustacheInvertedSections, IndentedInlineSections) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" {{^boolean}}NO{{/boolean}}\n {{^boolean}}WAY{{/boolean}}\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" NO\n WAY\n", Out);
}

TEST(MustacheInvertedSections, StandaloneLines) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| This Is\n{{^boolean}}\n|\n{{/boolean}}\n| A Line\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheInvertedSections, StandaloneIndentedLines) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| This Is\n  {{^boolean}}\n|\n  {{/boolean}}\n| A Line\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheInvertedSections, StandaloneLineEndings) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|\r\n{{^boolean}}\r\n{{/boolean}}\r\n|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheInvertedSections, StandaloneWithoutPreviousLine) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{^boolean}}\n^{{/boolean}}\n/", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("^\n/", Out);
}

TEST(MustacheInvertedSections, StandaloneWithoutNewline) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("^{{^boolean}}\n/\n  {{/boolean}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("^\n/\n", Out);
}

TEST(MustacheInvertedSections, Padding) {
  Value D = Object{{"boolean", false}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{^ boolean }}={{/ boolean }}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|=|", Out);
}

TEST(MustachePartials, BasicBehavior) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{>text}}", Ctx);
  T.registerPartial("text", "from partial");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("from partial", Out);
}

TEST(MustachePartials, FailedLookup) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{>text}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustachePartials, Context) {
  Value D = Object{{"text", "content"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{>partial}}", Ctx);
  T.registerPartial("partial", "*{{text}}*");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("*content*", Out);
}

TEST(MustachePartials, Recursion) {
  Value D =
      Object{{"content", "X"},
             {"nodes", Array{Object{{"content", "Y"}, {"nodes", Array{}}}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{>node}}", Ctx);
  T.registerPartial("node", "{{content}}({{#nodes}}{{>node}}{{/nodes}})");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("X(Y())", Out);
}

TEST(MustachePartials, Nested) {
  Value D = Object{{"a", "hello"}, {"b", "world"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{>outer}}", Ctx);
  T.registerPartial("outer", "*{{a}} {{>inner}}*");
  T.registerPartial("inner", "{{b}}!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("*hello world!*", Out);
}

TEST(MustachePartials, SurroundingWhitespace) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| {{>partial}} |", Ctx);
  T.registerPartial("partial", "\t|\t");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| \t|\t |", Out);
}

TEST(MustachePartials, InlineIndentation) {
  Value D = Object{{"data", "|"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{data}}  {{> partial}}\n", Ctx);
  T.registerPartial("partial", "<\n<");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  |  <\n<\n", Out);
}

TEST(MustachePartials, PaddingWhitespace) {
  Value D = Object{{"boolean", true}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{> partial }}|", Ctx);
  T.registerPartial("partial", "[]");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|[]|", Out);
}

TEST(MustachePartials, StandaloneIndentation) {
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  mustache::Template T("\\\n {{>partial}}\n/\n", Ctx);
  T.registerPartial("partial", "|\n{{{content}}}\n|\n");
  std::string O;
  raw_string_ostream OS(O);
  Value DataContext = Object{{"content", "<\n->"}};
  T.render(DataContext, OS);
  EXPECT_EQ("\\\n |\n <\n->\n |\n/\n", OS.str());
}

TEST(MustacheLambdas, BasicInterpolation) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{lambda}}!", Ctx);
  Lambda L = []() -> llvm::json::Value { return "World"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheLambdas, InterpolationExpansion) {
  Value D = Object{{"planet", "World"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{lambda}}!", Ctx);
  Lambda L = []() -> llvm::json::Value { return "{{planet}}"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheLambdas, BasicMultipleCalls) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{lambda}} == {{lambda}} == {{lambda}}", Ctx);
  int I = 0;
  Lambda L = [&I]() -> llvm::json::Value {
    I += 1;
    return I;
  };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1 == 2 == 3", Out);
}

TEST(MustacheLambdas, Escaping) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("<{{lambda}}{{&lambda}}", Ctx);
  Lambda L = []() -> llvm::json::Value { return ">"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<&gt;>", Out);
}

TEST(MustacheLambdas, Sections) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("<{{#lambda}}{{x}}{{/lambda}}>", Ctx);
  SectionLambda L = [](StringRef Text) -> llvm::json::Value {
    if (Text == "{{x}}") {
      return "yes";
    }
    return "no";
  };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<yes>", Out);
}

TEST(MustacheLambdas, SectionExpansion) {
  Value D = Object{
      {"planet", "Earth"},
  };
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("<{{#lambda}}-{{/lambda}}>", Ctx);
  SectionLambda L = [](StringRef Text) -> llvm::json::Value {
    SmallString<128> Result;
    Result += Text;
    Result += "{{planet}}";
    Result += Text;
    return Result;
  };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<-Earth->", Out);
}

TEST(MustacheLambdas, SectionsMultipleCalls) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#lambda}}FILE{{/lambda}} != {{#lambda}}LINE{{/lambda}}", Ctx);
  SectionLambda L = [](StringRef Text) -> llvm::json::Value {
    SmallString<128> Result;
    Result += "__";
    Result += Text;
    Result += "__";
    return Result;
  };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("__FILE__ != __LINE__", Out);
}

TEST(MustacheLambdas, InvertedSections) {
  Value D = Object{{"static", "static"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("<{{^lambda}}{{static}}{{/lambda}}>", Ctx);
  SectionLambda L = [](StringRef Text) -> llvm::json::Value { return false; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<>", Out);
}

TEST(MustacheComments, Inline) {
  // Comment blocks should be removed from the template.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("12345{{! Comment Block! }}67890", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1234567890", Out);
}

TEST(MustacheComments, Multiline) {
  // Multiline comments should be permitted.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("12345{{!\n  This is a\n  multi-line comment...\n}}67890\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1234567890\n", Out);
}

TEST(MustacheComments, Standalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n{{! Comment Block! }}\nEnd.\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n  {{! Indented Comment Block! }}\nEnd.\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, StandaloneLineEndings) {
  // "\r\n" should be considered a newline for standalone tags.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|\r\n{{! Standalone Comment }}\r\n|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheComments, StandaloneWithoutPreviousLine) {
  // Standalone tags should not require a newline to precede them.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{! I'm Still Standalone }}\n!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("!", Out);
}

TEST(MustacheComments, StandaloneWithoutNewline) {
  // Standalone tags should not require a newline to follow them.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("!\n  {{! I'm Still Standalone }}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("!\n", Out);
}

TEST(MustacheComments, MultilineStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n{{!\nSomething's going on here...\n}}\nEnd.\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedMultilineStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n  {{!\n    Something's going on here...\n  }}\nEnd.\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedInline) {
  // Inline comments should not strip whitespace.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  12 {{! 34 }}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  12 \n", Out);
}

TEST(MustacheComments, SurroundingWhitespace) {
  // Comment removal should preserve surrounding whitespace.
  Value D = {};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("12345 {{! Comment Block! }} 67890", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("12345  67890", Out);
}

TEST(MustacheComments, VariableNameCollision) {
  // Comments must never render, even if a variable with the same name exists.
  Value D = Object{
      {"! comment", 1}, {"! comment ", 2}, {"!comment", 3}, {"comment", 4}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("comments never show: >{{! comment }}<", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("comments never show: ><", Out);
}

// XFAIL: The following tests for the Triple Mustache feature are expected to
// fail. The assertions have been inverted from EXPECT_EQ to EXPECT_NE to allow
// them to pass against the current implementation. Once Triple Mustache is
// implemented, these assertions should be changed back to EXPECT_EQ.
TEST(MustacheTripleMustache, Basic) {
  Value D = Object{{"subject", "<b>World</b>"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Hello, {{{subject}}}!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, <b>World</b>!", Out);
}

TEST(MustacheTripleMustache, IntegerInterpolation) {
  Value D = Object{{"mph", 85}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{{mph}}} miles an hour!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheTripleMustache, DecimalInterpolation) {
  Value D = Object{{"power", 1.21}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{{power}}} jiggawatts!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1.21 jiggawatts!", Out);
}

TEST(MustacheTripleMustache, NullInterpolation) {
  Value D = Object{{"cannot", nullptr}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("I ({{{cannot}}}) be seen!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheTripleMustache, ContextMissInterpolation) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("I ({{{cannot}}}) be seen!", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheTripleMustache, DottedNames) {
  Value D = Object{{"person", Object{{"name", "<b>Joe</b>"}}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{{person.name}}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<b>Joe</b>", Out);
}

TEST(MustacheTripleMustache, ImplicitIterator) {
  Value D = Object{{"list", Array{"<a>", "<b>"}}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{#list}}({{{.}}}){{/list}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(<a>)(<b>)", Out);
}

TEST(MustacheTripleMustache, SurroundingWhitespace) {
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| {{{string}}} |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheTripleMustache, Standalone) {
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{{string}}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  ---\n", Out);
}

TEST(MustacheTripleMustache, WithPadding) {
  Value D = Object{{"string", "---"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{{ string }}}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheDelimiters, PairBehavior) {
  Value D = Object{{"text", "Hey!"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("{{=<% %>=}}(<%text%>)", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(Hey!)", Out);
}

TEST(MustacheDelimiters, SpecialCharacters) {
  Value D = Object{{"text", "It worked!"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("({{=[ ]=}}[text])", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(It worked!)", Out);
}

TEST(MustacheDelimiters, Sections) {
  Value D = Object{{"section", true}, {"data", "I got interpolated."}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[\n{{#section}}\n  {{data}}\n  |data|\n{{/section}}\n\n{{= "
             "| | =}}\n|#section|\n  {{data}}\n  |data|\n|/section|\n]\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[\n  I got interpolated.\n  |data|\n\n  {{data}}\n  I got "
            "interpolated.\n]\n",
            Out);
}

TEST(MustacheDelimiters, InvertedSections) {
  Value D = Object{{"section", false}, {"data", "I got interpolated."}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[\n{{^section}}\n  {{data}}\n  |data|\n{{/section}}\n\n{{= "
             "| | =}}\n|^section|\n  {{data}}\n  |data|\n|/section|\n]\n",
             Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[\n  I got interpolated.\n  |data|\n\n  {{data}}\n  I got "
            "interpolated.\n]\n",
            Out);
}

TEST(MustacheDelimiters, PartialInheritence) {
  Value D = Object{{"value", "yes"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[ {{>include}} ]\n{{= | | =}}\n[ |>include| ]\n", Ctx);
  T.registerPartial("include", ".{{value}}.");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[ .yes. ]\n[ .yes. ]\n", Out);
}

TEST(MustacheDelimiters, PostPartialBehavior) {
  Value D = Object{{"value", "yes"}};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("[ {{>include}} ]\n[ .{{value}}.  .|value|. ]\n", Ctx);
  T.registerPartial("include", ".{{value}}. {{= | | =}} .|value|.");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[ .yes.  .yes. ]\n[ .yes.  .|value|. ]\n", Out);
}

TEST(MustacheDelimiters, SurroundingWhitespace) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("| {{=@ @=}} |", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|  |", Out);
}

TEST(MustacheDelimiters, OutlyingWhitespaceInline) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T(" | {{=@ @=}}\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" | \n", Out);
}

TEST(MustacheDelimiters, StandaloneTag) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n{{=@ @=}}\nEnd.\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheDelimiters, IndentedStandaloneTag) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("Begin.\n  {{=@ @=}}\nEnd.\n", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheDelimiters, StandaloneLineEndings) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|\r\n{{= @ @ =}}\r\n|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheDelimiters, StandaloneWithoutPreviousLine) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("  {{=@ @=}}\n=", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("=", Out);
}

TEST(MustacheDelimiters, StandaloneWithoutNewline) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("=\n  {{=@ @=}}", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("=\n", Out);
}

TEST(MustacheDelimiters, PairwithPadding) {
  Value D = Object{};
  BumpPtrAllocator Allocator;
  StringSaver Saver(Allocator);
  MustacheContext Ctx(Allocator, Saver);
  Template T("|{{= @   @ =}}|", Ctx);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("||", Out);
}
