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
  auto T = Template("Hello from {Mustache}!\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello from {Mustache}!\n", Out);
}

TEST(MustacheInterpolation, BasicInterpolation) {
  // Unadorned tags should interpolate content into the template.
  Value D = Object{{"subject", "World"}};
  auto T = Template("Hello, {{subject}}!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheInterpolation, NoReinterpolation) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{{"template", "{{planet}}"}, {"planet", "Earth"}};
  auto T = Template("{{template}}: {{planet}}");
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
  auto T = Template("These characters should be HTML escaped: {{forbidden}}\n");
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
  auto T =
      Template("These characters should not be HTML escaped: {{&forbidden}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("These characters should not be HTML escaped: & \" < >\n", Out);
}

TEST(MustacheInterpolation, BasicIntegerInterpolation) {
  // Integers should interpolate seamlessly.
  Value D = Object{{"mph", 85}};
  auto T = Template("{{mph}} miles an hour!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheInterpolation, AmpersandIntegerInterpolation) {
  // Integers should interpolate seamlessly.
  Value D = Object{{"mph", 85}};
  auto T = Template("{{&mph}} miles an hour!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheInterpolation, BasicDecimalInterpolation) {
  // Decimals should interpolate seamlessly with proper significance.
  Value D = Object{{"power", 1.21}};
  auto T = Template("{{power}} jiggawatts!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1.21 jiggawatts!", Out);
}

TEST(MustacheInterpolation, BasicNullInterpolation) {
  // Nulls should interpolate as the empty string.
  Value D = Object{{"cannot", nullptr}};
  auto T = Template("I ({{cannot}}) be seen!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, AmpersandNullInterpolation) {
  // Nulls should interpolate as the empty string.
  Value D = Object{{"cannot", nullptr}};
  auto T = Template("I ({{&cannot}}) be seen!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, BasicContextMissInterpolation) {
  // Failed context lookups should default to empty strings.
  Value D = Object{};
  auto T = Template("I ({{cannot}}) be seen!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, DottedNamesBasicInterpolation) {
  // Dotted names should be considered a form of shorthand for sections.
  Value D = Object{{"person", Object{{"name", "Joe"}}}};
  auto T = Template("{{person.name}} == {{#person}}{{name}}{{/person}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Joe == Joe", Out);
}

TEST(MustacheInterpolation, DottedNamesAmpersandInterpolation) {
  // Dotted names should be considered a form of shorthand for sections.
  Value D = Object{{"person", Object{{"name", "Joe"}}}};
  auto T = Template("{{&person.name}} == {{#person}}{{&name}}{{/person}}");
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
  auto T = Template("{{a.b.c.d.e.name}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Phil", Out);
}

TEST(MustacheInterpolation, DottedNamesBrokenChains) {
  // Any falsey value prior to the last part of the name should yield ''.
  Value D = Object{{"a", Object{}}};
  auto T = Template("{{a.b.c}} == ");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheInterpolation, DottedNamesBrokenChainResolution) {
  // Each part of a dotted name should resolve only against its parent.
  Value D =
      Object{{"a", Object{{"b", Object{}}}}, {"c", Object{{"name", "Jim"}}}};
  auto T = Template("{{a.b.c.name}} == ");
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
  auto T = Template("{{#a}}{{b.c.d.e.name}}{{/a}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Phil", Out);
}

TEST(MustacheInterpolation, DottedNamesContextPrecedence) {
  // Dotted names should be resolved against former resolutions.
  Value D =
      Object{{"a", Object{{"b", Object{}}}}, {"b", Object{{"c", "ERROR"}}}};
  auto T = Template("{{#a}}{{b.c}}{{/a}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, DottedNamesAreNotSingleKeys) {
  // Dotted names shall not be parsed as single, atomic keys
  Value D = Object{{"a.b", "c"}};
  auto T = Template("{{a.b}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, DottedNamesNoMasking) {
  // Dotted Names in a given context are unavailable due to dot splitting
  Value D = Object{{"a.b", "c"}, {"a", Object{{"b", "d"}}}};
  auto T = Template("{{a.b}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("d", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsBasicInterpolation) {
  // Unadorned tags should interpolate content into the template.
  Value D = "world";
  auto T = Template("Hello, {{.}}!\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, world!\n", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsAmersand) {
  // Basic interpolation should be HTML escaped.
  Value D = "& \" < >";
  auto T = Template("These characters should not be HTML escaped: {{&.}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("These characters should not be HTML escaped: & \" < >\n", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsInteger) {
  // Integers should interpolate seamlessly.
  Value D = 85;
  auto T = Template("{{.}} miles an hour!\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("85 miles an hour!\n", Out);
}

TEST(MustacheInterpolation, InterpolationSurroundingWhitespace) {
  // Interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  auto T = Template("| {{string}} |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheInterpolation, AmersandSurroundingWhitespace) {
  // Interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  auto T = Template("| {{&string}} |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheInterpolation, StandaloneInterpolationWithWhitespace) {
  // Standalone interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  auto T = Template("  {{string}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  ---\n", Out);
}

TEST(MustacheInterpolation, StandaloneAmpersandWithWhitespace) {
  // Standalone interpolation should not alter surrounding whitespace.
  Value D = Object{{"string", "---"}};
  auto T = Template("  {{&string}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  ---\n", Out);
}

TEST(MustacheInterpolation, InterpolationWithPadding) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  auto T = Template("|{{ string }}|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheInterpolation, AmpersandWithPadding) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  auto T = Template("|{{& string }}|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheInterpolation, InterpolationWithPaddingAndNewlines) {
  // Superfluous in-tag whitespace should be ignored.
  Value D = Object{{"string", "---"}};
  auto T = Template("|{{ string \n\n\n }}|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|---|", Out);
}

TEST(MustacheSections, Truthy) {
  Value D = Object{{"boolean", true}};
  auto T = Template("{{#boolean}}This should be rendered.{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheSections, Falsey) {
  Value D = Object{{"boolean", false}};
  auto T = Template("{{#boolean}}This should not be rendered.{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInterpolation, IsFalseyNull) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", nullptr}};
  auto T = Template("Hello, {{#boolean}}World{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, ", Out);
}

TEST(MustacheInterpolation, IsFalseyArray) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", Array()}};
  auto T = Template("Hello, {{#boolean}}World{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, ", Out);
}

TEST(MustacheInterpolation, IsFalseyObject) {
  // Mustache-free templates should render as-is.
  Value D = Object{{"boolean", Object{}}};
  auto T = Template("Hello, {{#boolean}}World{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World", Out);
}

TEST(MustacheInterpolation, DoubleRendering) {
  // Mustache-free templates should render as-is.
  Value D1 = Object{{"subject", "World"}};
  auto T = Template("Hello, {{subject}}!");
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
  auto T = Template("{{#null}}This should not be rendered.{{/null}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheSections, Context) {
  Value D = Object{{"context", Object{{"name", "Joe"}}}};
  auto T = Template("{{#context}}Hi {{name}}.{{/context}}");
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
  auto T = Template("{{#sec}}{{a}}, {{b}}, {{c.d}}{{/sec}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("foo, bar, baz", Out);
}

TEST(MustacheSections, VariableTest) {
  Value D = Object{{"foo", "bar"}};
  auto T = Template("{{#foo}}{{.}} is {{foo}}{{/foo}}");
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
  auto T = Template("{{#tops}}"
                    "{{#middles}}"
                    "{{tname.lower}}{{mname}}."
                    "{{#bottoms}}"
                    "{{tname.upper}}{{mname}}{{bname}}."
                    "{{/bottoms}}"
                    "{{/middles}}"
                    "{{/tops}}");
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
  auto T = Template(
      "{{#a}}\n{{one}}\n{{#b}}\n{{one}}{{two}}{{one}}\n{{#c}}\n{{one}}{{two}}{{"
      "three}}{{two}}{{one}}\n{{#d}}\n{{one}}{{two}}{{three}}{{four}}{{three}}{"
      "{two}}{{one}}\n{{#five}}\n{{one}}{{two}}{{three}}{{four}}{{five}}{{four}"
      "}{{three}}{{two}}{{one}}\n{{one}}{{two}}{{three}}{{four}}{{.}}6{{.}}{{"
      "four}}{{three}}{{two}}{{one}}\n{{one}}{{two}}{{three}}{{four}}{{five}}{{"
      "four}}{{three}}{{two}}{{one}}\n{{/"
      "five}}\n{{one}}{{two}}{{three}}{{four}}{{three}}{{two}}{{one}}\n{{/"
      "d}}\n{{one}}{{two}}{{three}}{{two}}{{one}}\n{{/"
      "c}}\n{{one}}{{two}}{{one}}\n{{/b}}\n{{one}}\n{{/a}}\n");
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
  auto T = Template("{{#list}}{{item}}{{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("123", Out);
}

TEST(MustacheSections, EmptyList) {
  Value D = Object{{"list", Array{}}};
  auto T = Template("{{#list}}Yay lists!{{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheSections, Doubled) {
  Value D = Object{{"bool", true}, {"two", "second"}};
  auto T = Template("{{#bool}}\n* first\n{{/bool}}\n* "
                    "{{two}}\n{{#bool}}\n* third\n{{/bool}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("* first\n* second\n* third\n", Out);
}

TEST(MustacheSections, NestedTruthy) {
  Value D = Object{{"bool", true}};
  auto T = Template("| A {{#bool}}B {{#bool}}C{{/bool}} D{{/bool}} E |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A B C D E |", Out);
}

TEST(MustacheSections, NestedFalsey) {
  Value D = Object{{"bool", false}};
  auto T = Template("| A {{#bool}}B {{#bool}}C{{/bool}} D{{/bool}} E |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A  E |", Out);
}

TEST(MustacheSections, ContextMisses) {
  Value D = Object{};
  auto T = Template("[{{#missing}}Found key 'missing'!{{/missing}}]");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[]", Out);
}

TEST(MustacheSections, ImplicitIteratorString) {
  Value D = Object{{"list", Array{"a", "b", "c", "d", "e"}}};
  auto T = Template("{{#list}}({{.}}){{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(a)(b)(c)(d)(e)", Out);
}

TEST(MustacheSections, ImplicitIteratorInteger) {
  Value D = Object{{"list", Array{1, 2, 3, 4, 5}}};
  auto T = Template("{{#list}}({{.}}){{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(1)(2)(3)(4)(5)", Out);
}

TEST(MustacheSections, ImplicitIteratorArray) {
  Value D = Object{{"list", Array{Array{1, 2, 3}, Array{"a", "b", "c"}}}};
  auto T = Template("{{#list}}({{#.}}{{.}}{{/.}}){{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(123)(abc)", Out);
}

TEST(MustacheSections, ImplicitIteratorHTMLEscaping) {
  Value D = Object{{"list", Array{"&", "\"", "<", ">"}}};
  auto T = Template("{{#list}}({{.}}){{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(&amp;)(&quot;)(&lt;)(&gt;)", Out);
}

TEST(MustacheSections, ImplicitIteratorAmpersand) {
  Value D = Object{{"list", Array{"&", "\"", "<", ">"}}};
  auto T = Template("{{#list}}({{&.}}){{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(&)(\")(<)(>)", Out);
}

TEST(MustacheSections, ImplicitIteratorRootLevel) {
  Value D = Array{Object{{"value", "a"}}, Object{{"value", "b"}}};
  auto T = Template("{{#.}}({{value}}){{/.}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("(a)(b)", Out);
}

TEST(MustacheSections, DottedNamesTruthy) {
  Value D = Object{{"a", Object{{"b", Object{{"c", true}}}}}};
  auto T = Template("{{#a.b.c}}Here{{/a.b.c}} == Here");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Here == Here", Out);
}

TEST(MustacheSections, DottedNamesFalsey) {
  Value D = Object{{"a", Object{{"b", Object{{"c", false}}}}}};
  auto T = Template("{{#a.b.c}}Here{{/a.b.c}} == ");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheSections, DottedNamesBrokenChains) {
  Value D = Object{{"a", Object{}}};
  auto T = Template("{{#a.b.c}}Here{{/a.b.c}} == ");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheSections, SurroundingWhitespace) {
  Value D = Object{{"boolean", true}};
  auto T = Template(" | {{#boolean}}\t|\t{{/boolean}} | \n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" | \t|\t | \n", Out);
}

TEST(MustacheSections, InternalWhitespace) {
  Value D = Object{{"boolean", true}};
  auto T = Template(
      " | {{#boolean}} {{! Important Whitespace }}\n {{/boolean}} | \n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" |  \n  | \n", Out);
}

TEST(MustacheSections, IndentedInlineSections) {
  Value D = Object{{"boolean", true}};
  auto T =
      Template(" {{#boolean}}YES{{/boolean}}\n {{#boolean}}GOOD{{/boolean}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" YES\n GOOD\n", Out);
}

TEST(MustacheSections, StandaloneLines) {
  Value D = Object{{"boolean", true}};
  auto T = Template("| This Is\n{{#boolean}}\n|\n{{/boolean}}\n| A Line\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheSections, IndentedStandaloneLines) {
  Value D = Object{{"boolean", true}};
  auto T = Template("| This Is\n  {{#boolean}}\n|\n  {{/boolean}}\n| A Line\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheSections, StandaloneLineEndings) {
  Value D = Object{{"boolean", true}};
  auto T = Template("|\r\n{{#boolean}}\r\n{{/boolean}}\r\n|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheSections, StandaloneWithoutPreviousLine) {
  Value D = Object{{"boolean", true}};
  auto T = Template("  {{#boolean}}\n#{{/boolean}}\n/");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("#\n/", Out);
}

TEST(MustacheSections, StandaloneWithoutNewline) {
  Value D = Object{{"boolean", true}};
  auto T = Template("#{{#boolean}}\n/\n  {{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("#\n/\n", Out);
}

TEST(MustacheSections, Padding) {
  Value D = Object{{"boolean", true}};
  auto T = Template("|{{# boolean }}={{/ boolean }}|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|=|", Out);
}

TEST(MustacheInvertedSections, Falsey) {
  Value D = Object{{"boolean", false}};
  auto T = Template("{{^boolean}}This should be rendered.{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheInvertedSections, Truthy) {
  Value D = Object{{"boolean", true}};
  auto T = Template("{{^boolean}}This should not be rendered.{{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, NullIsFalsey) {
  Value D = Object{{"null", nullptr}};
  auto T = Template("{{^null}}This should be rendered.{{/null}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("This should be rendered.", Out);
}

TEST(MustacheInvertedSections, Context) {
  Value D = Object{{"context", Object{{"name", "Joe"}}}};
  auto T = Template("{{^context}}Hi {{name}}.{{/context}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, List) {
  Value D = Object{
      {"list", Array{Object{{"n", 1}}, Object{{"n", 2}}, Object{{"n", 3}}}}};
  auto T = Template("{{^list}}{{n}}{{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustacheInvertedSections, EmptyList) {
  Value D = Object{{"list", Array{}}};
  auto T = Template("{{^list}}Yay lists!{{/list}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Yay lists!", Out);
}

TEST(MustacheInvertedSections, Doubled) {
  Value D = Object{{"bool", false}, {"two", "second"}};
  auto T = Template("{{^bool}}\n* first\n{{/bool}}\n* "
                    "{{two}}\n{{^bool}}\n* third\n{{/bool}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("* first\n* second\n* third\n", Out);
}

TEST(MustacheInvertedSections, NestedFalsey) {
  Value D = Object{{"bool", false}};
  auto T = Template("| A {{^bool}}B {{^bool}}C{{/bool}} D{{/bool}} E |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A B C D E |", Out);
}

TEST(MustacheInvertedSections, NestedTruthy) {
  Value D = Object{{"bool", true}};
  auto T = Template("| A {{^bool}}B {{^bool}}C{{/bool}} D{{/bool}} E |");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| A  E |", Out);
}

TEST(MustacheInvertedSections, ContextMisses) {
  Value D = Object{};
  auto T = Template("[{{^missing}}Cannot find key 'missing'!{{/missing}}]");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("[Cannot find key 'missing'!]", Out);
}

TEST(MustacheInvertedSections, DottedNamesTruthy) {
  Value D = Object{{"a", Object{{"b", Object{{"c", true}}}}}};
  auto T = Template("{{^a.b.c}}Not Here{{/a.b.c}} == ");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" == ", Out);
}

TEST(MustacheInvertedSections, DottedNamesFalsey) {
  Value D = Object{{"a", Object{{"b", Object{{"c", false}}}}}};
  auto T = Template("{{^a.b.c}}Not Here{{/a.b.c}} == Not Here");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Not Here == Not Here", Out);
}

TEST(MustacheInvertedSections, DottedNamesBrokenChains) {
  Value D = Object{{"a", Object{}}};
  auto T = Template("{{^a.b.c}}Not Here{{/a.b.c}} == Not Here");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Not Here == Not Here", Out);
}

TEST(MustacheInvertedSections, SurroundingWhitespace) {
  Value D = Object{{"boolean", false}};
  auto T = Template(" | {{^boolean}}\t|\t{{/boolean}} | \n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" | \t|\t | \n", Out);
}

TEST(MustacheInvertedSections, InternalWhitespace) {
  Value D = Object{{"boolean", false}};
  auto T = Template(
      " | {{^boolean}} {{! Important Whitespace }}\n {{/boolean}} | \n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" |  \n  | \n", Out);
}

TEST(MustacheInvertedSections, IndentedInlineSections) {
  Value D = Object{{"boolean", false}};
  auto T =
      Template(" {{^boolean}}NO{{/boolean}}\n {{^boolean}}WAY{{/boolean}}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ(" NO\n WAY\n", Out);
}

TEST(MustacheInvertedSections, StandaloneLines) {
  Value D = Object{{"boolean", false}};
  auto T = Template("| This Is\n{{^boolean}}\n|\n{{/boolean}}\n| A Line\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheInvertedSections, StandaloneIndentedLines) {
  Value D = Object{{"boolean", false}};
  auto T = Template("| This Is\n  {{^boolean}}\n|\n  {{/boolean}}\n| A Line\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| This Is\n|\n| A Line\n", Out);
}

TEST(MustacheInvertedSections, StandaloneLineEndings) {
  Value D = Object{{"boolean", false}};
  auto T = Template("|\r\n{{^boolean}}\r\n{{/boolean}}\r\n|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheInvertedSections, StandaloneWithoutPreviousLine) {
  Value D = Object{{"boolean", false}};
  auto T = Template("  {{^boolean}}\n^{{/boolean}}\n/");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("^\n/", Out);
}

TEST(MustacheInvertedSections, StandaloneWithoutNewline) {
  Value D = Object{{"boolean", false}};
  auto T = Template("^{{^boolean}}\n/\n  {{/boolean}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("^\n/\n", Out);
}

TEST(MustacheInvertedSections, Padding) {
  Value D = Object{{"boolean", false}};
  auto T = Template("|{{^ boolean }}={{/ boolean }}|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|=|", Out);
}

TEST(MustachePartials, BasicBehavior) {
  Value D = Object{};
  auto T = Template("{{>text}}");
  T.registerPartial("text", "from partial");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("from partial", Out);
}

TEST(MustachePartials, FailedLookup) {
  Value D = Object{};
  auto T = Template("{{>text}}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("", Out);
}

TEST(MustachePartials, Context) {
  Value D = Object{{"text", "content"}};
  auto T = Template("{{>partial}}");
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
  auto T = Template("{{>node}}");
  T.registerPartial("node", "{{content}}({{#nodes}}{{>node}}{{/nodes}})");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("X(Y())", Out);
}

TEST(MustachePartials, Nested) {
  Value D = Object{{"a", "hello"}, {"b", "world"}};
  auto T = Template("{{>outer}}");
  T.registerPartial("outer", "*{{a}} {{>inner}}*");
  T.registerPartial("inner", "{{b}}!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("*hello world!*", Out);
}

TEST(MustachePartials, SurroundingWhitespace) {
  Value D = Object{};
  auto T = Template("| {{>partial}} |");
  T.registerPartial("partial", "\t|\t");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("| \t|\t |", Out);
}

TEST(MustachePartials, InlineIndentation) {
  Value D = Object{{"data", "|"}};
  auto T = Template("  {{data}}  {{> partial}}\n");
  T.registerPartial("partial", "<\n<");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  |  <\n<\n", Out);
}

TEST(MustachePartials, PaddingWhitespace) {
  Value D = Object{{"boolean", true}};
  auto T = Template("|{{> partial }}|");
  T.registerPartial("partial", "[]");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|[]|", Out);
}

TEST(MustacheLambdas, BasicInterpolation) {
  Value D = Object{};
  auto T = Template("Hello, {{lambda}}!");
  Lambda L = []() -> llvm::json::Value { return "World"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheLambdas, InterpolationExpansion) {
  Value D = Object{{"planet", "World"}};
  auto T = Template("Hello, {{lambda}}!");
  Lambda L = []() -> llvm::json::Value { return "{{planet}}"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheLambdas, BasicMultipleCalls) {
  Value D = Object{};
  auto T = Template("{{lambda}} == {{lambda}} == {{lambda}}");
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
  auto T = Template("<{{lambda}}{{&lambda}}");
  Lambda L = []() -> llvm::json::Value { return ">"; };
  T.registerLambda("lambda", L);
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("<&gt;>", Out);
}

TEST(MustacheLambdas, Sections) {
  Value D = Object{};
  auto T = Template("<{{#lambda}}{{x}}{{/lambda}}>");
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
  auto T = Template("<{{#lambda}}-{{/lambda}}>");
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
  auto T = Template("{{#lambda}}FILE{{/lambda}} != {{#lambda}}LINE{{/lambda}}");
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
  auto T = Template("<{{^lambda}}{{static}}{{/lambda}}>");
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
  auto T = Template("12345{{! Comment Block! }}67890");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1234567890", Out);
}

TEST(MustacheComments, Multiline) {
  // Multiline comments should be permitted.
  Value D = {};
  auto T =
      Template("12345{{!\n  This is a\n  multi-line comment...\n}}67890\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("1234567890\n", Out);
}

TEST(MustacheComments, Standalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  auto T = Template("Begin.\n{{! Comment Block! }}\nEnd.\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  auto T = Template("Begin.\n  {{! Indented Comment Block! }}\nEnd.\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, StandaloneLineEndings) {
  // "\r\n" should be considered a newline for standalone tags.
  Value D = {};
  auto T = Template("|\r\n{{! Standalone Comment }}\r\n|");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("|\r\n|", Out);
}

TEST(MustacheComments, StandaloneWithoutPreviousLine) {
  // Standalone tags should not require a newline to precede them.
  Value D = {};
  auto T = Template("  {{! I'm Still Standalone }}\n!");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("!", Out);
}

TEST(MustacheComments, StandaloneWithoutNewline) {
  // Standalone tags should not require a newline to follow them.
  Value D = {};
  auto T = Template("!\n  {{! I'm Still Standalone }}");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("!\n", Out);
}

TEST(MustacheComments, MultilineStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  auto T = Template("Begin.\n{{!\nSomething's going on here...\n}}\nEnd.\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedMultilineStandalone) {
  // All standalone comment lines should be removed.
  Value D = {};
  auto T =
      Template("Begin.\n  {{!\n    Something's going on here...\n  }}\nEnd.\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("Begin.\nEnd.\n", Out);
}

TEST(MustacheComments, IndentedInline) {
  // Inline comments should not strip whitespace.
  Value D = {};
  auto T = Template("  12 {{! 34 }}\n");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("  12 \n", Out);
}

TEST(MustacheComments, SurroundingWhitespace) {
  // Comment removal should preserve surrounding whitespace.
  Value D = {};
  auto T = Template("12345 {{! Comment Block! }} 67890");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("12345  67890", Out);
}

TEST(MustacheComments, VariableNameCollision) {
  // Comments must never render, even if a variable with the same name exists.
  Value D = Object{
      {"! comment", 1}, {"! comment ", 2}, {"!comment", 3}, {"comment", 4}};
  auto T = Template("comments never show: >{{! comment }}<");
  std::string Out;
  raw_string_ostream OS(Out);
  T.render(D, OS);
  EXPECT_EQ("comments never show: ><", Out);
}
