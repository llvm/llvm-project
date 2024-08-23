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
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::mustache;
using namespace llvm::json;

TEST(MustacheInterpolation, NoInterpolation) {
  // Mustache-free templates should render as-is.
  Value D = {};
  auto T = Template::createTemplate("Hello from {Mustache}!\n");
  auto Out = T.get().render(D);
  EXPECT_EQ("Hello from {Mustache}!\n", Out);
}

TEST(MustacheInterpolation, BasicInterpolation) {
  // Unadorned tags should interpolate content into the template.
  Value D = Object{{"subject", "World"}};
  auto T = Template::createTemplate("Hello, {{subject}}!");
  auto Out = T.get().render(D);
  EXPECT_EQ("Hello, World!", Out);
}

TEST(MustacheInterpolation, NoReinterpolation) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{{"template", "{{planet}}"}, {"planet", "Earth"}};
  auto T = Template::createTemplate("{{template}}: {{planet}}");
  auto Out = T.get().render(D);
  EXPECT_EQ("{{planet}}: Earth", Out);
}

TEST(MustacheInterpolation, HTMLEscaping) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{
      {"forbidden", "& \" < >"},
  };
  auto T = Template::createTemplate(
      "These characters should be HTML escaped: {{forbidden}}\n");
  auto Out = T.get().render(D);
  EXPECT_EQ("These characters should be HTML escaped: &amp; &quot; &lt; &gt;\n",
            Out);
}

TEST(MustacheInterpolation, Ampersand) {
  // Interpolated tag output should not be re-interpolated.
  Value D = Object{
      {"forbidden", "& \" < >"},
  };
  auto T = Template::createTemplate(
      "These characters should not be HTML escaped: {{&forbidden}}\n");
  auto Out = T.get().render(D);
  EXPECT_EQ("These characters should not be HTML escaped: & \" < >\n", Out);
}

TEST(MustacheInterpolation, BasicIntegerInterpolation) {
  Value D = Object{{"mph", 85}};
  auto T = Template::createTemplate("{{mph}} miles an hour!");
  auto Out = T.get().render(D);
  EXPECT_EQ("85 miles an hour!", Out);
}

TEST(MustacheInterpolation, BasicDecimalInterpolation) {
  Value D = Object{{"power", 1.21}};
  auto T = Template::createTemplate("{{power}} jiggawatts!");
  auto Out = T.get().render(D);
  EXPECT_EQ("1.21 jiggawatts!", Out);
}

TEST(MustacheInterpolation, BasicNullInterpolation) {
  Value D = Object{{"cannot", nullptr}};
  auto T = Template::createTemplate("I ({{cannot}}) be seen!");
  auto Out = T.get().render(D);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, BasicContextMissInterpolation) {
  Value D = Object{};
  auto T = Template::createTemplate("I ({{cannot}}) be seen!");
  auto Out = T.get().render(D);
  EXPECT_EQ("I () be seen!", Out);
}

TEST(MustacheInterpolation, DottedNamesBasicInterpolation) {
  Value D = Object{{"person", Object{{"name", "Joe"}}}};
  auto T = Template::createTemplate(
      "{{person.name}} == {{#person}}{{name}}{{/person}}");
  auto Out = T.get().render(D);
  EXPECT_EQ("Joe == Joe", Out);
}

TEST(MustacheInterpolation, DottedNamesArbitraryDepth) {
  Value D = Object{
      {"a",
       Object{{"b",
               Object{{"c",
                       Object{{"d",
                               Object{{"e", Object{{"name", "Phil"}}}}}}}}}}}};
  auto T = Template::createTemplate("{{a.b.c.d.e.name}} == Phil");
  auto Out = T.get().render(D);
  EXPECT_EQ("Phil == Phil", Out);
}

TEST(MustacheInterpolation, ImplicitIteratorsBasicInterpolation) {
  Value D = "world";
  auto T = Template::createTemplate("Hello, {{.}}!\n");
  auto Out = T.get().render(D);
  EXPECT_EQ("Hello, world!\n", Out);
}

TEST(MustacheInterpolation, InterpolationSurroundingWhitespace) {
  Value D = Object{{"string", "---"}};
  auto T = Template::createTemplate("| {{string}} |");
  auto Out = T.get().render(D);
  EXPECT_EQ("| --- |", Out);
}

TEST(MustacheInterpolation, InterpolationWithPadding) {
  Value D = Object{{"string", "---"}};
  auto T = Template::createTemplate("|{{ string }}|");
  auto Out = T.get().render(D);
  EXPECT_EQ("|---|", Out);
}
