//===- unittest/Support/YAMLParserTest ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/YAMLParser.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

static void SuppressDiagnosticsOutput(const SMDiagnostic &, void *) {
  // Prevent SourceMgr from writing errors to stderr
  // to reduce noise in unit test runs.
}

// Assumes Ctx is an SMDiagnostic where Diag can be stored.
static void CollectDiagnosticsOutput(const SMDiagnostic &Diag, void *Ctx) {
  SMDiagnostic* DiagOut = static_cast<SMDiagnostic*>(Ctx);
  *DiagOut = Diag;
}

// Checks that the given input gives a parse error. Makes sure that an error
// text is available and the parse fails.
static void ExpectParseError(StringRef Message, StringRef Input) {
  SourceMgr SM;
  yaml::Stream Stream(Input, SM);
  SM.setDiagHandler(SuppressDiagnosticsOutput);
  EXPECT_FALSE(Stream.validate()) << Message << ": " << Input;
  EXPECT_TRUE(Stream.failed()) << Message << ": " << Input;
}

// Checks that the given input can be parsed without error.
static void ExpectParseSuccess(StringRef Message, StringRef Input) {
  SourceMgr SM;
  yaml::Stream Stream(Input, SM);
  EXPECT_TRUE(Stream.validate()) << Message << ": " << Input;
}

TEST(YAMLParser, ParsesEmptyArray) {
  ExpectParseSuccess("Empty array", "[]");
}

TEST(YAMLParser, FailsIfNotClosingArray) {
  ExpectParseError("Not closing array", "[");
  ExpectParseError("Not closing array", "  [  ");
  ExpectParseError("Not closing array", "  [x");
}

TEST(YAMLParser, ParsesEmptyArrayWithWhitespace) {
  ExpectParseSuccess("Array with spaces", "  [  ]  ");
  ExpectParseSuccess("All whitespaces", "\t\r\n[\t\n \t\r ]\t\r \n\n");
}

TEST(YAMLParser, ParsesEmptyObject) {
  ExpectParseSuccess("Empty object", "[{}]");
}

TEST(YAMLParser, ParsesObject) {
  ExpectParseSuccess("Object with an entry", "[{\"a\":\"/b\"}]");
}

TEST(YAMLParser, ParsesMultipleKeyValuePairsInObject) {
  ExpectParseSuccess("Multiple key, value pairs",
                     "[{\"a\":\"/b\",\"c\":\"d\",\"e\":\"f\"}]");
}

TEST(YAMLParser, FailsIfNotClosingObject) {
  ExpectParseError("Missing close on empty", "[{]");
  ExpectParseError("Missing close after pair", "[{\"a\":\"b\"]");
}

TEST(YAMLParser, FailsIfMissingColon) {
  ExpectParseError("Missing colon between key and value", "[{\"a\"\"/b\"}]");
  ExpectParseError("Missing colon between key and value", "[{\"a\" \"b\"}]");
}

TEST(YAMLParser, FailsOnMissingQuote) {
  // Missing open quote counts as a plain scalar per YAML spec
  // (Following is equivalent to JSON [{"a\":\"b\"": null}])
  ExpectParseSuccess("Missing open quote", "[{a\":\"b\"}]");
  // Closing quote is more strict -- plain scalars cannot start with a quote
  ExpectParseError("Missing closing quote", "[{\"a\":\"b}]");
}

TEST(YAMLParser, ParsesEscapedQuotes) {
  ExpectParseSuccess("Parses escaped string in key and value",
                     "[{\"a\":\"\\\"b\\\"  \\\" \\\"\"}]");
}

TEST(YAMLParser, ParsesEmptyString) {
  ExpectParseSuccess("Parses empty string in value", "[{\"a\":\"\"}]");
}

TEST(YAMLParser, ParsesMultipleObjects) {
  ExpectParseSuccess(
      "Multiple objects in array",
      "["
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(YAMLParser, FailsOnMissingComma) {
  ExpectParseError(
      "Missing comma",
      "["
      " { \"a\" : \"b\" }"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(YAMLParser, ParsesSpacesInBetweenTokens) {
  ExpectParseSuccess(
      "Various whitespace between tokens",
      " \t \n\n \r [ \t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r,\t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r]\t \n\n \r");
}

TEST(YAMLParser, ParsesArrayOfArrays) {
  ExpectParseSuccess("Array of arrays", "[[]]");
}

TEST(YAMLParser, ParsesPlainScalars) {
  ExpectParseSuccess("Plain scalar", "hello");
  ExpectParseSuccess("Plain scalar beginning with a question mark", "?hello");
  ExpectParseSuccess("Plain scalar beginning with a colon", ":hello");
  ExpectParseSuccess("Plain scalar beginning with two colons", "::hello");
  ExpectParseSuccess("Plain scalar beginning with a hyphen", "-hello");
  ExpectParseSuccess("Multi-line plain scalar", "Hello\nworld");
  ExpectParseSuccess("Plain scalar with indicator characters",
                     "He-!l*lo, []world{}");
  ExpectParseSuccess("Plain scalar with indicator characters used as block key",
                     "He-!l*lo, []world{}: value");
  ExpectParseSuccess("Plain scalar in flow sequence", "hello");
  ExpectParseSuccess(
      "Plain scalar beginning with a question mark in flow sequence",
      "[ ?hello ]");
  ExpectParseSuccess("Plain scalar beginning with a colon in flow sequence",
                     "[ :hello ]");
  ExpectParseSuccess("Plain scalar beginning with two colons in flow sequence",
                     "[ ::hello ]");
  ExpectParseSuccess("Plain scalar beginning with a hyphen in flow sequence",
                     "[ -hello ]");
  ExpectParseSuccess("Multi-line plain scalar in flow sequence",
                     "[ Hello\nworld ]");
  ExpectParseSuccess(
      "Plain scalar with non-flow indicator characters in flow sequence",
      "[ He-!l*lo, world ]");
  ExpectParseSuccess(
      "Plain scalar with non-flow indicator characters used as flow key",
      "{ He-!l*lo, world: value } ");
  ExpectParseError(
      "Plain scalar with flow indicator characters inside flow sequence",
      "[ Hello[world ]");
  ExpectParseError(
      "Plain scalar with flow indicator characters inside flow key",
      "{ Hello[world: value }");
  // Multi-line plain scalar in keys is strictly invalid per the spec, but many
  // implementations accept it in flow keys nonetheless.  Block keys are not
  // accepted by any other implementation I can find.
  ExpectParseSuccess("Multi-line plain scalar in block key", "a\nb: c");
  ExpectParseSuccess("Multi-line plain scalar in flow key", "{\na\nb: c\n}");
}

TEST(YAMLParser, ParsesBlockLiteralScalars) {
  ExpectParseSuccess("Block literal scalar", "test: |\n  Hello\n  World\n");
  ExpectParseSuccess("Block literal scalar EOF", "test: |\n  Hello\n  World");
  ExpectParseSuccess("Empty block literal scalar header EOF", "test: | ");
  ExpectParseSuccess("Empty block literal scalar", "test: |\ntest2: 20");
  ExpectParseSuccess("Empty block literal scalar 2", "- | \n  \n\n \n- 42");
  ExpectParseSuccess("Block literal scalar in sequence",
                     "- |\n  Testing\n  Out\n\n- 22");
  ExpectParseSuccess("Block literal scalar in document",
                     "--- |\n  Document\n...");
  ExpectParseSuccess("Empty non indented lines still count",
                     "- |\n  First line\n \n\n  Another line\n\n- 2");
  ExpectParseSuccess("Comment in block literal scalar header",
                     "test: | # Comment \n  No Comment\ntest 2: | # Void");
  ExpectParseSuccess("Chomping indicators in block literal scalar header",
                     "test: |- \n  Hello\n\ntest 2: |+ \n\n  World\n\n\n");
  ExpectParseSuccess("Indent indicators in block literal scalar header",
                     "test: |1 \n  \n Hello \n  World\n");
  ExpectParseSuccess("Chomping and indent indicators in block literals",
                     "test: |-1\n Hello\ntest 2: |9+\n         World");
  ExpectParseSuccess("Trailing comments in block literals",
                     "test: |\n  Content\n # Trailing\n  #Comment\ntest 2: 3");
  ExpectParseError("Invalid block scalar header", "test: | failure");
  ExpectParseError("Invalid line indentation", "test: |\n  First line\n Error");
  ExpectParseError("Long leading space line", "test: |\n   \n  Test\n");
}

TEST(YAMLParser, NullTerminatedBlockScalars) {
  SourceMgr SM;
  yaml::Stream Stream("test: |\n  Hello\n  World\n", SM);
  yaml::Document &Doc = *Stream.begin();
  yaml::MappingNode *Map = cast<yaml::MappingNode>(Doc.getRoot());
  StringRef Value =
      cast<yaml::BlockScalarNode>(Map->begin()->getValue())->getValue();

  EXPECT_EQ(Value, "Hello\nWorld\n");
  EXPECT_EQ(Value.data()[Value.size()], '\0');
}

TEST(YAMLParser, HandlesEndOfFileGracefully) {
  ExpectParseError("In string starting with EOF", "[\"");
  ExpectParseError("In string hitting EOF", "[\"   ");
  ExpectParseError("In string escaping EOF", "[\"  \\");
  ExpectParseError("In array starting with EOF", "[");
  ExpectParseError("In array element starting with EOF", "[[], ");
  ExpectParseError("In array hitting EOF", "[[] ");
  ExpectParseError("In array hitting EOF", "[[]");
  ExpectParseError("In object hitting EOF", "{\"\"");
}

TEST(YAMLParser, HandlesNullValuesInKeyValueNodesGracefully) {
  ExpectParseError("KeyValueNode with null key", "? \"\n:");
  ExpectParseError("KeyValueNode with null value", "test: '");
}

// Checks that the given string can be parsed into an identical string inside
// of an array.
static void ExpectCanParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  SourceMgr SM;
  yaml::Stream Stream(StringInArray, SM);
  yaml::SequenceNode *ParsedSequence
    = dyn_cast<yaml::SequenceNode>(Stream.begin()->getRoot());
  StringRef ParsedString
    = dyn_cast<yaml::ScalarNode>(
      static_cast<yaml::Node*>(ParsedSequence->begin()))->getRawValue();
  ParsedString = ParsedString.substr(1, ParsedString.size() - 2);
  EXPECT_EQ(String, ParsedString.str());
}

// Checks that parsing the given string inside an array fails.
static void ExpectCannotParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  ExpectParseError((Twine("When parsing string \"") + String + "\"").str(),
                   StringInArray);
}

TEST(YAMLParser, ParsesStrings) {
  ExpectCanParseString("");
  ExpectCannotParseString("\\");
  ExpectCannotParseString("\"");
  ExpectCanParseString(" ");
  ExpectCanParseString("\\ ");
  ExpectCanParseString("\\\"");
  ExpectCannotParseString("\"\\");
  ExpectCannotParseString(" \\");
  ExpectCanParseString("\\\\");
  ExpectCannotParseString("\\\\\\");
  ExpectCanParseString("\\\\\\\\");
  ExpectCanParseString("\\\" ");
  ExpectCannotParseString("\\\\\" ");
  ExpectCanParseString("\\\\\\\" ");
  ExpectCanParseString("    \\\\  \\\"  \\\\\\\"   ");
}

TEST(YAMLParser, WorksWithIteratorAlgorithms) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\"]", SM);
  yaml::SequenceNode *Array
    = dyn_cast<yaml::SequenceNode>(Stream.begin()->getRoot());
  EXPECT_EQ(6, std::distance(Array->begin(), Array->end()));
}

TEST(YAMLParser, DefaultDiagnosticFilename) {
  SourceMgr SM;

  SMDiagnostic GeneratedDiag;
  SM.setDiagHandler(CollectDiagnosticsOutput, &GeneratedDiag);

  // When we construct a YAML stream over an unnamed string,
  // the filename is hard-coded as "YAML".
  yaml::Stream UnnamedStream("[]", SM);
  UnnamedStream.printError(UnnamedStream.begin()->getRoot(), "Hello, World!");
  EXPECT_EQ("YAML", GeneratedDiag.getFilename());
}

TEST(YAMLParser, DiagnosticFilenameFromBufferID) {
  SourceMgr SM;

  SMDiagnostic GeneratedDiag;
  SM.setDiagHandler(CollectDiagnosticsOutput, &GeneratedDiag);

  // When we construct a YAML stream over a named buffer,
  // we get its ID as filename in diagnostics.
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer("[]", "buffername.yaml");
  yaml::Stream Stream(Buffer->getMemBufferRef(), SM);
  Stream.printError(Stream.begin()->getRoot(), "Hello, World!");
  EXPECT_EQ("buffername.yaml", GeneratedDiag.getFilename());
}

TEST(YAMLParser, SameNodeIteratorOperatorNotEquals) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\"]", SM);

  yaml::SequenceNode *Node = dyn_cast<yaml::SequenceNode>(
                                              Stream.begin()->getRoot());

  auto Begin = Node->begin();
  auto End = Node->end();

  EXPECT_NE(Begin, End);
  EXPECT_EQ(Begin, Begin);
  EXPECT_EQ(End, End);
}

TEST(YAMLParser, SameNodeIteratorOperatorEquals) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\"]", SM);

  yaml::SequenceNode *Node = dyn_cast<yaml::SequenceNode>(
                                              Stream.begin()->getRoot());

  auto Begin = Node->begin();
  auto End = Node->end();

  EXPECT_NE(Begin, End);
  EXPECT_EQ(Begin, Begin);
  EXPECT_EQ(End, End);
}

TEST(YAMLParser, DifferentNodesIteratorOperatorNotEquals) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\"]", SM);
  yaml::Stream AnotherStream("[\"1\", \"2\"]", SM);

  yaml::SequenceNode *Node = dyn_cast<yaml::SequenceNode>(
                                                  Stream.begin()->getRoot());
  yaml::SequenceNode *AnotherNode = dyn_cast<yaml::SequenceNode>(
                                              AnotherStream.begin()->getRoot());

  auto Begin = Node->begin();
  auto End = Node->end();

  auto AnotherBegin = AnotherNode->begin();
  auto AnotherEnd = AnotherNode->end();

  EXPECT_NE(Begin, AnotherBegin);
  EXPECT_NE(Begin, AnotherEnd);
  EXPECT_EQ(End, AnotherEnd);
}

TEST(YAMLParser, DifferentNodesIteratorOperatorEquals) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\"]", SM);
  yaml::Stream AnotherStream("[\"1\", \"2\"]", SM);

  yaml::SequenceNode *Node = dyn_cast<yaml::SequenceNode>(
                                                    Stream.begin()->getRoot());
  yaml::SequenceNode *AnotherNode = dyn_cast<yaml::SequenceNode>(
                                             AnotherStream.begin()->getRoot());

  auto Begin = Node->begin();
  auto End = Node->end();

  auto AnotherBegin = AnotherNode->begin();
  auto AnotherEnd = AnotherNode->end();

  EXPECT_NE(Begin, AnotherBegin);
  EXPECT_NE(Begin, AnotherEnd);
  EXPECT_EQ(End, AnotherEnd);
}

TEST(YAMLParser, FlowSequenceTokensOutsideFlowSequence) {
  auto FlowSequenceStrs = {",", "]", "}"};
  SourceMgr SM;

  for (auto &Str : FlowSequenceStrs) {
    yaml::Stream Stream(Str, SM);
    yaml::Document &Doc = *Stream.begin();
    EXPECT_FALSE(Doc.skip());
  }
}

static void expectCanParseBool(StringRef S, bool Expected) {
  std::optional<bool> Parsed = yaml::parseBool(S);
  EXPECT_TRUE(Parsed.has_value());
  EXPECT_EQ(*Parsed, Expected);
}

static void expectCannotParseBool(StringRef S) {
  EXPECT_FALSE(yaml::parseBool(S).has_value());
}

TEST(YAMLParser, ParsesBools) {
  // Test true values.
  expectCanParseBool("ON", true);
  expectCanParseBool("On", true);
  expectCanParseBool("on", true);
  expectCanParseBool("TRUE", true);
  expectCanParseBool("True", true);
  expectCanParseBool("true", true);
  expectCanParseBool("Y", true);
  expectCanParseBool("y", true);
  expectCanParseBool("YES", true);
  expectCanParseBool("Yes", true);
  expectCanParseBool("yes", true);
  expectCannotParseBool("1");

  // Test false values.
  expectCanParseBool("FALSE", false);
  expectCanParseBool("False", false);
  expectCanParseBool("false", false);
  expectCanParseBool("N", false);
  expectCanParseBool("n", false);
  expectCanParseBool("NO", false);
  expectCanParseBool("No", false);
  expectCanParseBool("no", false);
  expectCanParseBool("OFF", false);
  expectCanParseBool("Off", false);
  expectCanParseBool("off", false);
  expectCannotParseBool("0");
}

} // end namespace llvm
