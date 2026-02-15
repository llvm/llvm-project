//===- LitTestGen.cpp - LIT test generator
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extracts embedded LIT tests from operation descriptions in tablegen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory
    litTestGenCategory("Options for -gen-lit-tests");
static llvm::cl::opt<std::string> outputDir(
    "output-dir", llvm::cl::desc("Output directory for generated test files"),
    llvm::cl::cat(litTestGenCategory), llvm::cl::value_desc("directory"));

struct LitTest {
  std::string sourceDefName;
  std::string testFileName;
  std::string irSnippet;
  llvm::SmallVector<std::string, 1> runLines;
  llvm::SmallVector<std::string> checkLines;
};

/// Extracts code snippets with mlir_example tag from a description field.
///
/// Returns a vector of LitTest objects found within ```mlir_example ... ```
/// blocks.
static llvm::SmallVector<LitTest>
extractOpTests(llvm::StringRef description, llvm::StringRef sourceDefName) {
  llvm::SmallVector<LitTest> tests;

  // Pattern to match ```mlir_example ... ``` code blocks
  // - ``` - Three literal backticks
  // - `mlir_example` - Literal text
  // - `(\(.+\))?` - Capture group matching the optional RUN tool name. Default
  // is `mlir-opt`.
  // - `([^`|`^`|``^`]+)` - Capture group matching the actual mlir IR example
  // content (everything except for three consecutive backticks).
  // - ``` - Three literal closing backticks
  llvm::Regex codeBlockRegex(
      "```mlir_example(\\([[:alnum:]_-]+\\))?[[:space:]]([^`|`^`|``^`]+)```");

  auto remaining = description;
  llvm::SmallVector<llvm::StringRef> matches;

  while (codeBlockRegex.match(remaining, &matches)) {
    if (matches.size() == 3) {
      std::string tool = "mlir-opt";
      // matches[1] contains the RUN tool name
      if (!matches[1].empty()) {
        tool = matches[1].ltrim('(').rtrim(')').str();
      }

      // matches[2] contains the code content
      auto codeRef = matches[2];
      // Remove leading/trailing whitespace and comment markers (# prefix)
      llvm::SmallVector<llvm::StringRef> lines;
      codeRef.split(lines, '\n', -1, false);

      std::string processedCode;
      for (llvm::StringRef line : lines) {
        auto isBody = true;
        line = line.ltrim();
        // Remove leading # comment markers if present
        if (line.starts_with("#")) {
          isBody = false;
          line = line.drop_front(1).ltrim();
        }
        if (!line.empty() || !processedCode.empty()) {
          auto tab = isBody ? "  " : "";
          processedCode += tab + line.str() + "\n";
        }
      }

      if (!processedCode.empty()) {
        // Generate test file name based on index
        auto testFileName = formatv("example_{0}.mlir", tests.size());
        // Generate default RUN line with --verify-roundtrip
        auto runLine =
            llvm::formatv("// RUN: {0} %s --verify-roundtrip", tool).str();

        tests.push_back(LitTest{
            sourceDefName.str(),
            testFileName,
            processedCode,
            {runLine},
            {} // No CHECK lines by default
        });
      }
    }

    // Move past this match to find the next one
    size_t matchEnd =
        remaining.find("```", remaining.find("```mlir_example") + 15);
    if (matchEnd == llvm::StringRef::npos)
      break;
    remaining = remaining.substr(matchEnd + 3);
  }

  return tests;
}

static llvm::SmallVector<LitTest>
extractTestsFromRecord(const llvm::Record *record) {
  llvm::SmallVector<LitTest> tests;

  // Try to extract mlir_example code blocks from the description field
  const llvm::RecordVal *descVal = record->getValue("description");
  if (!descVal)
    return tests;

  auto description = record->getValueAsString("description");
  if (description.empty())
    return tests;

  if (record->isSubClassOf("Op")) {
    tests = extractOpTests(description, record->getName());
  }

  return tests;
}

/// Generates a LIT test file for an IR test
static void generateTestFile(const LitTest &test, llvm::raw_ostream &os) {
  // Add RUN lines
  for (const auto &runLine : test.runLines) {
    os << "\n" << runLine << "\n";
  }

  os << "// Generated from TableGen definition: " << test.sourceDefName
     << "\n\n";

  // Add the test body
  os << test.irSnippet << "\n";

  // Add CHECK lines
  for (const auto &checkLine : test.checkLines) {
    os << "\n" << checkLine << "\n";
  }
}

/// Main function to generate all IR test test files
static void generateLitTests(const RecordKeeper &records, raw_ostream &os) {
  assert(records.getClass("Op") && "Undefined TableGen class type: Op");

  llvm::SmallVector<LitTest> allTests;

  llvm::SmallVector<StringRef, 2> testTypes{"Op"};
  for (const llvm::Record *def : records.getAllDerivedDefinitions(testTypes)) {
    if (def->isAnonymous())
      continue;

    auto opTests = extractTestsFromRecord(def);
    allTests.insert(allTests.end(), opTests.begin(), opTests.end());
  }

  if (allTests.empty()) {
    os << "// No mlir_example code blocks found in any TableGen definition\n";
    return;
  }

  // Generate summary
  os << "// Generated " << allTests.size() << " LIT test files\n";
  os << "// Use the following files for LIT testing:\n\n";

  // Generate file list and content for each test
  for (const auto &test : allTests) {
    std::string testFileName =
        formatv("generated_{0}_{1}", test.sourceDefName, test.testFileName);
    os << "// File: " << testFileName << "\n";

    os << "// --- BEGIN " << testFileName << " ---\n";
    generateTestFile(test, os);
    os << "// --- END " << testFileName << " ---\n\n";
  }
}

//===----------------------------------------------------------------------===//
// Generator Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genLitTests("gen-lit-tests",
                "Generate LIT test files for `Testable` TableGen records",
                [](const RecordKeeper &records, raw_ostream &os) {
                  generateLitTests(records, os);
                  return false;
                });
