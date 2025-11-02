//===- LitTestGen.cpp - LIT test generator ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LitTestGen extracts `LitTest` records from `Testable` TableGen records and 
// generates corresponding LIT test files.
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

#include <set>

using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory litTestGenCategory("Options for -gen-lit-tests");
static llvm::cl::opt<std::string>
    outputDir("output-dir", 
              llvm::cl::desc("Output directory for generated test files"),
              llvm::cl::cat(litTestGenCategory), 
              llvm::cl::value_desc("directory"));


/// Cpp type corresponding to the `LitTest` record type in TableGen
struct LitTest {
  std::string sourceDefName;
  std::string testFileName;
  std::string irSnippet;
  llvm::SmallVector<std::string> runLines;
  llvm::SmallVector<std::string> checkLines;
};

/// Extract code snippets with mlir_example tag from a description field.
/// Returns a vector of code snippets found within ```mlir_example ... ``` blocks.
static llvm::SmallVector<std::string> extractMlirExamples(llvm::StringRef description) {
  llvm::SmallVector<std::string> examples;

  // Pattern to match ```mlir_example ... ``` code blocks
  // [^\n]* matches rest of line after mlir_example
  // \n matches the newline after the opening fence
  // (.+?) captures the code content (non-greedy)
  // ``` matches the closing fence
  llvm::Regex codeBlockRegex("```mlir_example(.+)```");

  llvm::StringRef remaining = description;
  llvm::SmallVector<llvm::StringRef> matches;

  while (codeBlockRegex.match(remaining, &matches)) {
    if (matches.size() >= 2) {
      // matches[1] contains the captured group (the code content)
      std::string code = matches[1].str();

      llvm::errs() << "DEBUG: Extracted raw code:\n[" << code << "]\n";

      // Remove leading/trailing whitespace and comment markers (# prefix)
      llvm::SmallVector<llvm::StringRef> lines;
      llvm::StringRef codeRef(code);
      codeRef.split(lines, '\n', -1, false);

      std::string processedCode;
      for (llvm::StringRef line : lines) {
        line = line.ltrim();
        // Remove leading # comment markers if present
        if (line.starts_with("#")) {
          line = line.drop_front(1).ltrim();
        }
        if (!line.empty() || !processedCode.empty()) {
          processedCode += line.str() + "\n";
        }
      }

      // // Remove trailing empty lines
      // while (!processedCode.empty() && processedCode.back() == '\n') {
      //   size_t lastNewline = processedCode.find_last_not_of('\n');
      //   if (lastNewline == std::string::npos) {
      //     processedCode.clear();
      //     break;
      //   }
      //   processedCode = processedCode.substr(0, lastNewline + 1) + "\n";
      // }

      if (!processedCode.empty()) {
        examples.push_back(processedCode);
      }
    }

    // Move past this match to find the next one
    size_t matchEnd = remaining.find("```", remaining.find("```mlir_example") + 15);
    if (matchEnd == llvm::StringRef::npos)
      break;
    remaining = remaining.substr(matchEnd + 3);
  }

  return examples;
}

static llvm::SmallVector<LitTest> extractTestsFromRecord(const llvm::Record *record,
                                                         llvm::StringRef dialectName = "") {
  llvm::SmallVector<LitTest> tests;

  // Try to extract mlir_example code blocks from the description field
  const llvm::RecordVal *descVal = record->getValue("description");
  if (descVal) {
    llvm::StringRef description = record->getValueAsString("description");
    llvm::errs() << "DEBUG: Record: " << record->getName() << "\n";
    llvm::errs() << "DEBUG: Description length: " << description.size() << "\n";
    llvm::errs() << "DEBUG: Description content:\n" << description << "\n";
    llvm::errs() << "DEBUG: ---\n";
    if (!description.empty()) {
      llvm::SmallVector<std::string> examples = extractMlirExamples(description);
      llvm::errs() << "DEBUG: Found " << examples.size() << " examples\n";

      // Create a LitTest for each extracted example
      for (size_t i = 0; i < examples.size(); ++i) {
        std::string testFileName;
        if (examples.size() == 1) {
          testFileName = "example.mlir";
        } else {
          testFileName = formatv("example_{0}.mlir", i);
        }

        // Generate default RUN line with --verify-roundtrip
        llvm::SmallVector<std::string> runLines;
        runLines.push_back("// RUN: mlir-opt %s --verify-roundtrip");

        tests.push_back(LitTest {
          record->getName().str(),
          testFileName,
          examples[i],
          runLines,
          {} // No CHECK lines by default
        });
      }
    }
  }

  // Fall back to checking for the old tests field for backward compatibility
  const llvm::RecordVal *testsVal = record->getValue("tests");
  if (!testsVal)
    return tests;

  const llvm::ListInit *testsList =
    llvm::dyn_cast_or_null<llvm::ListInit>(testsVal->getValue());
  if (!testsList)
    return tests;

  for (const llvm::Init *init : testsList->getElements()) {
    const llvm::DefInit *defInit = llvm::dyn_cast<llvm::DefInit>(init);
    if (!defInit)
      continue;

    const llvm::Record *testRec = defInit->getDef();

    // Extract fields from LitTest record
    std::string name = testRec->getValueAsString("testFileName").str();
    std::string irSnippet = testRec->getValueAsString("irSnippet").str();

    llvm::SmallVector<std::string> runLines;
    llvm::for_each(*testRec->getValueAsListInit("runLines"), [&](const llvm::Init *init) {
      runLines.emplace_back(llvm::cast<llvm::StringInit>(init)->getValue());
    });

    llvm::SmallVector<std::string> checkLines;
    llvm::for_each(*testRec->getValueAsListInit("checkLines"), [&](const llvm::Init *init) {
      checkLines.emplace_back(llvm::cast<llvm::StringInit>(init)->getValue());
    });

    tests.push_back(LitTest {
      record->getName().str(),
      name,
      irSnippet,
      runLines,
      checkLines,
    });
  }

  return tests;
}

/// Extract tests from ops
static llvm::SmallVector<LitTest> extractOpTests(const RecordKeeper &records) {
  llvm::SmallVector<LitTest> tests;

  // Check if Op class exists before trying to get derived definitions
  if (records.getClass("Op")) {
    for (const llvm::Record *def : records.getAllDerivedDefinitions("Op")) {
      if (def->isAnonymous())
        continue;

      auto opTests = extractTestsFromRecord(def, "ops");
      tests.insert(tests.end(), opTests.begin(), opTests.end());
    }
  }

  return tests;
}

/// Generate a LIT test file for an IR test
static void generateTestFile(const LitTest &test, llvm::raw_ostream &os) {
  // Add RUN lines
  for (const auto& runLine : test.runLines) {
    os << "\n" << runLine << "\n";
  }

  os << "// Generated from TableGen definition: " << test.sourceDefName << "\n\n";
  
  // Add the test body
  os << test.irSnippet << "\n";
  
  // Add CHECK lines
  for (const auto& checkLine : test.checkLines) {
    os << "\n" << checkLine << "\n";
  }
}

/// Main function to generate all IR test test files
static void generateLitTests(const RecordKeeper &records, raw_ostream &os) {
  llvm::SmallVector<LitTest> allTests;

  // Extract tests from different definition types
  auto opTests = extractOpTests(records);
  allTests.insert(allTests.end(), opTests.begin(), opTests.end());

  if (allTests.empty()) {
    os << "// No mlir_example code blocks found in any TableGen definition\n";
    return;
  }

  // Generate summary
  os << "// Generated " << allTests.size() << " LIT test files\n";
  os << "// Use the following files for LIT testing:\n\n";

  // Generate file list and content for each test
  for (const auto& test : allTests) {
    std::string testFileName = formatv("generated_{0}_{1}", test.sourceDefName, test.testFileName);
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
    genLitTests("gen-lit-tests", "Generate LIT test files for `Testable` TableGen records",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    generateLitTests(records, os);
                    return false;
                  });