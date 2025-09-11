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

static llvm::SmallVector<LitTest> extractTestsFromRecord(const llvm::Record *record,
                                                         llvm::StringRef dialectName = "") {
  llvm::SmallVector<LitTest> tests;
  
  // Check if the record has a tests field
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

/// Extract tests from passes
static llvm::SmallVector<LitTest> extractPassTests(const RecordKeeper &records) {
  llvm::SmallVector<LitTest> tests;
  
  // Check if PassBase class exists before trying to get derived definitions
  if (records.getClass("PassBase")) {
    for (const llvm::Record *def : records.getAllDerivedDefinitions("PassBase")) {
      if (def->isAnonymous())
        continue;
        
      auto passTests = extractTestsFromRecord(def, "passes");
      tests.insert(tests.end(), passTests.begin(), passTests.end());
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
  
  // Extract tests from different definition types (only passes for now)
  auto passTests = extractPassTests(records);
  
  allTests.insert(allTests.end(), passTests.begin(), passTests.end());
  
  if (allTests.empty()) {
    os << "// No LitTest record found in any TableGen definition\n";
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