//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple drivers to test the mustache spec found at:
// https://github.com/mustache/spec
//
// It is used to verify that the current implementation conforms to the spec.
// Simply download the spec and pass the test JSON files to the driver. Each
// spec file should have a list of tests for compliance with the spec. These
// are loaded as test cases, and rendered with our Mustache implementation,
// which is then compared against the expected output from the spec.
//
// The current implementation only supports non-optional parts of the spec, so
// we do not expect any of the dynamic-names, inheritance, or lambda tests to
// pass. Additionally, Triple Mustache is not supported. Unsupported tests are
// marked as XFail and are removed from the XFail list as they are fixed.
//
// Usage:
//  llvm-test-mustache-spec path/to/test/file.json path/to/test/file2.json ...
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mustache.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

#define DEBUG_TYPE "llvm-test-mustache-spec"

static cl::OptionCategory Cat("llvm-test-mustache-spec Options");

static cl::list<std::string>
    InputFiles(cl::Positional, cl::desc("<input files>"), cl::OneOrMore);

static cl::opt<bool> ReportErrors("report-errors",
                                  cl::desc("Report errors in spec tests"),
                                  cl::cat(Cat));

static ExitOnError ExitOnErr;

static int NumXFail = 0;
static int NumSuccess = 0;

static const StringMap<StringSet<>> XFailTestNames = {{
    {"~dynamic-names.json",
     {
         "Basic Behavior - Partial",
         "Basic Behavior - Name Resolution",
         "Context",
         "Dotted Names",
         "Dotted Names - Failed Lookup",
         "Dotted names - Context Stacking",
         "Dotted names - Context Stacking Under Repetition",
         "Dotted names - Context Stacking Failed Lookup",
         "Recursion",
         "Surrounding Whitespace",
         "Inline Indentation",
         "Standalone Line Endings",
         "Standalone Without Previous Line",
         "Standalone Without Newline",
         "Standalone Indentation",
         "Padding Whitespace",
     }},
    {"~inheritance.json",
     {
         "Default",
         "Variable",
         "Triple Mustache",
         "Sections",
         "Negative Sections",
         "Mustache Injection",
         "Inherit",
         "Overridden content",
         "Data does not override block default",
         "Two overridden parents",
         "Override parent with newlines",
         "Inherit indentation",
         "Only one override",
         "Parent template",
         "Recursion",
         "Multi-level inheritance, no sub child",
         "Text inside parent",
         "Text inside parent",
         "Block scope",
         "Standalone parent",
         "Standalone block",
         "Block reindentation",
         "Intrinsic indentation",
         "Nested block reindentation",
     }},
    {"~lambdas.json",
     {
         "Interpolation",
         "Interpolation - Expansion",
         "Interpolation - Alternate Delimiters",
         "Interpolation - Multiple Calls",
         "Escaping",
         "Section",
         "Section - Expansion",
         "Section - Alternate Delimiters",
         "Section - Multiple Calls",
     }},
}};

struct TestData {
  TestData() = default;
  explicit TestData(const json::Object &TestCase)
      : TemplateStr(*TestCase.getString("template")),
        ExpectedStr(*TestCase.getString("expected")),
        Name(*TestCase.getString("name")), Data(TestCase.get("data")),
        Partials(TestCase.get("partials")) {}

  static Expected<TestData> createTestData(json::Object *TestCase,
                                           StringRef InputFile) {
    // If any of the needed elements are missing, we cannot continue.
    // NOTE: partials are optional in the test schema.
    if (!TestCase || !TestCase->getString("template") ||
        !TestCase->getString("expected") || !TestCase->getString("name") ||
        !TestCase->get("data"))
      return createStringError(
          llvm::inconvertibleErrorCode(),
          "invalid JSON schema in test file: " + InputFile + "\n");

    return TestData(*TestCase);
  }

  StringRef TemplateStr;
  StringRef ExpectedStr;
  StringRef Name;
  const Value *Data;
  const Value *Partials;
};

static void reportTestFailure(const TestData &TD, StringRef ActualStr,
                              bool IsXFail) {
  LLVM_DEBUG(dbgs() << "Template: " << TD.TemplateStr << "\n");
  if (TD.Partials) {
    LLVM_DEBUG(dbgs() << "Partial: ");
    LLVM_DEBUG(TD.Partials->print(dbgs()));
    LLVM_DEBUG(dbgs() << "\n");
  }
  LLVM_DEBUG(dbgs() << "JSON Data: ");
  LLVM_DEBUG(TD.Data->print(dbgs()));
  LLVM_DEBUG(dbgs() << "\n");
  outs() << formatv("Test {}: {}\n", (IsXFail ? "XFailed" : "Failed"), TD.Name);
  if (ReportErrors) {
    outs() << "  Expected: \'" << TD.ExpectedStr << "\'\n"
           << "  Actual: \'" << ActualStr << "\'\n"
           << " ====================\n";
  }
}

static void registerPartials(const Value *Partials, Template &T) {
  if (!Partials)
    return;
  for (const auto &[Partial, Str] : *Partials->getAsObject())
    T.registerPartial(Partial.str(), Str.getAsString()->str());
}

static json::Value readJsonFromFile(StringRef &InputFile) {
  std::unique_ptr<MemoryBuffer> Buffer =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(InputFile)));
  return ExitOnErr(parse(Buffer->getBuffer()));
}

static bool isTestXFail(StringRef FileName, StringRef TestName) {
  auto P = llvm::sys::path::filename(FileName);
  auto It = XFailTestNames.find(P);
  return It != XFailTestNames.end() && It->second.contains(TestName);
}

static bool evaluateTest(StringRef &InputFile, TestData &TestData,
                         std::string &ActualStr) {
  bool IsXFail = isTestXFail(InputFile, TestData.Name);
  bool Matches = TestData.ExpectedStr == ActualStr;
  if ((Matches && IsXFail) || (!Matches && !IsXFail)) {
    reportTestFailure(TestData, ActualStr, IsXFail);
    return false;
  }
  IsXFail ? NumXFail++ : NumSuccess++;
  return true;
}

static void runTest(StringRef InputFile) {
  NumXFail = 0;
  NumSuccess = 0;
  outs() << "Running Tests: " << InputFile << "\n";
  json::Value Json = readJsonFromFile(InputFile);

  json::Object *Obj = Json.getAsObject();
  Array *TestArray = Obj->getArray("tests");
  // Even though we parsed the JSON, it can have a bad format, so check it.
  if (!TestArray)
    ExitOnErr(createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid JSON schema in test file: " + InputFile + "\n"));

  const size_t Total = TestArray->size();

  for (Value V : *TestArray) {
    auto TestData =
        ExitOnErr(TestData::createTestData(V.getAsObject(), InputFile));
    BumpPtrAllocator Allocator;
    StringSaver Saver(Allocator);
    MustacheContext Ctx(Allocator, Saver);
    Template T(TestData.TemplateStr, Ctx);
    registerPartials(TestData.Partials, T);

    std::string ActualStr;
    raw_string_ostream OS(ActualStr);
    T.render(*TestData.Data, OS);
    evaluateTest(InputFile, TestData, ActualStr);
  }

  const int NumFailed = Total - NumSuccess - NumXFail;
  outs() << formatv("===Results===\n"
                    " Suceeded: {}\n"
                    " Expectedly Failed: {}\n"
                    " Failed: {}\n"
                    " Total: {}\n",
                    NumSuccess, NumXFail, NumFailed, Total);
}

int main(int argc, char **argv) {
  ExitOnErr.setBanner(std::string(argv[0]) + " error: ");
  cl::ParseCommandLineOptions(argc, argv);
  for (const auto &FileName : InputFiles)
    runTest(FileName);
  return 0;
}
