//===- mustache.cpp - The LLVM Modular Optimizer
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple drivers to test the mustache spec found here
// https://github.com/mustache/
// It is used to verify that the current implementation conforms to the spec
// simply download the spec and pass the test files to the driver
//
// Currently Triple Mustache is not supported we expect the following spec
// test to fail:
//    Triple Mustache
//    Triple Mustache Integer Interpolation
//    Triple Mustache Decimal Interpolation
//    Triple Mustache Null Interpolation
//    Triple Mustache Context Miss Interpolation
//    Dotted Names - Triple Mustache Interpolation
//    Implicit Iterators - Triple Mustache
//    Triple Mustache - Surrounding Whitespace
//    Triple Mustache - Standalone
//    Triple Mustache With Padding
//    Standalone Indentation
//    Implicit Iterator - Triple mustache
//
// Usage:
//  mustache path/to/test/file/test.json path/to/test/file/test2.json ...
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mustache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

using namespace llvm;
using namespace llvm::json;
using namespace llvm::mustache;

cl::list<std::string> InputFiles(cl::Positional, cl::desc("<input files>"),
                                 cl::OneOrMore);

void runThroughTest(StringRef InputFile) {
  llvm::outs() << "Running Tests: " << InputFile << "\n";
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferOrError =
      MemoryBuffer::getFile(InputFile);

  if (auto EC = BufferOrError.getError()) {
    return;
  }
  std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(BufferOrError.get());
  llvm::StringRef FileContent = Buffer->getBuffer();
  Expected<Value> Json = parse(FileContent);

  if (auto E = Json.takeError()) {
    errs() << "Parsing error: " << toString(std::move(E)) << "\n";
    return;
  }
  // Get test
  Array *Obj = (*Json).getAsObject()->getArray("tests");
  size_t Total = 0;
  size_t Success = 0;
  for (Value V : *Obj) {
    Object *TestCase = V.getAsObject();
    StringRef TemplateStr = TestCase->getString("template").value();
    StringRef ExpectedStr = TestCase->getString("expected").value();
    StringRef Name = TestCase->getString("name").value();
    Value *Data = TestCase->get("data");
    Value *Partials = TestCase->get("partials");

    if (!Data)
      continue;

    Template T = Template(TemplateStr);
    if (Partials) {
      for (auto PartialPairs : *Partials->getAsObject()) {
        StringRef Partial = PartialPairs.getSecond().getAsString().value();
        StringRef Str = llvm::StringRef(PartialPairs.getFirst());
        T.registerPartial(Str, Partial);
      }
    }
    StringRef ActualStr = T.render(*Data);
    if (ExpectedStr == ActualStr) {
      Success++;
    } else {
      llvm::outs() << "Test Failed: " << Name << "\n";
    }
    Total++;
  }

  llvm::outs() << "Result " << Success << "/" << Total << " succeeded\n";
}
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  for (const auto &FileName : InputFiles) {
    runThroughTest(FileName);
  }
  return 0;
}