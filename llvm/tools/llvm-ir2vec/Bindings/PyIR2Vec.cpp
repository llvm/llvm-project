//===- PyIR2Vec.cpp - Python Bindings for IR2Vec ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lib/Utils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <fstream>
#include <memory>
#include <string>

namespace nb = nanobind;
using namespace llvm;
using namespace llvm::ir2vec;

namespace {

std::unique_ptr<Module> getLLVMIR(const std::string &Filename,
                                  LLVMContext &Context) {
  SMDiagnostic Err;
  auto M = parseIRFile(Filename, Err, Context);
  if (!M)
    throw nb::value_error(("Failed to parse IR file '" + Filename +
                           "': " + Err.getMessage().str())
                              .c_str());
  return M;
}

class PyIR2VecTool {
private:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<IR2VecTool> Tool;

public:
  PyIR2VecTool(const std::string &Filename, const std::string &Mode,
               const std::string &VocabPath) {
    if (Mode != "sym" && Mode != "fa")
      throw nb::value_error("Invalid mode. Use 'sym' or 'fa'");

    if (VocabPath.empty())
      throw nb::value_error("Empty Vocab Path not allowed");

    Ctx = std::make_unique<LLVMContext>();
    M = getLLVMIR(Filename, *Ctx);
    Tool = std::make_unique<IR2VecTool>(*M);

    if (auto Err = Tool->initializeVocabulary(VocabPath)) {
      throw nb::value_error(("Failed to initialize IR2Vec vocabulary: " +
                             toString(std::move(Err)))
                                .c_str());
    }
  }
};

} // namespace

NB_MODULE(ir2vec, m) {
  m.doc() = std::string("Python bindings for ") + ToolName;

  nb::class_<PyIR2VecTool>(m, "IR2VecTool")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>(),
           nb::arg("filename"), nb::arg("mode"), nb::arg("vocabPath"));

  m.def(
      "initEmbedding",
      [](const std::string &filename, const std::string &mode,
         const std::string &vocabPath) {
        return std::make_unique<PyIR2VecTool>(filename, mode, vocabPath);
      },
      nb::arg("filename"), nb::arg("mode") = "sym", nb::arg("vocabPath"),
      nb::rv_policy::take_ownership);
}
