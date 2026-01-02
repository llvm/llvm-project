//===- ir2vec_bindings.cpp - Python Bindings for IR2Vec ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include <fstream>
#include <memory>
#include <string>

namespace py = pybind11;
using namespace llvm;
using namespace llvm::ir2vec;

namespace llvm {
namespace ir2vec {
void setIR2VecVocabPath(StringRef Path);
StringRef getIR2VecVocabPath();
} // namespace ir2vec
} // namespace llvm

namespace {

bool fileNotValid(const std::string &Filename) {
  std::ifstream F(Filename, std::ios_base::in | std::ios_base::binary);
  return !F.good();
}

std::unique_ptr<Module> getLLVMIR(const std::string &Filename,
                                  LLVMContext &Context) {
  SMDiagnostic Err;
  auto M = parseIRFile(Filename, Err, Context);
  if (!M)
    throw std::runtime_error("Failed to parse IR file.");
  return M;
}

class PyIR2VecTool {
private:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<IR2VecTool> Tool;

public:
  PyIR2VecTool(std::string Filename, std::string Mode,
               std::string VocabOverride) {
    if (fileNotValid(Filename))
      throw std::runtime_error("Invalid file path");

    if (Mode != "sym" && Mode != "fa")
      throw std::runtime_error("Invalid mode. Use 'sym' or 'fa'");

    if (VocabOverride.empty())
      throw std::runtime_error("Error - Empty Vocab Path not allowed");

    setIR2VecVocabPath(VocabOverride);

    Ctx = std::make_unique<LLVMContext>();
    M = getLLVMIR(Filename, *Ctx);
    Tool = std::make_unique<IR2VecTool>(*M);

    bool Ok = Tool->initializeVocabulary();
    if (!Ok)
      throw std::runtime_error("Failed to initialize IR2Vec vocabulary");
  }

  py::dict generateTriplets() {
    auto result = Tool->generateTriplets();
    py::list triplets_list;
    for (const auto &t : result.Triplets) {
      triplets_list.append(py::make_tuple(t.Head, t.Tail, t.Relation));
    }

    return py::dict(py::arg("max_relation") = result.MaxRelation,
                    py::arg("triplets") = triplets_list);
  }

  EntityList collectEntityMappings() {
    return IR2VecTool::collectEntityMappings();
  }
};

} // namespace

PYBIND11_MODULE(py_ir2vec, m) {
  m.doc() = "Python bindings for LLVM IR2Vec";

  py::class_<PyIR2VecTool>(m, "IR2VecTool")
      .def(py::init<std::string, std::string, std::string>())
      .def("generateTriplets", &PyIR2VecTool::generateTriplets)
      .def("getEntityMappings", &PyIR2VecTool::collectEntityMappings);

  m.def(
      "initEmbedding",
      [](std::string filename, std::string mode, std::string vocab_override) {
        return std::make_unique<PyIR2VecTool>(filename, mode, vocab_override);
      },
      py::arg("filename"), py::arg("mode") = "sym", py::arg("vocab_override"));
}