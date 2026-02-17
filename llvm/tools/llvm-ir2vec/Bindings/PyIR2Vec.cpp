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
#include <nanobind/ndarray.h>
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
  IR2VecKind OutputEmbeddingMode;

public:
  PyIR2VecTool(const std::string &Filename, const std::string &Mode,
               const std::string &VocabPath) {
    OutputEmbeddingMode = [](const std::string &Mode) -> IR2VecKind {
      if (Mode == "sym")
        return IR2VecKind::Symbolic;
      if (Mode == "fa")
        return IR2VecKind::FlowAware;
      throw nb::value_error("Invalid mode. Use 'sym' or 'fa'");
    }(Mode);

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

  nb::dict getFuncEmbMap() {
    auto ToolFuncEmbMap = Tool->getFunctionEmbeddingsMap(OutputEmbeddingMode);

    if (!ToolFuncEmbMap)
      throw nb::value_error(toString(ToolFuncEmbMap.takeError()).c_str());

    nb::dict NbFuncEmbMap;

    for (const auto &[FuncPtr, FuncEmb] : *ToolFuncEmbMap) {
      auto FuncEmbVec = FuncEmb.getData();
      double *NbFuncEmbVec = new double[FuncEmbVec.size()];
      std::copy(FuncEmbVec.begin(), FuncEmbVec.end(), NbFuncEmbVec);

      auto NbArray = nb::ndarray<nb::numpy, double>(
          NbFuncEmbVec, {FuncEmbVec.size()},
          nb::capsule(NbFuncEmbVec, [](void *P) noexcept {
            delete[] static_cast<double *>(P);
          }));

      NbFuncEmbMap[nb::str(FuncPtr->getName().str().c_str())] = NbArray;
    }

    return NbFuncEmbMap;
  }

  nb::ndarray<nb::numpy, double> getFuncEmb(const std::string &FuncName) {
    const Function *F = M->getFunction(FuncName);

    if (!F)
      throw nb::value_error(
          ("Function '" + FuncName + "' not found in module").c_str());

    auto ToolFuncEmb = Tool->getFunctionEmbedding(*F, OutputEmbeddingMode);

    if (!ToolFuncEmb)
      throw nb::value_error(toString(ToolFuncEmb.takeError()).c_str());

    auto FuncEmbVec = ToolFuncEmb->getData();
    double *NbFuncEmbVec = new double[FuncEmbVec.size()];
    std::copy(FuncEmbVec.begin(), FuncEmbVec.end(), NbFuncEmbVec);

    auto NbArray = nb::ndarray<nb::numpy, double>(
        NbFuncEmbVec, {FuncEmbVec.size()},
        nb::capsule(NbFuncEmbVec, [](void *P) noexcept {
          delete[] static_cast<double *>(P);
        }));

    return NbArray;
  }

  nb::dict getBBEmbMap(const std::string &FuncName) {
    const Function *F = M->getFunction(FuncName);

    if (!F)
      throw nb::value_error(
          ("Function '" + FuncName + "' not found in module").c_str());

    auto ToolBBEmbMap = Tool->getBBEmbeddingsMap(*F, OutputEmbeddingMode);

    if (!ToolBBEmbMap)
      throw nb::value_error(toString(ToolBBEmbMap.takeError()).c_str());

    nb::dict NbBBEmbMap;

    for (const auto &[BBPtr, BBEmb] : *ToolBBEmbMap) {
      auto BBEmbVec = BBEmb.getData();
      double *NbBBEmbVec = new double[BBEmbVec.size()];
      std::copy(BBEmbVec.begin(), BBEmbVec.end(), NbBBEmbVec);

      auto NbArray = nb::ndarray<nb::numpy, double>(
          NbBBEmbVec, {BBEmbVec.size()},
          nb::capsule(NbBBEmbVec, [](void *P) noexcept {
            delete[] static_cast<double *>(P);
          }));

      NbBBEmbMap[nb::str(BBPtr->getName().str().c_str())] = NbArray;
    }

    return NbBBEmbMap;
  }
};

} // namespace

NB_MODULE(ir2vec, m) {
  m.doc() = std::string("Python bindings for ") + ToolName;

  nb::class_<PyIR2VecTool>(m, "IR2VecTool")
      .def(nb::init<const std::string &, const std::string &,
                    const std::string &>(),
           nb::arg("filename"), nb::arg("mode"), nb::arg("vocabPath"))
      .def("getFuncEmbMap", &PyIR2VecTool::getFuncEmbMap,
           "Generate function-level embeddings for all functions\n"
           "Returns: dict[str, ndarray[float64]] - "
           "{function_name: embedding vector}")
      .def("getFuncEmb", &PyIR2VecTool::getFuncEmb, nb::arg("funcName"),
           "Generate embedding for a single function by name\n"
           "Args: funcName (str) - IR-Name of the function\n"
           "Returns: ndarray[float64] - Function embedding vector")
      .def("getBBEmbMap", &PyIR2VecTool::getBBEmbMap, nb::arg("funcName"),
           "Generate embeddings for all basic blocks in a function\n"
           "Args: funcName (str) - IR-Name of the function\n"
           "Returns: dict[str, ndarray[float64]] - "
           "{basic_block_name: embedding vector}");
  m.def(
      "initEmbedding",
      [](const std::string &filename, const std::string &mode,
         const std::string &vocabPath) {
        return std::make_unique<PyIR2VecTool>(filename, mode, vocabPath);
      },
      nb::arg("filename"), nb::arg("mode") = "sym", nb::arg("vocabPath"),
      nb::rv_policy::take_ownership);
}
