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

    nb::dict NBFuncEmbMap;

    for (const auto &[FuncPtr, FuncEmb] : *ToolFuncEmbMap) {
      auto FuncEmbVec = FuncEmb.getData();
      double *NBFuncEmbVec = new double[FuncEmbVec.size()];
      std::copy(FuncEmbVec.begin(), FuncEmbVec.end(), NBFuncEmbVec);

      auto NbArray = nb::ndarray<nb::numpy, double>(
          NBFuncEmbVec, {FuncEmbVec.size()},
          nb::capsule(NBFuncEmbVec, [](void *P) noexcept {
            delete[] static_cast<double *>(P);
          }));

      NBFuncEmbMap[nb::str(FuncPtr->getName().str().c_str())] = NbArray;
    }

    return NBFuncEmbMap;
  }

  nb::list getBBEmbMap() {
    auto result = Tool->getBBEmbeddings(EmbKind);
    nb::list nb_result;

    for (const auto &[bb_ptr, embedding] : result) {
      std::string bb_name = bb_ptr->getName().str();
      auto data = embedding.getData();

      double *data_ptr = new double[data.size()];
      std::copy(data.data(), data.data() + data.size(), data_ptr);
      auto nb_array = nb::ndarray<nb::numpy, double, nb::shape<-1>>(
          data_ptr, {data.size()}, nb::capsule(data_ptr, [](void *p) noexcept {
            delete[] static_cast<double *>(p);
          }));
      nb_result.append(nb::make_tuple(nb::str(bb_name.c_str()), nb_array));
    }

    return nb_result;
  }

  nb::list getInstEmbMap() {
    auto result = Tool->getInstEmbeddings(EmbKind);
    nb::list nb_result;

    for (const auto &[inst_ptr, embedding] : result) {
      std::string inst_str;
      llvm::raw_string_ostream RSO(inst_str);
      inst_ptr->print(RSO);
      RSO.flush();

      auto data = embedding.getData();

      double *data_ptr = new double[data.size()];
      std::copy(data.data(), data.data() + data.size(), data_ptr);

      // Create nanobind numpy array with dynamic 1D shape
      auto nb_array = nb::ndarray<nb::numpy, double, nb::shape<-1>>(
          data_ptr, {data.size()}, nb::capsule(data_ptr, [](void *p) noexcept {
            delete[] static_cast<double *>(p);
          }));

      nb_result.append(nb::make_tuple(nb::str(inst_str.c_str()), nb_array));
    }

    return nb_result;
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
           "{function_name: embedding}")
      .def("getBBEmbMap", &PyIR2VecTool::getBBEmbMap,
           "Generate basic block embeddings for all functions\n"
           "Returns: list[tuple[str, ndarray[float64]]] - "
           "[{bb_name, embedding}]")
      .def("getInstEmbMap", &PyIR2VecTool::getInstEmbMap,
           "Generate instruction embeddings for all functions\n"
           "Returns: list[tuple[str, ndarray[float64]]]");

  m.def(
      "initEmbedding",
      [](const std::string &filename, const std::string &mode,
         const std::string &vocabPath) {
        return std::make_unique<PyIR2VecTool>(filename, mode, vocabPath);
      },
      nb::arg("filename"), nb::arg("mode") = "sym", nb::arg("vocabPath"),
      nb::rv_policy::take_ownership);
}
