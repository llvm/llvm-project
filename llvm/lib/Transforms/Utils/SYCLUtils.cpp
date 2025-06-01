//===------------ SYCLUtils.cpp - SYCL utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SYCL utility functions.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SYCLUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace llvm;
using namespace sycl;

namespace {

SmallString<0> computeFunctionCategoryForSplitting(IRSplitMode SM,
                                                   const Function &F) {
  static constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";
  SmallString<0> Key;
  switch (SM) {
  case IRSplitMode::IRSM_PER_KERNEL:
    Key = F.getName().str();
    break;
  case IRSplitMode::IRSM_PER_TU:
    Key = F.getFnAttribute(ATTR_SYCL_MODULE_ID).getValueAsString().str();
    break;
  default:
    llvm_unreachable("other modes aren't expected");
  }

  return Key;
}

bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
         F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::PTX_Kernel;
}

bool isEntryPoint(const Function &F) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  return isKernel(F);
}

} // anonymous namespace

namespace llvm {
namespace sycl {

std::optional<IRSplitMode> convertStringToSplitMode(StringRef S) {
  static const StringMap<IRSplitMode> Values = {
      {"source", IRSplitMode::IRSM_PER_TU},
      {"kernel", IRSplitMode::IRSM_PER_KERNEL},
      {"none", IRSplitMode::IRSM_NONE}};

  auto It = Values.find(S);
  if (It == Values.end())
    return std::nullopt;

  return It->second;
}

FunctionCategorizer::FunctionCategorizer(IRSplitMode SM) : SM(SM) {
  if (SM == IRSplitMode::IRSM_NONE)
    llvm_unreachable("FunctionCategorizer isn't supported to none splitting.");
}

std::optional<int> FunctionCategorizer::operator()(const Function &F) {
  if (!isEntryPoint(F))
    return std::nullopt; // skip the function.

  auto StringKey = computeFunctionCategoryForSplitting(SM, F);
  if (auto it = StrKeyToID.find(StringRef(StringKey)); it != StrKeyToID.end())
    return it->second;

  int ID = static_cast<int>(StrKeyToID.size());
  return StrKeyToID.try_emplace(std::move(StringKey), ID).first->second;
}

std::string makeSymbolTable(const Module &M) {
  SmallString<0> Data;
  raw_svector_ostream OS(Data);
  for (const auto &F : M)
    if (isEntryPoint(F))
      OS << F.getName() << '\n';

  return std::string(OS.str());
}

void writeStringTable(const StringTable &Table, raw_ostream &OS) {
  assert(!Table.empty() && "table should contain at least column titles");
  assert(!Table[0].empty() && "table should be non-empty");
  OS << '[' << join(Table[0].begin(), Table[0].end(), "|") << "]\n";
  for (size_t I = 1, E = Table.size(); I != E; ++I) {
    assert(Table[I].size() == Table[0].size() && "row's size should be equal");
    OS << join(Table[I].begin(), Table[I].end(), "|") << '\n';
  }
}

} // namespace sycl
} // namespace llvm
