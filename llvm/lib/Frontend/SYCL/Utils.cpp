//===------------ Utils.cpp - SYCL utility functions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SYCL utility functions.
//===----------------------------------------------------------------------===//
#include "llvm/Frontend/SYCL/Utils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace sycl;

namespace {

bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
         F.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         F.getCallingConv() == CallingConv::PTX_Kernel;
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

bool isEntryPoint(const Function &F) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  return isKernel(F);
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
