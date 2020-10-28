//===---------------- TPCDynamicLibrarySearchGenerator.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TPCDynamicLibrarySearchGenerator.h"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<TPCDynamicLibrarySearchGenerator>>
TPCDynamicLibrarySearchGenerator::Load(TargetProcessControl &TPC,
                                       const char *LibraryPath) {
  auto Handle = TPC.loadDylib(LibraryPath);
  if (!Handle)
    return Handle.takeError();

  return std::make_unique<TPCDynamicLibrarySearchGenerator>(TPC, *Handle);
}

Error TPCDynamicLibrarySearchGenerator::tryToGenerate(
    LookupKind K, JITDylib &JD, JITDylibLookupFlags JDLookupFlags,
    const SymbolLookupSet &Symbols) {

  if (Symbols.empty())
    return Error::success();

  SymbolMap NewSymbols;

  TargetProcessControl::LookupRequestElement Request(H, Symbols);
  auto Result = TPC.lookupSymbols(Request);
  if (!Result)
    return Result.takeError();

  assert(Result->size() == 1 && "Results for more than one library returned");
  assert(Result->front().size() == Symbols.size() &&
         "Result has incorrect number of elements");

  auto ResultI = Result->front().begin();
  for (auto &KV : Symbols)
    NewSymbols[KV.first] =
        JITEvaluatedSymbol(*ResultI++, JITSymbolFlags::Exported);

  return JD.define(absoluteSymbols(std::move(NewSymbols)));
}

} // end namespace orc
} // end namespace llvm
