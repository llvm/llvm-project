//===- MergedFunctionsInfo.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/MergedFunctionsInfo.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/Support/DataExtractor.h"

using namespace llvm;
using namespace gsym;

void MergedFunctionsInfo::clear() { MergedFunctions.clear(); }

llvm::Error MergedFunctionsInfo::encode(FileWriter &Out) const {
  Out.writeU32(MergedFunctions.size());
  for (const auto &F : MergedFunctions) {
    Out.writeU32(0);
    const auto StartOffset = Out.tell();
    // Encode the FunctionInfo with no padding so later we can just read them
    // one after the other without knowing the offset in the stream for each.
    llvm::Expected<uint64_t> result = F.encode(Out, /*NoPadding =*/true);
    if (!result)
      return result.takeError();
    const auto Length = Out.tell() - StartOffset;
    Out.fixup32(static_cast<uint32_t>(Length), StartOffset - 4);
  }
  return Error::success();
}

llvm::Expected<MergedFunctionsInfo>
MergedFunctionsInfo::decode(DataExtractor &Data, uint64_t BaseAddr) {
  MergedFunctionsInfo MFI;
  uint64_t Offset = 0;
  uint32_t Count = Data.getU32(&Offset);

  for (uint32_t i = 0; i < Count; ++i) {
    uint32_t FnSize = Data.getU32(&Offset);
    DataExtractor FnData(Data.getData().substr(Offset, FnSize),
                         Data.isLittleEndian(), Data.getAddressSize());
    llvm::Expected<FunctionInfo> FI =
        FunctionInfo::decode(FnData, BaseAddr + Offset);
    if (!FI)
      return FI.takeError();
    MFI.MergedFunctions.push_back(std::move(*FI));
    Offset += FnSize;
  }

  return MFI;
}

bool operator==(const MergedFunctionsInfo &LHS,
                const MergedFunctionsInfo &RHS) {
  return LHS.MergedFunctions == RHS.MergedFunctions;
}
