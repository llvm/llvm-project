//===---------- TargetProcessControlTypes.cpp - Shared TPC types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TargetProcessControl types.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"

namespace llvm {
namespace orc {
namespace tpctypes {

WrapperFunctionResult WrapperFunctionResult::from(StringRef S) {
  CWrapperFunctionResult R;
  zeroInit(R);
  R.Size = S.size();
  if (R.Size > sizeof(uint64_t)) {
    R.Data.ValuePtr = new uint8_t[R.Size];
    memcpy(R.Data.ValuePtr, S.data(), R.Size);
    R.Destroy = destroyWithDeleteArray;
  } else
    memcpy(R.Data.Value, S.data(), R.Size);
  return R;
}

void WrapperFunctionResult::destroyWithFree(CWrapperFunctionResultData Data,
                                            uint64_t Size) {
  free(Data.ValuePtr);
}

void WrapperFunctionResult::destroyWithDeleteArray(
    CWrapperFunctionResultData Data, uint64_t Size) {
  delete[] Data.ValuePtr;
}

} // end namespace tpctypes
} // end namespace orc
} // end namespace llvm
