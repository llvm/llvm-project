//===----- AllocationActions.gpp -- JITLink allocation support calls  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/AllocationActions.h"

namespace llvm {
namespace orc {
namespace shared {

void runFinalizeActions(AllocActions &AAs,
                        OnRunFinalizeActionsCompleteFn OnComplete) {
  std::vector<WrapperFunctionCall> DeallocActions;
  DeallocActions.reserve(numDeallocActions(AAs));

  for (auto &AA : AAs) {
    if (AA.Finalize)

      if (auto Err = AA.Finalize.runWithSPSRetErrorMerged()) {
        while (!DeallocActions.empty()) {
          Err = joinErrors(std::move(Err),
                           DeallocActions.back().runWithSPSRetErrorMerged());
          DeallocActions.pop_back();
        }
        return OnComplete(std::move(Err));
      }

    if (AA.Dealloc)
      DeallocActions.push_back(std::move(AA.Dealloc));
  }

  AAs.clear();
  OnComplete(std::move(DeallocActions));
}

void runDeallocActions(ArrayRef<WrapperFunctionCall> DAs,
                       OnRunDeallocActionsComeleteFn OnComplete) {
  Error Err = Error::success();
  while (!DAs.empty()) {
    Err = joinErrors(std::move(Err), DAs.back().runWithSPSRetErrorMerged());
    DAs = DAs.drop_back();
  }
  OnComplete(std::move(Err));
}

} // namespace shared
} // namespace orc
} // namespace llvm
