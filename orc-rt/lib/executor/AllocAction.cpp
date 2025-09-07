//===- AllocAction.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AllocAction and related APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/AllocAction.h"
#include "orc-rt/ScopeExit.h"

namespace orc_rt {

Expected<std::vector<AllocAction>>
runFinalizeActions(std::vector<AllocActionPair> AAPs) {
  std::vector<AllocAction> DeallocActions;
  auto RunDeallocActions = make_scope_exit([&]() {
    while (!DeallocActions.empty()) {
      // TODO: Log errors from cleanup dealloc actions.
      {
        [[maybe_unused]] auto B = DeallocActions.back()();
      }
      DeallocActions.pop_back();
    }
  });

  for (auto &AAP : AAPs) {
    if (AAP.Finalize) {
      auto B = AAP.Finalize();
      if (const char *ErrMsg = B.getOutOfBandError())
        return make_error<StringError>(ErrMsg);
    }
    if (AAP.Dealloc)
      DeallocActions.push_back(std::move(AAP.Dealloc));
  }

  RunDeallocActions.release();
  return DeallocActions;
}

void runDeallocActions(std::vector<AllocAction> DAAs) {
  while (!DAAs.empty()) {
    // TODO: Log errors from cleanup dealloc actions.
    {
      [[maybe_unused]] auto B = DAAs.back()();
    }
    DAAs.pop_back();
  }
}

} // namespace orc_rt
