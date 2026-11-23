//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the registry for PassConfigCallbacks that enable changes
/// to the TargetPassConfig during the initialization of TargetMachine.
///
//===----------------------------------------------------------------------===//

#include "llvm/Target/RegisterTargetPassConfigCallback.h"

namespace llvm {
// TargetPassConfig callbacks
static SmallVector<RegisterTargetPassConfigCallback *, 1>
    TargetPassConfigCallbacks{};

void invokeGlobalTargetPassConfigCallbacks(TargetMachine &TM,
                                           PassManagerBase &PM,
                                           TargetPassConfig *PassConfig) {
  for (const RegisterTargetPassConfigCallback *Reg : TargetPassConfigCallbacks)
    Reg->Callback(TM, PM, PassConfig);
}

RegisterTargetPassConfigCallback::RegisterTargetPassConfigCallback(
    PassConfigCallback &&C)
    : Callback(std::move(C)) {
  TargetPassConfigCallbacks.push_back(this);
}

RegisterTargetPassConfigCallback::~RegisterTargetPassConfigCallback() {
  const auto &It = find(TargetPassConfigCallbacks, this);
  if (It != TargetPassConfigCallbacks.end())
    TargetPassConfigCallbacks.erase(It);
}
} // namespace llvm
