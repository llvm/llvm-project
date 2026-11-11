//===-------------------------- JITLinkDylib.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"

namespace llvm::jitlink {

JITLinkDylib::~JITLinkDylib() {
  for (JITLinkMemoryManager *MemMgr : ToNotifyOnDestruction)
    MemMgr->notifyDestroying(*this);
}

} // namespace llvm::jitlink
