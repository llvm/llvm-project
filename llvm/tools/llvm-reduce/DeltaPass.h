//===--- DeltaPass.h - Delta Pass Structure --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_DELTAPASS_H
#define LLVM_TOOLS_LLVM_REDUCE_DELTAPASS_H

#include "ReducerWorkItem.h"
#include "deltas/Delta.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
struct DeltaPass {
  StringRef Name;                            // e.g., "strip-debug-info"
  void (*Func)(Oracle &, ReducerWorkItem &); // e.g., stripDebugInfoDeltaPass
  StringRef Desc;                            // e.g., "Stripping Debug Info"
};
} // namespace llvm

#endif
