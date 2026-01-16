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

#ifndef LLVM_TARGET_REGISTERTARGETPASSCONFIGCALLBACK_H
#define LLVM_TARGET_REGISTERTARGETPASSCONFIGCALLBACK_H

#include "TargetMachine.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

using PassConfigCallback =
    std::function<void(TargetMachine &, PassManagerBase &, TargetPassConfig *)>;

class RegisterTargetPassConfigCallback {
public:
  PassConfigCallback Callback;

  LLVM_ABI explicit RegisterTargetPassConfigCallback(PassConfigCallback &&C);
  LLVM_ABI ~RegisterTargetPassConfigCallback();
};

LLVM_ABI void
invokeGlobalTargetPassConfigCallbacks(TargetMachine &TM, PassManagerBase &PM,
                                      TargetPassConfig *PassConfig);
} // namespace llvm

#endif // LLVM_TARGET_REGISTERTARGETPASSCONFIGCALLBACK_H
