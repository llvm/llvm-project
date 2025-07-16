//===- IOSandbox.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IOSANDBOX_H
#define LLVM_SUPPORT_IOSANDBOX_H

#include "llvm/Support/SaveAndRestore.h"

namespace llvm::sys {
SaveAndRestore<bool> sandbox_scoped_enable();
SaveAndRestore<bool> sandbox_scoped_disable();
void sandbox_violation_if_enabled();
} // namespace llvm::sys

#endif
