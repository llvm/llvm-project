//===- DefaultHostBootstrapValues.h - Defaults for host process -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Set sensible default bootstrap values for JIT execution in the host process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DEFAULTHOSTBOOTSTRAPVALUES_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DEFAULTHOSTBOOTSTRAPVALUES_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include <vector>

namespace llvm::orc {

void addDefaultBootstrapValuesForHostProcess(
    StringMap<std::vector<char>> &BootstrapMap,
    StringMap<ExecutorAddr> &BootstrapSymbols);

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DEFAULTHOSTBOOTSTRAPVALUES_H
