//===----- DefaultHostBootstrapValues.cpp - Defaults for host process -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/DefaultHostBootstrapValues.h"

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"

#ifdef __APPLE__
#include <dlfcn.h>
#endif // __APPLE__

namespace llvm::orc {

void addDefaultBootstrapValuesForHostProcess(
    StringMap<std::vector<char>> &BootstrapMap,
    StringMap<ExecutorAddr> &BootstrapSymbols) {

  // FIXME: We probably shouldn't set these on Windows?
  BootstrapSymbols[rt::RegisterEHFrameSectionWrapperName] =
      ExecutorAddr::fromPtr(&llvm_orc_registerEHFrameSectionWrapper);
  BootstrapSymbols[rt::DeregisterEHFrameSectionWrapperName] =
      ExecutorAddr::fromPtr(&llvm_orc_deregisterEHFrameSectionWrapper);

#ifdef __APPLE__
  if (!dlsym(RTLD_DEFAULT, "__unw_add_find_dynamic_unwind_sections"))
    BootstrapMap["darwin-use-ehframes-only"].push_back(1);
#endif // __APPLE__
}

} // namespace llvm::orc
