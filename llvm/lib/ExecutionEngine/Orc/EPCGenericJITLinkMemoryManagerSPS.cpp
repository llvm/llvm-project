//===- EPCGenericJITLinkMemoryManagerSPS.cpp - SPS mem manager ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManagerSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm::orc::sps {

Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(JITDylib &JD) {
  auto B = createSimpleMemoryMapBindings(JD);
  if (!B)
    return B.takeError();
  return std::make_unique<EPCGenericJITLinkMemoryManager>(
      JD.getExecutionSession(), std::move(*B));
}

Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(ExecutionSession &ES) {
  return createEPCGenericJITLinkMemoryManager(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
