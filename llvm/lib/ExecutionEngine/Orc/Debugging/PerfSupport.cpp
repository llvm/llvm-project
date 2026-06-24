//===------ PerfSupport.cpp - Utils for enabling perf support -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/ExecutionEngine/Orc/Debugging/PerfSupport.h"
#include "llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

namespace llvm::orc {

Error enablePerfSupport(LLJIT &J) {
  auto *ObjLinkingLayer = dyn_cast<ObjectLinkingLayer>(&J.getObjLinkingLayer());
  if (!ObjLinkingLayer)
    return make_error<StringError>("Cannot enable LLJIT perf support: "
                                   "Perf support requires JITLink",
                                   inconvertibleErrorCode());
  auto ProcessSymsJD = J.getProcessSymbolsJITDylib();
  if (!ProcessSymsJD)
    return make_error<StringError>("Cannot enable LLJIT perf support: "
                                   "Process symbols are not available",
                                   inconvertibleErrorCode());

  auto &ES = J.getExecutionSession();
  const auto &TT = J.getTargetTriple();

  switch (TT.getObjectFormat()) {
  case Triple::ELF: {
    ObjLinkingLayer->addPlugin(std::make_unique<DebugInfoPreservationPlugin>());
    auto PS = PerfSupportPlugin::Create(
        ES.getExecutorProcessControl(), *ProcessSymsJD, true, true);
    if (!PS)
      return PS.takeError();
    ObjLinkingLayer->addPlugin(std::move(*PS));
    return Error::success();
  }
  default:
    return make_error<StringError>(
        "Cannot enable LLJIT perf support: " +
            Triple::getObjectFormatTypeName(TT.getObjectFormat()) +
            " is not supported",
        inconvertibleErrorCode());
  }
}

} // namespace llvm::orc
