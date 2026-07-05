//===-------------- EPCGenericDylibManagerTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManager.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(EPCGenericDylibManagerTest, CreateFromExecutionSession) {
  // Verify that Create(ExecutionSession&) looks up symbols in the bootstrap
  // JITDylib using the default NativeDylibManager symbol names.
  class EPCWithBootstrapSymbols : public UnsupportedExecutorProcessControl {
  public:
    EPCWithBootstrapSymbols(std::shared_ptr<SymbolStringPool> SSP,
                            StringMap<ExecutorAddr> BS)
        : UnsupportedExecutorProcessControl(std::move(SSP)) {
      this->BootstrapSymbols = std::move(BS);
    }
  };

  auto &SNs = rt::orc_rt_NativeDylibManagerSPSSymbols;

  ExecutorAddr InstanceAddr(1), OpenAddr(2), ResolveAddr(3);

  StringMap<ExecutorAddr> BootstrapSyms;
  BootstrapSyms[SNs.InstanceName] = InstanceAddr;
  BootstrapSyms[SNs.OpenName] = OpenAddr;
  BootstrapSyms[SNs.ResolveName] = ResolveAddr;

  auto SSP = std::make_shared<SymbolStringPool>();
  auto EPC =
      std::make_unique<EPCWithBootstrapSymbols>(SSP, std::move(BootstrapSyms));
  ExecutionSession ES(std::move(EPC));

  auto Result = EPCGenericDylibManager::Create(ES, SNs);
  EXPECT_THAT_EXPECTED(Result, Succeeded());

  cantFail(ES.endSession());
}

} // namespace
