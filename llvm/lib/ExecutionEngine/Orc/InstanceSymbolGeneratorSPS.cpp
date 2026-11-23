//===- InstanceSymbolGeneratorSPS.cpp - SPS instance symbols --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/InstanceSymbolGeneratorSPS.h"

namespace llvm::orc::sps {

std::unique_ptr<InstanceSymbolGenerator>
createInstanceSymbolGenerator(ExecutorAddr LookupFnAddr) {
  return std::make_unique<InstanceSymbolGenerator>(
      InstanceSymbolGenerator::LookupProxy(
          SessionLookupInstanceSymbolsProxySpec::dispatch, LookupFnAddr));
}

} // namespace llvm::orc::sps
