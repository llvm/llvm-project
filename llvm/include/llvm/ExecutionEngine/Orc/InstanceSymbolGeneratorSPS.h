//===- InstanceSymbolGeneratorSPS.h - SPS instance symbols ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds InstanceSymbolGenerator to the ORC runtime's SPS controller interface:
// a ProxySpec for the lookup operation, plus a factory that constructs a
// generator from the lookup function's address.
//
// The spec pairs InstanceSymbolGenerator's proxy with its controller-interface
// descriptor in Shared/SPSCI/SessionSPSCI.h, which supplies the wrapper name
// and wire signature.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATORSPS_H
#define LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATORSPS_H

#include "llvm/ExecutionEngine/Orc/InstanceSymbolGenerator.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SessionSPSCI.h"
#include "llvm/Support/Compiler.h"

#include <memory>

namespace llvm::orc::sps {

using SessionLookupInstanceSymbolsProxySpec =
    ProxySpec<InstanceSymbolGenerator::LookupProxy,
              rt::sps_ci::SessionLookupInstanceSymbols>;

/// Create an InstanceSymbolGenerator that calls the ORC runtime's SPS
/// instance-symbol lookup function at the given address.
LLVM_ABI std::unique_ptr<InstanceSymbolGenerator>
createInstanceSymbolGenerator(ExecutorAddr LookupFnAddr);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_INSTANCESYMBOLGENERATORSPS_H
