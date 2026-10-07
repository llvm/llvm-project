//===- EPCGenericDylibManagerSPS.h - SPS dylib management -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binds EPCGenericDylibManager to the ORC runtime's SPS controller interface:
// a ProxySpec per operation, plus factories that resolve them and construct an
// instance.
//
// Each spec pairs one of EPCGenericDylibManager's proxies with its
// controller-interface descriptor in Shared/SPSCI/NativeDylibManagerSPSCI.h,
// which supplies the wrapper name and wire signature. The specs are public so
// that clients can resolve the operations under non-default names, using
// recordProxy<Spec>(&P, Name) with lookupAndApply.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGERSPS_H
#define LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGERSPS_H

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManager.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/NativeDylibManagerSPSCI.h"
#include "llvm/Support/Compiler.h"

#include <memory>

namespace llvm::orc::sps {

using DylibMgrOpenProxySpec =
    ProxySpec<EPCGenericDylibManager::OpenProxy, rt::sps_ci::DylibMgrOpen>;
using DylibMgrResolveProxySpec = ProxySpec<EPCGenericDylibManager::ResolveProxy,
                                           rt::sps_ci::DylibMgrResolve>;

/// Create an EPCGenericDylibManager for the ORC runtime's NativeDylibManager
/// interface, resolving its symbols in the given JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericDylibManager>>
createEPCGenericDylibManager(JITDylib &JD);

/// Create an EPCGenericDylibManager for the ORC runtime's NativeDylibManager
/// interface, resolving its symbols in the given ExecutionSession's bootstrap
/// JITDylib.
LLVM_ABI Expected<std::unique_ptr<EPCGenericDylibManager>>
createEPCGenericDylibManager(ExecutionSession &ES);

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_EPCGENERICDYLIBMANAGERSPS_H
