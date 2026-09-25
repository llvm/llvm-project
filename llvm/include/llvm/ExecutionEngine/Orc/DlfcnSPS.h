//===------- DlfcnSPS.h - SPS specs for dlfcn proxies -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS ProxySpecs for the dlfcn proxies: each binds a Proxy to its
// controller-interface descriptor in Shared/SPSCI/DlfcnSPSCI.h, which supplies
// the wrapper name and wire signature.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_DLFCNSPS_H
#define LLVM_EXECUTIONENGINE_ORC_DLFCNSPS_H

#include "llvm/ExecutionEngine/Orc/Dlfcn.h"
#include "llvm/ExecutionEngine/Orc/SPSProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/DlfcnSPSCI.h"

namespace llvm::orc::sps {

/// SPS proxy for DlfcnOpenProxy (dlopen).
using DlfcnOpenProxySpec = ProxySpec<DlfcnOpenProxy, rt::sps_ci::DlfcnOpen>;

/// SPS proxy for DlfcnUpdateProxy (dlupdate).
using DlfcnUpdateProxySpec =
    ProxySpec<DlfcnUpdateProxy, rt::sps_ci::DlfcnUpdate>;

/// SPS proxy for DlfcnCloseProxy (dlclose).
using DlfcnCloseProxySpec = ProxySpec<DlfcnCloseProxy, rt::sps_ci::DlfcnClose>;

} // namespace llvm::orc::sps

#endif // LLVM_EXECUTIONENGINE_ORC_DLFCNSPS_H
