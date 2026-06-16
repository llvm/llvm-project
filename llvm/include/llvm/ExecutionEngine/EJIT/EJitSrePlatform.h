//===-- EJitSrePlatform.h - SRE platform adapter for the code pool --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Thin adapter that wires EJitCodePoolManager to the real SRE platform
//  primitives (SRE_MemDbgAlloc for raw memory, enable_ex for sealing). This
//  header is only meaningful when EJIT_SRE_CODE_POOL is defined; it keeps the
//  SRE symbol declarations out of generic LLVM translation units.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITSREPLATFORM_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITSREPLATFORM_H

#ifdef EJIT_SRE_CODE_POOL

#include "llvm/ExecutionEngine/EJIT/EJitCodePool.h"
#include <memory>

namespace llvm {
namespace ejit {

/// Construct an EJitCodePoolManager wired to the SRE platform: raw memory from
/// SRE_MemDbgAlloc (partition EJIT_SRE_CODE_POOL_PTNO) and sealing via
/// enable_ex. On a host without real SRE symbols, weak fallbacks make this a
/// link-safe no-op-seal / aligned-host-alloc manager (see EJitSrePlatform.cpp).
std::unique_ptr<EJitCodePoolManager> makeSreCodePoolManager();

} // namespace ejit
} // namespace llvm

#endif // EJIT_SRE_CODE_POOL
#endif // LLVM_EXECUTIONENGINE_EJIT_EJITSREPLATFORM_H
