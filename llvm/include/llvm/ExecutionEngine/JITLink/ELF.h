//===------- ELF.h - Generic JIT link function for ELF ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic jit-link functions for ELF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_ELF_H
#define LLVM_EXECUTIONENGINE_JITLINK_ELF_H

#include "llvm/ExecutionEngine/JITLink/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

/// jit-link the given ObjBuffer, which must be a ELF object file.
///
/// Uses conservative defaults for GOT and stub handling based on the target
/// platform.
void jitLink_ELF(std::unique_ptr<JITLinkContext> Ctx);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_ELF_H
