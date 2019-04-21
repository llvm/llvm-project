//===--- JITLink_MachO.h - Generic JIT link function for MachO --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic jit-link functions for MachO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_JITLINK_MACHO_H
#define LLVM_EXECUTIONENGINE_JITLINK_JITLINK_MACHO_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

namespace llvm {
namespace jitlink {

/// jit-link the given ObjBuffer, which must be a MachO object file.
///
/// Uses conservative defaults for GOT and stub handling based on the target
/// platform.
void jitLink_MachO(std::unique_ptr<JITLinkContext> Ctx);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINK_MACHO_X86_64_H
