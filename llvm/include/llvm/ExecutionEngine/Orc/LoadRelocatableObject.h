//===---- LoadRelocatableObject.h - Load relocatable objects ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A wrapper for `MemoryBuffer::getFile` / `MemoryBuffer::getFileSlice` that:
//
//   1. Adds file paths to errors by default.
//   2. Checks architecture compatibility up-front.
//   3. Handles MachO universal binaries, returning the MemoryBuffer for the
//      requested slice only.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LOADRELOCATABLEOBJECT_H
#define LLVM_EXECUTIONENGINE_ORC_LOADRELOCATABLEOBJECT_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace orc {

// Load an object file compatible with the given triple (if given) from the
// given path. May return a file slice if the path contains a universal binary.
Expected<std::unique_ptr<MemoryBuffer>> loadRelocatableObject(
    StringRef Path, const Triple &TT,
    std::optional<StringRef> IdentifierOverride = std::nullopt);

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LOADRELOCATABLEOBJECT_H
