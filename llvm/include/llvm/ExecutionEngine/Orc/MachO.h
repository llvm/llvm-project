//===------------- MachO.h - MachO format utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for load MachO relocatable object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MACHO_H
#define LLVM_EXECUTIONENGINE_ORC_MACHO_H

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

namespace object {

class MachOUniversalBinary;

} // namespace object

namespace orc {

/// Check that the given buffer contains a MachO object file compatible with the
/// given triple.
/// ObjIsSlice should be set to true if Obj is a slice of a universal binary
/// (that fact will then be reported in the error messages).
Error checkMachORelocatableObject(MemoryBufferRef Obj, const Triple &TT,
                                  bool ObjIsSlice);

/// Check that the given buffer contains a MachO object file compatible with the
/// given triple.
/// This convenience overload returns the buffer if it passes all checks,
/// otherwise it returns an error.
Expected<std::unique_ptr<MemoryBuffer>>
checkMachORelocatableObject(std::unique_ptr<MemoryBuffer> Obj, const Triple &TT,
                            bool ObjIsSlice);

/// Load a relocatable object compatible with TT from Path.
/// If Path is a universal binary, this function will return a buffer for the
/// slice compatible with Triple (if one is present).
Expected<std::unique_ptr<MemoryBuffer>> loadMachORelocatableObject(
    StringRef Path, const Triple &TT,
    std::optional<StringRef> IdentifierOverride = std::nullopt);

/// Load a compatible relocatable object (if available) from a MachO universal
/// binary.
Expected<std::unique_ptr<MemoryBuffer>>
loadMachORelocatableObjectFromUniversalBinary(
    StringRef UBPath, std::unique_ptr<MemoryBuffer> UBBuf, const Triple &TT,
    std::optional<StringRef> IdentifierOverride = std::nullopt);

/// Utility for identifying the file-slice compatible with TT in a universal
/// binary.
Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(object::MachOUniversalBinary &UB, const Triple &TT);

/// Utility for identifying the file-slice compatible with TT in a universal
/// binary.
Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(MemoryBufferRef UBBuf, const Triple &TT);

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHO_H
