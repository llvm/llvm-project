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

#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/LoadLinkableFile.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

namespace object {

class Archive;
class MachOUniversalBinary;

} // namespace object

namespace orc {

class ExecutionSession;
class JITDylib;
class ObjectLayer;

/// Check that the given buffer contains a MachO object file compatible with the
/// given triple.
/// ObjIsSlice should be set to true if Obj is a slice of a universal binary
/// (that fact will then be reported in the error messages).
LLVM_ABI Error checkMachORelocatableObject(MemoryBufferRef Obj,
                                           const Triple &TT, bool ObjIsSlice);

/// Check that the given buffer contains a MachO object file compatible with the
/// given triple.
/// This convenience overload returns the buffer if it passes all checks,
/// otherwise it returns an error.
LLVM_ABI Expected<std::unique_ptr<MemoryBuffer>>
checkMachORelocatableObject(std::unique_ptr<MemoryBuffer> Obj, const Triple &TT,
                            bool ObjIsSlice);

/// Load a relocatable object compatible with TT from Path.
/// If Path is a universal binary, this function will return a buffer for the
/// slice compatible with Triple (if one is present).
LLVM_ABI Expected<std::pair<std::unique_ptr<MemoryBuffer>, LinkableFileKind>>
loadMachOLinkableFile(
    StringRef Path, const Triple &TT, LoadArchives LA,
    std::optional<StringRef> IdentifierOverride = std::nullopt);

/// Load a compatible relocatable object (if available) from a MachO universal
/// binary.
/// Path is only used for error reporting. Identifier will be used to name the
/// resulting buffer.
LLVM_ABI Expected<std::pair<std::unique_ptr<MemoryBuffer>, LinkableFileKind>>
loadLinkableSliceFromMachOUniversalBinary(sys::fs::file_t FD,
                                          std::unique_ptr<MemoryBuffer> UBBuf,
                                          const Triple &TT, LoadArchives LA,
                                          StringRef UBPath,
                                          StringRef Identifier);

/// Utility for identifying the file-slice compatible with TT in a universal
/// binary.
LLVM_ABI Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(object::MachOUniversalBinary &UB, const Triple &TT);

/// Utility for identifying the file-slice compatible with TT in a universal
/// binary.
LLVM_ABI Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(MemoryBufferRef UBBuf, const Triple &TT);

/// For use with StaticLibraryDefinitionGenerators.
class ForceLoadMachOArchiveMembers {
public:
  ForceLoadMachOArchiveMembers(ObjectLayer &L, JITDylib &JD, bool ObjCOnly)
      : L(L), JD(JD), ObjCOnly(ObjCOnly) {}

  LLVM_ABI Expected<bool> operator()(object::Archive &A,
                                     MemoryBufferRef MemberBuf, size_t Index);

private:
  ObjectLayer &L;
  JITDylib &JD;
  bool ObjCOnly;
};

using GetFallbackArchsFn =
    unique_function<SmallVector<std::pair<uint32_t, uint32_t>>(
        uint32_t CPUType, uint32_t CPUSubType)>;

/// Match the exact CPU type/subtype only.
LLVM_ABI SmallVector<std::pair<uint32_t, uint32_t>>
noFallbackArchs(uint32_t CPUType, uint32_t CPUSubType);

/// Match standard dynamic loader fallback rules.
LLVM_ABI SmallVector<std::pair<uint32_t, uint32_t>>
standardMachOFallbackArchs(uint32_t CPUType, uint32_t CPUSubType);

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// given dylib.
LLVM_ABI Expected<SymbolNameSet> getDylibInterfaceFromDylib(
    ExecutionSession &ES, Twine Path,
    GetFallbackArchsFn GetFallbackArchs = standardMachOFallbackArchs);

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// relevant slice of the TapiUniversal file.
LLVM_ABI Expected<SymbolNameSet> getDylibInterfaceFromTapiFile(
    ExecutionSession &ES, Twine Path,
    GetFallbackArchsFn GetFallbackArchs = standardMachOFallbackArchs);

/// Returns a SymbolNameSet containing the exported symbols defined in the
/// relevant slice of the given file, which may be either a dylib or a tapi
/// file.
LLVM_ABI Expected<SymbolNameSet> getDylibInterface(
    ExecutionSession &ES, Twine Path,
    GetFallbackArchsFn GetFallbackArchs = standardMachOFallbackArchs);

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_MACHO_H
