//===- PackFile.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeView pack files: the type records, symbols and function debug info that
// COFF objects share, stored in a COFF object named by its content id.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_PACKFILE_H
#define LLVM_DEBUGINFO_CVPACK_PACKFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

/// A pack is an ordinary COFF object, stored as <content id>.cvp: 64 lowercase
/// hex digits and a distinct extension, so that it is not mistaken for, or
/// globbed with, the objects of a build.
inline constexpr StringLiteral PackFileExtension = ".cvp";

/// A non-owning view of a CodeView pack COFF object. All record views point
/// into FileData, which must outlive this object.
struct PackFileView {
  ArrayRef<uint8_t> FileData;
  PackId ContentId{};
  /// The machine of the objects packed against this pack.
  uint16_t Machine = 0;
  std::vector<CVType> Types;
  std::vector<CVSymbol> Symbols;
  /// Function debug info: COMDAT .debug$S contents without their magic, with
  /// type indices in this pack's logical index space and file checksum
  /// offsets replaced by indices into Files.
  std::vector<PackFileEntry> Files;
  std::vector<ArrayRef<uint8_t>> Functions;
  /// The BLAKE3 global hash of every record of Types, 8 bytes each, from the
  /// optional .debug$H section; empty without one.
  ArrayRef<uint8_t> GlobalHashes;
};

struct PackContents {
  /// The machine of the objects packed against this pack.
  uint16_t Machine = COFF::IMAGE_FILE_MACHINE_AMD64;
  std::vector<ArrayRef<uint8_t>> Types;
  std::vector<ArrayRef<uint8_t>> Symbols;
  std::vector<PackFileEntry> Files;
  std::vector<ArrayRef<uint8_t>> Functions;
  /// Write a .debug$H section with the global hash of every type record, so
  /// that readers need not compute them.
  bool WriteGlobalHashes = false;
};

struct PackWriteResult {
  PackId ContentId{};
  std::vector<uint8_t> FileData;
};

/// Read a pack. With ExpectedContentId, the file must hash to it, unless
/// VerifyContentId is false: the id is then taken as the pack's without
/// reading the whole file.
LLVM_ABI Expected<PackFileView>
readPackFile(ArrayRef<uint8_t> FileData,
             std::optional<PackId> ExpectedContentId = std::nullopt,
             bool VerifyContentId = true);

LLVM_ABI Expected<PackWriteResult> writePackFile(const PackContents &Contents);

/// A loaded pack. The records of View point into Buffer.
struct ResolvedCVPack {
  std::unique_ptr<MemoryBuffer> Buffer;
  PackFileView View;
};

/// Loads packs by content id. An object names its pack by id only, so the
/// pack is searched as <id>.cvp: first in the directories that depend on the
/// object, such as its own, then in those of the resolver. Thread-safe; a pack
/// is loaded once.
class CVPackResolver {
public:
  LLVM_ABI explicit CVPackResolver(ArrayRef<StringRef> Directories = {});
  LLVM_ABI ~CVPackResolver();
  /// Load Id, if it is not loaded yet, searching ObjectDirectories before the
  /// directories of the resolver. An empty directory is the current one.
  /// Without VerifyContentId, the content of the pack is not checked against
  /// its id: for packs already checked by other means, such as a key index
  /// built from the checked pack, when every object read through it is
  /// verified against its own hash.
  LLVM_ABI Expected<const ResolvedCVPack *>
  get(const PackId &Id, bool VerifyContentId = true,
      ArrayRef<StringRef> ObjectDirectories = {});

private:
  struct Impl;
  std::unique_ptr<Impl> P;
};

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_PACKFILE_H
