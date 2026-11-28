//===- FunctionDebug.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The debug info of a COMDAT function, which every object that emits the
// function repeats, moves to a pack with its type indices and file references
// rewritten. These functions rewrite it in both directions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_FUNCTIONDEBUG_H
#define LLVM_DEBUGINFO_CVPACK_FUNCTIONDEBUG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

/// How transformFunctionDebug writes the file operands of inline-site
/// annotations: in the fewest bytes, as compilers do, or always in 4 bytes, so
/// that a pack's copy does not depend on any object's checksum offsets.
enum class FileOperandEncoding { Minimal, Fixed };

/// Append Data, the subsections of a COMDAT .debug$S section without its
/// leading magic, to Out with every non-simple type index mapped through
/// MapType and every file reference, of a line block or an inline-site
/// annotation, through MapFile. Both are called with the value and its offset
/// in Data. Inline sites that change files are rebuilt with Encoding, and
/// padded with zeros to 4 bytes. Fails for content that cannot be moved to a
/// pack: anything but symbols and lines subsections, unknown symbols, and
/// symbols that reference the string table.
LLVM_ABI Error transformFunctionDebug(
    ArrayRef<uint8_t> Data, std::vector<uint8_t> &Out,
    function_ref<Expected<uint32_t>(uint32_t Value, uint32_t Offset)> MapType,
    function_ref<Expected<uint32_t>(uint32_t Value, uint32_t Offset)> MapFile,
    FileOperandEncoding Encoding);

/// The source files of an object, from its strings and file checksums
/// subsections.
class ObjectFileTable {
public:
  /// Build the table from the non-COMDAT .debug$S sections of an object. Fails
  /// unless they hold exactly one strings and one file checksums subsection.
  LLVM_ABI static Expected<ObjectFileTable>
  create(ArrayRef<ArrayRef<uint8_t>> DebugSSections);
  /// The checksum offset of the first entry for File.
  LLVM_ABI std::optional<uint32_t> lookup(const PackFileEntry &File) const;
  /// The entry at a checksum offset.
  LLVM_ABI std::optional<PackFileEntry> entryAt(uint32_t Offset) const;
  ArrayRef<std::pair<uint32_t, PackFileEntry>> entries() const {
    return Entries;
  }

private:
  std::vector<std::pair<uint32_t, PackFileEntry>> Entries;
  std::map<std::string, uint32_t, std::less<>> ByIdentity;
};

/// A unique byte string for a file's identity.
LLVM_ABI std::string fileIdentity(const PackFileEntry &File);

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_FUNCTIONDEBUG_H
