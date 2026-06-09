//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MSVC-specific.
// Definitions and a parser for the C++ 20 ".modmeta" section.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_COFFMODULEMAP_H
#define LLVM_OBJECT_COFFMODULEMAP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm::object {

struct COFFCxxModuleMetadataHeader {
  uint8_t Version;
  uint8_t Reserved;
  /// Number of bytes used to encode module IDs.
  uint8_t ModuleIndexWidth;
  /// Number of bytes used to encode symbol IDs.
  uint8_t SymbolIndexWidth;
  /// Size of this header and the module lists.
  support::ulittle32_t ModuleDataSize;
};

struct COFFCxxModuleMetadata {
  uint8_t Version;
  uint8_t Reserved;
  /// Number of bytes used to encode module IDs.
  uint8_t ModuleIndexWidth;
  /// Number of bytes used to encode symbol IDs.
  uint8_t SymbolIndexWidth;

  /// Data for modules.
  ///
  /// Starts with a list of module IDs that are header units.
  /// This is followed by a list of modules, which is terminated by the maximum
  /// module ID (e.g. 0xff for Width=1). Each module starts with the ID followed
  /// by a list of modules it depends on, a list of non-exported symbols, and a
  /// list of exported symbols.
  StringRef ModuleData;

  /// List of null-terminated module names.
  ///
  /// The names are present in the order of the modules in \p ModuleData. Note
  /// that the module with ID 0 does not have a name.
  StringRef NamesData;
};

struct LLVM_ABI COFFCxxModuleMetadataReader {
  COFFCxxModuleMetadataReader(const COFFCxxModuleMetadata &Map);

  StringRef ModuleData;
  StringRef NamesData;

  uint8_t ModuleIndexWidth;
  uint8_t SymbolIndexWidth;

  bool hasModuleData() const;

  Expected<uint32_t> readModuleID();
  Expected<StringRef> readModuleName();

  /// Read a list of modules.
  ///
  /// \param Visitor A callback that accepts an `ArrayRef` of `uint8_t`,
  /// `ulittle16_t`, or `ulittle32_t` depending on the index width.
  template <typename T> Error readModuleList(T &&Visitor) {
    return readList(std::forward<T>(Visitor), ModuleIndexWidth);
  }

  /// Read a list of symbols.
  ///
  /// \param Visitor A callback that accepts an `ArrayRef` of `uint8_t`,
  /// `ulittle16_t`, or `ulittle32_t` depending on the index width.
  template <typename T> Error readSymbolList(T &&Visitor) {
    return readList(std::forward<T>(Visitor), SymbolIndexWidth);
  }

private:
  template <typename T> Error readList(T &&Visitor, uint8_t Width) {
    Expected<ArrayRef<uint8_t>> List = readListImpl(Width);
    if (!List)
      return List.takeError();

    if (Width == 1)
      Visitor(*List);
    else if (Width == 2)
      Visitor(ArrayRef<support::ulittle16_t>(
          reinterpret_cast<const support::ulittle16_t *>(List->data()),
          List->size() / sizeof(support::ulittle16_t)));
    else if (Width == 4)
      Visitor(ArrayRef<support::ulittle32_t>(
          reinterpret_cast<const support::ulittle32_t *>(List->data()),
          List->size() / sizeof(support::ulittle32_t)));
    else
      assert(false && "unexpected list width");

    return Error::success();
  }

  Expected<ArrayRef<uint8_t>> readListImpl(uint8_t Width);
};

LLVM_ABI Expected<COFFCxxModuleMetadata>
parseCOFFCxxModuleMetadata(StringRef SectionData);

} // namespace llvm::object

#endif
