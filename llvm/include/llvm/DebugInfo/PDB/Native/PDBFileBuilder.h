//===- PDBFileBuilder.h - PDB File Creation ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_PDBFILEBUILDER_H
#define LLVM_DEBUGINFO_PDB_NATIVE_PDBFILEBUILDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/PDB/Native/HashTable.h"
#include "llvm/DebugInfo/PDB/Native/NamedStreamMap.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTableBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {
class WritableBinaryStream;
namespace codeview {
struct GUID;
}

namespace msf {
class MSFBuilder;
struct MSFLayout;
}
namespace pdb {
struct SrcHeaderBlockEntry;
class DbiStreamBuilder;
class InfoStreamBuilder;
class GSIStreamBuilder;
class TpiStreamBuilder;

class PDBFileBuilder {
public:
  LLVM_ABI explicit PDBFileBuilder(BumpPtrAllocator &Allocator);
  LLVM_ABI ~PDBFileBuilder();
  PDBFileBuilder(const PDBFileBuilder &) = delete;
  PDBFileBuilder &operator=(const PDBFileBuilder &) = delete;

  LLVM_ABI Error initialize(uint32_t BlockSize);

  LLVM_ABI msf::MSFBuilder &getMsfBuilder();
  LLVM_ABI InfoStreamBuilder &getInfoBuilder();
  LLVM_ABI DbiStreamBuilder &getDbiBuilder();
  LLVM_ABI TpiStreamBuilder &getTpiBuilder();
  LLVM_ABI TpiStreamBuilder &getIpiBuilder();
  LLVM_ABI PDBStringTableBuilder &getStringTableBuilder();
  LLVM_ABI GSIStreamBuilder &getGsiBuilder();

  // If HashPDBContentsToGUID is true on the InfoStreamBuilder, Guid is filled
  // with the computed PDB GUID on return.
  LLVM_ABI Error commit(StringRef Filename, codeview::GUID *Guid);

  LLVM_ABI Expected<uint32_t> getNamedStreamIndex(StringRef Name) const;
  LLVM_ABI Error addNamedStream(StringRef Name, StringRef Data);
  LLVM_ABI void addInjectedSource(StringRef Name,
                                  std::unique_ptr<MemoryBuffer> Buffer);

private:
  struct InjectedSourceDescriptor {
    // The full name of the stream that contains the contents of this injected
    // source.  This is built as a concatenation of the literal "/src/files"
    // plus the "vname".
    std::string StreamName;

    // The exact name of the file name as specified by the user.
    uint32_t NameIndex;

    // The string table index of the "vname" of the file.  As far as we
    // understand, this is the same as the name, except it is lowercased and
    // forward slashes are converted to backslashes.
    uint32_t VNameIndex;
    std::unique_ptr<MemoryBuffer> Content;
  };

  Error finalizeMsfLayout();
  Expected<uint32_t> allocateNamedStream(StringRef Name, uint32_t Size);

  void commitInjectedSources(WritableBinaryStream &MsfBuffer,
                             const msf::MSFLayout &Layout);
  void commitSrcHeaderBlock(WritableBinaryStream &MsfBuffer,
                            const msf::MSFLayout &Layout);

  BumpPtrAllocator &Allocator;

  std::unique_ptr<msf::MSFBuilder> Msf;
  std::unique_ptr<InfoStreamBuilder> Info;
  std::unique_ptr<DbiStreamBuilder> Dbi;
  std::unique_ptr<GSIStreamBuilder> Gsi;
  std::unique_ptr<TpiStreamBuilder> Tpi;
  std::unique_ptr<TpiStreamBuilder> Ipi;

  PDBStringTableBuilder Strings;
  StringTableHashTraits InjectedSourceHashTraits;
  HashTable<SrcHeaderBlockEntry> InjectedSourceTable;

  SmallVector<InjectedSourceDescriptor, 2> InjectedSources;

  NamedStreamMap NamedStreams;
  DenseMap<uint32_t, std::string> NamedStreamData;
};
}
}

#endif
