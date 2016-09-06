//===- ModuleSubstreamVisitor.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/ModuleSubstream.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"

namespace llvm {
namespace codeview {
struct LineColumnEntry {
  support::ulittle32_t NameIndex;
  msf::FixedStreamArray<LineNumberEntry> LineNumbers;
  msf::FixedStreamArray<ColumnNumberEntry> Columns;
};

struct FileChecksumEntry {
  uint32_t FileNameOffset;    // Byte offset of filename in global stringtable.
  FileChecksumKind Kind;      // The type of checksum.
  ArrayRef<uint8_t> Checksum; // The bytes of the checksum.
};

typedef msf::VarStreamArray<LineColumnEntry> LineInfoArray;
typedef msf::VarStreamArray<FileChecksumEntry> FileChecksumArray;

class IModuleSubstreamVisitor {
public:
  virtual ~IModuleSubstreamVisitor() {}

  virtual Error visitUnknown(ModuleSubstreamKind Kind,
                             msf::ReadableStreamRef Data) = 0;
  virtual Error visitSymbols(msf::ReadableStreamRef Data);
  virtual Error visitLines(msf::ReadableStreamRef Data,
                           const LineSubstreamHeader *Header,
                           const LineInfoArray &Lines);
  virtual Error visitStringTable(msf::ReadableStreamRef Data);
  virtual Error visitFileChecksums(msf::ReadableStreamRef Data,
                                   const FileChecksumArray &Checksums);
  virtual Error visitFrameData(msf::ReadableStreamRef Data);
  virtual Error visitInlineeLines(msf::ReadableStreamRef Data);
  virtual Error visitCrossScopeImports(msf::ReadableStreamRef Data);
  virtual Error visitCrossScopeExports(msf::ReadableStreamRef Data);
  virtual Error visitILLines(msf::ReadableStreamRef Data);
  virtual Error visitFuncMDTokenMap(msf::ReadableStreamRef Data);
  virtual Error visitTypeMDTokenMap(msf::ReadableStreamRef Data);
  virtual Error visitMergedAssemblyInput(msf::ReadableStreamRef Data);
  virtual Error visitCoffSymbolRVA(msf::ReadableStreamRef Data);
};

Error visitModuleSubstream(const ModuleSubstream &R,
                           IModuleSubstreamVisitor &V);
} // namespace codeview

namespace msf {
template <> class VarStreamArrayExtractor<codeview::LineColumnEntry> {
public:
  VarStreamArrayExtractor(const codeview::LineSubstreamHeader *Header)
      : Header(Header) {}

  Error operator()(ReadableStreamRef Stream, uint32_t &Len,
                   codeview::LineColumnEntry &Item) const {
    using namespace codeview;
    const LineFileBlockHeader *BlockHeader;
    StreamReader Reader(Stream);
    if (auto EC = Reader.readObject(BlockHeader))
      return EC;
    bool HasColumn = Header->Flags & LineFlags::HaveColumns;
    uint32_t LineInfoSize =
        BlockHeader->NumLines *
        (sizeof(LineNumberEntry) + (HasColumn ? sizeof(ColumnNumberEntry) : 0));
    if (BlockHeader->BlockSize < sizeof(LineFileBlockHeader))
      return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                       "Invalid line block record size");
    uint32_t Size = BlockHeader->BlockSize - sizeof(LineFileBlockHeader);
    if (LineInfoSize > Size)
      return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                       "Invalid line block record size");
    // The value recorded in BlockHeader->BlockSize includes the size of
    // LineFileBlockHeader.
    Len = BlockHeader->BlockSize;
    Item.NameIndex = BlockHeader->NameIndex;
    if (auto EC = Reader.readArray(Item.LineNumbers, BlockHeader->NumLines))
      return EC;
    if (HasColumn) {
      if (auto EC = Reader.readArray(Item.Columns, BlockHeader->NumLines))
        return EC;
    }
    return Error::success();
  }

private:
  const codeview::LineSubstreamHeader *Header;
};

template <> class VarStreamArrayExtractor<codeview::FileChecksumEntry> {
public:
  Error operator()(ReadableStreamRef Stream, uint32_t &Len,
                   codeview::FileChecksumEntry &Item) const {
    using namespace codeview;
    const FileChecksum *Header;
    StreamReader Reader(Stream);
    if (auto EC = Reader.readObject(Header))
      return EC;
    Item.FileNameOffset = Header->FileNameOffset;
    Item.Kind = static_cast<FileChecksumKind>(Header->ChecksumKind);
    if (auto EC = Reader.readBytes(Item.Checksum, Header->ChecksumSize))
      return EC;
    Len = sizeof(FileChecksum) + Header->ChecksumSize;
    return Error::success();
  }
};

} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULESUBSTREAMVISITOR_H
