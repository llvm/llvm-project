//===- CodeGenDataReader.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading codegen data.
//
//===----------------------------------------------------------------------===//

#include "llvm/CGData/CodeGenDataReader.h"
#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "cg-data-reader"

using namespace llvm;

static cl::opt<bool> IndexedCodeGenDataReadFunctionMapNames(
    "indexed-codegen-data-read-function-map-names", cl::init(true), cl::Hidden,
    cl::desc("Read function map names in indexed CodeGenData. Can be "
             "disabled to save memory and time for final consumption of the "
             "indexed CodeGenData in production."));

namespace llvm {

static Expected<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(const Twine &Filename, vfs::FileSystem &FS) {
  auto BufferOrErr = Filename.str() == "-" ? MemoryBuffer::getSTDIN()
                                           : FS.getBufferForFile(Filename);
  if (std::error_code EC = BufferOrErr.getError())
    return errorCodeToError(EC);
  return std::move(BufferOrErr.get());
}

Error CodeGenDataReader::mergeFromObjectFile(
    const object::ObjectFile *Obj, OutlinedHashTreeRecord &GlobalOutlineRecord,
    StableFunctionMapRecord &GlobalFunctionMapRecord,
    stable_hash *CombinedHash) {
  Triple TT = Obj->makeTriple();
  auto CGOutlineName =
      getCodeGenDataSectionName(CG_outline, TT.getObjectFormat(), false);
  auto CGMergeName =
      getCodeGenDataSectionName(CG_merge, TT.getObjectFormat(), false);

  auto processSectionContents = [&](const StringRef &Name,
                                    const StringRef &Contents) {
    if (Name != CGOutlineName && Name != CGMergeName)
      return;
    if (CombinedHash)
      *CombinedHash = stable_hash_combine(*CombinedHash, xxh3_64bits(Contents));
    auto *Data = reinterpret_cast<const unsigned char *>(Contents.data());
    auto *EndData = Data + Contents.size();
    // In case dealing with an executable that has concatenated cgdata,
    // we want to merge them into a single cgdata.
    // Although it's not a typical workflow, we support this scenario
    // by looping over all data in the sections.
    if (Name == CGOutlineName) {
      while (Data != EndData) {
        OutlinedHashTreeRecord LocalOutlineRecord;
        LocalOutlineRecord.deserialize(Data);
        GlobalOutlineRecord.merge(LocalOutlineRecord);
      }
    } else if (Name == CGMergeName) {
      while (Data != EndData) {
        StableFunctionMapRecord LocalFunctionMapRecord;
        LocalFunctionMapRecord.deserialize(Data);
        GlobalFunctionMapRecord.merge(LocalFunctionMapRecord);
      }
    }
  };

  for (auto &Section : Obj->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();
    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr)
      return ContentsOrErr.takeError();
    processSectionContents(*NameOrErr, *ContentsOrErr);
  }

  return Error::success();
}

Error IndexedCodeGenDataReader::read() {
  using namespace support;

  // The smallest header with the version 1 is 24 bytes.
  // Do not update this value even with the new version of the header.
  const unsigned MinHeaderSize = 24;
  if (DataBuffer->getBufferSize() < MinHeaderSize)
    return error(cgdata_error::bad_header);

  auto *Start =
      reinterpret_cast<const unsigned char *>(DataBuffer->getBufferStart());
  auto *End =
      reinterpret_cast<const unsigned char *>(DataBuffer->getBufferEnd());
  if (auto E = IndexedCGData::Header::readFromBuffer(Start).moveInto(Header))
    return E;

  if (hasOutlinedHashTree()) {
    const unsigned char *Ptr = Start + Header.OutlinedHashTreeOffset;
    if (Ptr >= End)
      return error(cgdata_error::eof);
    HashTreeRecord.deserialize(Ptr);
  }
  if (hasStableFunctionMap()) {
    const unsigned char *Ptr = Start + Header.StableFunctionMapOffset;
    if (Ptr >= End)
      return error(cgdata_error::eof);
    FunctionMapRecord.deserialize(Ptr, IndexedCodeGenDataReadFunctionMapNames);
  }

  return success();
}

Expected<std::unique_ptr<CodeGenDataReader>>
CodeGenDataReader::create(const Twine &Path, vfs::FileSystem &FS) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path, FS);
  if (Error E = BufferOrError.takeError())
    return std::move(E);
  return CodeGenDataReader::create(std::move(BufferOrError.get()));
}

Expected<std::unique_ptr<CodeGenDataReader>>
CodeGenDataReader::create(std::unique_ptr<MemoryBuffer> Buffer) {
  if (Buffer->getBufferSize() == 0)
    return make_error<CGDataError>(cgdata_error::empty_cgdata);

  std::unique_ptr<CodeGenDataReader> Reader;
  // Create the reader.
  if (IndexedCodeGenDataReader::hasFormat(*Buffer))
    Reader = std::make_unique<IndexedCodeGenDataReader>(std::move(Buffer));
  else if (TextCodeGenDataReader::hasFormat(*Buffer))
    Reader = std::make_unique<TextCodeGenDataReader>(std::move(Buffer));
  else
    return make_error<CGDataError>(cgdata_error::malformed);

  // Initialize the reader and return the result.
  if (Error E = Reader->read())
    return std::move(E);

  return std::move(Reader);
}

bool IndexedCodeGenDataReader::hasFormat(const MemoryBuffer &DataBuffer) {
  using namespace support;
  if (DataBuffer.getBufferSize() < sizeof(IndexedCGData::Magic))
    return false;

  uint64_t Magic = endian::read<uint64_t, llvm::endianness::little, aligned>(
      DataBuffer.getBufferStart());
  // Verify that it's magical.
  return Magic == IndexedCGData::Magic;
}

bool TextCodeGenDataReader::hasFormat(const MemoryBuffer &Buffer) {
  // Verify that this really looks like plain ASCII text by checking a
  // 'reasonable' number of characters (up to the magic size).
  StringRef Prefix = Buffer.getBuffer().take_front(sizeof(uint64_t));
  return llvm::all_of(Prefix, [](char c) { return isPrint(c) || isSpace(c); });
}
Error TextCodeGenDataReader::read() {
  using namespace support;

  // Parse the custom header line by line.
  for (; !Line.is_at_eof(); ++Line) {
    // Skip empty or whitespace-only lines
    if (Line->trim().empty())
      continue;

    if (!Line->starts_with(":"))
      break;
    StringRef Str = Line->drop_front().rtrim();
    if (Str.equals_insensitive("outlined_hash_tree"))
      DataKind |= CGDataKind::FunctionOutlinedHashTree;
    else if (Str.equals_insensitive("stable_function_map"))
      DataKind |= CGDataKind::StableFunctionMergingMap;
    else
      return error(cgdata_error::bad_header);
  }

  // We treat an empty header (that is a comment # only) as a valid header.
  if (Line.is_at_eof()) {
    if (DataKind == CGDataKind::Unknown)
      return Error::success();
    return error(cgdata_error::bad_header);
  }

  // The YAML docs follow after the header.
  const char *Pos = Line->data();
  size_t Size = reinterpret_cast<size_t>(DataBuffer->getBufferEnd()) -
                reinterpret_cast<size_t>(Pos);
  yaml::Input YOS(StringRef(Pos, Size));
  if (hasOutlinedHashTree())
    HashTreeRecord.deserializeYAML(YOS);
  if (hasStableFunctionMap())
    FunctionMapRecord.deserializeYAML(YOS);

  return Error::success();
}
} // end namespace llvm
