//===- OffloadBundle.cpp - Utilities for offload bundles---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//

#include "llvm/Object/OffloadBundle.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"

using namespace llvm;
using namespace llvm::object;

static TimerGroup OffloadBundlerTimerGroup("Offload Bundler Timer Group",
                                           "Timer group for offload bundler");

// Extract an Offload bundle (usually a Offload Bundle) from a fat_bin
// section.
Error extractOffloadBundle(MemoryBufferRef Contents, uint64_t SectionOffset,
                           StringRef FileName,
                           SmallVectorImpl<OffloadBundleFatBin> &Bundles) {

  size_t Offset = 0;
  size_t NextbundleStart = 0;
  StringRef Magic;
  std::unique_ptr<MemoryBuffer> Buffer;

  // There could be multiple offloading bundles stored at this section.
  while ((NextbundleStart != StringRef::npos) &&
         (Offset < Contents.getBuffer().size())) {
    Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator=*/false);

    if (identify_magic((*Buffer).getBuffer()) ==
        file_magic::offload_bundle_compressed) {
      Magic = "CCOB";
      // Decompress this bundle first.
      NextbundleStart = (*Buffer).getBuffer().find(Magic, Magic.size());
      if (NextbundleStart == StringRef::npos)
        NextbundleStart = (*Buffer).getBuffer().size();

      ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
          MemoryBuffer::getMemBuffer(
              (*Buffer).getBuffer().take_front(NextbundleStart), FileName,
              false);
      if (std::error_code EC = CodeOrErr.getError())
        return createFileError(FileName, EC);

      Expected<std::unique_ptr<MemoryBuffer>> DecompressedBufferOrErr =
          CompressedOffloadBundle::decompress(**CodeOrErr, nullptr);
      if (!DecompressedBufferOrErr)
        return createStringError("failed to decompress input: " +
                                 toString(DecompressedBufferOrErr.takeError()));

      auto FatBundleOrErr = OffloadBundleFatBin::create(
          **DecompressedBufferOrErr, Offset, FileName, true);
      if (!FatBundleOrErr)
        return FatBundleOrErr.takeError();

      // Add current Bundle to list.
      Bundles.emplace_back(std::move(**FatBundleOrErr));

    } else if (identify_magic((*Buffer).getBuffer()) ==
               file_magic::offload_bundle) {
      // Create the OffloadBundleFatBin object. This will also create the Bundle
      // Entry list info.
      auto FatBundleOrErr = OffloadBundleFatBin::create(
          *Buffer, SectionOffset + Offset, FileName);
      if (!FatBundleOrErr)
        return FatBundleOrErr.takeError();

      // Add current Bundle to list.
      Bundles.emplace_back(std::move(**FatBundleOrErr));

      Magic = "__CLANG_OFFLOAD_BUNDLE__";
      NextbundleStart = (*Buffer).getBuffer().find(Magic, Magic.size());
    }

    if (NextbundleStart != StringRef::npos)
      Offset += NextbundleStart;
  }

  return Error::success();
}

Error OffloadBundleFatBin::readEntries(StringRef Buffer,
                                       uint64_t SectionOffset) {
  uint64_t NumOfEntries = 0;

  BinaryStreamReader Reader(Buffer, llvm::endianness::little);

  // Read the Magic String first.
  StringRef Magic;
  if (auto EC = Reader.readFixedString(Magic, 24))
    return errorCodeToError(object_error::parse_failed);

  // Read the number of Code Objects (Entries) in the current Bundle.
  if (auto EC = Reader.readInteger(NumOfEntries))
    return errorCodeToError(object_error::parse_failed);

  NumberOfEntries = NumOfEntries;

  // For each Bundle Entry (code object).
  for (uint64_t I = 0; I < NumOfEntries; I++) {
    uint64_t EntrySize;
    uint64_t EntryOffset;
    uint64_t EntryIDSize;
    StringRef EntryID;

    if (Error Err = Reader.readInteger(EntryOffset))
      return Err;

    if (Error Err = Reader.readInteger(EntrySize))
      return Err;

    if (Error Err = Reader.readInteger(EntryIDSize))
      return Err;

    if (Error Err = Reader.readFixedString(EntryID, EntryIDSize))
      return Err;

    auto Entry = std::make_unique<OffloadBundleEntry>(
        EntryOffset + SectionOffset, EntrySize, EntryIDSize, EntryID);

    Entries.push_back(*Entry);
  }

  return Error::success();
}

Expected<std::unique_ptr<OffloadBundleFatBin>>
OffloadBundleFatBin::create(MemoryBufferRef Buf, uint64_t SectionOffset,
                            StringRef FileName, bool Decompress) {
  if (Buf.getBufferSize() < 24)
    return errorCodeToError(object_error::parse_failed);

  // Check for magic bytes.
  if ((identify_magic(Buf.getBuffer()) != file_magic::offload_bundle) &&
      (identify_magic(Buf.getBuffer()) !=
       file_magic::offload_bundle_compressed))
    return errorCodeToError(object_error::parse_failed);

  std::unique_ptr<OffloadBundleFatBin> TheBundle(
      new OffloadBundleFatBin(Buf, FileName));

  // Read the Bundle Entries.
  Error Err =
      TheBundle->readEntries(Buf.getBuffer(), Decompress ? 0 : SectionOffset);
  if (Err)
    return Err;

  return std::move(TheBundle);
}

Error OffloadBundleFatBin::extractBundle(const ObjectFile &Source) {
  // This will extract all entries in the Bundle.
  for (OffloadBundleEntry &Entry : Entries) {

    if (Entry.Size == 0)
      continue;

    // create output file name. Which should be
    // <fileName>-offset<Offset>-size<Size>.co"
    std::string Str = getFileName().str() + "-offset" + itostr(Entry.Offset) +
                      "-size" + itostr(Entry.Size) + ".co";
    if (Error Err = object::extractCodeObject(Source, Entry.Offset, Entry.Size,
                                              StringRef(Str)))
      return Err;
  }

  return Error::success();
}

Error object::extractOffloadBundleFatBinary(
    const ObjectFile &Obj, SmallVectorImpl<OffloadBundleFatBin> &Bundles) {
  assert((Obj.isELF() || Obj.isCOFF()) && "Invalid file type");

  // Iterate through Sections until we find an offload_bundle section.
  for (SectionRef Sec : Obj.sections()) {
    Expected<StringRef> Buffer = Sec.getContents();
    if (!Buffer)
      return Buffer.takeError();

    // If it does not start with the reserved suffix, just skip this section.
    if ((llvm::identify_magic(*Buffer) == file_magic::offload_bundle) ||
        (llvm::identify_magic(*Buffer) ==
         file_magic::offload_bundle_compressed)) {

      uint64_t SectionOffset = 0;
      if (Obj.isELF()) {
        SectionOffset = ELFSectionRef(Sec).getOffset();
      } else if (Obj.isCOFF()) // TODO: add COFF Support.
        return createStringError(object_error::parse_failed,
                                 "COFF object files not supported");

      MemoryBufferRef Contents(*Buffer, Obj.getFileName());
      if (Error Err = extractOffloadBundle(Contents, SectionOffset,
                                           Obj.getFileName(), Bundles))
        return Err;
    }
  }
  return Error::success();
}

Error object::extractCodeObject(const ObjectFile &Source, int64_t Offset,
                                int64_t Size, StringRef OutputFileName) {
  Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(OutputFileName, Size);

  if (!BufferOrErr)
    return BufferOrErr.takeError();

  Expected<MemoryBufferRef> InputBuffOrErr = Source.getMemoryBufferRef();
  if (Error Err = InputBuffOrErr.takeError())
    return Err;

  std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
  std::copy(InputBuffOrErr->getBufferStart() + Offset,
            InputBuffOrErr->getBufferStart() + Offset + Size,
            Buf->getBufferStart());
  if (Error E = Buf->commit())
    return E;

  return Error::success();
}

Error object::extractCodeObject(const MemoryBufferRef Buffer, int64_t Offset,
                                int64_t Size, StringRef OutputFileName) {
  Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(OutputFileName, Size);
  if (!BufferOrErr)
    return BufferOrErr.takeError();

  std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
  std::copy(Buffer.getBufferStart() + Offset,
            Buffer.getBufferStart() + Offset + Size, Buf->getBufferStart());

  return Buf->commit();
}

// given a file name, offset, and size, extract data into a code object file,
// into file "<SourceFile>-offset<Offset>-size<Size>.co".
Error object::extractOffloadBundleByURI(StringRef URIstr) {
  // create a URI object
  Expected<std::unique_ptr<OffloadBundleURI>> UriOrErr(
      OffloadBundleURI::createOffloadBundleURI(URIstr, FILE_URI));
  if (!UriOrErr)
    return UriOrErr.takeError();

  OffloadBundleURI &Uri = **UriOrErr;
  std::string OutputFile = Uri.FileName.str();
  OutputFile +=
      "-offset" + itostr(Uri.Offset) + "-size" + itostr(Uri.Size) + ".co";

  // Create an ObjectFile object from uri.file_uri.
  auto ObjOrErr = ObjectFile::createObjectFile(Uri.FileName);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto Obj = ObjOrErr->getBinary();
  if (Error Err =
          object::extractCodeObject(*Obj, Uri.Offset, Uri.Size, OutputFile))
    return Err;

  return Error::success();
}

// Utility function to format numbers with commas.
static std::string formatWithCommas(unsigned long long Value) {
  std::string Num = std::to_string(Value);
  int InsertPosition = Num.length() - 3;
  while (InsertPosition > 0) {
    Num.insert(InsertPosition, ",");
    InsertPosition -= 3;
  }
  return Num;
}

Expected<std::unique_ptr<MemoryBuffer>>
CompressedOffloadBundle::compress(compression::Params P,
                                  const MemoryBuffer &Input, uint16_t Version,
                                  raw_ostream *VerboseStream) {
  if (!compression::zstd::isAvailable() && !compression::zlib::isAvailable())
    return createStringError("compression not supported.");
  Timer HashTimer("Hash Calculation Timer", "Hash calculation time",
                  OffloadBundlerTimerGroup);
  if (VerboseStream)
    HashTimer.startTimer();
  MD5 Hash;
  MD5::MD5Result Result;
  Hash.update(Input.getBuffer());
  Hash.final(Result);
  uint64_t TruncatedHash = Result.low();
  if (VerboseStream)
    HashTimer.stopTimer();

  SmallVector<uint8_t, 0> CompressedBuffer;
  auto BufferUint8 = ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(Input.getBuffer().data()),
      Input.getBuffer().size());
  Timer CompressTimer("Compression Timer", "Compression time",
                      OffloadBundlerTimerGroup);
  if (VerboseStream)
    CompressTimer.startTimer();
  compression::compress(P, BufferUint8, CompressedBuffer);
  if (VerboseStream)
    CompressTimer.stopTimer();

  uint16_t CompressionMethod = static_cast<uint16_t>(P.format);

  // Store sizes in 64-bit variables first.
  uint64_t UncompressedSize64 = Input.getBuffer().size();
  uint64_t TotalFileSize64;

  // Calculate total file size based on version.
  if (Version == 2) {
    // For V2, ensure the sizes don't exceed 32-bit limit.
    if (UncompressedSize64 > std::numeric_limits<uint32_t>::max())
      return createStringError("uncompressed size (%llu) exceeds version 2 "
                               "unsigned 32-bit integer limit",
                               UncompressedSize64);
    TotalFileSize64 = MagicNumber.size() + sizeof(uint32_t) + sizeof(Version) +
                      sizeof(CompressionMethod) + sizeof(uint32_t) +
                      sizeof(TruncatedHash) + CompressedBuffer.size();
    if (TotalFileSize64 > std::numeric_limits<uint32_t>::max())
      return createStringError("total file size (%llu) exceeds version 2 "
                               "unsigned 32-bit integer limit",
                               TotalFileSize64);

  } else { // Version 3.
    TotalFileSize64 = MagicNumber.size() + sizeof(uint64_t) + sizeof(Version) +
                      sizeof(CompressionMethod) + sizeof(uint64_t) +
                      sizeof(TruncatedHash) + CompressedBuffer.size();
  }

  SmallVector<char, 0> FinalBuffer;
  raw_svector_ostream OS(FinalBuffer);
  OS << MagicNumber;
  OS.write(reinterpret_cast<const char *>(&Version), sizeof(Version));
  OS.write(reinterpret_cast<const char *>(&CompressionMethod),
           sizeof(CompressionMethod));

  // Write size fields according to version.
  if (Version == 2) {
    uint32_t TotalFileSize32 = static_cast<uint32_t>(TotalFileSize64);
    uint32_t UncompressedSize32 = static_cast<uint32_t>(UncompressedSize64);
    OS.write(reinterpret_cast<const char *>(&TotalFileSize32),
             sizeof(TotalFileSize32));
    OS.write(reinterpret_cast<const char *>(&UncompressedSize32),
             sizeof(UncompressedSize32));
  } else { // Version 3.
    OS.write(reinterpret_cast<const char *>(&TotalFileSize64),
             sizeof(TotalFileSize64));
    OS.write(reinterpret_cast<const char *>(&UncompressedSize64),
             sizeof(UncompressedSize64));
  }

  OS.write(reinterpret_cast<const char *>(&TruncatedHash),
           sizeof(TruncatedHash));
  OS.write(reinterpret_cast<const char *>(CompressedBuffer.data()),
           CompressedBuffer.size());

  if (VerboseStream) {
    auto MethodUsed = P.format == compression::Format::Zstd ? "zstd" : "zlib";
    double CompressionRate =
        static_cast<double>(UncompressedSize64) / CompressedBuffer.size();
    double CompressionTimeSeconds = CompressTimer.getTotalTime().getWallTime();
    double CompressionSpeedMBs =
        (UncompressedSize64 / (1024.0 * 1024.0)) / CompressionTimeSeconds;
    *VerboseStream << "Compressed bundle format version: " << Version << "\n"
                   << "Total file size (including headers): "
                   << formatWithCommas(TotalFileSize64) << " bytes\n"
                   << "Compression method used: " << MethodUsed << "\n"
                   << "Compression level: " << P.level << "\n"
                   << "Binary size before compression: "
                   << formatWithCommas(UncompressedSize64) << " bytes\n"
                   << "Binary size after compression: "
                   << formatWithCommas(CompressedBuffer.size()) << " bytes\n"
                   << "Compression rate: " << format("%.2lf", CompressionRate)
                   << "\n"
                   << "Compression ratio: "
                   << format("%.2lf%%", 100.0 / CompressionRate) << "\n"
                   << "Compression speed: "
                   << format("%.2lf MB/s", CompressionSpeedMBs) << "\n"
                   << "Truncated MD5 hash: " << format_hex(TruncatedHash, 16)
                   << "\n";
  }

  return MemoryBuffer::getMemBufferCopy(
      StringRef(FinalBuffer.data(), FinalBuffer.size()));
}

// Use packed structs to avoid padding, such that the structs map the serialized
// format.
LLVM_PACKED_START
union RawCompressedBundleHeader {
  struct CommonFields {
    uint32_t Magic;
    uint16_t Version;
    uint16_t Method;
  };

  struct V1Header {
    CommonFields Common;
    uint32_t UncompressedFileSize;
    uint64_t Hash;
  };

  struct V2Header {
    CommonFields Common;
    uint32_t FileSize;
    uint32_t UncompressedFileSize;
    uint64_t Hash;
  };

  struct V3Header {
    CommonFields Common;
    uint64_t FileSize;
    uint64_t UncompressedFileSize;
    uint64_t Hash;
  };

  CommonFields Common;
  V1Header V1;
  V2Header V2;
  V3Header V3;
};
LLVM_PACKED_END

// Helper method to get header size based on version.
static size_t getHeaderSize(uint16_t Version) {
  switch (Version) {
  case 1:
    return sizeof(RawCompressedBundleHeader::V1Header);
  case 2:
    return sizeof(RawCompressedBundleHeader::V2Header);
  case 3:
    return sizeof(RawCompressedBundleHeader::V3Header);
  default:
    llvm_unreachable("Unsupported version");
  }
}

Expected<CompressedOffloadBundle::CompressedBundleHeader>
CompressedOffloadBundle::CompressedBundleHeader::tryParse(StringRef Blob) {
  assert(Blob.size() >= sizeof(RawCompressedBundleHeader::CommonFields));
  assert(identify_magic(Blob) == file_magic::offload_bundle_compressed);

  RawCompressedBundleHeader Header;
  std::memcpy(&Header, Blob.data(), std::min(Blob.size(), sizeof(Header)));

  CompressedBundleHeader Normalized;
  Normalized.Version = Header.Common.Version;

  size_t RequiredSize = getHeaderSize(Normalized.Version);

  if (Blob.size() < RequiredSize)
    return createStringError("compressed bundle header size too small");

  switch (Normalized.Version) {
  case 1:
    Normalized.UncompressedFileSize = Header.V1.UncompressedFileSize;
    Normalized.Hash = Header.V1.Hash;
    break;
  case 2:
    Normalized.FileSize = Header.V2.FileSize;
    Normalized.UncompressedFileSize = Header.V2.UncompressedFileSize;
    Normalized.Hash = Header.V2.Hash;
    break;
  case 3:
    Normalized.FileSize = Header.V3.FileSize;
    Normalized.UncompressedFileSize = Header.V3.UncompressedFileSize;
    Normalized.Hash = Header.V3.Hash;
    break;
  default:
    return createStringError("unknown compressed bundle version");
  }

  // Determine compression format.
  switch (Header.Common.Method) {
  case static_cast<uint16_t>(compression::Format::Zlib):
  case static_cast<uint16_t>(compression::Format::Zstd):
    Normalized.CompressionFormat =
        static_cast<compression::Format>(Header.Common.Method);
    break;
  default:
    return createStringError("unknown compressing method");
  }

  return Normalized;
}

Expected<std::unique_ptr<MemoryBuffer>>
CompressedOffloadBundle::decompress(const MemoryBuffer &Input,
                                    raw_ostream *VerboseStream) {
  StringRef Blob = Input.getBuffer();

  // Check minimum header size (using V1 as it's the smallest).
  if (Blob.size() < sizeof(RawCompressedBundleHeader::CommonFields))
    return MemoryBuffer::getMemBufferCopy(Blob);

  if (identify_magic(Blob) != file_magic::offload_bundle_compressed) {
    if (VerboseStream)
      *VerboseStream << "Uncompressed bundle\n";
    return MemoryBuffer::getMemBufferCopy(Blob);
  }

  Expected<CompressedBundleHeader> HeaderOrErr =
      CompressedBundleHeader::tryParse(Blob);
  if (!HeaderOrErr)
    return HeaderOrErr.takeError();

  const CompressedBundleHeader &Normalized = *HeaderOrErr;
  unsigned ThisVersion = Normalized.Version;
  size_t HeaderSize = getHeaderSize(ThisVersion);

  compression::Format CompressionFormat = Normalized.CompressionFormat;

  size_t TotalFileSize = Normalized.FileSize.value_or(0);
  size_t UncompressedSize = Normalized.UncompressedFileSize;
  auto StoredHash = Normalized.Hash;

  Timer DecompressTimer("Decompression Timer", "Decompression time",
                        OffloadBundlerTimerGroup);
  if (VerboseStream)
    DecompressTimer.startTimer();

  SmallVector<uint8_t, 0> DecompressedData;
  StringRef CompressedData =
      Blob.substr(HeaderSize, TotalFileSize - HeaderSize);

  if (Error DecompressionError = compression::decompress(
          CompressionFormat, arrayRefFromStringRef(CompressedData),
          DecompressedData, UncompressedSize))
    return createStringError("could not decompress embedded file contents: " +
                             toString(std::move(DecompressionError)));

  if (VerboseStream) {
    DecompressTimer.stopTimer();

    double DecompressionTimeSeconds =
        DecompressTimer.getTotalTime().getWallTime();

    // Recalculate MD5 hash for integrity check.
    Timer HashRecalcTimer("Hash Recalculation Timer", "Hash recalculation time",
                          OffloadBundlerTimerGroup);
    HashRecalcTimer.startTimer();
    MD5 Hash;
    MD5::MD5Result Result;
    Hash.update(ArrayRef<uint8_t>(DecompressedData));
    Hash.final(Result);
    uint64_t RecalculatedHash = Result.low();
    HashRecalcTimer.stopTimer();
    bool HashMatch = (StoredHash == RecalculatedHash);

    double CompressionRate =
        static_cast<double>(UncompressedSize) / CompressedData.size();
    double DecompressionSpeedMBs =
        (UncompressedSize / (1024.0 * 1024.0)) / DecompressionTimeSeconds;

    *VerboseStream << "Compressed bundle format version: " << ThisVersion
                   << "\n";
    if (ThisVersion >= 2)
      *VerboseStream << "Total file size (from header): "
                     << formatWithCommas(TotalFileSize) << " bytes\n";
    *VerboseStream
        << "Decompression method: "
        << (CompressionFormat == compression::Format::Zlib ? "zlib" : "zstd")
        << "\n"
        << "Size before decompression: "
        << formatWithCommas(CompressedData.size()) << " bytes\n"
        << "Size after decompression: " << formatWithCommas(UncompressedSize)
        << " bytes\n"
        << "Compression rate: " << format("%.2lf", CompressionRate) << "\n"
        << "Compression ratio: " << format("%.2lf%%", 100.0 / CompressionRate)
        << "\n"
        << "Decompression speed: "
        << format("%.2lf MB/s", DecompressionSpeedMBs) << "\n"
        << "Stored hash: " << format_hex(StoredHash, 16) << "\n"
        << "Recalculated hash: " << format_hex(RecalculatedHash, 16) << "\n"
        << "Hashes match: " << (HashMatch ? "Yes" : "No") << "\n";
  }

  return MemoryBuffer::getMemBufferCopy(toStringRef(DecompressedData));
}
