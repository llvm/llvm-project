//===- OffloadBundle.cpp - Utilities for offload bundles---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//

#include "llvm/Object/OffloadBundle.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/Constants.h"
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
#include "llvm/Support/Alignment.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"

using namespace llvm;
using namespace llvm::object;

static llvm::TimerGroup
    OffloadBundlerTimerGroup("Offload Bundler Timer Group",
                             "Timer group for offload bundler");

// Extract an Offload bundle (usually a Offload Bundle) from a fat_bin
// section
Error extractOffloadBundle(MemoryBufferRef Contents, uint64_t SectionOffset,
                           StringRef FileName,
                           SmallVectorImpl<OffloadBundleFatBin> &Bundles) {

  uint64_t Offset = 0;
  int64_t NextbundleStart = 0;

  // There could be multiple offloading bundles stored at this section.
  while (NextbundleStart >= 0) {

    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator=*/false);

    // Create the FatBinBindle object. This will also create the Bundle Entry
    // list info.
    auto FatBundleOrErr =
        OffloadBundleFatBin::create(*Buffer, SectionOffset + Offset, FileName);
    if (!FatBundleOrErr)
      return FatBundleOrErr.takeError();

    // Add current Bundle to list.
    Bundles.emplace_back(std::move(**FatBundleOrErr));

    // Find the next bundle by searching for the magic string
    StringRef Str = Buffer->getBuffer();
    NextbundleStart =
        (int64_t)Str.find(StringRef("__CLANG_OFFLOAD_BUNDLE__"), 24);

    if (NextbundleStart >= 0)
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

  // For each Bundle Entry (code object)
  for (uint64_t I = 0; I < NumOfEntries; I++) {
    uint64_t EntrySize;
    uint64_t EntryOffset;
    uint64_t EntryIDSize;
    StringRef EntryID;

    if (auto EC = Reader.readInteger(EntryOffset))
      return errorCodeToError(object_error::parse_failed);

    if (auto EC = Reader.readInteger(EntrySize))
      return errorCodeToError(object_error::parse_failed);

    if (auto EC = Reader.readInteger(EntryIDSize))
      return errorCodeToError(object_error::parse_failed);

    if (auto EC = Reader.readFixedString(EntryID, EntryIDSize))
      return errorCodeToError(object_error::parse_failed);

    auto Entry = std::make_unique<OffloadBundleEntry>(
        EntryOffset + SectionOffset, EntrySize, EntryIDSize, EntryID);

    Entries.push_back(*Entry);
  }

  return Error::success();
}

Expected<std::unique_ptr<OffloadBundleFatBin>>
OffloadBundleFatBin::create(MemoryBufferRef Buf, uint64_t SectionOffset,
                            StringRef FileName) {
  if (Buf.getBufferSize() < 24)
    return errorCodeToError(object_error::parse_failed);

  // Check for magic bytes.
  if (identify_magic(Buf.getBuffer()) != file_magic::offload_bundle)
    return errorCodeToError(object_error::parse_failed);

  OffloadBundleFatBin *TheBundle = new OffloadBundleFatBin(Buf, FileName);

  // Read the Bundle Entries
  Error Err = TheBundle->readEntries(Buf.getBuffer(), SectionOffset);
  if (Err)
    return errorCodeToError(object_error::parse_failed);

  return std::unique_ptr<OffloadBundleFatBin>(TheBundle);
}

Error OffloadBundleFatBin::extractBundle(const ObjectFile &Source) {
  // This will extract all entries in the Bundle
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
    if ((llvm::identify_magic(*Buffer) == llvm::file_magic::offload_bundle) ||
        (llvm::identify_magic(*Buffer) ==
         llvm::file_magic::offload_bundle_compressed)) {

      uint64_t SectionOffset = 0;
      if (Obj.isELF()) {
        SectionOffset = ELFSectionRef(Sec).getOffset();
      } else if (Obj.isCOFF()) // TODO: add COFF Support
        return createStringError(object_error::parse_failed,
                                 "COFF object files not supported.\n");

      MemoryBufferRef Contents(*Buffer, Obj.getFileName());

      if (llvm::identify_magic(*Buffer) ==
          llvm::file_magic::offload_bundle_compressed) {
        // Decompress the input if necessary.
        Expected<std::unique_ptr<MemoryBuffer>> DecompressedBufferOrErr =
            CompressedOffloadBundle::decompress(Contents, false);

        if (!DecompressedBufferOrErr)
          return createStringError(
              inconvertibleErrorCode(),
              "Failed to decompress input: " +
                  llvm::toString(DecompressedBufferOrErr.takeError()));

        MemoryBuffer &DecompressedInput = **DecompressedBufferOrErr;
        if (Error Err = extractOffloadBundle(DecompressedInput, SectionOffset,
                                             Obj.getFileName(), Bundles))
          return Err;
      } else {
        if (Error Err = extractOffloadBundle(Contents, SectionOffset,
                                             Obj.getFileName(), Bundles))
          return Err;
      }
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

// given a file name, offset, and size, extract data into a code object file,
// into file <SourceFile>-offset<Offset>-size<Size>.co
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

  // Create an ObjectFile object from uri.file_uri
  auto ObjOrErr = ObjectFile::createObjectFile(Uri.FileName);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto Obj = ObjOrErr->getBinary();
  if (Error Err =
          object::extractCodeObject(*Obj, Uri.Offset, Uri.Size, OutputFile))
    return Err;

  return Error::success();
}

// Utility function to format numbers with commas
static std::string formatWithCommas(unsigned long long Value) {
  std::string Num = std::to_string(Value);
  int InsertPosition = Num.length() - 3;
  while (InsertPosition > 0) {
    Num.insert(InsertPosition, ",");
    InsertPosition -= 3;
  }
  return Num;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
CompressedOffloadBundle::decompress(llvm::MemoryBufferRef &Input,
                                    bool Verbose) {
  StringRef Blob = Input.getBuffer();

  if (Blob.size() < V1HeaderSize)
    return llvm::MemoryBuffer::getMemBufferCopy(Blob);

  if (llvm::identify_magic(Blob) !=
      llvm::file_magic::offload_bundle_compressed) {
    if (Verbose)
      llvm::errs() << "Uncompressed bundle.\n";
    return llvm::MemoryBuffer::getMemBufferCopy(Blob);
  }

  size_t CurrentOffset = MagicSize;

  uint16_t ThisVersion;
  memcpy(&ThisVersion, Blob.data() + CurrentOffset, sizeof(uint16_t));
  CurrentOffset += VersionFieldSize;

  uint16_t CompressionMethod;
  memcpy(&CompressionMethod, Blob.data() + CurrentOffset, sizeof(uint16_t));
  CurrentOffset += MethodFieldSize;

  uint32_t TotalFileSize;
  if (ThisVersion >= 2) {
    if (Blob.size() < V2HeaderSize)
      return createStringError(inconvertibleErrorCode(),
                               "Compressed bundle header size too small");
    memcpy(&TotalFileSize, Blob.data() + CurrentOffset, sizeof(uint32_t));
    CurrentOffset += FileSizeFieldSize;
  }

  uint32_t UncompressedSize;
  memcpy(&UncompressedSize, Blob.data() + CurrentOffset, sizeof(uint32_t));
  CurrentOffset += UncompressedSizeFieldSize;

  uint64_t StoredHash;
  memcpy(&StoredHash, Blob.data() + CurrentOffset, sizeof(uint64_t));
  CurrentOffset += HashFieldSize;

  llvm::compression::Format CompressionFormat;
  if (CompressionMethod ==
      static_cast<uint16_t>(llvm::compression::Format::Zlib))
    CompressionFormat = llvm::compression::Format::Zlib;
  else if (CompressionMethod ==
           static_cast<uint16_t>(llvm::compression::Format::Zstd))
    CompressionFormat = llvm::compression::Format::Zstd;
  else
    return createStringError(inconvertibleErrorCode(),
                             "Unknown compressing method");

  llvm::Timer DecompressTimer("Decompression Timer", "Decompression time",
                              OffloadBundlerTimerGroup);
  if (Verbose)
    DecompressTimer.startTimer();

  SmallVector<uint8_t, 0> DecompressedData;
  StringRef CompressedData = Blob.substr(CurrentOffset);
  if (llvm::Error DecompressionError = llvm::compression::decompress(
          CompressionFormat, llvm::arrayRefFromStringRef(CompressedData),
          DecompressedData, UncompressedSize))
    return createStringError(inconvertibleErrorCode(),
                             "Could not decompress embedded file contents: " +
                                 llvm::toString(std::move(DecompressionError)));

  if (Verbose) {
    DecompressTimer.stopTimer();

    double DecompressionTimeSeconds =
        DecompressTimer.getTotalTime().getWallTime();

    // Recalculate MD5 hash for integrity check.
    llvm::Timer HashRecalcTimer("Hash Recalculation Timer",
                                "Hash recalculation time",
                                OffloadBundlerTimerGroup);
    HashRecalcTimer.startTimer();
    llvm::MD5 Hash;
    llvm::MD5::MD5Result Result;
    Hash.update(llvm::ArrayRef<uint8_t>(DecompressedData.data(),
                                        DecompressedData.size()));
    Hash.final(Result);
    uint64_t RecalculatedHash = Result.low();
    HashRecalcTimer.stopTimer();
    bool HashMatch = (StoredHash == RecalculatedHash);

    double CompressionRate =
        static_cast<double>(UncompressedSize) / CompressedData.size();
    double DecompressionSpeedMBs =
        (UncompressedSize / (1024.0 * 1024.0)) / DecompressionTimeSeconds;

    llvm::errs() << "Compressed bundle format version: " << ThisVersion << "\n";
    if (ThisVersion >= 2)
      llvm::errs() << "Total file size (from header): "
                   << formatWithCommas(TotalFileSize) << " bytes\n";
    llvm::errs() << "Decompression method: "
                 << (CompressionFormat == llvm::compression::Format::Zlib
                         ? "zlib"
                         : "zstd")
                 << "\n"
                 << "Size before decompression: "
                 << formatWithCommas(CompressedData.size()) << " bytes\n"
                 << "Size after decompression: "
                 << formatWithCommas(UncompressedSize) << " bytes\n"
                 << "Compression rate: "
                 << llvm::format("%.2lf", CompressionRate) << "\n"
                 << "Compression ratio: "
                 << llvm::format("%.2lf%%", 100.0 / CompressionRate) << "\n"
                 << "Decompression speed: "
                 << llvm::format("%.2lf MB/s", DecompressionSpeedMBs) << "\n"
                 << "Stored hash: " << llvm::format_hex(StoredHash, 16) << "\n"
                 << "Recalculated hash: "
                 << llvm::format_hex(RecalculatedHash, 16) << "\n"
                 << "Hashes match: " << (HashMatch ? "Yes" : "No") << "\n";
  }

  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::toStringRef(DecompressedData));
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
CompressedOffloadBundle::compress(llvm::compression::Params P,
                                  const llvm::MemoryBuffer &Input,
                                  bool Verbose) {
  if (!llvm::compression::zstd::isAvailable() &&
      !llvm::compression::zlib::isAvailable())
    return createStringError(llvm::inconvertibleErrorCode(),
                             "Compression not supported");

  llvm::Timer HashTimer("Hash Calculation Timer", "Hash calculation time",
                        OffloadBundlerTimerGroup);
  if (Verbose)
    HashTimer.startTimer();
  llvm::MD5 Hash;
  llvm::MD5::MD5Result Result;
  Hash.update(Input.getBuffer());
  Hash.final(Result);
  uint64_t TruncatedHash = Result.low();
  if (Verbose)
    HashTimer.stopTimer();

  SmallVector<uint8_t, 0> CompressedBuffer;
  auto BufferUint8 = llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(Input.getBuffer().data()),
      Input.getBuffer().size());

  llvm::Timer CompressTimer("Compression Timer", "Compression time",
                            OffloadBundlerTimerGroup);
  if (Verbose)
    CompressTimer.startTimer();
  llvm::compression::compress(P, BufferUint8, CompressedBuffer);
  if (Verbose)
    CompressTimer.stopTimer();

  uint16_t CompressionMethod = static_cast<uint16_t>(P.format);
  uint32_t UncompressedSize = Input.getBuffer().size();
  uint32_t TotalFileSize = MagicNumber.size() + sizeof(TotalFileSize) +
                           sizeof(Version) + sizeof(CompressionMethod) +
                           sizeof(UncompressedSize) + sizeof(TruncatedHash) +
                           CompressedBuffer.size();

  SmallVector<char, 0> FinalBuffer;
  llvm::raw_svector_ostream OS(FinalBuffer);
  OS << MagicNumber;
  OS.write(reinterpret_cast<const char *>(&Version), sizeof(Version));
  OS.write(reinterpret_cast<const char *>(&CompressionMethod),
           sizeof(CompressionMethod));
  OS.write(reinterpret_cast<const char *>(&TotalFileSize),
           sizeof(TotalFileSize));
  OS.write(reinterpret_cast<const char *>(&UncompressedSize),
           sizeof(UncompressedSize));
  OS.write(reinterpret_cast<const char *>(&TruncatedHash),
           sizeof(TruncatedHash));
  OS.write(reinterpret_cast<const char *>(CompressedBuffer.data()),
           CompressedBuffer.size());

  if (Verbose) {
    auto MethodUsed =
        P.format == llvm::compression::Format::Zstd ? "zstd" : "zlib";
    double CompressionRate =
        static_cast<double>(UncompressedSize) / CompressedBuffer.size();
    double CompressionTimeSeconds = CompressTimer.getTotalTime().getWallTime();
    double CompressionSpeedMBs =
        (UncompressedSize / (1024.0 * 1024.0)) / CompressionTimeSeconds;

    llvm::errs() << "Compressed bundle format version: " << Version << "\n"
                 << "Total file size (including headers): "
                 << formatWithCommas(TotalFileSize) << " bytes\n"
                 << "Compression method used: " << MethodUsed << "\n"
                 << "Compression level: " << P.level << "\n"
                 << "Binary size before compression: "
                 << formatWithCommas(UncompressedSize) << " bytes\n"
                 << "Binary size after compression: "
                 << formatWithCommas(CompressedBuffer.size()) << " bytes\n"
                 << "Compression rate: "
                 << llvm::format("%.2lf", CompressionRate) << "\n"
                 << "Compression ratio: "
                 << llvm::format("%.2lf%%", 100.0 / CompressionRate) << "\n"
                 << "Compression speed: "
                 << llvm::format("%.2lf MB/s", CompressionSpeedMBs) << "\n"
                 << "Truncated MD5 hash: "
                 << llvm::format_hex(TruncatedHash, 16) << "\n";
  }
  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(FinalBuffer.data(), FinalBuffer.size()));
}
