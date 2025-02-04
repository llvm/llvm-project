//===- Offloading.cpp - Utilities for handling offloading code  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/OffloadBinary.h"

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

namespace {

static llvm::TimerGroup
    OffloadBundlerTimerGroup("Offload Bundler Timer Group",
                             "Timer group for offload bundler");

/// Attempts to extract all the embedded device images contained inside the
/// buffer \p Contents. The buffer is expected to contain a valid offloading
/// binary format.
Error extractOffloadFiles(MemoryBufferRef Contents,
                          SmallVectorImpl<OffloadFile> &Binaries) {
  uint64_t Offset = 0;
  // There could be multiple offloading binaries stored at this section.
  while (Offset < Contents.getBuffer().size()) {
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator*/ false);
    if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                       Buffer->getBufferStart()))
      Buffer = MemoryBuffer::getMemBufferCopy(Buffer->getBuffer(),
                                              Buffer->getBufferIdentifier());
    auto BinaryOrErr = OffloadBinary::create(*Buffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();
    OffloadBinary &Binary = **BinaryOrErr;

    // Create a new owned binary with a copy of the original memory.
    std::unique_ptr<MemoryBuffer> BufferCopy = MemoryBuffer::getMemBufferCopy(
        Binary.getData().take_front(Binary.getSize()),
        Contents.getBufferIdentifier());
    auto NewBinaryOrErr = OffloadBinary::create(*BufferCopy);
    if (!NewBinaryOrErr)
      return NewBinaryOrErr.takeError();
    Binaries.emplace_back(std::move(*NewBinaryOrErr), std::move(BufferCopy));

    Offset += Binary.getSize();
  }

  return Error::success();
}

// Extract offloading binaries from an Object file \p Obj.
Error extractFromObject(const ObjectFile &Obj,
                        SmallVectorImpl<OffloadFile> &Binaries) {
  assert((Obj.isELF() || Obj.isCOFF()) && "Invalid file type");

  for (SectionRef Sec : Obj.sections()) {
    // ELF files contain a section with the LLVM_OFFLOADING type.
    if (Obj.isELF() &&
        static_cast<ELFSectionRef>(Sec).getType() != ELF::SHT_LLVM_OFFLOADING)
      continue;

    // COFF has no section types so we rely on the name of the section.
    if (Obj.isCOFF()) {
      Expected<StringRef> NameOrErr = Sec.getName();
      if (!NameOrErr)
        return NameOrErr.takeError();

      if (!NameOrErr->starts_with(".llvm.offloading"))
        continue;
    }

    Expected<StringRef> Buffer = Sec.getContents();
    if (!Buffer)
      return Buffer.takeError();

    MemoryBufferRef Contents(*Buffer, Obj.getFileName());
    if (Error Err = extractOffloadFiles(Contents, Binaries))
      return Err;
  }

  return Error::success();
}

// Extract an Offload bundle (usually a Offload Bundle) from a fat_bin
// section
Error extractOffloadBundle(MemoryBufferRef Contents, uint64_t SectionOffset,
                           StringRef fileName,
                           SmallVectorImpl<OffloadBundleFatBin> &Bundles) {

  uint64_t Offset = 0;
  int64_t nextbundleStart = 0;

  // There could be multiple offloading bundles stored at this section.
  while (nextbundleStart >= 0) {

    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator*/ false);

    // Create the FatBinBindle object. This will also create the Bundle Entry
    // list info.
    auto FatBundleOrErr =
        OffloadBundleFatBin::create(*Buffer, SectionOffset + Offset, fileName);
    if (!FatBundleOrErr)
      return FatBundleOrErr.takeError();
    OffloadBundleFatBin &Bundle = **FatBundleOrErr;

    // add current Bundle to list.
    Bundles.emplace_back(std::move(**FatBundleOrErr));

    // find the next bundle by searching for the magic string
    StringRef str = Buffer->getBuffer();
    nextbundleStart =
        (int64_t)str.find(StringRef("__CLANG_OFFLOAD_BUNDLE__"), 24);

    if (nextbundleStart >= 0)
      Offset += nextbundleStart;
    else {
      return Error::success();
    }
  } // end of while loop

  return Error::success();
}

Error extractFromBitcode(MemoryBufferRef Buffer,
                         SmallVectorImpl<OffloadFile> &Binaries) {
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = getLazyIRModule(
      MemoryBuffer::getMemBuffer(Buffer, /*RequiresNullTerminator=*/false), Err,
      Context);
  if (!M)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create module");

  // Extract offloading data from globals referenced by the
  // `llvm.embedded.object` metadata with the `.llvm.offloading` section.
  auto *MD = M->getNamedMetadata("llvm.embedded.objects");
  if (!MD)
    return Error::success();

  for (const MDNode *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;

    MDString *SectionID = dyn_cast<MDString>(Op->getOperand(1));
    if (!SectionID || SectionID->getString() != ".llvm.offloading")
      continue;

    GlobalVariable *GV =
        mdconst::dyn_extract_or_null<GlobalVariable>(Op->getOperand(0));
    if (!GV)
      continue;

    auto *CDS = dyn_cast<ConstantDataSequential>(GV->getInitializer());
    if (!CDS)
      continue;

    MemoryBufferRef Contents(CDS->getAsString(), M->getName());
    if (Error Err = extractOffloadFiles(Contents, Binaries))
      return Err;
  }

  return Error::success();
}

Error extractFromArchive(const Archive &Library,
                         SmallVectorImpl<OffloadFile> &Binaries) {
  // Try to extract device code from each file stored in the static archive.
  Error Err = Error::success();
  for (auto Child : Library.children(Err)) {
    auto ChildBufferOrErr = Child.getMemoryBufferRef();
    if (!ChildBufferOrErr)
      return ChildBufferOrErr.takeError();
    std::unique_ptr<MemoryBuffer> ChildBuffer =
        MemoryBuffer::getMemBuffer(*ChildBufferOrErr, false);

    // Check if the buffer has the required alignment.
    if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                       ChildBuffer->getBufferStart()))
      ChildBuffer = MemoryBuffer::getMemBufferCopy(
          ChildBufferOrErr->getBuffer(),
          ChildBufferOrErr->getBufferIdentifier());

    if (Error Err = extractOffloadBinaries(*ChildBuffer, Binaries))
      return Err;
  }

  if (Err)
    return Err;
  return Error::success();
}

} // namespace

Error OffloadBundleFatBin::ReadEntries(StringRef Buffer,
                                       uint64_t SectionOffset) {
  uint64_t BundleNumber = 0;
  uint64_t NumOfEntries = 0;

  // get Reader
  BinaryStreamReader Reader(Buffer, llvm::endianness::little);

  // Read the Magic String first.
  StringRef Magic;
  if (auto EC = Reader.readFixedString(Magic, 24)) {
    return errorCodeToError(object_error::parse_failed);
  }

  // read the number of Code Objects (Entries) in the current Bundle.
  if (auto EC = Reader.readInteger(NumOfEntries)) {
    printf("OffloadBundleFatBin::ReadEntries .... failed to read number of "
           "Entries\n");
    return errorCodeToError(object_error::parse_failed);
  }
  NumberOfEntries = NumOfEntries;

  // For each Bundle Entry (code object)
  for (uint64_t I = 0; I < NumOfEntries; I++) {
    uint64_t EntrySize;
    uint64_t EntryOffset;
    uint64_t EntryIDSize;
    StringRef EntryID;
    uint64_t absOffset;

    if (auto EC = Reader.readInteger(EntryOffset)) {
      return errorCodeToError(object_error::parse_failed);
    }

    if (auto EC = Reader.readInteger(EntrySize)) {
      return errorCodeToError(object_error::parse_failed);
    }

    if (auto EC = Reader.readInteger(EntryIDSize)) {
      return errorCodeToError(object_error::parse_failed);
    }

    if (auto EC = Reader.readFixedString(EntryID, EntryIDSize)) {
      return errorCodeToError(object_error::parse_failed);
    }

    // create a Bundle Entry object:
    auto entry = new OffloadBundleEntry(EntryOffset + SectionOffset, EntrySize,
                                        EntryIDSize, EntryID);

    Entries.push_back(*entry);
  } // end of for loop

  return Error::success();
}

Expected<std::unique_ptr<OffloadBundleFatBin>>
OffloadBundleFatBin::create(MemoryBufferRef Buf, uint64_t SectionOffset,
                            StringRef fileName) {
  if (Buf.getBufferSize() < 24)
    return errorCodeToError(object_error::parse_failed);

  // Check for magic bytes.
  if (identify_magic(Buf.getBuffer()) != file_magic::offload_bundle)
    return errorCodeToError(object_error::parse_failed);

  OffloadBundleFatBin *TheBundle = new OffloadBundleFatBin(Buf, fileName);

  // Read the Bundle Entries
  Error Err = TheBundle->ReadEntries(Buf.getBuffer(), SectionOffset);
  if (Err)
    return errorCodeToError(object_error::parse_failed);

  return std::unique_ptr<OffloadBundleFatBin>(TheBundle);
}

Error OffloadBundleFatBin::extractBundle(const ObjectFile &Source) {
  // This will extract all entries in the Bundle
  SmallVectorImpl<OffloadBundleEntry>::iterator it = Entries.begin();
  for (int64_t I = 0; I < getNumEntries(); I++) {

    if (it->Size > 0) {
      // create output file name. Which should be
      // <fileName>-offset<Offset>-size<Size>.co"
      std::string str = getFileName().str() + "-offset" + itostr(it->Offset) +
                        "-size" + itostr(it->Size) + ".co";
      if (Error Err = object::extractCodeObject(Source, it->Offset, it->Size,
                                                StringRef(str)))
        return Err;
    }
    ++it;
  }

  return Error::success();
}

Expected<std::unique_ptr<OffloadBinary>>
OffloadBinary::create(MemoryBufferRef Buf) {
  if (Buf.getBufferSize() < sizeof(Header) + sizeof(Entry))
    return errorCodeToError(object_error::parse_failed);

  // Check for 0x10FF1OAD magic bytes.
  if (identify_magic(Buf.getBuffer()) != file_magic::offload_binary)
    return errorCodeToError(object_error::parse_failed);

  // Make sure that the data has sufficient alignment.
  if (!isAddrAligned(Align(getAlignment()), Buf.getBufferStart()))
    return errorCodeToError(object_error::parse_failed);

  const char *Start = Buf.getBufferStart();
  const Header *TheHeader = reinterpret_cast<const Header *>(Start);
  if (TheHeader->Version != OffloadBinary::Version)
    return errorCodeToError(object_error::parse_failed);

  if (TheHeader->Size > Buf.getBufferSize() ||
      TheHeader->Size < sizeof(Entry) || TheHeader->Size < sizeof(Header))
    return errorCodeToError(object_error::unexpected_eof);

  if (TheHeader->EntryOffset > TheHeader->Size - sizeof(Entry) ||
      TheHeader->EntrySize > TheHeader->Size - sizeof(Header))
    return errorCodeToError(object_error::unexpected_eof);

  const Entry *TheEntry =
      reinterpret_cast<const Entry *>(&Start[TheHeader->EntryOffset]);

  if (TheEntry->ImageOffset > Buf.getBufferSize() ||
      TheEntry->StringOffset > Buf.getBufferSize())
    return errorCodeToError(object_error::unexpected_eof);

  return std::unique_ptr<OffloadBinary>(
      new OffloadBinary(Buf, TheHeader, TheEntry));
}

SmallString<0> OffloadBinary::write(const OffloadingImage &OffloadingData) {
  // Create a null-terminated string table with all the used strings.
  StringTableBuilder StrTab(StringTableBuilder::ELF);
  for (auto &KeyAndValue : OffloadingData.StringData) {
    StrTab.add(KeyAndValue.first);
    StrTab.add(KeyAndValue.second);
  }
  StrTab.finalize();

  uint64_t StringEntrySize =
      sizeof(StringEntry) * OffloadingData.StringData.size();

  // Make sure the image we're wrapping around is aligned as well.
  uint64_t BinaryDataSize = alignTo(sizeof(Header) + sizeof(Entry) +
                                        StringEntrySize + StrTab.getSize(),
                                    getAlignment());

  // Create the header and fill in the offsets. The entry will be directly
  // placed after the header in memory. Align the size to the alignment of the
  // header so this can be placed contiguously in a single section.
  Header TheHeader;
  TheHeader.Size = alignTo(
      BinaryDataSize + OffloadingData.Image->getBufferSize(), getAlignment());
  TheHeader.EntryOffset = sizeof(Header);
  TheHeader.EntrySize = sizeof(Entry);

  // Create the entry using the string table offsets. The string table will be
  // placed directly after the entry in memory, and the image after that.
  Entry TheEntry;
  TheEntry.TheImageKind = OffloadingData.TheImageKind;
  TheEntry.TheOffloadKind = OffloadingData.TheOffloadKind;
  TheEntry.Flags = OffloadingData.Flags;
  TheEntry.StringOffset = sizeof(Header) + sizeof(Entry);
  TheEntry.NumStrings = OffloadingData.StringData.size();

  TheEntry.ImageOffset = BinaryDataSize;
  TheEntry.ImageSize = OffloadingData.Image->getBufferSize();

  SmallString<0> Data;
  Data.reserve(TheHeader.Size);
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<char *>(&TheHeader), sizeof(Header));
  OS << StringRef(reinterpret_cast<char *>(&TheEntry), sizeof(Entry));
  for (auto &KeyAndValue : OffloadingData.StringData) {
    uint64_t Offset = sizeof(Header) + sizeof(Entry) + StringEntrySize;
    StringEntry Map{Offset + StrTab.getOffset(KeyAndValue.first),
                    Offset + StrTab.getOffset(KeyAndValue.second)};
    OS << StringRef(reinterpret_cast<char *>(&Map), sizeof(StringEntry));
  }
  StrTab.write(OS);
  // Add padding to required image alignment.
  OS.write_zeros(TheEntry.ImageOffset - OS.tell());
  OS << OffloadingData.Image->getBuffer();

  // Add final padding to required alignment.
  assert(TheHeader.Size >= OS.tell() && "Too much data written?");
  OS.write_zeros(TheHeader.Size - OS.tell());
  assert(TheHeader.Size == OS.tell() && "Size mismatch");

  return Data;
}

Error object::extractOffloadBinaries(MemoryBufferRef Buffer,
                                     SmallVectorImpl<OffloadFile> &Binaries) {
  file_magic Type = identify_magic(Buffer.getBuffer());
  switch (Type) {
  case file_magic::bitcode:
    return extractFromBitcode(Buffer, Binaries);
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Buffer, Type);
    if (!ObjFile)
      return ObjFile.takeError();
    return extractFromObject(*ObjFile->get(), Binaries);
  }
  case file_magic::archive: {
    Expected<std::unique_ptr<llvm::object::Archive>> LibFile =
        object::Archive::create(Buffer);
    if (!LibFile)
      return LibFile.takeError();
    return extractFromArchive(*LibFile->get(), Binaries);
  }
  case file_magic::offload_binary:
    return extractOffloadFiles(Buffer, Binaries);
  default:
    return Error::success();
  }
}

Error object::extractOffloadBundleFatBinary(
    const ObjectFile &Obj, SmallVectorImpl<OffloadBundleFatBin> &Bundles) {
  assert((Obj.isELF() || Obj.isCOFF()) && "Invalid file type");

  // iterate through Sections until we find an offload_bundle section.
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
      } else if (Obj.isCOFF()) {
        if (const COFFObjectFile *COFFObj = dyn_cast<COFFObjectFile>(&Obj)) {
          const coff_section *CoffSection = COFFObj->getCOFFSection(Sec);
        }
      }

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
  object::OffloadBundleURI *uri =
      new object::OffloadBundleURI(URIstr, FILE_URI);

  std::string OutputFile = uri->FileName.str();
  OutputFile +=
      "-offset" + itostr(uri->Offset) + "-size" + itostr(uri->Size) + ".co";

  // Create an ObjectFile object from uri.file_uri
  auto ObjOrErr = ObjectFile::createObjectFile(uri->FileName);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto Obj = ObjOrErr->getBinary();
  if (Error Err =
          object::extractCodeObject(*Obj, uri->Offset, uri->Size, OutputFile))
    return Err;

  return Error::success();
}

OffloadKind object::getOffloadKind(StringRef Name) {
  return llvm::StringSwitch<OffloadKind>(Name)
      .Case("openmp", OFK_OpenMP)
      .Case("cuda", OFK_Cuda)
      .Case("hip", OFK_HIP)
      .Default(OFK_None);
}

StringRef object::getOffloadKindName(OffloadKind Kind) {
  switch (Kind) {
  case OFK_OpenMP:
    return "openmp";
  case OFK_Cuda:
    return "cuda";
  case OFK_HIP:
    return "hip";
  default:
    return "none";
  }
}

ImageKind object::getImageKind(StringRef Name) {
  return llvm::StringSwitch<ImageKind>(Name)
      .Case("o", IMG_Object)
      .Case("bc", IMG_Bitcode)
      .Case("cubin", IMG_Cubin)
      .Case("fatbin", IMG_Fatbinary)
      .Case("s", IMG_PTX)
      .Default(IMG_None);
}

StringRef object::getImageKindName(ImageKind Kind) {
  switch (Kind) {
  case IMG_Object:
    return "o";
  case IMG_Bitcode:
    return "bc";
  case IMG_Cubin:
    return "cubin";
  case IMG_Fatbinary:
    return "fatbin";
  case IMG_PTX:
    return "s";
  default:
    return "";
  }
}

bool object::areTargetsCompatible(const OffloadFile::TargetID &LHS,
                                  const OffloadFile::TargetID &RHS) {
  // Exact matches are not considered compatible because they are the same
  // target. We are interested in different targets that are compatible.
  if (LHS == RHS)
    return false;

  // The triples must match at all times.
  if (LHS.first != RHS.first)
    return false;

  // If the architecture is "all" we assume it is always compatible.
  if (LHS.second == "generic" || RHS.second == "generic")
    return true;

  // Only The AMDGPU target requires additional checks.
  llvm::Triple T(LHS.first);
  if (!T.isAMDGPU())
    return false;

  // The base processor must always match.
  if (LHS.second.split(":").first != RHS.second.split(":").first)
    return false;

  // Check combintions of on / off features that must match.
  if (LHS.second.contains("xnack+") && RHS.second.contains("xnack-"))
    return false;
  if (LHS.second.contains("xnack-") && RHS.second.contains("xnack+"))
    return false;
  if (LHS.second.contains("sramecc-") && RHS.second.contains("sramecc+"))
    return false;
  if (LHS.second.contains("sramecc+") && RHS.second.contains("sramecc-"))
    return false;
  return true;
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

    // Recalculate MD5 hash for integrity check
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
