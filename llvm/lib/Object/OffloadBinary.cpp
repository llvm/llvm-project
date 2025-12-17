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
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::object;

namespace {

/// A MemoryBuffer that shares ownership of the underlying memory.
/// This allows multiple OffloadBinary instances to share the same buffer.
class SharedMemoryBuffer : public MemoryBuffer {
public:
  SharedMemoryBuffer(std::shared_ptr<MemoryBuffer> Buf)
      : SharedBuf(std::move(Buf)) {
    init(SharedBuf->getBufferStart(), SharedBuf->getBufferEnd(),
         /*RequiresNullTerminator=*/false);
  }

  BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

  StringRef getBufferIdentifier() const override {
    return SharedBuf->getBufferIdentifier();
  }

private:
  const std::shared_ptr<MemoryBuffer> SharedBuf;
};

/// Attempts to extract all the embedded device images contained inside the
/// buffer \p Contents. The buffer is expected to contain a valid offloading
/// binary format.
Error extractOffloadFiles(MemoryBufferRef Contents,
                          SmallVectorImpl<OffloadFile> &Binaries) {
  uint64_t Offset = 0;
  // There could be multiple offloading binaries stored at this section.
  while (Offset < Contents.getBufferSize()) {
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator*/ false);
    if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                       Buffer->getBufferStart()))
      Buffer = MemoryBuffer::getMemBufferCopy(Buffer->getBuffer(),
                                              Buffer->getBufferIdentifier());

    auto HeaderOrErr = OffloadBinary::extractHeader(*Buffer);
    if (!HeaderOrErr)
      return HeaderOrErr.takeError();
    const OffloadBinary::Header *Header = *HeaderOrErr;

    // Create a copy of original memory containing only the current binary.
    std::unique_ptr<MemoryBuffer> BufferCopy = MemoryBuffer::getMemBufferCopy(
        Buffer->getBuffer().take_front(Header->Size),
        Contents.getBufferIdentifier());

    auto BinariesOrErr = OffloadBinary::create(*BufferCopy);
    if (!BinariesOrErr)
      return BinariesOrErr.takeError();

    // Share ownership among multiple OffloadFiles.
    std::shared_ptr<MemoryBuffer> SharedBuffer =
        std::shared_ptr<MemoryBuffer>(std::move(BufferCopy));

    for (auto &Binary : *BinariesOrErr) {
      std::unique_ptr<SharedMemoryBuffer> SharedBufferPtr =
          std::make_unique<SharedMemoryBuffer>(SharedBuffer);
      Binaries.emplace_back(std::move(Binary), std::move(SharedBufferPtr));
    }

    Offset += Header->Size;
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

Expected<const OffloadBinary::Header *>
OffloadBinary::extractHeader(MemoryBufferRef Buf) {
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
  if (TheHeader->Version == 0 || TheHeader->Version > OffloadBinary::Version)
    return errorCodeToError(object_error::parse_failed);

  if (TheHeader->Size > Buf.getBufferSize() ||
      TheHeader->Size < sizeof(Entry) || TheHeader->Size < sizeof(Header))
    return errorCodeToError(object_error::unexpected_eof);

  uint64_t EntriesCount =
      (TheHeader->Version == 1) ? 1 : TheHeader->EntriesCount;
  uint64_t EntriesSize = sizeof(Entry) * EntriesCount;
  if (TheHeader->EntriesOffset > TheHeader->Size - EntriesSize ||
      EntriesSize > TheHeader->Size - sizeof(Header))
    return errorCodeToError(object_error::unexpected_eof);

  return TheHeader;
}

Expected<SmallVector<std::unique_ptr<OffloadBinary>>>
OffloadBinary::create(MemoryBufferRef Buf, std::optional<uint64_t> Index) {
  auto HeaderOrErr = OffloadBinary::extractHeader(Buf);
  if (!HeaderOrErr)
    return HeaderOrErr.takeError();
  const Header *TheHeader = *HeaderOrErr;

  const char *Start = Buf.getBufferStart();
  const Entry *Entries =
      reinterpret_cast<const Entry *>(&Start[TheHeader->EntriesOffset]);

  SmallVector<std::unique_ptr<OffloadBinary>> Binaries;
  if (TheHeader->Version > 1 && Index.has_value()) {
    if (*Index >= TheHeader->EntriesCount)
      return errorCodeToError(object_error::parse_failed);
    const Entry *TheEntry = &Entries[*Index];
    if (TheEntry->ImageOffset > Buf.getBufferSize() ||
        TheEntry->StringOffset > Buf.getBufferSize())
      return errorCodeToError(object_error::unexpected_eof);

    Binaries.emplace_back(new OffloadBinary(Buf, TheHeader, TheEntry, *Index));
    return Binaries;
  }

  uint64_t EntriesCount =
      (TheHeader->Version == 1) ? 1 : TheHeader->EntriesCount;
  for (uint64_t I = 0; I < EntriesCount; ++I) {
    const Entry *TheEntry = &Entries[I];
    if (TheEntry->ImageOffset > Buf.getBufferSize() ||
        TheEntry->StringOffset > Buf.getBufferSize())
      return errorCodeToError(object_error::unexpected_eof);

    Binaries.emplace_back(new OffloadBinary(Buf, TheHeader, TheEntry, I));
  }

  return Binaries;
}

SmallString<0> OffloadBinary::write(ArrayRef<OffloadingImage> OffloadingData) {
  uint64_t EntriesCount = OffloadingData.size();
  assert(EntriesCount > 0 && "At least one offloading image is required");

  // Create a null-terminated string table with all the used strings.
  // Also calculate total size of images.
  StringTableBuilder StrTab(StringTableBuilder::ELF);
  uint64_t TotalStringEntries = 0;
  uint64_t TotalImagesSize = 0;
  for (const OffloadingImage &Img : OffloadingData) {
    for (auto &KeyAndValue : Img.StringData) {
      StrTab.add(KeyAndValue.first);
      StrTab.add(KeyAndValue.second);
    }
    TotalStringEntries += Img.StringData.size();
    TotalImagesSize += Img.Image->getBufferSize();
  }
  StrTab.finalize();

  uint64_t StringEntrySize = sizeof(StringEntry) * TotalStringEntries;
  uint64_t EntriesSize = sizeof(Entry) * EntriesCount;
  uint64_t StrTabOffset = sizeof(Header) + EntriesSize + StringEntrySize;

  // Make sure the image we're wrapping around is aligned as well.
  uint64_t BinaryDataSize =
      alignTo(StrTabOffset + StrTab.getSize(), getAlignment());

  // Create the header and fill in the offsets. The entries will be directly
  // placed after the header in memory. Align the size to the alignment of the
  // header so this can be placed contiguously in a single section.
  Header TheHeader;
  TheHeader.Size = alignTo(BinaryDataSize + TotalImagesSize, getAlignment());
  TheHeader.EntriesOffset = sizeof(Header);
  TheHeader.EntriesCount = EntriesCount;

  SmallString<0> Data;
  Data.reserve(TheHeader.Size);
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<char *>(&TheHeader), sizeof(Header));

  // Create the entries using the string table offsets. The string table will be
  // placed directly after the set of entries in memory, and all the images are
  // after that.
  uint64_t StringEntryOffset = sizeof(Header) + EntriesSize;
  uint64_t ImageOffset = BinaryDataSize;
  for (const OffloadingImage &Img : OffloadingData) {
    Entry TheEntry;

    TheEntry.TheImageKind = Img.TheImageKind;
    TheEntry.TheOffloadKind = Img.TheOffloadKind;
    TheEntry.Flags = Img.Flags;

    TheEntry.StringOffset = StringEntryOffset;
    StringEntryOffset += sizeof(StringEntry) * Img.StringData.size();
    TheEntry.NumStrings = Img.StringData.size();

    TheEntry.ImageOffset = ImageOffset;
    ImageOffset += Img.Image->getBufferSize();
    TheEntry.ImageSize = Img.Image->getBufferSize();

    OS << StringRef(reinterpret_cast<char *>(&TheEntry), sizeof(Entry));
  }

  // Create the string map entries.
  for (const OffloadingImage &Img : OffloadingData) {
    for (auto &KeyAndValue : Img.StringData) {
      StringEntry Map{StrTabOffset + StrTab.getOffset(KeyAndValue.first),
                      StrTabOffset + StrTab.getOffset(KeyAndValue.second),
                      KeyAndValue.second.size()};
      OS << StringRef(reinterpret_cast<char *>(&Map), sizeof(StringEntry));
    }
  }

  StrTab.write(OS);
  // Add padding to required image alignment.
  OS.write_zeros(BinaryDataSize - OS.tell());

  for (const OffloadingImage &Img : OffloadingData)
    OS << Img.Image->getBuffer();

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

OffloadKind object::getOffloadKind(StringRef Name) {
  return llvm::StringSwitch<OffloadKind>(Name)
      .Case("openmp", OFK_OpenMP)
      .Case("cuda", OFK_Cuda)
      .Case("hip", OFK_HIP)
      .Case("sycl", OFK_SYCL)
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
  case OFK_SYCL:
    return "sycl";
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
