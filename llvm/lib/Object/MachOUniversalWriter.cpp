//===- MachOUniversalWriter.cpp - MachO universal binary writer---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Slice class and writeUniversalBinary function for writing a MachO
// universal binary file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOUniversalWriter.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Support/FileOutputBuffer.h"

using namespace llvm;
using namespace object;

// For compatibility with cctools lipo, a file's alignment is calculated as the
// minimum aligment of all segments. For object files, the file's alignment is
// the maximum alignment of its sections.
static uint32_t calculateFileAlignment(const MachOObjectFile &O) {
  uint32_t P2CurrentAlignment;
  uint32_t P2MinAlignment = MachOUniversalBinary::MaxSectionAlignment;
  const bool Is64Bit = O.is64Bit();

  for (const auto &LC : O.load_commands()) {
    if (LC.C.cmd != (Is64Bit ? MachO::LC_SEGMENT_64 : MachO::LC_SEGMENT))
      continue;
    if (O.getHeader().filetype == MachO::MH_OBJECT) {
      unsigned NumberOfSections =
          (Is64Bit ? O.getSegment64LoadCommand(LC).nsects
                   : O.getSegmentLoadCommand(LC).nsects);
      P2CurrentAlignment = NumberOfSections ? 2 : P2MinAlignment;
      for (unsigned SI = 0; SI < NumberOfSections; ++SI) {
        P2CurrentAlignment = std::max(P2CurrentAlignment,
                                      (Is64Bit ? O.getSection64(LC, SI).align
                                               : O.getSection(LC, SI).align));
      }
    } else {
      P2CurrentAlignment =
          countTrailingZeros(Is64Bit ? O.getSegment64LoadCommand(LC).vmaddr
                                     : O.getSegmentLoadCommand(LC).vmaddr);
    }
    P2MinAlignment = std::min(P2MinAlignment, P2CurrentAlignment);
  }
  // return a value >= 4 byte aligned, and less than MachO MaxSectionAlignment
  return std::max(
      static_cast<uint32_t>(2),
      std::min(P2MinAlignment, static_cast<uint32_t>(
                                   MachOUniversalBinary::MaxSectionAlignment)));
}

static uint32_t calculateAlignment(const MachOObjectFile &ObjectFile) {
  switch (ObjectFile.getHeader().cputype) {
  case MachO::CPU_TYPE_I386:
  case MachO::CPU_TYPE_X86_64:
  case MachO::CPU_TYPE_POWERPC:
  case MachO::CPU_TYPE_POWERPC64:
    return 12; // log2 value of page size(4k) for x86 and PPC
  case MachO::CPU_TYPE_ARM:
  case MachO::CPU_TYPE_ARM64:
  case MachO::CPU_TYPE_ARM64_32:
    return 14; // log2 value of page size(16k) for Darwin ARM
  default:
    return calculateFileAlignment(ObjectFile);
  }
}

Slice::Slice(const MachOObjectFile &O, uint32_t Align)
    : B(&O), CPUType(O.getHeader().cputype),
      CPUSubType(O.getHeader().cpusubtype),
      ArchName(std::string(O.getArchTriple().getArchName())),
      P2Alignment(Align) {}

Slice::Slice(const MachOObjectFile &O) : Slice(O, calculateAlignment(O)) {}

Expected<Slice> Slice::create(const Archive *A) {
  Error Err = Error::success();
  std::unique_ptr<MachOObjectFile> FO = nullptr;
  for (const Archive::Child &Child : A->children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary();
    if (!ChildOrErr)
      return createFileError(A->getFileName(), ChildOrErr.takeError());
    Binary *Bin = ChildOrErr.get().get();
    if (Bin->isMachOUniversalBinary())
      return createStringError(std::errc::invalid_argument,
                               ("archive member " + Bin->getFileName() +
                                " is a fat file (not allowed in an archive)")
                                   .str()
                                   .c_str());
    if (!Bin->isMachO())
      return createStringError(
          std::errc::invalid_argument,
          ("archive member " + Bin->getFileName() +
           " is not a MachO file (not allowed in an archive)")
              .str()
              .c_str());
    MachOObjectFile *O = cast<MachOObjectFile>(Bin);
    if (FO && std::tie(FO->getHeader().cputype, FO->getHeader().cpusubtype) !=
                  std::tie(O->getHeader().cputype, O->getHeader().cpusubtype)) {
      return createStringError(
          std::errc::invalid_argument,
          ("archive member " + O->getFileName() + " cputype (" +
           Twine(O->getHeader().cputype) + ") and cpusubtype(" +
           Twine(O->getHeader().cpusubtype) +
           ") does not match previous archive members cputype (" +
           Twine(FO->getHeader().cputype) + ") and cpusubtype(" +
           Twine(FO->getHeader().cpusubtype) + ") (all members must match) " +
           FO->getFileName())
              .str()
              .c_str());
    }
    if (!FO) {
      ChildOrErr.get().release();
      FO.reset(O);
    }
  }
  if (Err)
    return createFileError(A->getFileName(), std::move(Err));
  if (!FO)
    return createStringError(
        std::errc::invalid_argument,
        ("empty archive with no architecture specification: " +
         A->getFileName() + " (can't determine architecture for it)")
            .str()
            .c_str());

  Slice ArchiveSlice = Slice(*(FO.get()), FO->is64Bit() ? 3 : 2);
  ArchiveSlice.B = A;
  return ArchiveSlice;
}

static Expected<SmallVector<MachO::fat_arch, 2>>
buildFatArchList(ArrayRef<Slice> Slices) {
  SmallVector<MachO::fat_arch, 2> FatArchList;
  uint64_t Offset =
      sizeof(MachO::fat_header) + Slices.size() * sizeof(MachO::fat_arch);

  for (const auto &S : Slices) {
    Offset = alignTo(Offset, 1ull << S.getP2Alignment());
    if (Offset > UINT32_MAX)
      return createStringError(
          std::errc::invalid_argument,
          ("fat file too large to be created because the offset "
           "field in struct fat_arch is only 32-bits and the offset " +
           Twine(Offset) + " for " + S.getBinary()->getFileName() +
           " for architecture " + S.getArchString() + "exceeds that.")
              .str()
              .c_str());

    MachO::fat_arch FatArch;
    FatArch.cputype = S.getCPUType();
    FatArch.cpusubtype = S.getCPUSubType();
    FatArch.offset = Offset;
    FatArch.size = S.getBinary()->getMemoryBufferRef().getBufferSize();
    FatArch.align = S.getP2Alignment();
    Offset += FatArch.size;
    FatArchList.push_back(FatArch);
  }
  return FatArchList;
}

Error object::writeUniversalBinary(ArrayRef<Slice> Slices,
                                   StringRef OutputFileName) {
  MachO::fat_header FatHeader;
  FatHeader.magic = MachO::FAT_MAGIC;
  FatHeader.nfat_arch = Slices.size();

  Expected<SmallVector<MachO::fat_arch, 2>> FatArchListOrErr =
      buildFatArchList(Slices);
  if (!FatArchListOrErr)
    return FatArchListOrErr.takeError();
  SmallVector<MachO::fat_arch, 2> FatArchList = *FatArchListOrErr;

  const bool IsExecutable = any_of(Slices, [](Slice S) {
    return sys::fs::can_execute(S.getBinary()->getFileName());
  });
  const uint64_t OutputFileSize =
      static_cast<uint64_t>(FatArchList.back().offset) +
      FatArchList.back().size;
  Expected<std::unique_ptr<FileOutputBuffer>> OutFileOrError =
      FileOutputBuffer::create(OutputFileName, OutputFileSize,
                               IsExecutable ? FileOutputBuffer::F_executable
                                            : 0);
  if (!OutFileOrError)
    return createFileError(OutputFileName, OutFileOrError.takeError());
  std::unique_ptr<FileOutputBuffer> OutFile = std::move(OutFileOrError.get());
  std::memset(OutFile->getBufferStart(), 0, OutputFileSize);

  if (sys::IsLittleEndianHost)
    MachO::swapStruct(FatHeader);
  std::memcpy(OutFile->getBufferStart(), &FatHeader, sizeof(MachO::fat_header));

  for (size_t Index = 0, Size = Slices.size(); Index < Size; ++Index) {
    MemoryBufferRef BufferRef = Slices[Index].getBinary()->getMemoryBufferRef();
    std::copy(BufferRef.getBufferStart(), BufferRef.getBufferEnd(),
              OutFile->getBufferStart() + FatArchList[Index].offset);
  }

  // FatArchs written after Slices in order to reduce the number of swaps for
  // the LittleEndian case
  if (sys::IsLittleEndianHost)
    for (MachO::fat_arch &FA : FatArchList)
      MachO::swapStruct(FA);
  std::memcpy(OutFile->getBufferStart() + sizeof(MachO::fat_header),
              FatArchList.begin(),
              sizeof(MachO::fat_arch) * FatArchList.size());

  if (Error E = OutFile->commit())
    return createFileError(OutputFileName, std::move(E));

  return Error::success();
}
