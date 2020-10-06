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
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"

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

Slice::Slice(const Archive &A, uint32_t CPUType, uint32_t CPUSubType,
             std::string ArchName, uint32_t Align)
    : B(&A), CPUType(CPUType), CPUSubType(CPUSubType),
      ArchName(std::move(ArchName)), P2Alignment(Align) {}

Slice::Slice(const MachOObjectFile &O, uint32_t Align)
    : B(&O), CPUType(O.getHeader().cputype),
      CPUSubType(O.getHeader().cpusubtype),
      ArchName(std::string(O.getArchTriple().getArchName())),
      P2Alignment(Align) {}

Slice::Slice(const IRObjectFile &IRO, uint32_t CPUType, uint32_t CPUSubType,
             std::string ArchName, uint32_t Align)
    : B(&IRO), CPUType(CPUType), CPUSubType(CPUSubType),
      ArchName(std::move(ArchName)), P2Alignment(Align) {}

Slice::Slice(const MachOObjectFile &O) : Slice(O, calculateAlignment(O)) {}

using MachoCPUTy = std::pair<unsigned, unsigned>;

static Expected<MachoCPUTy> getMachoCPUFromTriple(Triple TT) {
  auto CPU = std::make_pair(MachO::getCPUType(TT), MachO::getCPUSubType(TT));
  if (!CPU.first) {
    return CPU.first.takeError();
  }
  if (!CPU.second) {
    return CPU.second.takeError();
  }
  return std::make_pair(*CPU.first, *CPU.second);
}

static Expected<MachoCPUTy> getMachoCPUFromTriple(StringRef TT) {
  return getMachoCPUFromTriple(Triple{TT});
}

Expected<Slice> Slice::create(const Archive &A, LLVMContext *LLVMCtx) {
  Error Err = Error::success();
  std::unique_ptr<MachOObjectFile> MFO = nullptr;
  std::unique_ptr<IRObjectFile> IRFO = nullptr;
  for (const Archive::Child &Child : A.children(Err)) {
    Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary(LLVMCtx);
    if (!ChildOrErr)
      return createFileError(A.getFileName(), ChildOrErr.takeError());
    Binary *Bin = ChildOrErr.get().get();
    if (Bin->isMachOUniversalBinary())
      return createStringError(std::errc::invalid_argument,
                               ("archive member " + Bin->getFileName() +
                                " is a fat file (not allowed in an archive)")
                                   .str()
                                   .c_str());
    if (Bin->isMachO()) {
      MachOObjectFile *O = cast<MachOObjectFile>(Bin);
      if (IRFO) {
        return createStringError(
            std::errc::invalid_argument,
            "archive member %s is a MachO, while previous archive member "
            "%s was an IR LLVM object",
            O->getFileName().str().c_str(), IRFO->getFileName().str().c_str());
      }
      if (MFO &&
          std::tie(MFO->getHeader().cputype, MFO->getHeader().cpusubtype) !=
              std::tie(O->getHeader().cputype, O->getHeader().cpusubtype)) {
        return createStringError(
            std::errc::invalid_argument,
            ("archive member " + O->getFileName() + " cputype (" +
             Twine(O->getHeader().cputype) + ") and cpusubtype(" +
             Twine(O->getHeader().cpusubtype) +
             ") does not match previous archive members cputype (" +
             Twine(MFO->getHeader().cputype) + ") and cpusubtype(" +
             Twine(MFO->getHeader().cpusubtype) +
             ") (all members must match) " + MFO->getFileName())
                .str()
                .c_str());
      }
      if (!MFO) {
        ChildOrErr.get().release();
        MFO.reset(O);
      }
    } else if (Bin->isIR()) {
      IRObjectFile *O = cast<IRObjectFile>(Bin);
      if (MFO) {
        return createStringError(std::errc::invalid_argument,
                                 "archive member '%s' is an LLVM IR object, "
                                 "while previous archive member "
                                 "'%s' was a MachO",
                                 O->getFileName().str().c_str(),
                                 MFO->getFileName().str().c_str());
      }
      if (IRFO) {
        Expected<MachoCPUTy> CPUO = getMachoCPUFromTriple(O->getTargetTriple());
        Expected<MachoCPUTy> CPUFO =
            getMachoCPUFromTriple(IRFO->getTargetTriple());
        if (!CPUO)
          return CPUO.takeError();
        if (!CPUFO)
          return CPUFO.takeError();
        if (*CPUO != *CPUFO) {
          return createStringError(
              std::errc::invalid_argument,
              ("archive member " + O->getFileName() + " cputype (" +
               Twine(CPUO->first) + ") and cpusubtype(" + Twine(CPUO->second) +
               ") does not match previous archive members cputype (" +
               Twine(CPUFO->first) + ") and cpusubtype(" +
               Twine(CPUFO->second) + ") (all members must match) " +
               IRFO->getFileName())
                  .str()
                  .c_str());
        }
      } else {
        ChildOrErr.get().release();
        IRFO.reset(O);
      }
    } else
      return createStringError(std::errc::invalid_argument,
                               ("archive member " + Bin->getFileName() +
                                " is neither a MachO file or an LLVM IR file "
                                "(not allowed in an archive)")
                                   .str()
                                   .c_str());
  }
  if (Err)
    return createFileError(A.getFileName(), std::move(Err));
  if (!MFO && !IRFO)
    return createStringError(
        std::errc::invalid_argument,
        ("empty archive with no architecture specification: " +
         A.getFileName() + " (can't determine architecture for it)")
            .str()
            .c_str());

  if (MFO) {
    Slice ArchiveSlice(*(MFO.get()), MFO->is64Bit() ? 3 : 2);
    ArchiveSlice.B = &A;
    return ArchiveSlice;
  }

  // For IR objects
  Expected<Slice> ArchiveSliceOrErr = Slice::create(*IRFO, 0);
  if (!ArchiveSliceOrErr)
    return createFileError(A.getFileName(), ArchiveSliceOrErr.takeError());
  auto &ArchiveSlice = ArchiveSliceOrErr.get();
  ArchiveSlice.B = &A;
  return std::move(ArchiveSlice);
}

Expected<Slice> Slice::create(const IRObjectFile &IRO, uint32_t Align) {
  Expected<MachoCPUTy> CPUOrErr = getMachoCPUFromTriple(IRO.getTargetTriple());
  if (!CPUOrErr)
    return CPUOrErr.takeError();
  unsigned CPUType, CPUSubType;
  std::tie(CPUType, CPUSubType) = CPUOrErr.get();
  // We don't directly use the architecture name of the target triple T, as,
  // for instance, thumb is treated as ARM by the MachOUniversal object.
  std::string ArchName(
      MachOObjectFile::getArchTriple(CPUType, CPUSubType).getArchName());
  return Slice{IRO, CPUType, CPUSubType, std::move(ArchName), Align};
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

static Error writeUniversalBinaryToStream(ArrayRef<Slice> Slices,
                                          raw_ostream &Out) {
  MachO::fat_header FatHeader;
  FatHeader.magic = MachO::FAT_MAGIC;
  FatHeader.nfat_arch = Slices.size();

  Expected<SmallVector<MachO::fat_arch, 2>> FatArchListOrErr =
      buildFatArchList(Slices);
  if (!FatArchListOrErr)
    return FatArchListOrErr.takeError();
  SmallVector<MachO::fat_arch, 2> FatArchList = *FatArchListOrErr;

  if (sys::IsLittleEndianHost)
    MachO::swapStruct(FatHeader);
  Out.write(reinterpret_cast<const char *>(&FatHeader),
            sizeof(MachO::fat_header));

  if (sys::IsLittleEndianHost)
    for (MachO::fat_arch &FA : FatArchList)
      MachO::swapStruct(FA);
  Out.write(reinterpret_cast<const char *>(FatArchList.data()),
            sizeof(MachO::fat_arch) * FatArchList.size());

  if (sys::IsLittleEndianHost)
    for (MachO::fat_arch &FA : FatArchList)
      MachO::swapStruct(FA);

  size_t Offset =
      sizeof(MachO::fat_header) + sizeof(MachO::fat_arch) * FatArchList.size();
  for (size_t Index = 0, Size = Slices.size(); Index < Size; ++Index) {
    MemoryBufferRef BufferRef = Slices[Index].getBinary()->getMemoryBufferRef();
    assert((Offset <= FatArchList[Index].offset) && "Incorrect slice offset");
    Out.write_zeros(FatArchList[Index].offset - Offset);
    Out.write(BufferRef.getBufferStart(), BufferRef.getBufferSize());
    Offset = FatArchList[Index].offset + BufferRef.getBufferSize();
  }

  Out.flush();
  return Error::success();
}

Error object::writeUniversalBinary(ArrayRef<Slice> Slices,
                                   StringRef OutputFileName) {
  const bool IsExecutable = any_of(Slices, [](Slice S) {
    return sys::fs::can_execute(S.getBinary()->getFileName());
  });
  unsigned Mode = sys::fs::all_read | sys::fs::all_write;
  if (IsExecutable)
    Mode |= sys::fs::all_exe;
  Expected<sys::fs::TempFile> Temp = sys::fs::TempFile::create(
      OutputFileName + ".temp-universal-%%%%%%", Mode);
  if (!Temp)
    return Temp.takeError();
  raw_fd_ostream Out(Temp->FD, false);
  if (Error E = writeUniversalBinaryToStream(Slices, Out)) {
    if (Error DiscardError = Temp->discard())
      return joinErrors(std::move(E), std::move(DiscardError));
    return E;
  }
  return Temp->keep(OutputFileName);
}

Expected<std::unique_ptr<MemoryBuffer>>
object::writeUniversalBinaryToBuffer(ArrayRef<Slice> Slices) {
  SmallVector<char, 0> Buffer;
  raw_svector_ostream Out(Buffer);

  if (Error E = writeUniversalBinaryToStream(Slices, Out))
    return std::move(E);

  return std::make_unique<SmallVectorMemoryBuffer>(std::move(Buffer));
}
