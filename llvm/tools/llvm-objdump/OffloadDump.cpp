//===-- OffloadDump.cpp - Offloading dumper ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the offloading-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "OffloadDump.h"
#include "llvm-objdump.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/OffloadBundle.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objdump;

void disassembleObject(llvm::object::ObjectFile *, bool InlineRelocs);

/// Get the printable name of the image kind.
static StringRef getImageName(const OffloadBinary &OB) {
  switch (OB.getImageKind()) {
  case IMG_Object:
    return "elf";
  case IMG_Bitcode:
    return "llvm ir";
  case IMG_Cubin:
    return "cubin";
  case IMG_Fatbinary:
    return "fatbinary";
  case IMG_PTX:
    return "ptx";
  case IMG_SPIRV:
    return "spir-v";
  default:
    return "<none>";
  }
}

/// Print metadata from an OffloadBinary.
static void printOffloadBinaryMetadata(const OffloadBinary &OB,
                                       uint64_t Level) {
  outs().indent(Level * 2) << left_justify("kind", 16) << getImageName(OB)
                           << "\n";
  outs().indent(Level * 2) << left_justify("arch", 16) << OB.getArch() << "\n";
  outs().indent(Level * 2) << left_justify("triple", 16) << OB.getTriple()
                           << "\n";
  outs().indent(Level * 2) << left_justify("producer", 16)
                           << getOffloadKindName(OB.getOffloadKind()) << "\n";

  StringRef InnerImage = OB.getImage();
  outs().indent(Level * 2) << left_justify("image size", 16)
                           << InnerImage.size() << " bytes\n";
}

static void printBinary(const OffloadBinary &OB, uint64_t Index,
                        uint64_t Level = 0, Twine ParentIndexPrefix = "") {
  outs() << "\n";
  outs().indent(Level * 2) << "OFFLOADING IMAGE [" << ParentIndexPrefix << Index
                           << "]:\n";

  printOffloadBinaryMetadata(OB, Level);

  StringRef ImageData = OB.getImage();
  if (identify_magic(ImageData) != file_magic::offload_binary)
    return;

  MemoryBufferRef InnerBuffer(ImageData, "inner-offload-binary");
  SmallVector<OffloadFile> InnerBinaries;
  Error Err = extractOffloadBinaries(InnerBuffer, InnerBinaries);
  if (Err) {
    reportWarning("failed to extract nested OffloadBinary: " +
                      toString(std::move(Err)),
                  OB.getFileName());
    return;
  }
  assert(!InnerBinaries.empty() &&
         "An offload binary with a magic number should contain at least one "
         "binary");

  outs().indent(Level * 2) << left_justify("nested images", 16)
                           << InnerBinaries.size() << "\n";

  for (uint64_t I = 0, E = InnerBinaries.size(); I != E; ++I) {
    const OffloadBinary *InnerOB = InnerBinaries[I].getBinary();
    printBinary(*InnerOB, I, Level + 1, ParentIndexPrefix + Twine(Index) + ".");
  }
}

/// Print the embedded offloading contents of an ObjectFile \p O.
void llvm::dumpOffloadBinary(const ObjectFile &O, StringRef ArchName) {
  if (!O.isELF() && !O.isCOFF()) {
    reportWarning(
        "--offloading is currently only supported for COFF and ELF targets",
        O.getFileName());
    return;
  }

  SmallVector<OffloadFile> Binaries;
  if (Error Err = extractOffloadBinaries(O.getMemoryBufferRef(), Binaries))
    reportError(O.getFileName(), "while extracting offloading files: " +
                                     toString(std::move(Err)));

  // Print out all the binaries that are contained in this buffer.
  for (uint64_t I = 0, E = Binaries.size(); I != E; ++I)
    printBinary(*Binaries[I].getBinary(), I);

  dumpOffloadBundleFatBinary(O, ArchName);
}

// Given an Object file, collect all Bundles of FatBin Binaries
// and dump them into Code Object files
// if -arch=-name is specified, only dump the Entries that match the target arch
void llvm::dumpOffloadBundleFatBinary(const ObjectFile &O, StringRef ArchName) {
  if (!O.isELF() && !O.isCOFF()) {
    reportWarning(
        "--offloading is currently only supported for COFF and ELF targets",
        O.getFileName());
    return;
  }

  SmallVector<llvm::object::OffloadBundleFatBin> FoundBundles;

  if (Error Err = llvm::object::extractOffloadBundleFatBinary(O, FoundBundles))
    reportError(O.getFileName(), "while extracting offload FatBin bundles: " +
                                     toString(std::move(Err)));
  for (const auto &[BundleNum, Bundle] : llvm::enumerate(FoundBundles)) {
    for (OffloadBundleEntry &Entry : Bundle.getEntries()) {
      if (!ArchName.empty() && Entry.ID.find(ArchName) != std::string::npos)
        continue;

      // create file name for this object file:  <source-filename>.<Bundle
      // Number>.<EntryID>
      std::string str =
          Bundle.getFileName().str() + "." + itostr(BundleNum) + "." + Entry.ID;

      if (Bundle.isDecompressed()) {
        if (Error Err = object::extractCodeObject(
                Bundle.DecompressedBuffer->getMemBufferRef(), Entry.Offset,
                Entry.Size, StringRef(str)))
          reportError(O.getFileName(),
                      "while extracting offload Bundle Entries: " +
                          toString(std::move(Err)));
      } else {
        if (Error Err = object::extractCodeObject(O, Entry.Offset, Entry.Size,
                                                  StringRef(str)))
          reportError(O.getFileName(),
                      "while extracting offload Bundle Entries: " +
                          toString(std::move(Err)));
      }
      outs() << "Extracting offload bundle: " << str << "\n";
    }
  }
}

/// Print the contents of an offload binary file \p OB. This may contain
/// multiple binaries stored in the same buffer.
void llvm::dumpOffloadSections(const OffloadBinary &OB) {
  SmallVector<OffloadFile> Binaries;
  if (Error Err = extractOffloadBinaries(OB.getMemoryBufferRef(), Binaries))
    reportError(OB.getFileName(), "while extracting offloading files: " +
                                      toString(std::move(Err)));

  // Print out all the binaries that are contained in this buffer.
  for (uint64_t I = 0, E = Binaries.size(); I != E; ++I)
    printBinary(*Binaries[I].getBinary(), I);
}
