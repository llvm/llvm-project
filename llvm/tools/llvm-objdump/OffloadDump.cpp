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
#include "llvm/Object/ELFObjectFile.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objdump;

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
  default:
    return "<none>";
  }
}

static void printBinary(const OffloadBinary &OB, uint64_t Index) {
  outs() << "\nOFFLOADING IMAGE [" << Index << "]:\n";
  outs() << left_justify("kind", 16) << getImageName(OB) << "\n";
  outs() << left_justify("arch", 16) << OB.getArch() << "\n";
  outs() << left_justify("triple", 16) << OB.getTriple() << "\n";
  outs() << left_justify("producer", 16)
         << getOffloadKindName(OB.getOffloadKind()) << "\n";
}

static Error visitAllBinaries(const OffloadBinary &OB) {
  uint64_t Offset = 0;
  uint64_t Index = 0;
  while (Offset < OB.getMemoryBufferRef().getBufferSize()) {
    MemoryBufferRef Buffer =
        MemoryBufferRef(OB.getData().drop_front(Offset), OB.getFileName());
    auto BinaryOrErr = OffloadBinary::create(Buffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();

    OffloadBinary &Binary = **BinaryOrErr;
    printBinary(Binary, Index++);

    Offset += Binary.getSize();
  }
  return Error::success();
}

/// Print the embedded offloading contents of an ObjectFile \p O.
void llvm::dumpOffloadBinary(const ObjectFile &O) {
  if (!O.isELF()) {
    reportWarning("--offloading is currently only supported for ELF targets",
                  O.getFileName());
    return;
  }

  for (ELFSectionRef Sec : O.sections()) {
    if (Sec.getType() != ELF::SHT_LLVM_OFFLOADING)
      continue;

    Expected<StringRef> Contents = Sec.getContents();
    if (!Contents)
      reportError(Contents.takeError(), O.getFileName());

    MemoryBufferRef Buffer = MemoryBufferRef(*Contents, O.getFileName());
    auto BinaryOrErr = OffloadBinary::create(Buffer);
    if (!BinaryOrErr)
      reportError(O.getFileName(), "while extracting offloading files: " +
                                       toString(BinaryOrErr.takeError()));
    OffloadBinary &Binary = **BinaryOrErr;

    // Print out all the binaries that are contained in this buffer. If we fail
    // to parse a binary before reaching the end of the buffer emit a warning.
    if (Error Err = visitAllBinaries(Binary))
      reportWarning("while parsing offloading files: " +
                        toString(std::move(Err)),
                    O.getFileName());
  }
}

/// Print the contents of an offload binary file \p OB. This may contain
/// multiple binaries stored in the same buffer.
void llvm::dumpOffloadSections(const OffloadBinary &OB) {
  // Print out all the binaries that are contained at this buffer. If we fail to
  // parse a binary before reaching the end of the buffer emit a warning.
  if (Error Err = visitAllBinaries(OB))
    reportWarning("while parsing offloading files: " + toString(std::move(Err)),
                  OB.getFileName());
}
