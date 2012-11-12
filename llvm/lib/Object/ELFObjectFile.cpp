//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELF.h"

namespace llvm {

using namespace object;

// Creates an in-memory object-file by default: createELFObjectFile(Buffer)
ObjectFile *ObjectFile::createELFObjectFile(MemoryBuffer *Object) {
  std::pair<unsigned char, unsigned char> Ident = getElfArchType(Object);
  error_code ec;

  if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB)
    return new ELFObjectFile<support::little, false>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB)
    return new ELFObjectFile<support::big, false>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB)
    return new ELFObjectFile<support::big, true>(Object, ec);
  else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB) {
    ELFObjectFile<support::little, true> *result =
          new ELFObjectFile<support::little, true>(Object, ec);
    return result;
  }

  report_fatal_error("Buffer is not an ELF object file!");
}

} // end namespace llvm
