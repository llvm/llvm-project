//===- StreamReader.cpp - Reads bytes and objects from a stream -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;

StreamReader::StreamReader(const StreamInterface &S) : Stream(S), Offset(0) {}

std::error_code StreamReader::readBytes(MutableArrayRef<uint8_t> Buffer) {
  if (auto EC = Stream.readBytes(Offset, Buffer))
    return EC;
  Offset += Buffer.size();
  return std::error_code();
}

std::error_code StreamReader::readInteger(uint32_t &Dest) {
  support::ulittle32_t P;
  if (std::error_code EC = readObject(&P))
    return EC;
  Dest = P;
  return std::error_code();
}

std::error_code StreamReader::readZeroString(std::string &Dest) {
  Dest.clear();
  char C;
  do {
    readObject(&C);
    if (C != '\0')
      Dest.push_back(C);
  } while (C != '\0');
  return std::error_code();
}
