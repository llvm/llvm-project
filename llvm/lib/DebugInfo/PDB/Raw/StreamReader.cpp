//===- StreamReader.cpp - Reads bytes and objects from a stream -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::pdb;

StreamReader::StreamReader(const StreamInterface &S) : Stream(S), Offset(0) {}

Error StreamReader::readBytes(MutableArrayRef<uint8_t> Buffer) {
  if (auto EC = Stream.readBytes(Offset, Buffer))
    return EC;
  Offset += Buffer.size();
  return Error::success();
}

Error StreamReader::readInteger(uint32_t &Dest) {
  support::ulittle32_t P;
  if (auto EC = readObject(&P))
    return EC;
  Dest = P;
  return Error::success();
}

Error StreamReader::readZeroString(std::string &Dest) {
  Dest.clear();
  char C;
  do {
    if (auto EC = readObject(&C))
      return EC;
    if (C != '\0')
      Dest.push_back(C);
  } while (C != '\0');
  return Error::success();
}

Error StreamReader::getArrayRef(ArrayRef<uint8_t> &Array, uint32_t Length) {
  if (auto EC = Stream.getArrayRef(Offset, Array, Length))
    return EC;
  Offset += Length;
  return Error::success();
}
