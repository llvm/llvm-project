//===- AMDGPUPrintf.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for the AMDGPU printf runtime interface, shared by the lowerings of
// printf calls on the AMDGPU target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AMDGPUPrintf.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

using namespace llvm;

// Scan the format string to locate all specifiers, and mark the ones that
// specify a string, i.e, the "%s" specifier with optional '*' characters.
void llvm::locateAMDGPUPrintfCStrings(SparseBitVector<8> &BV, StringRef Str) {
  static const char ConvSpecifiers[] = "diouxXfFeEgGaAcspn";
  // Skip the first argument, the format string.
  unsigned ArgIdx = 1;

  size_t SpecPos;
  while ((SpecPos = Str.find('%')) != StringRef::npos) {
    Str = Str.drop_front(SpecPos + 1);
    // "%%" prints a '%' and takes no argument.
    if (Str.consume_front("%"))
      continue;
    size_t SpecEnd = Str.find_first_of(ConvSpecifiers);
    if (SpecEnd == StringRef::npos)
      return;
    // Each '*' in the specifier takes an argument of its own.
    ArgIdx += Str.take_front(SpecEnd).count('*');
    if (Str[SpecEnd] == 's')
      BV.set(ArgIdx);
    Str = Str.drop_front(SpecEnd + 1);
    ++ArgIdx;
  }
}

uint64_t llvm::getAMDGPUPrintfFormatHash(StringRef Fmt) {
  MD5 Hasher;
  MD5::MD5Result Hash;
  Hasher.update(Fmt);
  Hasher.final(Hash);
  return Hash.low();
}

std::string llvm::getAMDGPUPrintfFormatMetadata(StringRef Fmt) {
  // Try sticking to llvm.printf.fmts format, although we are not going to
  // use the ID and argument size fields while printing,
  return "0:0:" +
         llvm::utohexstr(getAMDGPUPrintfFormatHash(Fmt), /*LowerCase=*/true) +
         "," + Fmt.str();
}

void llvm::packAMDGPUPrintfConstantString(StringRef S,
                                          SmallVectorImpl<uint32_t> &Words) {
  std::string Str(S.str() + '\0');

  DataExtractor Extractor(Str, /*IsLittleEndian=*/true);
  DataExtractor::Cursor Offset(0);
  while (Offset && Offset.tell() < Str.size()) {
    const uint64_t ReadSize = 4;
    uint64_t ReadNow = std::min(ReadSize, Str.size() - Offset.tell());
    uint64_t ReadBytes = 0;
    switch (ReadNow) {
    default:
      llvm_unreachable("min(4, X) > 4?");
    case 1:
      ReadBytes = Extractor.getU8(Offset);
      break;
    case 2:
      ReadBytes = Extractor.getU16(Offset);
      break;
    case 3:
      ReadBytes = Extractor.getU24(Offset);
      break;
    case 4:
      ReadBytes = Extractor.getU32(Offset);
      break;
    }
    cantFail(Offset.takeError(), "failed to read bytes from constant array");
    Words.push_back(static_cast<uint32_t>(ReadBytes));
  }
  // Additional padding for 8 byte alignment
  int Rem = (Str.size() % 8);
  if (Rem > 0 && Rem <= 4)
    Words.push_back(0);
}
