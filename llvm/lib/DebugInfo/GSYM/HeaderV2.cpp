//===- HeaderV2.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#define HEX8(v) llvm::format_hex(v, 4)
#define HEX16(v) llvm::format_hex(v, 6)
#define HEX32(v) llvm::format_hex(v, 10)
#define HEX64(v) llvm::format_hex(v, 18)

using namespace llvm;
using namespace gsym;

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const HeaderV2 &H) {
  OS << "HeaderV2:\n";
  OS << "  Magic          = " << HEX32(H.Magic) << "\n";
  OS << "  Version        = " << HEX16(H.Version) << '\n';
  OS << "  BaseAddress    = " << HEX64(H.BaseAddress) << '\n';
  OS << "  NumAddresses   = " << HEX32(H.NumAddresses) << '\n';
  OS << "  AddrOffSize    = " << HEX8(H.AddrOffSize) << '\n';
  OS << "  AddrInfoOffSize = " << HEX8(H.AddrInfoOffSize) << '\n';
  OS << "  StrpSize       = " << HEX8(H.StrpSize) << '\n';
  return OS;
}

llvm::Error HeaderV2::checkForError() const {
  if (Magic != GSYM_MAGIC)
    return createStringError(std::errc::invalid_argument,
                             "invalid GSYM magic 0x%8.8x", Magic);
  if (Version != HeaderV2::getVersion())
    return createStringError(std::errc::invalid_argument,
                             "unsupported GSYM version %u", Version);
  if (AddrOffSize < 1 || AddrOffSize > 8)
    return createStringError(std::errc::invalid_argument,
                             "invalid address offset size %u", AddrOffSize);
  if (AddrInfoOffSize < 1 || AddrInfoOffSize > 8)
    return createStringError(std::errc::invalid_argument,
                             "invalid address info offset size %u",
                             AddrInfoOffSize);
  if (StrpSize < 1 || StrpSize > 8)
    return createStringError(std::errc::invalid_argument,
                             "invalid strp size %u", StrpSize);
  if (Padding != 0)
    return createStringError(std::errc::invalid_argument,
                             "padding must be zero, got %u", Padding);
  switch (StrTableEncoding) {
  case StringTableEncoding::Default:
    break;
  }
  return Error::success();
}

llvm::Expected<HeaderV2> HeaderV2::decode(DataExtractor &Data) {
  uint64_t Offset = 0;
  // The fixed portion of the HeaderV2 is 24 bytes:
  //   Magic(4) + Version(2) + Padding(2) + BaseAddress(8) +
  //   NumAddresses(4) + AddrOffSize(1) + AddrInfoOffSize(1) +
  //   StrpSize(1) + StrTableEncoding(1)
  const uint64_t FixedHeaderSize = sizeof(HeaderV2);
  if (!Data.isValidOffsetForDataOfSize(Offset, FixedHeaderSize))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a gsym::HeaderV2");
  HeaderV2 H;
  H.Magic = Data.getU32(&Offset);
  H.Version = Data.getU16(&Offset);
  H.Padding = Data.getU16(&Offset);
  H.BaseAddress = Data.getU64(&Offset);
  H.NumAddresses = Data.getU32(&Offset);
  H.AddrOffSize = Data.getU8(&Offset);
  H.AddrInfoOffSize = Data.getU8(&Offset);
  H.StrpSize = Data.getU8(&Offset);
  H.StrTableEncoding = static_cast<StringTableEncoding>(Data.getU8(&Offset));
  if (llvm::Error Err = H.checkForError())
    return std::move(Err);
  return H;
}

llvm::Error HeaderV2::encode(FileWriter &O) const {
  if (llvm::Error Err = checkForError())
    return Err;
  O.writeU32(Magic);
  O.writeU16(Version);
  O.writeU16(Padding);
  O.writeU64(BaseAddress);
  O.writeU32(NumAddresses);
  O.writeU8(AddrOffSize);
  O.writeU8(AddrInfoOffSize);
  O.writeU8(StrpSize);
  O.writeU8(static_cast<uint8_t>(StrTableEncoding));
  return Error::success();
}

bool llvm::gsym::operator==(const HeaderV2 &LHS, const HeaderV2 &RHS) {
  return LHS.Magic == RHS.Magic && LHS.Version == RHS.Version &&
         LHS.Padding == RHS.Padding && LHS.BaseAddress == RHS.BaseAddress &&
         LHS.NumAddresses == RHS.NumAddresses &&
         LHS.AddrOffSize == RHS.AddrOffSize &&
         LHS.AddrInfoOffSize == RHS.AddrInfoOffSize &&
         LHS.StrpSize == RHS.StrpSize &&
         LHS.StrTableEncoding == RHS.StrTableEncoding;
}
