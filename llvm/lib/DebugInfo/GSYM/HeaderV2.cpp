//===- HeaderV2.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#define HEX8(v) llvm::format_hex(v, 4)
#define HEX16(v) llvm::format_hex(v, 6)
#define HEX32(v) llvm::format_hex(v, 10)
#define HEX64(v) llvm::format_hex(v, 18)

using namespace llvm;
using namespace gsym;

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const HeaderV2 &H) {
  OS << "Header:\n";
  OS << "  Magic          = " << HEX32(H.Magic) << "\n";
  OS << "  Version        = " << HEX16(H.Version) << '\n';
  OS << "  AddrOffSize    = " << HEX8(H.AddrOffSize) << '\n';
  OS << "  StrTableEnc    = " << HEX8(static_cast<uint8_t>(H.StrTableEncoding))
     << '\n';
  OS << "  BaseAddress    = " << HEX64(H.BaseAddress) << '\n';
  OS << "  NumAddresses   = " << HEX32(H.NumAddresses) << '\n';
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
  uint8_t Encoding = static_cast<uint8_t>(StrTableEncoding);
  switch (Encoding) {
  case static_cast<uint8_t>(StringTableEncoding::Default):
    break;
  default:
    return createStringError(std::errc::invalid_argument,
                             "unsupported string table encoding %u", Encoding);
  }
  return Error::success();
}

llvm::Expected<HeaderV2> HeaderV2::decode(GsymDataExtractor &Data) {
  uint64_t Offset = 0;
  // The fixed portion of the HeaderV2 is 20 bytes:
  //   Magic(4) + Version(2) + AddrOffSize(1) + StrTableEncoding(1) +
  //   BaseAddress(8) + NumAddresses(4)
  const uint64_t FixedHeaderSize = HeaderV2::getEncodedSize();
  if (!Data.isValidOffsetForDataOfSize(Offset, FixedHeaderSize))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a gsym::HeaderV2");
  HeaderV2 H;
  H.Magic = Data.getU32(&Offset);
  H.Version = Data.getU16(&Offset);
  H.AddrOffSize = Data.getU8(&Offset);
  H.StrTableEncoding = static_cast<StringTableEncoding>(Data.getU8(&Offset));
  H.BaseAddress = Data.getU64(&Offset);
  H.NumAddresses = Data.getU32(&Offset);
  if (llvm::Error Err = H.checkForError())
    return std::move(Err);
  return H;
}

llvm::Error HeaderV2::encode(FileWriter &O) const {
  if (llvm::Error Err = checkForError())
    return Err;
  O.writeU32(Magic);
  O.writeU16(Version);
  O.writeU8(AddrOffSize);
  O.writeU8(static_cast<uint8_t>(StrTableEncoding));
  O.writeU64(BaseAddress);
  O.writeU32(NumAddresses);
  return Error::success();
}

bool llvm::gsym::operator==(const HeaderV2 &LHS, const HeaderV2 &RHS) {
  return LHS.Magic == RHS.Magic && LHS.Version == RHS.Version &&
         LHS.AddrOffSize == RHS.AddrOffSize &&
         LHS.StrTableEncoding == RHS.StrTableEncoding &&
         LHS.BaseAddress == RHS.BaseAddress &&
         LHS.NumAddresses == RHS.NumAddresses;
}
