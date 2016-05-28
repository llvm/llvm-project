//===-- RecordSerialization.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utilities for serializing and deserializing CodeView records.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::support;

/// Reinterpret a byte array as an array of characters. Does not interpret as
/// a C string, as StringRef has several helpers (split) that make that easy.
StringRef llvm::codeview::getBytesAsCharacters(ArrayRef<uint8_t> LeafData) {
  return StringRef(reinterpret_cast<const char *>(LeafData.data()),
                   LeafData.size());
}

StringRef llvm::codeview::getBytesAsCString(ArrayRef<uint8_t> LeafData) {
  return getBytesAsCharacters(LeafData).split('\0').first;
}

std::error_code llvm::codeview::consume(ArrayRef<uint8_t> &Data, APSInt &Num) {
  // Used to avoid overload ambiguity on APInt construtor.
  bool FalseVal = false;
  if (Data.size() < 2)
    return std::make_error_code(std::errc::illegal_byte_sequence);
  uint16_t Short = *reinterpret_cast<const ulittle16_t *>(Data.data());
  Data = Data.drop_front(2);
  if (Short < LF_NUMERIC) {
    Num = APSInt(APInt(/*numBits=*/16, Short, /*isSigned=*/false),
                 /*isUnsigned=*/true);
    return std::error_code();
  }
  switch (Short) {
  case LF_CHAR:
    if (Data.size() < 1)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/8,
                       *reinterpret_cast<const int8_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(1);
    return std::error_code();
  case LF_SHORT:
    if (Data.size() < 2)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const little16_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(2);
    return std::error_code();
  case LF_USHORT:
    if (Data.size() < 2)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const ulittle16_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(2);
    return std::error_code();
  case LF_LONG:
    if (Data.size() < 4)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const little32_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(4);
    return std::error_code();
  case LF_ULONG:
    if (Data.size() < 4)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const ulittle32_t *>(Data.data()),
                       /*isSigned=*/FalseVal),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(4);
    return std::error_code();
  case LF_QUADWORD:
    if (Data.size() < 8)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const little64_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(8);
    return std::error_code();
  case LF_UQUADWORD:
    if (Data.size() < 8)
      return std::make_error_code(std::errc::illegal_byte_sequence);
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const ulittle64_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(8);
    return std::error_code();
  }
  return std::make_error_code(std::errc::illegal_byte_sequence);
}

std::error_code llvm::codeview::consume(StringRef &Data, APSInt &Num) {
  ArrayRef<uint8_t> Bytes(Data.bytes_begin(), Data.bytes_end());
  auto EC = consume(Bytes, Num);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return EC;
}

/// Decode a numeric leaf value that is known to be a uint64_t.
std::error_code llvm::codeview::consume_numeric(ArrayRef<uint8_t> &Data,
                                                uint64_t &Num) {
  APSInt N;
  if (auto EC = consume(Data, N))
    return EC;
  if (N.isSigned() || !N.isIntN(64))
    return std::make_error_code(std::errc::illegal_byte_sequence);
  Num = N.getLimitedValue();
  return std::error_code();
}

std::error_code llvm::codeview::consume(ArrayRef<uint8_t> &Data,
                                        uint32_t &Item) {
  const support::ulittle32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Item = *IntPtr;
  return std::error_code();
}

std::error_code llvm::codeview::consume(StringRef &Data, uint32_t &Item) {
  ArrayRef<uint8_t> Bytes(Data.bytes_begin(), Data.bytes_end());
  auto EC = consume(Bytes, Item);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return EC;
}

std::error_code llvm::codeview::consume(ArrayRef<uint8_t> &Data,
                                        int32_t &Item) {
  const support::little32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Item = *IntPtr;
  return std::error_code();
}

std::error_code llvm::codeview::consume(ArrayRef<uint8_t> &Data,
                                        StringRef &Item) {
  if (Data.empty())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  StringRef Rest;
  std::tie(Item, Rest) = getBytesAsCharacters(Data).split('\0');
  // We expect this to be null terminated.  If it was not, it is an error.
  if (Data.size() == Item.size())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  Data = ArrayRef<uint8_t>(Rest.bytes_begin(), Rest.bytes_end());
  return std::error_code();
}
