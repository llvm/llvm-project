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
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
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

Error llvm::codeview::consume(ArrayRef<uint8_t> &Data, APSInt &Num) {
  // Used to avoid overload ambiguity on APInt construtor.
  bool FalseVal = false;
  if (Data.size() < 2)
    return make_error<CodeViewError>(
        cv_error_code::insufficient_buffer,
        "Buffer does not contain enough data for an APSInt");
  uint16_t Short = *reinterpret_cast<const ulittle16_t *>(Data.data());
  Data = Data.drop_front(2);
  if (Short < LF_NUMERIC) {
    Num = APSInt(APInt(/*numBits=*/16, Short, /*isSigned=*/false),
                 /*isUnsigned=*/true);
    return Error::success();
  }
  switch (Short) {
  case LF_CHAR:
    if (Data.size() < 1)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_CHAR");
    Num = APSInt(APInt(/*numBits=*/8,
                       *reinterpret_cast<const int8_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(1);
    return Error::success();
  case LF_SHORT:
    if (Data.size() < 2)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_SHORT");
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const little16_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(2);
    return Error::success();
  case LF_USHORT:
    if (Data.size() < 2)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_USHORT");
    Num = APSInt(APInt(/*numBits=*/16,
                       *reinterpret_cast<const ulittle16_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(2);
    return Error::success();
  case LF_LONG:
    if (Data.size() < 4)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_LONG");
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const little32_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(4);
    return Error::success();
  case LF_ULONG:
    if (Data.size() < 4)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_ULONG");
    Num = APSInt(APInt(/*numBits=*/32,
                       *reinterpret_cast<const ulittle32_t *>(Data.data()),
                       /*isSigned=*/FalseVal),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(4);
    return Error::success();
  case LF_QUADWORD:
    if (Data.size() < 8)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_QUADWORD");
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const little64_t *>(Data.data()),
                       /*isSigned=*/true),
                 /*isUnsigned=*/false);
    Data = Data.drop_front(8);
    return Error::success();
  case LF_UQUADWORD:
    if (Data.size() < 8)
      return make_error<CodeViewError>(
          cv_error_code::insufficient_buffer,
          "Buffer does not contain enough data for an LF_UQUADWORD");
    Num = APSInt(APInt(/*numBits=*/64,
                       *reinterpret_cast<const ulittle64_t *>(Data.data()),
                       /*isSigned=*/false),
                 /*isUnsigned=*/true);
    Data = Data.drop_front(8);
    return Error::success();
  }
  return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                   "Buffer contains invalid APSInt type");
}

Error llvm::codeview::consume(StringRef &Data, APSInt &Num) {
  ArrayRef<uint8_t> Bytes(Data.bytes_begin(), Data.bytes_end());
  auto EC = consume(Bytes, Num);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return EC;
}

/// Decode a numeric leaf value that is known to be a uint64_t.
Error llvm::codeview::consume_numeric(ArrayRef<uint8_t> &Data, uint64_t &Num) {
  APSInt N;
  if (auto EC = consume(Data, N))
    return EC;
  if (N.isSigned() || !N.isIntN(64))
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Data is not a numeric value!");
  Num = N.getLimitedValue();
  return Error::success();
}

Error llvm::codeview::consume(ArrayRef<uint8_t> &Data, uint32_t &Item) {
  const support::ulittle32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Item = *IntPtr;
  return Error::success();
}

Error llvm::codeview::consume(StringRef &Data, uint32_t &Item) {
  ArrayRef<uint8_t> Bytes(Data.bytes_begin(), Data.bytes_end());
  auto EC = consume(Bytes, Item);
  Data = StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
  return EC;
}

Error llvm::codeview::consume(ArrayRef<uint8_t> &Data, int32_t &Item) {
  const support::little32_t *IntPtr;
  if (auto EC = consumeObject(Data, IntPtr))
    return EC;
  Item = *IntPtr;
  return Error::success();
}

Error llvm::codeview::consume(ArrayRef<uint8_t> &Data, StringRef &Item) {
  if (Data.empty())
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Null terminated string buffer is empty!");

  StringRef Rest;
  std::tie(Item, Rest) = getBytesAsCharacters(Data).split('\0');
  // We expect this to be null terminated.  If it was not, it is an error.
  if (Data.size() == Item.size())
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Expected null terminator!");

  Data = ArrayRef<uint8_t>(Rest.bytes_begin(), Rest.bytes_end());
  return Error::success();
}
