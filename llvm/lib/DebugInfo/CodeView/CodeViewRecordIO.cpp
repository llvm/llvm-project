//===- CodeViewRecordIO.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/CodeViewRecordIO.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"

using namespace llvm;
using namespace llvm::codeview;

Error CodeViewRecordIO::beginRecord(uint16_t Kind) {
  assert(!CurrentRecord.hasValue() && "There is already a record active!");
  CurrentRecord.emplace();

  CurrentRecord->Kind = Kind;
  return Error::success();
}

Error CodeViewRecordIO::endRecord() {
  assert(CurrentRecord.hasValue() && "Not in a record!");
  CurrentRecord.reset();
  return Error::success();
}

Error CodeViewRecordIO::skipPadding() {
  assert(!isWriting() && "Cannot skip padding while writing!");

  if (Reader->bytesRemaining() == 0)
    return Error::success();

  uint8_t Leaf = Reader->peek();
  if (Leaf < LF_PAD0)
    return Error::success();
  // Leaf is greater than 0xf0. We should advance by the number of bytes in
  // the low 4 bits.
  unsigned BytesToAdvance = Leaf & 0x0F;
  return Reader->skip(BytesToAdvance);
}

Error CodeViewRecordIO::mapByteVectorTail(ArrayRef<uint8_t> &Bytes) {
  if (isWriting()) {
    if (auto EC = Writer->writeBytes(Bytes))
      return EC;
  } else {
    if (auto EC = Reader->readBytes(Bytes, Reader->bytesRemaining()))
      return EC;
  }
  return Error::success();
}

Error CodeViewRecordIO::mapInteger(TypeIndex &TypeInd) {
  if (isWriting()) {
    if (auto EC = Writer->writeInteger(TypeInd.getIndex()))
      return EC;
    return Error::success();
  }

  uint32_t I;
  if (auto EC = Reader->readInteger(I))
    return EC;
  TypeInd.setIndex(I);
  return Error::success();
}

Error CodeViewRecordIO::mapEncodedInteger(int64_t &Value) {
  if (isWriting()) {
    if (Value >= 0) {
      if (auto EC = writeEncodedUnsignedInteger(static_cast<uint64_t>(Value)))
        return EC;
    } else {
      if (auto EC = writeEncodedSignedInteger(Value))
        return EC;
    }
  } else {
    APSInt N;
    if (auto EC = consume(*Reader, N))
      return EC;
    Value = N.getExtValue();
  }

  return Error::success();
}

Error CodeViewRecordIO::mapEncodedInteger(uint64_t &Value) {
  if (isWriting()) {
    if (auto EC = writeEncodedUnsignedInteger(Value))
      return EC;
  } else {
    APSInt N;
    if (auto EC = consume(*Reader, N))
      return EC;
    Value = N.getZExtValue();
  }
  return Error::success();
}

Error CodeViewRecordIO::mapEncodedInteger(APSInt &Value) {
  if (isWriting()) {
    if (Value.isSigned())
      return writeEncodedSignedInteger(Value.getSExtValue());
    return writeEncodedUnsignedInteger(Value.getZExtValue());
  }

  return consume(*Reader, Value);
}

Error CodeViewRecordIO::mapStringZ(StringRef &Value) {
  if (isWriting()) {
    if (auto EC = Writer->writeZeroString(Value))
      return EC;
  } else {
    if (auto EC = Reader->readZeroString(Value))
      return EC;
  }
  return Error::success();
}

Error CodeViewRecordIO::mapGuid(StringRef &Guid) {
  if (isWriting()) {
    assert(Guid.size() == 16 && "Invalid Guid Size!");
    if (auto EC = Writer->writeFixedString(Guid))
      return EC;
  } else {
    if (auto EC = Reader->readFixedString(Guid, 16))
      return EC;
  }
  return Error::success();
}

Error CodeViewRecordIO::writeEncodedSignedInteger(const int64_t &Value) {
  assert(Value < 0 && "Encoded integer is not signed!");
  if (Value >= std::numeric_limits<int8_t>::min()) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_CHAR)))
      return EC;
    if (auto EC = Writer->writeInteger(static_cast<int8_t>(Value)))
      return EC;
  } else if (Value >= std::numeric_limits<int16_t>::min()) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_SHORT)))
      return EC;
    if (auto EC = Writer->writeInteger(static_cast<int16_t>(Value)))
      return EC;
  } else if (Value >= std::numeric_limits<int32_t>::min()) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_LONG)))
      return EC;
    if (auto EC = Writer->writeInteger(static_cast<int32_t>(Value)))
      return EC;
  } else {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_QUADWORD)))
      return EC;
    if (auto EC = Writer->writeInteger(Value))
      return EC;
  }
  return Error::success();
}

Error CodeViewRecordIO::writeEncodedUnsignedInteger(const uint64_t &Value) {
  if (Value < LF_NUMERIC) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(Value)))
      return EC;
  } else if (Value <= std::numeric_limits<uint16_t>::max()) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_USHORT)))
      return EC;
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(Value)))
      return EC;
  } else if (Value <= std::numeric_limits<uint32_t>::max()) {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_ULONG)))
      return EC;
    if (auto EC = Writer->writeInteger(static_cast<uint32_t>(Value)))
      return EC;
  } else {
    if (auto EC = Writer->writeInteger(static_cast<uint16_t>(LF_UQUADWORD)))
      return EC;
    if (auto EC = Writer->writeInteger(Value))
      return EC;
  }

  return Error::success();
}
