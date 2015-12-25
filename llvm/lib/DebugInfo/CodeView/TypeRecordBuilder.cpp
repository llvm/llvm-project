//===-- TypeRecordBuilder.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeRecordBuilder.h"

using namespace llvm;
using namespace codeview;

TypeRecordBuilder::TypeRecordBuilder(TypeRecordKind Kind) : Stream(Buffer),
  Writer(Stream) {
  writeTypeRecordKind(Kind);
}

StringRef TypeRecordBuilder::str() {
  return StringRef(Buffer.data(), Buffer.size());
}

void TypeRecordBuilder::writeUInt8(uint8_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeInt16(int16_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeUInt16(uint16_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeInt32(int32_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeUInt32(uint32_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeInt64(int64_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeUInt64(uint64_t Value) {
  Writer.write(Value);
}

void TypeRecordBuilder::writeEncodedInteger(int64_t Value) {
  if (Value >= 0) {
    writeEncodedUnsignedInteger(static_cast<uint64_t>(Value));
  } else {
    writeEncodedSignedInteger(Value);
  }
}

void TypeRecordBuilder::writeEncodedSignedInteger(int64_t Value) {
  if (Value >= std::numeric_limits<int8_t>::min() &&
      Value <= std::numeric_limits<int8_t>::max()) {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::SByte));
    writeInt16(static_cast<int8_t>(Value));
  } else if (Value >= std::numeric_limits<int16_t>::min() &&
             Value <= std::numeric_limits<int16_t>::max()) {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::Int16));
    writeInt16(static_cast<int16_t>(Value));
  } else if (Value >= std::numeric_limits<int32_t>::min() &&
             Value <= std::numeric_limits<int32_t>::max()) {
    writeUInt16(static_cast<uint32_t>(TypeRecordKind::Int32));
    writeInt32(static_cast<int32_t>(Value));
  } else {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::Int64));
    writeInt64(Value);
  }
}

void TypeRecordBuilder::writeEncodedUnsignedInteger(uint64_t Value) {
  if (Value < static_cast<uint16_t>(TypeRecordKind::SByte)) {
    writeUInt16(static_cast<uint16_t>(Value));
  } else if (Value <= std::numeric_limits<uint16_t>::max()) {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::UInt16));
    writeUInt16(static_cast<uint16_t>(Value));
  } else if (Value <= std::numeric_limits<uint32_t>::max()) {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::UInt32));
    writeUInt32(static_cast<uint32_t>(Value));
  } else {
    writeUInt16(static_cast<uint16_t>(TypeRecordKind::UInt64));
    writeUInt64(Value);
  }
}

void TypeRecordBuilder::writeNullTerminatedString(const char *Value) {
  assert(Value != nullptr);

  size_t Length = strlen(Value);
  Stream.write(Value, Length);
  writeUInt8(0);
}

void TypeRecordBuilder::writeNullTerminatedString(StringRef Value) {
  Stream.write(Value.data(), Value.size());
  writeUInt8(0);
}

void TypeRecordBuilder::writeTypeIndex(TypeIndex TypeInd) {
  writeUInt32(TypeInd.getIndex());
}

void TypeRecordBuilder::writeTypeRecordKind(TypeRecordKind Kind) {
  writeUInt16(static_cast<uint16_t>(Kind));
}
