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

TypeRecordBuilder::TypeRecordBuilder(TypeRecordKind Kind)
    : Stream(Buffer), Writer(Stream) {
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
    writeUInt16(LF_CHAR);
    writeInt16(static_cast<int8_t>(Value));
  } else if (Value >= std::numeric_limits<int16_t>::min() &&
             Value <= std::numeric_limits<int16_t>::max()) {
    writeUInt16(LF_SHORT);
    writeInt16(static_cast<int16_t>(Value));
  } else if (Value >= std::numeric_limits<int32_t>::min() &&
             Value <= std::numeric_limits<int32_t>::max()) {
    writeUInt16(LF_LONG);
    writeInt32(static_cast<int32_t>(Value));
  } else {
    writeUInt16(LF_QUADWORD);
    writeInt64(Value);
  }
}

void TypeRecordBuilder::writeEncodedUnsignedInteger(uint64_t Value) {
  if (Value < LF_CHAR) {
    writeUInt16(static_cast<uint16_t>(Value));
  } else if (Value <= std::numeric_limits<uint16_t>::max()) {
    writeUInt16(LF_USHORT);
    writeUInt16(static_cast<uint16_t>(Value));
  } else if (Value <= std::numeric_limits<uint32_t>::max()) {
    writeUInt16(LF_ULONG);
    writeUInt32(static_cast<uint32_t>(Value));
  } else {
    writeUInt16(LF_UQUADWORD);
    writeUInt64(Value);
  }
}

void TypeRecordBuilder::writeNullTerminatedString(StringRef Value) {
  // Microsoft's linker seems to have trouble with symbol names longer than
  // 0xffd8 bytes.
  Value = Value.substr(0, 0xffd8);
  Stream.write(Value.data(), Value.size());
  writeUInt8(0);
}

void TypeRecordBuilder::writeGuid(StringRef Guid) {
  assert(Guid.size() == 16);
  Stream.write(Guid.data(), 16);
}

void TypeRecordBuilder::writeTypeIndex(TypeIndex TypeInd) {
  writeUInt32(TypeInd.getIndex());
}

void TypeRecordBuilder::writeTypeRecordKind(TypeRecordKind Kind) {
  writeUInt16(static_cast<uint16_t>(Kind));
}
