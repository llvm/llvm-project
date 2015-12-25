//===-- ListRecordBuilder.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ListRecordBuilder.h"

using namespace llvm;
using namespace codeview;

ListRecordBuilder::ListRecordBuilder(TypeRecordKind Kind) : Builder(Kind) {}

void ListRecordBuilder::finishSubRecord() {
  // The builder starts at offset 2 in the actual CodeView buffer, so add an
  // additional offset of 2 before computing the alignment.
  uint32_t Remainder = (Builder.size() + 2) % 4;
  if (Remainder != 0) {
    for (int32_t PaddingBytesLeft = 4 - Remainder; PaddingBytesLeft > 0;
         --PaddingBytesLeft) {
      Builder.writeUInt8(0xf0 + PaddingBytesLeft);
    }
  }

  // TODO: Split the list into multiple records if it's longer than 64KB, using
  // a subrecord of TypeRecordKind::Index to chain the records together.
  assert(Builder.size() < 65536);
}
