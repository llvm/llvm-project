//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/SimpleTypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::codeview;

namespace {

void checkVFTableShape(ArrayRef<uint8_t> Encoded,
                       ArrayRef<VFTableSlotKind> Slots) {
  SimpleTypeSerializer Serializer;
  VFTableShapeRecord Original(Slots);
  EXPECT_EQ(Encoded, Serializer.serialize(Original));

  CVType CVT(Encoded);
  VFTableShapeRecord Decoded;
  ASSERT_THAT_ERROR(TypeDeserializer::deserializeAs(CVT, Decoded), Succeeded());
  EXPECT_THAT(Decoded.getSlots(), testing::ElementsAreArray(Slots));
  EXPECT_EQ(Encoded, Serializer.serialize(Decoded));
}

TEST(TypeRecordMappingTest, VFTableShapeSingleSlot) {
  // MSVC emits the first slot in the high nibble; the unused low nibble is
  // zero.
  const uint8_t Encoded[] = {0x06, 0x00, 0x0a, 0x00, 0x01, 0x00, 0x50, 0xf1};
  const VFTableSlotKind Slots[] = {VFTableSlotKind::Near};
  checkVFTableShape(Encoded, Slots);
}

TEST(TypeRecordMappingTest, VFTableShapeEvenSlotOrder) {
  // Distinct slot kinds expose reversed nibble order even with an even count.
  const uint8_t Encoded[] = {0x06, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x56, 0xf1};
  const VFTableSlotKind Slots[] = {VFTableSlotKind::Near, VFTableSlotKind::Far};
  checkVFTableShape(Encoded, Slots);
}

TEST(TypeRecordMappingTest, VFTableShapeOddSlotCount) {
  // Identical pairs survive reversal, but the last slot must not be discarded.
  const uint8_t Encoded[] = {0x06, 0x00, 0x0a, 0x00, 0x03, 0x00, 0x55, 0x50};
  const VFTableSlotKind Slots[] = {VFTableSlotKind::Near, VFTableSlotKind::Near,
                                   VFTableSlotKind::Near};
  checkVFTableShape(Encoded, Slots);
}

} // namespace
