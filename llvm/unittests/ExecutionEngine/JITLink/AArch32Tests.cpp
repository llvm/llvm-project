//===------- AArch32Tests.cpp - Unit tests for the AArch32 backend --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/BinaryFormat/ELF.h>
#include <llvm/ExecutionEngine/JITLink/aarch32.h>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::aarch32;
using namespace llvm::support;
using namespace llvm::support::endian;

struct MutableHalfWords {
  MutableHalfWords(HalfWords Preset) : Hi(Preset.Hi), Lo(Preset.Lo) {}

  void patch(HalfWords Value, HalfWords Mask) {
    Hi = (Hi & ~Mask.Hi) | Value.Hi;
    Lo = (Lo & ~Mask.Lo) | Value.Lo;
  }

  uint16_t Hi; // First halfword
  uint16_t Lo; // Second halfword
};

struct MutableWord {
  MutableWord(uint32_t Preset) : Wd(Preset) {}

  void patch(uint32_t Value, uint32_t Mask) { Wd = (Wd & ~Mask) | Value; }

  uint32_t Wd;
};
namespace llvm {
namespace jitlink {

Expected<aarch32::EdgeKind_aarch32> getJITLinkEdgeKind(uint32_t ELFType);
Expected<uint32_t> getELFRelocationType(Edge::Kind Kind);

} // namespace jitlink
} // namespace llvm

TEST(AArch32_ELF, EdgeKinds) {
  // Fails: Invalid ELF type -> JITLink kind
  Expected<uint32_t> ErrKind = getJITLinkEdgeKind(ELF::R_ARM_NONE);
  EXPECT_TRUE(errorToBool(ErrKind.takeError()));

  // Fails: Invalid JITLink kind -> ELF type
  Expected<uint32_t> ErrType = getELFRelocationType(Edge::Invalid);
  EXPECT_TRUE(errorToBool(ErrType.takeError()));

  for (Edge::Kind K = FirstDataRelocation; K < LastThumbRelocation; K += 1) {
    Expected<uint32_t> ELFType = getELFRelocationType(K);
    EXPECT_FALSE(errorToBool(ELFType.takeError()))
        << "Failed to translate JITLink kind -> ELF type";

    Expected<Edge::Kind> JITLinkKind = getJITLinkEdgeKind(*ELFType);
    EXPECT_FALSE(errorToBool(JITLinkKind.takeError()))
        << "Failed to translate ELF type -> JITLink kind";

    EXPECT_EQ(*JITLinkKind, K) << "Round-trip value inconsistent?";
  }
}

namespace llvm {
namespace jitlink {
namespace aarch32 {

HalfWords encodeImmBT4BlT1BlxT2(int64_t Value);
HalfWords encodeImmBT4BlT1BlxT2_J1J2(int64_t Value);
uint32_t encodeImmBA1BlA1BlxA2(int64_t Value);
HalfWords encodeImmMovtT1MovwT3(uint16_t Value);
HalfWords encodeRegMovtT1MovwT3(int64_t Value);
uint32_t encodeImmMovtA1MovwA2(uint16_t Value);
uint32_t encodeRegMovtA1MovwA2(int64_t Value);

int64_t decodeImmBT4BlT1BlxT2(uint32_t Hi, uint32_t Lo);
int64_t decodeImmBT4BlT1BlxT2_J1J2(uint32_t Hi, uint32_t Lo);
int64_t decodeImmBA1BlA1BlxA2(int64_t Value);
uint16_t decodeImmMovtT1MovwT3(uint32_t Hi, uint32_t Lo);
int64_t decodeRegMovtT1MovwT3(uint32_t Hi, uint32_t Lo);
uint16_t decodeImmMovtA1MovwA2(uint64_t Value);
int64_t decodeRegMovtA1MovwA2(uint64_t Value);

} // namespace aarch32
} // namespace jitlink
} // namespace llvm

// Big-endian for v7 and v8 (and v6 unless in legacy backwards compatible mode
// be32) have little-endian instructions and big-endian data. In ELF relocatable
// objects big-endian instructions may still be encountered. A be8 supporting
// linker is expected to endian-reverse instructions for the executable.
template <endianness Endian>
static HalfWords makeHalfWords(std::array<uint8_t, 4> Mem) {
  return HalfWords{read16<Endian>(Mem.data()), read16<Endian>(Mem.data() + 2)};
}

/// 25-bit branch with link (with J1J2 range extension)
TEST(AArch32_Relocations, Thumb_Call_J1J2) {
  static_assert(isInt<25>(16777215), "Max value");
  static_assert(isInt<25>(-16777215), "Min value");
  static_assert(!isInt<25>(16777217), "First overflow");
  static_assert(!isInt<25>(-16777217), "First underflow");

  constexpr HalfWords ImmMask = FixupInfo<Thumb_Call>::ImmMask;

  static std::array<HalfWords, 3> MemPresets{
      makeHalfWords<llvm::endianness::little>(
          {0xff, 0xf7, 0xfe, 0xef}), // common
      makeHalfWords<llvm::endianness::little>(
          {0x00, 0x00, 0x00, 0x00}), // zeros
      makeHalfWords<llvm::endianness::little>({0xff, 0xff, 0xff, 0xff}), // ones
  };

  auto EncodeDecode = [ImmMask](int64_t In, MutableHalfWords &Mem) {
    Mem.patch(encodeImmBT4BlT1BlxT2_J1J2(In), ImmMask);
    return decodeImmBT4BlT1BlxT2_J1J2(Mem.Hi, Mem.Lo);
  };

  for (MutableHalfWords Mem : MemPresets) {
    HalfWords UnaffectedBits(Mem.Hi & ~ImmMask.Hi, Mem.Lo & ~ImmMask.Lo);

    EXPECT_EQ(EncodeDecode(1, Mem), 0);                 // Zero value
    EXPECT_EQ(EncodeDecode(0x41, Mem), 0x40);           // Common value
    EXPECT_EQ(EncodeDecode(16777215, Mem), 16777214);   // Maximum value
    EXPECT_EQ(EncodeDecode(-16777215, Mem), -16777216); // Minimum value
    EXPECT_NE(EncodeDecode(16777217, Mem), 16777217);   // First overflow
    EXPECT_NE(EncodeDecode(-16777217, Mem), -16777217); // First underflow

    EXPECT_TRUE(UnaffectedBits.Hi == (Mem.Hi & ~ImmMask.Hi) &&
                UnaffectedBits.Lo == (Mem.Lo & ~ImmMask.Lo))
        << "Diff outside immediate field";
  }
}

/// 22-bit branch with link (without J1J2 range extension)
TEST(AArch32_Relocations, Thumb_Call_Bare) {
  static_assert(isInt<22>(2097151), "Max value");
  static_assert(isInt<22>(-2097151), "Min value");
  static_assert(!isInt<22>(2097153), "First overflow");
  static_assert(!isInt<22>(-2097153), "First underflow");

  constexpr HalfWords ImmMask = FixupInfo<Thumb_Call>::ImmMask;

  static std::array<HalfWords, 3> MemPresets{
      makeHalfWords<llvm::endianness::little>(
          {0xff, 0xf7, 0xfe, 0xef}), // common
      makeHalfWords<llvm::endianness::little>(
          {0x00, 0x00, 0x00, 0x00}), // zeros
      makeHalfWords<llvm::endianness::little>({0xff, 0xff, 0xff, 0xff}), // ones
  };

  auto EncodeDecode = [ImmMask](int64_t In, MutableHalfWords &Mem) {
    Mem.patch(encodeImmBT4BlT1BlxT2_J1J2(In), ImmMask);
    return decodeImmBT4BlT1BlxT2_J1J2(Mem.Hi, Mem.Lo);
  };

  for (MutableHalfWords Mem : MemPresets) {
    HalfWords UnaffectedBits(Mem.Hi & ~ImmMask.Hi, Mem.Lo & ~ImmMask.Lo);

    EXPECT_EQ(EncodeDecode(1, Mem), 0);               // Zero value
    EXPECT_EQ(EncodeDecode(0x41, Mem), 0x40);         // Common value
    EXPECT_EQ(EncodeDecode(2097151, Mem), 2097150);   // Maximum value
    EXPECT_EQ(EncodeDecode(-2097151, Mem), -2097152); // Minimum value
    EXPECT_NE(EncodeDecode(2097153, Mem), 2097153);   // First overflow
    EXPECT_NE(EncodeDecode(-2097153, Mem), -2097153); // First underflow

    EXPECT_TRUE(UnaffectedBits.Hi == (Mem.Hi & ~ImmMask.Hi) &&
                UnaffectedBits.Lo == (Mem.Lo & ~ImmMask.Lo))
        << "Diff outside immediate field";
  }
}

/// 26-bit branch with link
TEST(AArch32_Relocations, Arm_Call_Bare) {
  static_assert(isInt<26>(33554430), "Max value");
  static_assert(isInt<26>(-33554432), "Min value");
  static_assert(!isInt<26>(33554432), "First overflow");
  static_assert(!isInt<26>(-33554434), "First underflow");

  constexpr uint32_t ImmMask = FixupInfo<Arm_Call>::ImmMask;

  static std::array<uint32_t, 3> MemPresets{
      0xfeeffff7, // common
      0x00000000, // zeros
      0xffffffff, // ones
  };

  auto EncodeDecode = [=](int64_t In, MutableWord &Mem) {
    Mem.patch(encodeImmBA1BlA1BlxA2(In), ImmMask);
    return decodeImmBA1BlA1BlxA2(Mem.Wd);
  };

  for (MutableWord Mem : MemPresets) {
    MutableWord UnaffectedBits(Mem.Wd & ~ImmMask);

    EXPECT_EQ(EncodeDecode(0, Mem), 0);                 // Zero value
    EXPECT_EQ(EncodeDecode(0x40, Mem), 0x40);           // Common value
    EXPECT_EQ(EncodeDecode(33554428, Mem), 33554428);   // Maximum value
    EXPECT_EQ(EncodeDecode(-33554432, Mem), -33554432); // Minimum value
    EXPECT_NE(EncodeDecode(33554434, Mem), 33554434);   // First overflow
    EXPECT_NE(EncodeDecode(-33554434, Mem), -33554434); // First underflow

    EXPECT_TRUE(UnaffectedBits.Wd == (Mem.Wd & ~ImmMask))
        << "Diff outside immediate field";
  }
}

/// Write immediate value to the top halfword of the destination register
TEST(AArch32_Relocations, Thumb_MovtAbs) {
  static_assert(isUInt<16>(65535), "Max value");
  static_assert(!isUInt<16>(65536), "First overflow");

  constexpr HalfWords ImmMask = FixupInfo<Thumb_MovtAbs>::ImmMask;
  constexpr HalfWords RegMask = FixupInfo<Thumb_MovtAbs>::RegMask;

  static std::array<uint8_t, 3> Registers{0, 5, 12};
  static std::array<HalfWords, 3> MemPresets{
      makeHalfWords<llvm::endianness::little>(
          {0xff, 0xf7, 0xfe, 0xef}), // common
      makeHalfWords<llvm::endianness::little>(
          {0x00, 0x00, 0x00, 0x00}), // zeros
      makeHalfWords<llvm::endianness::little>({0xff, 0xff, 0xff, 0xff}), // ones
  };

  auto EncodeDecode = [ImmMask](uint32_t In, MutableHalfWords &Mem) {
    Mem.patch(encodeImmMovtT1MovwT3(In), ImmMask);
    return decodeImmMovtT1MovwT3(Mem.Hi, Mem.Lo);
  };

  for (MutableHalfWords Mem : MemPresets) {
    for (uint8_t Reg : Registers) {
      HalfWords UnaffectedBits(Mem.Hi & ~(ImmMask.Hi | RegMask.Hi),
                               Mem.Lo & ~(ImmMask.Lo | RegMask.Lo));

      Mem.patch(encodeRegMovtT1MovwT3(Reg), RegMask);
      EXPECT_EQ(EncodeDecode(0x76bb, Mem), 0x76bb);   // Common value
      EXPECT_EQ(EncodeDecode(0, Mem), 0);             // Minimum value
      EXPECT_EQ(EncodeDecode(0xffff, Mem), 0xffff);   // Maximum value
      EXPECT_NE(EncodeDecode(0x10000, Mem), 0x10000); // First overflow

      // Destination register as well as unaffected bits should be intact
      EXPECT_EQ(decodeRegMovtT1MovwT3(Mem.Hi, Mem.Lo), Reg);
      EXPECT_TRUE(UnaffectedBits.Hi == (Mem.Hi & ~(ImmMask.Hi | RegMask.Hi)) &&
                  UnaffectedBits.Lo == (Mem.Lo & ~(ImmMask.Lo | RegMask.Lo)))
          << "Diff outside immediate/register field";
    }
  }
}

/// Write immediate value to the top halfword of the destination register
TEST(AArch32_Relocations, Arm_MovtAbs) {
  static_assert(isUInt<16>(65535), "Max value");
  static_assert(!isUInt<16>(65536), "First overflow");

  constexpr uint32_t ImmMask = FixupInfo<Arm_MovtAbs>::ImmMask;
  constexpr uint32_t RegMask = FixupInfo<Arm_MovtAbs>::RegMask;

  static std::array<uint8_t, 3> Registers{0, 5, 12};
  static std::array<uint32_t, 3> MemPresets{
      0xfeeffff7, // common
      0x00000000, // zeros
      0xffffffff, // ones
  };

  auto EncodeDecode = [=](uint64_t In, MutableWord &Mem) {
    Mem.patch(encodeImmMovtA1MovwA2(In), ImmMask);
    return decodeImmMovtA1MovwA2(Mem.Wd);
  };

  for (MutableWord Mem : MemPresets) {
    for (uint8_t Reg : Registers) {
      MutableWord UnaffectedBits(Mem.Wd & ~(ImmMask | RegMask));

      Mem.patch(encodeRegMovtA1MovwA2(Reg), RegMask);
      EXPECT_EQ(EncodeDecode(0x76bb, Mem), 0x76bb);   // Common value
      EXPECT_EQ(EncodeDecode(0, Mem), 0);             // Minimum value
      EXPECT_EQ(EncodeDecode(0xffff, Mem), 0xffff);   // Maximum value
      EXPECT_NE(EncodeDecode(0x10000, Mem), 0x10000); // First overflow

      // Destination register as well as unaffected bits should be intact
      EXPECT_EQ(decodeRegMovtA1MovwA2(Mem.Wd), Reg);
      EXPECT_TRUE(UnaffectedBits.Wd == (Mem.Wd & ~(ImmMask | RegMask)))
          << "Diff outside immediate/register field";
    }
  }
}
