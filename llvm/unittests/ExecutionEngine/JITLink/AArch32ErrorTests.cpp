//===------- AArch32ErrorTests.cpp - Test AArch32 error handling ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <llvm/ExecutionEngine/JITLink/aarch32.h>

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::aarch32;
using namespace llvm::support;
using namespace llvm::support::endian;

constexpr unsigned PointerSize = 4;
auto G = std::make_unique<LinkGraph>(
    "foo", std::make_shared<orc::SymbolStringPool>(),
    Triple("armv7-linux-gnueabi"), PointerSize, endianness::little,
    aarch32::getEdgeKindName);
auto &Sec =
    G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

auto ArmCfg = getArmConfigForCPUArch(ARMBuildAttrs::v7);

constexpr uint64_t ArmAlignment = 4;
constexpr uint64_t ThumbAlignment = 2;
constexpr uint64_t AlignmentOffset = 0;

constexpr orc::ExecutorAddrDiff SymbolOffset = 0;
constexpr orc::ExecutorAddrDiff SymbolSize = 4;

class AArch32Errors : public testing::Test {
protected:
  const ArmConfig Cfg = getArmConfigForCPUArch(ARMBuildAttrs::v7);
  std::unique_ptr<LinkGraph> G;
  Section *S = nullptr;

  const uint8_t Zeros[4]{0x00, 0x00, 0x00, 0x00};
  uint8_t MutableZeros[4]{0x00, 0x00, 0x00, 0x00};

public:
  static void SetUpTestCase() {}

  void SetUp() override {
    G = std::make_unique<LinkGraph>(
        "foo", std::make_shared<orc::SymbolStringPool>(),
        Triple("armv7-linux-gnueabi"), PointerSize, endianness::little,
        aarch32::getEdgeKindName);
    S = &G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  }

  void TearDown() override {}

protected:
  template <size_t Size>
  Block &createBlock(const uint8_t (&Content)[Size], uint64_t Addr,
                     uint64_t Alignment = 4) {
    ArrayRef<char> CharContent{reinterpret_cast<const char *>(&Content),
                               sizeof(Content)};
    return G->createContentBlock(*S, CharContent, orc::ExecutorAddr(Addr),
                                 Alignment, AlignmentOffset);
  }

  template <size_t Size>
  Block &createMutableBlock(uint8_t (&Content)[Size], uint64_t Addr,
                            uint64_t Alignment = 4) {
    MutableArrayRef<char> CharContent{reinterpret_cast<char *>(&Content),
                                      sizeof(Content)};
    return G->createMutableContentBlock(
        *S, CharContent, orc::ExecutorAddr(Addr), Alignment, AlignmentOffset);
  }

  Symbol &createSymbolWithDistance(Block &Origin, uint64_t Dist) {
    uint64_t TargetAddr = Origin.getAddress().getValue() + Dist;
    return G->addAnonymousSymbol(createBlock(Zeros, TargetAddr), 0 /*Offset*/,
                                 PointerSize, false, false);
  };

  template <endianness Endian> void write(uint8_t *Mem, HalfWords Data) {
    write16<Endian>(Mem, Data.Hi);
    write16<Endian>(Mem + 2, Data.Lo);
  }
};

TEST_F(AArch32Errors, readAddendDataGeneric) {
  Block &ZerosBlock = createBlock(Zeros, 0x1000);
  constexpr uint64_t ZerosOffset = 0;

  // Invalid edge kind is the only error we can raise here right now.
  Edge::Kind Invalid = Edge::GenericEdgeKind::Invalid;
  EXPECT_THAT_EXPECTED(readAddend(*G, ZerosBlock, ZerosOffset, Invalid, Cfg),
                       FailedWithMessage(testing::HasSubstr(
                           "can not read implicit addend for aarch32 edge kind "
                           "INVALID RELOCATION")));
}

TEST(AArch32_ELF, readAddendArmErrors) {

  constexpr orc::ExecutorAddr B1DummyAddr(0x1000);

  // Permanently undefined instruction in ARM
  //    udf #0
  uint8_t ArmWord[] = {0xf0, 0x00, 0xf0, 0xe7};
  ArrayRef<char> ArmContent(reinterpret_cast<const char *>(&ArmWord),
                            sizeof(ArmWord));
  auto &BArm = G->createContentBlock(Sec, ArmContent, B1DummyAddr, ArmAlignment,
                                     AlignmentOffset);

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    EXPECT_THAT_EXPECTED(readAddend(*G, BArm, SymbolOffset, K, ArmCfg),
                         FailedWithMessage(testing::AllOf(
                             testing::StartsWith("Invalid opcode"),
                             testing::EndsWith(aarch32::getEdgeKindName(K)))));
  }
}

TEST(AArch32_ELF, readAddendThumbErrors) {

  constexpr orc::ExecutorAddr B2DummyAddr(0x2000);

  // Permanently undefined instruction in Thumb
  //    udf #0
  //
  //    11110:op:imm4:1:op1:imm12
  //    op  = 1111111 Permanent undefined
  //    op1 = 010
  //
  constexpr HalfWords ThumbHalfWords{0xf7f0, 0xa000};
  ArrayRef<char> ThumbContent(reinterpret_cast<const char *>(&ThumbHalfWords),
                              sizeof(ThumbHalfWords));
  auto &BThumb = G->createContentBlock(Sec, ThumbContent, B2DummyAddr,
                                       ThumbAlignment, AlignmentOffset);

  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    EXPECT_THAT_EXPECTED(readAddend(*G, BThumb, SymbolOffset, K, ArmCfg),
                         FailedWithMessage(testing::AllOf(
                             testing::StartsWith("Invalid opcode"),
                             testing::EndsWith(aarch32::getEdgeKindName(K)))));
  }
}

TEST_F(AArch32Errors, applyFixupDataGeneric) {
  Block &OriginBlock = createMutableBlock(MutableZeros, 0x1000);
  Block &TargetBlock = createBlock(Zeros, 0x2000);

  constexpr uint64_t OffsetInTarget = 0;
  Symbol &TargetSymbol = G->addAnonymousSymbol(TargetBlock, OffsetInTarget,
                                               PointerSize, false, false);

  constexpr uint64_t OffsetInOrigin = 0;
  Edge::Kind Invalid = Edge::GenericEdgeKind::Invalid;
  Edge InvalidEdge(Invalid, OffsetInOrigin, TargetSymbol, 0 /*Addend*/);
  EXPECT_THAT_ERROR(
      applyFixup(*G, OriginBlock, InvalidEdge, Cfg),
      FailedWithMessage(testing::HasSubstr(
          "encountered unfixable aarch32 edge kind INVALID RELOCATION")));
}

TEST(AArch32_ELF, applyFixupArmErrors) {

  constexpr orc::ExecutorAddr B3DummyAddr(0x5000);

  uint8_t ArmWord[] = {0xf0, 0x00, 0xf0, 0xe7};
  MutableArrayRef<char> MutableArmContent(reinterpret_cast<char *>(ArmWord),
                                          sizeof(ArmWord));

  auto &BArm = G->createMutableContentBlock(Sec, MutableArmContent, B3DummyAddr,
                                            ArmAlignment, AlignmentOffset);

  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BArm, SymbolOffset, SymbolSize, false, false);

  for (Edge::Kind K = FirstArmRelocation; K < LastArmRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, BArm, E, ArmCfg),
                      FailedWithMessage(testing::AllOf(
                          testing::StartsWith("Invalid opcode"),
                          testing::EndsWith(aarch32::getEdgeKindName(K)))));
  }
}

TEST(AArch32_ELF, applyFixupThumbErrors) {

  struct MutableHalfWords {
    constexpr MutableHalfWords(HalfWords Preset)
        : Hi(Preset.Hi), Lo(Preset.Lo) {}

    uint16_t Hi; // First halfword
    uint16_t Lo; // Second halfword
  };

  constexpr orc::ExecutorAddr B4DummyAddr(0x6000);

  // Permanently undefined instruction in Thumb
  //    udf #0
  //
  //    11110:op:imm4:1:op1:imm12
  //    op  = 1111111 Permanent undefined
  //    op1 = 010
  //
  constexpr HalfWords ThumbHalfWords{0xf7f0, 0xa000};
  MutableHalfWords MutableThumbHalfWords{ThumbHalfWords};
  MutableArrayRef<char> MutableThumbContent(
      reinterpret_cast<char *>(&MutableThumbHalfWords),
      sizeof(MutableThumbHalfWords));

  auto &BThumb = G->createMutableContentBlock(
      Sec, MutableThumbContent, B4DummyAddr, ThumbAlignment, AlignmentOffset);
  Symbol &TargetSymbol =
      G->addAnonymousSymbol(BThumb, SymbolOffset, SymbolSize, false, false);

  for (Edge::Kind K = FirstThumbRelocation; K < LastThumbRelocation; K += 1) {
    Edge E(K, 0, TargetSymbol, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, BThumb, E, ArmCfg),
                      FailedWithMessage(testing::AllOf(
                          testing::StartsWith("Invalid opcode"),
                          testing::EndsWith(aarch32::getEdgeKindName(K)))));
  }
}

TEST_F(AArch32Errors, applyFixupThumbCall) {
  // Check range of R_ARM_THM_CALL relocation
  constexpr uint64_t Call1Offset = 0; //< first out-of-range
  constexpr uint64_t Call2Offset = 4; //< last in-range

  uint8_t TwoCallsMem[8];
  Block &Site = createMutableBlock(TwoCallsMem, 0);
  constexpr HalfWords CallOpcode = FixupInfo<Thumb_Call>::Opcode;
  write<endianness::little>(TwoCallsMem + Call1Offset, CallOpcode);
  write<endianness::little>(TwoCallsMem + Call2Offset, CallOpcode);

  // Thumb call with J1J2-encoding has range of 25 bit
  ArmConfig ArmCfg;
  ArmCfg.J1J2BranchEncoding = true;
  Symbol &J1J2Target = createSymbolWithDistance(Site, 0x01ull << 24);
  {
    Edge LastInRange(Thumb_Call, Call2Offset, J1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, LastInRange, ArmCfg), Succeeded());
    Edge FirstOutOfRange(Thumb_Call, Call1Offset, J1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, FirstOutOfRange, ArmCfg),
                      FailedWithMessage(testing::HasSubstr("out of range")));
  }

  // Thumb call without J1J2-encoding has range of 22 bit
  ArmCfg.J1J2BranchEncoding = false;
  Symbol &NonJ1J2Target = createSymbolWithDistance(Site, 0x01ull << 21);
  {
    Edge LastInRange(Thumb_Call, Call2Offset, NonJ1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, LastInRange, ArmCfg), Succeeded());
    Edge FirstOutOfRange(Thumb_Call, Call1Offset, NonJ1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, FirstOutOfRange, ArmCfg),
                      FailedWithMessage(testing::HasSubstr("out of range")));
  }
}

TEST_F(AArch32Errors, applyFixupThumbJump24) {
  // Check range of R_ARM_THM_JUMP24 relocation
  constexpr uint64_t Jump1Offset = 0; //< first out-of-range
  constexpr uint64_t Jump2Offset = 4; //< last in-range

  uint8_t TwoJumpsMem[8];
  constexpr HalfWords JumpOpcode = FixupInfo<Thumb_Jump24>::Opcode;
  write<endianness::little>(TwoJumpsMem + Jump1Offset, JumpOpcode);
  write<endianness::little>(TwoJumpsMem + Jump2Offset, JumpOpcode);
  Block &Site = createMutableBlock(TwoJumpsMem, 0);

  // Thumb Jump24 with J1J2-encoding has range of 25 bit
  ArmCfg.J1J2BranchEncoding = true;
  Symbol &J1J2Target = createSymbolWithDistance(Site, 0x01ull << 24);
  J1J2Target.setTargetFlags(TargetFlags_aarch32::ThumbSymbol);
  {
    Edge LastInRange(Thumb_Jump24, Jump2Offset, J1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, LastInRange, ArmCfg), Succeeded());
    Edge FirstOutOfRange(Thumb_Jump24, Jump1Offset, J1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, FirstOutOfRange, ArmCfg),
                      FailedWithMessage(testing::HasSubstr("out of range")));
  }

  // Thumb Jump24 without J1J2-encoding has range of 22 bit
  ArmCfg.J1J2BranchEncoding = false;
  Symbol &NonJ1J2Target = createSymbolWithDistance(Site, 0x01ull << 21);
  NonJ1J2Target.setTargetFlags(TargetFlags_aarch32::ThumbSymbol);
  {
    Edge LastInRange(Thumb_Jump24, Jump2Offset, NonJ1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, LastInRange, ArmCfg), Succeeded());
    Edge FirstOutOfRange(Thumb_Jump24, Jump1Offset, NonJ1J2Target, 0);
    EXPECT_THAT_ERROR(applyFixup(*G, Site, FirstOutOfRange, ArmCfg),
                      FailedWithMessage(testing::HasSubstr("out of range")));
  }

  // Check that branching to an ARM target with Jump24 fails
  Symbol &ArmTarget = createSymbolWithDistance(Site, 0x1000);
  assert((ArmTarget.getTargetFlags() & TargetFlags_aarch32::ThumbSymbol) == 0);
  Edge Interworking(Thumb_Jump24, Jump2Offset, ArmTarget, 0);
  EXPECT_THAT_ERROR(applyFixup(*G, Site, Interworking, ArmCfg),
                    FailedWithMessage(testing::HasSubstr(
                        "Branch relocation needs interworking "
                        "stub when bridging to ARM")));
}
