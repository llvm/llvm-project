//===- WODRoundTripTest.cpp - V3 WOD encode/decode round-trip tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCWinEH.h"
#include "llvm/Support/Win64EH.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::Win64EH;

/// Helper: encode a WinEH::Instruction via EncodeWOD, then decode the resulting
/// bytes via decodeWOD, returning the DecodedWOD. Fails the test on error.
static DecodedWOD roundTrip(const WinEH::Instruction &Inst) {
  SmallVector<uint8_t, 8> Pool;
  Win64EH::EncodeWOD(Inst, Pool);

  auto Result = Win64EH::decodeWOD(Pool, 0);
  EXPECT_TRUE(!!Result) << "decodeWOD failed: " << toString(Result.takeError());
  EXPECT_EQ(Result->ByteSize, Pool.size())
      << "Decoded byte size doesn't match encoded pool size";
  return *Result;
}

TEST(WODRoundTrip, PushNonVol) {
  // push rbx (reg 3)
  WinEH::Instruction Inst(UOP_PushNonVol, nullptr, 3, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH);
  EXPECT_EQ(W.Register, 3u);
  EXPECT_EQ(W.ByteSize, 1u);
}

TEST(WODRoundTrip, PushNonVolHighReg) {
  // push r15 (reg 15)
  WinEH::Instruction Inst(UOP_PushNonVol, nullptr, 15, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH);
  EXPECT_EQ(W.Register, 15u);
}

TEST(WODRoundTrip, AllocSmall) {
  // sub rsp, 40 (smallest alloc: 8..128 in steps of 8)
  WinEH::Instruction Inst(UOP_AllocSmall, nullptr, 0, 40);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_ALLOC_SMALL);
  EXPECT_EQ(W.Size, 40u);
  EXPECT_EQ(W.ByteSize, 1u);
}

TEST(WODRoundTrip, AllocSmallMin) {
  // sub rsp, 8
  WinEH::Instruction Inst(UOP_AllocSmall, nullptr, 0, 8);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_ALLOC_SMALL);
  EXPECT_EQ(W.Size, 8u);
}

TEST(WODRoundTrip, AllocSmallMax) {
  // sub rsp, 128 (max for ALLOC_SMALL: (15+1)*8 = 128)
  WinEH::Instruction Inst(UOP_AllocSmall, nullptr, 0, 128);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_ALLOC_SMALL);
  EXPECT_EQ(W.Size, 128u);
}

TEST(WODRoundTrip, AllocLarge) {
  // sub rsp, 4096 (fits in ALLOC_LARGE: 3 bytes, size/8)
  WinEH::Instruction Inst(UOP_AllocLarge, nullptr, 0, 4096);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_ALLOC_LARGE);
  EXPECT_EQ(W.Size, 4096u);
  EXPECT_EQ(W.ByteSize, 3u);
}

TEST(WODRoundTrip, AllocHuge) {
  // sub rsp, 524288 (>= 512*1024-8, uses ALLOC_HUGE: 5 bytes)
  WinEH::Instruction Inst(UOP_AllocLarge, nullptr, 0, 524288);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_ALLOC_HUGE);
  EXPECT_EQ(W.Size, 524288u);
  EXPECT_EQ(W.ByteSize, 5u);
}

TEST(WODRoundTrip, SetFPReg) {
  // lea rbp, [rsp+32]  =>  SetFPReg reg=5(RBP), offset=32
  WinEH::Instruction Inst(UOP_SetFPReg, nullptr, 5, 32);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_SET_FPREG);
  EXPECT_EQ(W.Register, 5u);
  EXPECT_EQ(W.Displacement, 32u);
  EXPECT_EQ(W.ByteSize, 2u);
}

TEST(WODRoundTrip, SaveNonVol) {
  // mov [rsp+40], rbx  =>  SaveNonVol reg=3(RBX), disp=40
  WinEH::Instruction Inst(UOP_SaveNonVol, nullptr, 3, 40);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_SAVE_NONVOL);
  EXPECT_EQ(W.Register, 3u);
  EXPECT_EQ(W.Displacement, 40u);
  EXPECT_EQ(W.ByteSize, 3u);
}

TEST(WODRoundTrip, SaveNonVolBig) {
  // mov [rsp+0x90000], rbx  =>  SaveNonVolBig reg=3, disp=0x90000
  WinEH::Instruction Inst(UOP_SaveNonVolBig, nullptr, 3, 0x90000);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_SAVE_NONVOL_FAR);
  EXPECT_EQ(W.Register, 3u);
  EXPECT_EQ(W.Displacement, 0x90000u);
  EXPECT_EQ(W.ByteSize, 5u);
}

TEST(WODRoundTrip, SaveXMM128) {
  // movaps [rsp+32], xmm6  =>  SaveXMM128 reg=6, disp=32
  WinEH::Instruction Inst(UOP_SaveXMM128, nullptr, 6, 32);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_SAVE_XMM128);
  EXPECT_EQ(W.Register, 6u);
  EXPECT_EQ(W.Displacement, 32u);
  EXPECT_EQ(W.ByteSize, 3u);
}

TEST(WODRoundTrip, SaveXMM128Big) {
  // movaps [rsp+0x90000], xmm6  =>  SaveXMM128Big reg=6, disp=0x90000
  WinEH::Instruction Inst(UOP_SaveXMM128Big, nullptr, 6, 0x90000);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_SAVE_XMM128_FAR);
  EXPECT_EQ(W.Register, 6u);
  EXPECT_EQ(W.Displacement, 0x90000u);
  EXPECT_EQ(W.ByteSize, 5u);
}

TEST(WODRoundTrip, PushMachFrameCode) {
  // .seh_pushframe @code  =>  PushMachFrame offset=1
  WinEH::Instruction Inst(UOP_PushMachFrame, nullptr, 0, 1);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH_CANONICAL_FRAME);
  EXPECT_EQ(W.Type, 1u);
  EXPECT_EQ(W.ByteSize, 2u);
}

TEST(WODRoundTrip, PushMachFrameNoCode) {
  // .seh_pushframe  =>  PushMachFrame offset=0
  WinEH::Instruction Inst(UOP_PushMachFrame, nullptr, 0, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH_CANONICAL_FRAME);
  EXPECT_EQ(W.Type, 0u);
  EXPECT_EQ(W.ByteSize, 2u);
}

TEST(WODRoundTrip, MultipleOpsInPool) {
  // Encode push rbx + sub rsp, 32 into one pool, then decode both.
  SmallVector<uint8_t, 8> Pool;
  WinEH::Instruction Push(UOP_PushNonVol, nullptr, 3, 0);
  WinEH::Instruction Alloc(UOP_AllocSmall, nullptr, 0, 32);
  Win64EH::EncodeWOD(Push, Pool);
  Win64EH::EncodeWOD(Alloc, Pool);
  EXPECT_EQ(Pool.size(), 2u); // 1 byte each

  auto W0 = Win64EH::decodeWOD(Pool, 0);
  ASSERT_TRUE(!!W0);
  EXPECT_EQ(W0->Opcode, WOD_PUSH);
  EXPECT_EQ(W0->Register, 3u);

  auto W1 = Win64EH::decodeWOD(Pool, W0->ByteSize);
  ASSERT_TRUE(!!W1);
  EXPECT_EQ(W1->Opcode, WOD_ALLOC_SMALL);
  EXPECT_EQ(W1->Size, 32u);
}

TEST(WODRoundTrip, Push2NonConsecutive) {
  // push2 rbx, rdi (regs 3, 7 - non-consecutive) => WOD_PUSH2 (2 bytes)
  WinEH::Instruction Inst(UOP_Push2, nullptr, 3, 7, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH2);
  EXPECT_EQ(W.Register, 3u);
  EXPECT_EQ(W.Register2, 7u);
  EXPECT_EQ(W.ByteSize, 2u);
}

TEST(WODRoundTrip, Push2Consecutive) {
  // push2 r12, r13 (regs 12, 13 - consecutive) => WOD_PUSH_CONSECUTIVE_2
  // (1 byte)
  WinEH::Instruction Inst(UOP_Push2, nullptr, 12, 13, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH_CONSECUTIVE_2);
  EXPECT_EQ(W.Register, 12u);
  EXPECT_EQ(W.ByteSize, 1u);
}

TEST(WODRoundTrip, Push2HighRegs) {
  // push2 r14, r8 (regs 14, 8 - non-consecutive, high regs)
  WinEH::Instruction Inst(UOP_Push2, nullptr, 14, 8, 0);
  auto W = roundTrip(Inst);
  EXPECT_EQ(W.Opcode, WOD_PUSH2);
  EXPECT_EQ(W.Register, 14u);
  EXPECT_EQ(W.Register2, 8u);
  EXPECT_EQ(W.ByteSize, 2u);
}
