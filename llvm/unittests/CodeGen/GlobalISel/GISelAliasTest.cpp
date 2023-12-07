//===- GISelAliasTest.cpp--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/LoadStoreOpt.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/AtomicOrdering.h"
#include "gtest/gtest.h"

namespace {

// Test simple aliasing.
TEST_F(AArch64GISelMITest, SimpleAlias) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT S64 = LLT::scalar(64);
  LLT P0 = LLT::pointer(0, 64);

  auto Base = B.buildIntToPtr(P0, Copies[0]);
  auto Base2 = B.buildIntToPtr(P0, Copies[1]);
  // These two addresses are identical.
  auto Addr = B.buildPtrAdd(P0, Base, B.buildConstant(S64, 8));
  auto Addr2 = B.buildPtrAdd(P0, Base, B.buildConstant(S64, 8));

  MachinePointerInfo PtrInfo;
  auto *LoadMMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::Flags::MOLoad, S64, Align());
  auto Ld1 = B.buildLoad(S64, Addr, *LoadMMO);
  auto Ld2 = B.buildLoad(S64, Addr2, *LoadMMO);

  // We expect the same address to return alias.
  EXPECT_TRUE(GISelAddressing::instMayAlias(*Ld1, *Ld2, *MRI, nullptr));

  // Expect both being volatile to say alias, since we can't reorder them.
  auto *LoadVolMMO = MF->getMachineMemOperand(
      LoadMMO,
      MachineMemOperand::Flags::MOLoad | MachineMemOperand::Flags::MOVolatile);
  // Pick a different address so we don't trivially match the alias case above.
  auto VolLd1 = B.buildLoad(S64, Addr, *LoadVolMMO);
  auto VolLd2 = B.buildLoad(S64, Base2, *LoadVolMMO);
  EXPECT_TRUE(GISelAddressing::instMayAlias(*VolLd1, *VolLd2, *MRI, nullptr));

  // Same for atomics.
  auto *LoadAtomicMMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::Flags::MOLoad, S64, Align(8), AAMDNodes(),
      nullptr, SyncScope::System, AtomicOrdering::Acquire);
  auto AtomicLd1 = B.buildLoad(S64, Addr, *LoadAtomicMMO);
  auto AtomicLd2 = B.buildLoad(S64, Base2, *LoadAtomicMMO);
  EXPECT_TRUE(
      GISelAddressing::instMayAlias(*AtomicLd1, *AtomicLd2, *MRI, nullptr));

  // Invariant memory with stores.
  auto *LoadInvariantMMO = MF->getMachineMemOperand(
      LoadMMO,
      MachineMemOperand::Flags::MOLoad | MachineMemOperand::Flags::MOInvariant);
  auto InvariantLd = B.buildLoad(S64, Addr, *LoadInvariantMMO);
  auto Store = B.buildStore(B.buildConstant(S64, 0), Base2, PtrInfo, Align());
  EXPECT_FALSE(
      GISelAddressing::instMayAlias(*InvariantLd, *Store, *MRI, nullptr));
}

// Test aliasing checks for same base + different offsets.
TEST_F(AArch64GISelMITest, OffsetAliasing) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT S64 = LLT::scalar(64);
  LLT P0 = LLT::pointer(0, 64);

  auto Base = B.buildIntToPtr(P0, Copies[0]);
  auto Addr = B.buildPtrAdd(P0, Base, B.buildConstant(S64, 8));
  auto Addr2 = B.buildPtrAdd(P0, Base, B.buildConstant(S64, 16));

  MachinePointerInfo PtrInfo;
  auto *LoadMMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::Flags::MOLoad, S64, Align());
  auto Ld1 = B.buildLoad(S64, Addr, *LoadMMO);
  auto Ld2 = B.buildLoad(S64, Addr2, *LoadMMO);

  // The offset between the two addresses is >= than the size of access.
  // Can't alias.
  EXPECT_FALSE(GISelAddressing::instMayAlias(*Ld1, *Ld2, *MRI, nullptr));
  EXPECT_FALSE(GISelAddressing::instMayAlias(*Ld2, *Ld1, *MRI, nullptr));

  auto Addr3 = B.buildPtrAdd(P0, Base, B.buildConstant(S64, 4));
  auto Ld3 = B.buildLoad(S64, Addr3, *LoadMMO);
  // Offset of 4 is < the size of access, 8 bytes.
  EXPECT_TRUE(GISelAddressing::instMayAlias(*Ld1, *Ld3, *MRI, nullptr));
}

// Test aliasing checks for frame indexes.
TEST_F(AArch64GISelMITest, FrameIndexAliasing) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  LLT S64 = LLT::scalar(64);
  LLT P0 = LLT::pointer(0, 64);

  auto &MFI = MF->getFrameInfo();
  auto FixedFI1 = MFI.CreateFixedObject(8, 0, true);
  auto FixedFI2 = MFI.CreateFixedObject(8, 8, true);

  auto FI1 = MFI.CreateStackObject(8, Align(8), false);
  auto GFI1 = B.buildFrameIndex(P0, FI1);
  // This G_FRAME_INDEX is separate but refers to the same index.
  auto GFI2 = B.buildFrameIndex(P0, FI1);

  MachinePointerInfo PtrInfo;
  auto *LoadMMO = MF->getMachineMemOperand(
      PtrInfo, MachineMemOperand::Flags::MOLoad, S64, Align());
  auto Ld1 = B.buildLoad(S64, GFI1, *LoadMMO);
  auto Ld2 = B.buildLoad(S64, GFI2, *LoadMMO);

  // The offset between the two addresses is >= than the size of access.
  // Can't alias.
  EXPECT_FALSE(GISelAddressing::instMayAlias(*Ld1, *Ld2, *MRI, nullptr));


  auto GFixedFI1 = B.buildFrameIndex(P0, FixedFI1);
  auto GFixedFI2 = B.buildFrameIndex(P0, FixedFI2);
  auto FixedFILd1 = B.buildLoad(S64, GFixedFI1, *LoadMMO);
  auto FixedFILd2 = B.buildLoad(S64, GFixedFI2, *LoadMMO);
  // If we have two different FrameIndex bases, but at least one is not a fixed
  // object, then we can say they don't alias. If both were fixed, then we could
  // have multiple frameindex slots being accessed at once since their relative
  // positions are known. However, if one is not fixed, then they can't alias
  // because non-fixed FIs are only given offsets during PEI.
  EXPECT_FALSE(GISelAddressing::instMayAlias(*FixedFILd1, *Ld1, *MRI, nullptr));
  EXPECT_TRUE(
      GISelAddressing::instMayAlias(*FixedFILd1, *FixedFILd2, *MRI, nullptr));
}

} // namespace
