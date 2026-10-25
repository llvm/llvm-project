//===- llvm/unittest/CodeGen/SelectionDAGAddressAnalysisTest.cpp  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "SelectionDAGTestBase.h"
#include "llvm/Analysis/MemoryLocation.h"

namespace llvm {

class SelectionDAGAddressAnalysisTest : public SelectionDAGTestBase {};

TEST_F(SelectionDAGAddressAnalysisTest, scalableOffsets) {
  SDLoc Loc;
  EVT VT = MVT::nxv4i32;
  SDValue Base = DAG->getRegister(1, MVT::i64);
  auto AddOffset = [&](SDValue Ptr, uint64_t Bytes, bool Scalable) {
    return DAG->getMemBasePlusOffset(Ptr, TypeSize::get(Bytes, Scalable), Loc);
  };
  auto Match = [&](SDValue Ptr, bool Scalable = true) {
    SDValue Load = DAG->getLoad(VT, Loc, DAG->getEntryNode(), Ptr,
                                MachinePointerInfo(), Align(4));
    return BaseIndexOffset::match(Load.getNode(), *DAG, Scalable);
  };

  // Compare nested offsets with different fixed and scalable components.
  SDValue Ptr0 = AddOffset(AddOffset(Base, 32, true), 8, false);
  SDValue Ptr1 = AddOffset(AddOffset(Ptr0, 16, true), 4, false);
  auto Base0 = Match(Ptr0);
  auto Base1 = Match(Ptr1);
  StackOffset Diff;
  ASSERT_TRUE(Base0.equalBaseIndex(Base1, *DAG, Diff));
  EXPECT_EQ(Diff, StackOffset::get(4, 16));
  ASSERT_TRUE(Base1.equalBaseIndex(Base0, *DAG, Diff));
  EXPECT_EQ(Diff, StackOffset::get(-4, -16));

  // The fixed-offset interface must not discard the scalable difference.
  int64_t FixedDiff;
  EXPECT_FALSE(Base0.equalBaseIndex(Base1, *DAG, FixedDiff));
  EXPECT_FALSE(
      Match(Ptr0, false).equalBaseIndex(Match(Ptr1, false), *DAG, FixedDiff));

  // Reassociation must not affect the result. Equal scalable components
  // cancel, leaving a difference that the fixed-offset interface can use.
  SDValue Ptr2 = AddOffset(AddOffset(Base, 48, true), 12, false);
  EXPECT_TRUE(Base1.equalBaseIndex(Match(Ptr2), *DAG, FixedDiff));
  EXPECT_EQ(FixedDiff, 0);
  EXPECT_TRUE(
      Base1.equalBaseIndex(Match(AddOffset(Ptr2, 4, false)), *DAG, FixedDiff));
  EXPECT_EQ(FixedDiff, 4);

  SDValue OtherBase = DAG->getRegister(2, MVT::i64);
  EXPECT_FALSE(
      Base0.equalBaseIndex(Match(AddOffset(OtherBase, 48, true)), *DAG, Diff));
}

TEST_F(SelectionDAGAddressAnalysisTest, sameFrameObject) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  TypeSize Offset = TypeSize::getFixed(0);
  SDValue Value = DAG->getConstant(0, Loc, VecVT);
  SDValue Index = DAG->getMemBasePlusOffset(FIPtr, Offset, Loc);
  SDValue Store = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index,
                                PtrInfo.getWithOffset(Offset));
  TypeSize NumBytes = cast<StoreSDNode>(Store)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store.getNode(), LocationSize::precise(NumBytes), Store.getNode(),
      LocationSize::precise(NumBytes), *DAG, IsAlias);

  EXPECT_TRUE(IsValid);
  EXPECT_TRUE(IsAlias);
}

TEST_F(SelectionDAGAddressAnalysisTest, sameFrameObjectUnknownSize) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  TypeSize Offset = TypeSize::getFixed(0);
  SDValue Value = DAG->getConstant(0, Loc, VecVT);
  SDValue Index = DAG->getMemBasePlusOffset(FIPtr, Offset, Loc);
  SDValue Store = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index,
                                PtrInfo.getWithOffset(Offset));

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store.getNode(), LocationSize::beforeOrAfterPointer(), Store.getNode(),
      LocationSize::beforeOrAfterPointer(), *DAG, IsAlias);

  EXPECT_FALSE(IsValid);
}

TEST_F(SelectionDAGAddressAnalysisTest, noAliasingFrameObjects) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <4 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4);
  // <2 x i8>
  auto SubVecVT = EVT::getVectorVT(Context, Int8VT, 2);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  SDValue Value = DAG->getConstant(0, Loc, SubVecVT);
  TypeSize Offset0 = TypeSize::getFixed(0);
  TypeSize Offset1 = SubVecVT.getStoreSize();
  SDValue Index0 = DAG->getMemBasePlusOffset(FIPtr, Offset0, Loc);
  SDValue Index1 = DAG->getMemBasePlusOffset(FIPtr, Offset1, Loc);
  SDValue Store0 = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index0,
                                 PtrInfo.getWithOffset(Offset0));
  SDValue Store1 = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index1,
                                 PtrInfo.getWithOffset(Offset1));
  TypeSize NumBytes0 = cast<StoreSDNode>(Store0)->getMemoryVT().getStoreSize();
  TypeSize NumBytes1 = cast<StoreSDNode>(Store1)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store0.getNode(), LocationSize::precise(NumBytes0), Store1.getNode(),
      LocationSize::precise(NumBytes1), *DAG, IsAlias);

  EXPECT_TRUE(IsValid);
  EXPECT_FALSE(IsAlias);
}

TEST_F(SelectionDAGAddressAnalysisTest, unknownSizeFrameObjects) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <vscale x 4 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4, true);
  // <vscale x 2 x i8>
  auto SubVecVT = EVT::getVectorVT(Context, Int8VT, 2, true);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  SDValue Value = DAG->getConstant(0, Loc, SubVecVT);
  TypeSize Offset1 = SubVecVT.getStoreSize();
  SDValue Index1 = DAG->getMemBasePlusOffset(FIPtr, Offset1, Loc);
  SDValue Store0 =
      DAG->getStore(DAG->getEntryNode(), Loc, Value, FIPtr, PtrInfo);
  SDValue Store1 = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index1,
                                 MachinePointerInfo(PtrInfo.getAddrSpace()));
  TypeSize NumBytes0 = cast<StoreSDNode>(Store0)->getMemoryVT().getStoreSize();
  TypeSize NumBytes1 = cast<StoreSDNode>(Store1)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store0.getNode(), LocationSize::precise(NumBytes0), Store1.getNode(),
      LocationSize::precise(NumBytes1), *DAG, IsAlias);

  EXPECT_FALSE(IsValid);
}

TEST_F(SelectionDAGAddressAnalysisTest, globalWithFrameObject) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <vscale x 4 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4, true);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  SDValue Value = DAG->getConstant(0, Loc, VecVT);
  TypeSize Offset = TypeSize::getFixed(0);
  SDValue Index = DAG->getMemBasePlusOffset(FIPtr, Offset, Loc);
  SDValue Store = DAG->getStore(DAG->getEntryNode(), Loc, Value, Index,
                                PtrInfo.getWithOffset(Offset));
  TypeSize NumBytes = cast<StoreSDNode>(Store)->getMemoryVT().getStoreSize();
  EVT GTy = DAG->getTargetLoweringInfo().getValueType(DAG->getDataLayout(),
                                                      G->getType());
  SDValue GValue = DAG->getConstant(0, Loc, GTy);
  SDValue GAddr = DAG->getGlobalAddress(G, Loc, GTy);
  SDValue GStore = DAG->getStore(DAG->getEntryNode(), Loc, GValue, GAddr,
                                 MachinePointerInfo(G, 0));
  TypeSize GNumBytes = cast<StoreSDNode>(GStore)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store.getNode(), LocationSize::precise(NumBytes), GStore.getNode(),
      LocationSize::precise(GNumBytes), *DAG, IsAlias);

  EXPECT_TRUE(IsValid);
  EXPECT_FALSE(IsAlias);
}

TEST_F(SelectionDAGAddressAnalysisTest, globalWithAliasedGlobal) {
  SDLoc Loc;

  EVT GTy = DAG->getTargetLoweringInfo().getValueType(DAG->getDataLayout(),
                                                      G->getType());
  SDValue GValue = DAG->getConstant(0, Loc, GTy);
  SDValue GAddr = DAG->getGlobalAddress(G, Loc, GTy);
  SDValue GStore = DAG->getStore(DAG->getEntryNode(), Loc, GValue, GAddr,
                                 MachinePointerInfo(G, 0));
  TypeSize GNumBytes = cast<StoreSDNode>(GStore)->getMemoryVT().getStoreSize();

  SDValue AliasedGValue = DAG->getConstant(1, Loc, GTy);
  SDValue AliasedGAddr = DAG->getGlobalAddress(AliasedG, Loc, GTy);
  SDValue AliasedGStore =
      DAG->getStore(DAG->getEntryNode(), Loc, AliasedGValue, AliasedGAddr,
                    MachinePointerInfo(AliasedG, 0));

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      GStore.getNode(), LocationSize::precise(GNumBytes),
      AliasedGStore.getNode(), LocationSize::precise(GNumBytes), *DAG, IsAlias);

  // With some deeper analysis we could detect if G and AliasedG is aliasing or
  // not. But computeAliasing is currently defensive and assumes that a
  // GlobalAlias might alias with any global variable.
  EXPECT_FALSE(IsValid);
}

TEST_F(SelectionDAGAddressAnalysisTest, fixedSizeFrameObjectsWithinDiff) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <vscale x 4 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4, true);
  // <vscale x 2 x i8>
  auto SubVecVT = EVT::getVectorVT(Context, Int8VT, 2, true);
  // <2 x i8>
  auto SubFixedVecVT2xi8 = EVT::getVectorVT(Context, Int8VT, 2);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  SDValue Value0 = DAG->getConstant(0, Loc, SubFixedVecVT2xi8);
  SDValue Value1 = DAG->getConstant(0, Loc, SubVecVT);
  TypeSize Offset0 = TypeSize::getFixed(0);
  TypeSize Offset1 = SubFixedVecVT2xi8.getStoreSize();
  SDValue Index0 = DAG->getMemBasePlusOffset(FIPtr, Offset0, Loc);
  SDValue Index1 = DAG->getMemBasePlusOffset(FIPtr, Offset1, Loc);
  SDValue Store0 = DAG->getStore(DAG->getEntryNode(), Loc, Value0, Index0,
                                 PtrInfo.getWithOffset(Offset0));
  SDValue Store1 = DAG->getStore(DAG->getEntryNode(), Loc, Value1, Index1,
                                 PtrInfo.getWithOffset(Offset1));
  TypeSize NumBytes0 = cast<StoreSDNode>(Store0)->getMemoryVT().getStoreSize();
  TypeSize NumBytes1 = cast<StoreSDNode>(Store1)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store0.getNode(), LocationSize::precise(NumBytes0), Store1.getNode(),
      LocationSize::precise(NumBytes1), *DAG, IsAlias);
  EXPECT_TRUE(IsValid);
  EXPECT_FALSE(IsAlias);

  IsValid = BaseIndexOffset::computeAliasing(
      Store1.getNode(), LocationSize::precise(NumBytes1), Store0.getNode(),
      LocationSize::precise(NumBytes0), *DAG, IsAlias);
  EXPECT_TRUE(IsValid);
  EXPECT_FALSE(IsAlias);
}

TEST_F(SelectionDAGAddressAnalysisTest, fixedSizeFrameObjectsOutOfDiff) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <vscale x 4 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 4, true);
  // <vscale x 2 x i8>
  auto SubVecVT = EVT::getVectorVT(Context, Int8VT, 2, true);
  // <2 x i8>
  auto SubFixedVecVT2xi8 = EVT::getVectorVT(Context, Int8VT, 2);
  // <4 x i8>
  auto SubFixedVecVT4xi8 = EVT::getVectorVT(Context, Int8VT, 4);
  SDValue FIPtr = DAG->CreateStackTemporary(VecVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(*MF, FI);
  SDValue Value0 = DAG->getConstant(0, Loc, SubFixedVecVT4xi8);
  SDValue Value1 = DAG->getConstant(0, Loc, SubVecVT);
  TypeSize Offset0 = TypeSize::getFixed(0);
  TypeSize Offset1 = SubFixedVecVT2xi8.getStoreSize();
  SDValue Index0 = DAG->getMemBasePlusOffset(FIPtr, Offset0, Loc);
  SDValue Index1 = DAG->getMemBasePlusOffset(FIPtr, Offset1, Loc);
  SDValue Store0 = DAG->getStore(DAG->getEntryNode(), Loc, Value0, Index0,
                                 PtrInfo.getWithOffset(Offset0));
  SDValue Store1 = DAG->getStore(DAG->getEntryNode(), Loc, Value1, Index1,
                                 PtrInfo.getWithOffset(Offset1));
  TypeSize NumBytes0 = cast<StoreSDNode>(Store0)->getMemoryVT().getStoreSize();
  TypeSize NumBytes1 = cast<StoreSDNode>(Store1)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store0.getNode(), LocationSize::precise(NumBytes0), Store1.getNode(),
      LocationSize::precise(NumBytes1), *DAG, IsAlias);
  EXPECT_TRUE(IsValid);
  EXPECT_TRUE(IsAlias);
}

TEST_F(SelectionDAGAddressAnalysisTest, twoFixedStackObjects) {
  SDLoc Loc;
  auto Int8VT = EVT::getIntegerVT(Context, 8);
  // <vscale x 2 x i8>
  auto VecVT = EVT::getVectorVT(Context, Int8VT, 2, true);
  // <2 x i8>
  auto FixedVecVT = EVT::getVectorVT(Context, Int8VT, 2);
  SDValue FIPtr0 = DAG->CreateStackTemporary(FixedVecVT);
  SDValue FIPtr1 = DAG->CreateStackTemporary(VecVT);
  int FI0 = cast<FrameIndexSDNode>(FIPtr0.getNode())->getIndex();
  int FI1 = cast<FrameIndexSDNode>(FIPtr1.getNode())->getIndex();
  MachinePointerInfo PtrInfo0 = MachinePointerInfo::getFixedStack(*MF, FI0);
  MachinePointerInfo PtrInfo1 = MachinePointerInfo::getFixedStack(*MF, FI1);
  SDValue Value0 = DAG->getConstant(0, Loc, FixedVecVT);
  SDValue Value1 = DAG->getConstant(0, Loc, VecVT);
  TypeSize Offset0 = TypeSize::getFixed(0);
  SDValue Index0 = DAG->getMemBasePlusOffset(FIPtr0, Offset0, Loc);
  SDValue Index1 = DAG->getMemBasePlusOffset(FIPtr1, Offset0, Loc);
  SDValue Store0 = DAG->getStore(DAG->getEntryNode(), Loc, Value0, Index0,
                                 PtrInfo0.getWithOffset(Offset0));
  SDValue Store1 = DAG->getStore(DAG->getEntryNode(), Loc, Value1, Index1,
                                 PtrInfo1.getWithOffset(Offset0));
  TypeSize NumBytes0 = cast<StoreSDNode>(Store0)->getMemoryVT().getStoreSize();
  TypeSize NumBytes1 = cast<StoreSDNode>(Store1)->getMemoryVT().getStoreSize();

  bool IsAlias;
  bool IsValid = BaseIndexOffset::computeAliasing(
      Store0.getNode(), LocationSize::precise(NumBytes0), Store1.getNode(),
      LocationSize::precise(NumBytes1), *DAG, IsAlias);
  EXPECT_TRUE(IsValid);
  EXPECT_FALSE(IsAlias);
}

} // end namespace llvm
