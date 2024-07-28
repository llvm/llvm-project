//===- llvm/unittest/CodeGen/SelectionDAGAddressAnalysisTest.cpp  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class SelectionDAGAddressAnalysisTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "@g = global i32 0\n"
                         "@g_alias = alias i32, i32* @g\n"
                         "define i32 @f() {\n"
                         "  %1 = load i32, i32* @g\n"
                         "  ret i32 %1\n"
                         "}";

    Triple TargetTriple("aarch64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    // FIXME: These tests do not depend on AArch64 specifically, but we have to
    // initialize a target. A skeleton Target for unittests would allow us to
    // always run these tests.
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine("AArch64", "", "+sve", Options, std::nullopt,
                               std::nullopt, CodeGenOptLevel::Aggressive)));
    if (!TM)
      GTEST_SKIP();

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("F?");
    G = M->getGlobalVariable("g");
    if (!G)
      report_fatal_error("G?");
    AliasedG = M->getNamedAlias("g_alias");
    if (!AliasedG)
      report_fatal_error("AliasedG?");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI.getContext(), 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, MMI,
              nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  GlobalVariable *G;
  GlobalAlias *AliasedG;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

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
