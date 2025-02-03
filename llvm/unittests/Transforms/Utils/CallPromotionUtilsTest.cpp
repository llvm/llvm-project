//===- CallPromotionUtilsTest.cpp - CallPromotionUtils unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/ProfileData/PGOCtxProfWriter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UtilsTests", errs());
  return Mod;
}

// Returns a constant representing the vtable's address point specified by the
// offset.
static Constant *getVTableAddressPointOffset(GlobalVariable *VTable,
                                             uint32_t AddressPointOffset) {
  Module &M = *VTable->getParent();
  LLVMContext &Context = M.getContext();
  assert(AddressPointOffset <
             M.getDataLayout().getTypeAllocSize(VTable->getValueType()) &&
         "Out-of-bound access");

  return ConstantExpr::getInBoundsGetElementPtr(
      Type::getInt8Ty(Context), VTable,
      llvm::ConstantInt::get(Type::getInt32Ty(Context), AddressPointOffset));
}

TEST(CallPromotionUtilsTest, TryPromoteCall) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

@_ZTV4Impl = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%class.Impl*)* @_ZN4Impl3RunEv to i8*)] }

define void @f() {
entry:
  %o = alloca %class.Impl
  %base = getelementptr %class.Impl, %class.Impl* %o, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
  %f = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 1
  store i32 3, i32* %f
  %base.i = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  %c = bitcast %class.Interface* %base.i to void (%class.Interface*)***
  %vtable.i = load void (%class.Interface*)**, void (%class.Interface*)*** %c
  %fp = load void (%class.Interface*)*, void (%class.Interface*)** %vtable.i
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}

declare void @_ZN4Impl3RunEv(%class.Impl* %this)
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_TRUE(IsPromoted);
  GV = M->getNamedValue("_ZN4Impl3RunEv");
  ASSERT_TRUE(GV);
  auto *F1 = dyn_cast<Function>(GV);
  EXPECT_EQ(F1, CI->getCalledFunction());
}

TEST(CallPromotionUtilsTest, TryPromoteCall_NoFPLoad) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

define void @f(void (%class.Interface*)* %fp, %class.Interface* nonnull %base.i) {
entry:
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

TEST(CallPromotionUtilsTest, TryPromoteCall_NoVTablePtrLoad) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

define void @f(void (%class.Interface*)** %vtable.i, %class.Interface* nonnull %base.i) {
entry:
  %fp = load void (%class.Interface*)*, void (%class.Interface*)** %vtable.i
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

TEST(CallPromotionUtilsTest, TryPromoteCall_NoVTableInitFound) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

define void @f() {
entry:
  %o = alloca %class.Impl
  %f = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 1
  store i32 3, i32* %f
  %base.i = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  %c = bitcast %class.Interface* %base.i to void (%class.Interface*)***
  %vtable.i = load void (%class.Interface*)**, void (%class.Interface*)*** %c
  %fp = load void (%class.Interface*)*, void (%class.Interface*)** %vtable.i
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}

declare void @_ZN4Impl3RunEv(%class.Impl* %this)
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

TEST(CallPromotionUtilsTest, TryPromoteCall_EmptyVTable) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

@_ZTV4Impl = external global { [3 x i8*] }

define void @f() {
entry:
  %o = alloca %class.Impl
  %base = getelementptr %class.Impl, %class.Impl* %o, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
  %f = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 1
  store i32 3, i32* %f
  %base.i = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  %c = bitcast %class.Interface* %base.i to void (%class.Interface*)***
  %vtable.i = load void (%class.Interface*)**, void (%class.Interface*)*** %c
  %fp = load void (%class.Interface*)*, void (%class.Interface*)** %vtable.i
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}

declare void @_ZN4Impl3RunEv(%class.Impl* %this)
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

TEST(CallPromotionUtilsTest, TryPromoteCall_NullFP) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

@_ZTV4Impl = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* null] }

define void @f() {
entry:
  %o = alloca %class.Impl
  %base = getelementptr %class.Impl, %class.Impl* %o, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
  %f = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 1
  store i32 3, i32* %f
  %base.i = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  %c = bitcast %class.Interface* %base.i to void (%class.Interface*)***
  %vtable.i = load void (%class.Interface*)**, void (%class.Interface*)*** %c
  %fp = load void (%class.Interface*)*, void (%class.Interface*)** %vtable.i
  call void %fp(%class.Interface* nonnull %base.i)
  ret void
}

declare void @_ZN4Impl3RunEv(%class.Impl* %this)
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

// Based on clang/test/CodeGenCXX/member-function-pointer-calls.cpp
TEST(CallPromotionUtilsTest, TryPromoteCall_MemberFunctionCalls) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%struct.A = type { i32 (...)** }

@_ZTV1A = linkonce_odr unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast (i32 (%struct.A*)* @_ZN1A3vf1Ev to i8*), i8* bitcast (i32 (%struct.A*)* @_ZN1A3vf2Ev to i8*)] }, align 8

define i32 @_Z2g1v() {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast %struct.A* %a to i8*
  %1 = getelementptr %struct.A, %struct.A* %a, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8
  %2 = bitcast %struct.A* %a to i8*
  %3 = bitcast i8* %2 to i8**
  %vtable.i = load i8*, i8** %3, align 8
  %4 = bitcast i8* %vtable.i to i32 (%struct.A*)**
  %memptr.virtualfn.i = load i32 (%struct.A*)*, i32 (%struct.A*)** %4, align 8
  %call.i = call i32 %memptr.virtualfn.i(%struct.A* %a)
  ret i32 %call.i
}

define i32 @_Z2g2v() {
entry:
  %a = alloca %struct.A, align 8
  %0 = bitcast %struct.A* %a to i8*
  %1 = getelementptr %struct.A, %struct.A* %a, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8
  %2 = bitcast %struct.A* %a to i8*
  %3 = bitcast i8* %2 to i8**
  %vtable.i = load i8*, i8** %3, align 8
  %4 = getelementptr i8, i8* %vtable.i, i64 8
  %5 = bitcast i8* %4 to i32 (%struct.A*)**
  %memptr.virtualfn.i = load i32 (%struct.A*)*, i32 (%struct.A*)** %5, align 8
  %call.i = call i32 %memptr.virtualfn.i(%struct.A* %a)
  ret i32 %call.i
}

declare i32 @_ZN1A3vf1Ev(%struct.A* %this)
declare i32 @_ZN1A3vf2Ev(%struct.A* %this)
)IR");

  auto *GV = M->getNamedValue("_Z2g1v");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted1 = tryPromoteCall(*CI);
  EXPECT_TRUE(IsPromoted1);
  GV = M->getNamedValue("_ZN1A3vf1Ev");
  ASSERT_TRUE(GV);
  F = dyn_cast<Function>(GV);
  EXPECT_EQ(F, CI->getCalledFunction());

  GV = M->getNamedValue("_Z2g2v");
  ASSERT_TRUE(GV);
  F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Inst = &F->front().front();
  AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted2 = tryPromoteCall(*CI);
  EXPECT_TRUE(IsPromoted2);
  GV = M->getNamedValue("_ZN1A3vf2Ev");
  ASSERT_TRUE(GV);
  F = dyn_cast<Function>(GV);
  EXPECT_EQ(F, CI->getCalledFunction());
}

// Check that it isn't crashing due to missing promotion legality.
TEST(CallPromotionUtilsTest, TryPromoteCall_Legality) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
%struct1 = type <{ i32, i64 }>
%struct2 = type <{ i32, i64 }>

%class.Impl = type <{ %class.Interface, i32, [4 x i8] }>
%class.Interface = type { i32 (...)** }

@_ZTV4Impl = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (%struct2 (%class.Impl*)* @_ZN4Impl3RunEv to i8*)] }

define %struct1 @f() {
entry:
  %o = alloca %class.Impl
  %base = getelementptr %class.Impl, %class.Impl* %o, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
  %f = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 1
  store i32 3, i32* %f
  %base.i = getelementptr inbounds %class.Impl, %class.Impl* %o, i64 0, i32 0
  %c = bitcast %class.Interface* %base.i to %struct1 (%class.Interface*)***
  %vtable.i = load %struct1 (%class.Interface*)**, %struct1 (%class.Interface*)*** %c
  %fp = load %struct1 (%class.Interface*)*, %struct1 (%class.Interface*)** %vtable.i
  %rv = call %struct1 %fp(%class.Interface* nonnull %base.i)
  ret %struct1 %rv
}

declare %struct2 @_ZN4Impl3RunEv(%class.Impl* %this)
)IR");

  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = &*++F->front().rbegin();
  auto *CI = dyn_cast<CallInst>(Inst);
  ASSERT_TRUE(CI);
  ASSERT_FALSE(CI->getCalledFunction());
  bool IsPromoted = tryPromoteCall(*CI);
  EXPECT_FALSE(IsPromoted);
}

TEST(CallPromotionUtilsTest, promoteCallWithVTableCmp) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV5Base1 = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN5Base15func0Ev, ptr @_ZN5Base15func1Ev] }, !type !0
@_ZTV8Derived1 = constant { [4 x ptr], [3 x ptr] } { [4 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZN5Base15func0Ev, ptr @_ZN5Base15func1Ev], [3 x ptr] [ptr null, ptr null, ptr @_ZN5Base25func2Ev] }, !type !0, !type !1, !type !2
@_ZTV8Derived2 = constant { [3 x ptr], [3 x ptr], [4 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN5Base35func3Ev], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr @_ZN5Base25func2Ev], [4 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr null, ptr @_ZN5Base15func0Ev, ptr @_ZN5Base15func1Ev] }, !type !3, !type !4, !type !5, !type !6

define i32 @testfunc(ptr %d) {
entry:
  %vtable = load ptr, ptr %d, !prof !7
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
  %0 = load ptr, ptr %vfn
  %call = tail call i32 %0(ptr %d), !prof !8
  ret i32 %call
}

define i32 @_ZN5Base15func1Ev(ptr %this) {
entry:
  ret i32 2
}

declare i32 @_ZN5Base25func2Ev(ptr)
declare i32 @_ZN5Base15func0Ev(ptr)
declare void @_ZN5Base35func3Ev(ptr)

!0 = !{i64 16, !"_ZTS5Base1"}
!1 = !{i64 48, !"_ZTS5Base2"}
!2 = !{i64 16, !"_ZTS8Derived1"}
!3 = !{i64 64, !"_ZTS5Base1"}
!4 = !{i64 40, !"_ZTS5Base2"}
!5 = !{i64 16, !"_ZTS5Base3"}
!6 = !{i64 16, !"_ZTS8Derived2"}
!7 = !{!"VP", i32 2, i64 1600, i64 -9064381665493407289, i64 800, i64 5035968517245772950, i64 500, i64 3215870116411581797, i64 300}
!8 = !{!"VP", i32 0, i64 1600, i64 6804820478065511155, i64 1600})IR");

  Function *F = M->getFunction("testfunc");
  CallInst *CI = dyn_cast<CallInst>(&*std::next(F->front().rbegin()));
  ASSERT_TRUE(CI && CI->isIndirectCall());

  // Create the constant and the branch weights
  SmallVector<Constant *, 3> VTableAddressPoints;

  for (auto &[VTableName, AddressPointOffset] : {std::pair{"_ZTV5Base1", 16},
                                                 {"_ZTV8Derived1", 16},
                                                 {"_ZTV8Derived2", 64}})
    VTableAddressPoints.push_back(getVTableAddressPointOffset(
        M->getGlobalVariable(VTableName), AddressPointOffset));

  MDBuilder MDB(C);
  MDNode *BranchWeights = MDB.createBranchWeights(1600, 0);

  size_t OrigEntryBBSize = F->front().size();

  LoadInst *VPtr = dyn_cast<LoadInst>(&*F->front().begin());

  Function *Callee = M->getFunction("_ZN5Base15func1Ev");
  // Tests that promoted direct call is returned.
  CallBase &DirectCB = promoteCallWithVTableCmp(
      *CI, VPtr, Callee, VTableAddressPoints, BranchWeights);
  EXPECT_EQ(DirectCB.getCalledOperand(), Callee);

  // Promotion inserts 3 icmp instructions and 2 or instructions, and removes
  // 1 call instruction from the entry block.
  EXPECT_EQ(F->front().size(), OrigEntryBBSize + 4);
}

TEST(CallPromotionUtilsTest, PromoteWithIcmpAndCtxProf) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C,
                                      R"IR(
define i32 @testfunc1(ptr %d) !guid !0 {
  call void @llvm.instrprof.increment(ptr @testfunc1, i64 0, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @testfunc1, i64 0, i32 1, i32 0, ptr %d)
  %call = call i32 %d()
  ret i32 %call
}

define i32 @f1() !guid !1 {
  call void @llvm.instrprof.increment(ptr @f1, i64 0, i32 1, i32 0)
  ret i32 2
}

define i32 @f2() !guid !2 {
  call void @llvm.instrprof.increment(ptr @f2, i64 0, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @f2, i64 0, i32 1, i32 0, ptr @f4)
  %r = call i32 @f4()
  ret i32 %r
}

define i32 @testfunc2(ptr %p) !guid !4 {
  call void @llvm.instrprof.increment(ptr @testfunc2, i64 0, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @testfunc2, i64 0, i32 1, i32 0, ptr @testfunc1)
  %r = call i32 @testfunc1(ptr %p)
  ret i32 %r
}

declare i32 @f3()

define i32 @f4() !guid !3 {
  ret i32 3
}

!0 = !{i64 1000}
!1 = !{i64 1001}
!2 = !{i64 1002}
!3 = !{i64 1004}
!4 = !{i64 1005}
)IR");

  const char *Profile = R"json(
    [
    {
      "Guid": 1000,
      "Counters": [1],
      "Callsites": [
        [{ "Guid": 1001,
            "Counters": [10]}, 
          { "Guid": 1002,
            "Counters": [11],
            "Callsites": [[{"Guid": 1004, "Counters":[13]}]]
          },
          { "Guid": 1003,
            "Counters": [12]
          }]]
    },
    {
      "Guid": 1005,
      "Counters": [2],
      "Callsites": [
        [{ "Guid": 1000,
            "Counters": [1],
            "Callsites": [
              [{ "Guid": 1001,
                  "Counters": [101]}, 
                { "Guid": 1002,
                  "Counters": [102],
                  "Callsites": [[{"Guid": 1004, "Counters":[104]}]]
                },
                { "Guid": 1003,
                  "Counters": [103]
                }]]}]]}]
    )json";

  llvm::unittest::TempFile ProfileFile("ctx_profile", "", "", /*Unique=*/true);
  {
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    // "False" means no error.
    ASSERT_FALSE(llvm::createCtxProfFromYAML(Profile, Out));
  }

  ModuleAnalysisManager MAM;
  MAM.registerPass([&]() { return CtxProfAnalysis(ProfileFile.path()); });
  MAM.registerPass([&]() { return PassInstrumentationAnalysis(); });
  auto &CtxProf = MAM.getResult<CtxProfAnalysis>(*M);
  auto *Caller = M->getFunction("testfunc1");
  ASSERT_NE(Caller, nullptr);
  auto *Callee = M->getFunction("f2");
  ASSERT_NE(Callee, nullptr);
  auto *IndirectCS = [&]() -> CallBase * {
    for (auto &BB : *Caller)
      for (auto &I : BB)
        if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isIndirectCall())
          return CB;
    return nullptr;
  }();
  ASSERT_NE(IndirectCS, nullptr);
  promoteCallWithIfThenElse(*IndirectCS, *Callee, CtxProf);

  std::string Str;
  raw_string_ostream OS(Str);
  CtxProfAnalysisPrinterPass Printer(OS);
  Printer.run(*M, MAM);
  const char *Expected = R"yaml(
- Guid:            1000
  Counters:        [ 1, 11, 22 ]
  Callsites:
    - - Guid:            1001
        Counters:        [ 10 ]
      - Guid:            1003
        Counters:        [ 12 ]
    - - Guid:            1002
        Counters:        [ 11 ]
        Callsites:
          - - Guid:            1004
              Counters:        [ 13 ]
- Guid:            1005
  Counters:        [ 2 ]
  Callsites:
    - - Guid:            1000
        Counters:        [ 1, 102, 204 ]
        Callsites:
          - - Guid:            1001
              Counters:        [ 101 ]
            - Guid:            1003
              Counters:        [ 103 ]
          - - Guid:            1002
              Counters:        [ 102 ]
              Callsites:
                - - Guid:            1004
                    Counters:        [ 104 ]
)yaml";
  EXPECT_EQ(Expected, Str);
}
