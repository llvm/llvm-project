//===- CallPromotionUtilsTest.cpp - CallPromotionUtils unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UtilsTests", errs());
  return Mod;
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8
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
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Impl, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %base
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
