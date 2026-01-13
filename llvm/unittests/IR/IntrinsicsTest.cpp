//===- llvm/unittest/IR/IntrinsicsTest.cpp - ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsARM.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/IR/IntrinsicsLoongArch.h"
#include "llvm/IR/IntrinsicsMips.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsS390.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class IntrinsicsTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  BasicBlock *BB = nullptr;

  void SetUp() override {
    M = std::make_unique<Module>("Test", Context);
    auto F = M->getOrInsertFunction(
        "test", FunctionType::get(Type::getVoidTy(Context), false));
    BB = BasicBlock::Create(Context, "", cast<Function>(F.getCallee()));
    EXPECT_NE(BB, nullptr);
  }

  void TearDown() override { M.reset(); }

public:
  Instruction *makeIntrinsic(Intrinsic::ID ID) const {
    IRBuilder<> Builder(BB);
    SmallVector<Value *, 4> ProcessedArgs;
    auto *Decl = Intrinsic::getOrInsertDeclaration(M.get(), ID);
    for (auto *Ty : Decl->getFunctionType()->params()) {
      auto *Val = Constant::getNullValue(Ty);
      ProcessedArgs.push_back(Val);
    }
    return Builder.CreateCall(Decl, ProcessedArgs);
  }
  template <typename T> void checkIsa(const Instruction &I) {
    EXPECT_TRUE(isa<T>(I));
  }
};

TEST(IntrinsicNameLookup, Basic) {
  using namespace Intrinsic;
  EXPECT_EQ(Intrinsic::memcpy, lookupIntrinsicID("llvm.memcpy"));

  // Partial, either between dots or not the last dot are not intrinsics.
  EXPECT_EQ(not_intrinsic, lookupIntrinsicID("llvm.mem"));
  EXPECT_EQ(not_intrinsic, lookupIntrinsicID("llvm.gc"));

  // Look through intrinsic names with internal dots.
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID("llvm.memcpy.inline"));

  // Check that overloaded names are mapped to the underlying ID.
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID("llvm.memcpy.inline.p0.p0.i8"));
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID("llvm.memcpy.inline.p0.p0.i32"));
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID("llvm.memcpy.inline.p0.p0.i64"));
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID("llvm.memcpy.inline.p0.p0.i1024"));
}

TEST(IntrinsicNameLookup, NonNullterminatedStringRef) {
  using namespace Intrinsic;
  // This reproduces an issue where lookupIntrinsicID() can access memory beyond
  // the bounds of the passed in StringRef. For ASAN to catch this as an error,
  // create a StringRef using heap allocated memory and make it not null
  // terminated.

  // ASAN will report a "AddressSanitizer: heap-buffer-overflow" error in
  // `lookupLLVMIntrinsicByName` when LLVM is built with these options:
  //  -DCMAKE_BUILD_TYPE=Debug
  //  -DLLVM_USE_SANITIZER=Address
  //  -DLLVM_OPTIMIZE_SANITIZED_BUILDS=OFF

  // Make an intrinsic name "llvm.memcpy.inline" on the heap.
  std::string Name = "llvm.memcpy.inline";
  assert(Name.size() == 18);
  // Create a StringRef backed by heap allocated memory such that OOB access
  // in that StringRef can be flagged by asan. Here, the String `S` is of size
  // 18, and backed by a heap allocated buffer `Data`, so access to S[18] will
  // be flagged bby asan.
  auto Data = std::make_unique<char[]>(Name.size());
  std::strncpy(Data.get(), Name.data(), Name.size());
  StringRef S(Data.get(), Name.size());
  EXPECT_EQ(memcpy_inline, lookupIntrinsicID(S));
}

// Tests to verify getIntrinsicForClangBuiltin.
TEST(IntrinsicNameLookup, ClangBuiltinLookup) {
  using namespace Intrinsic;
  static constexpr std::tuple<StringRef, StringRef, ID> ClangTests[] = {
      {"__builtin_adjust_trampoline", "", adjust_trampoline},
      {"__builtin_trap", "", trap},
      {"__builtin_arm_chkfeat", "aarch64", aarch64_chkfeat},
      {"__builtin_amdgcn_alignbyte", "amdgcn", amdgcn_alignbyte},
      {"__builtin_amdgcn_workgroup_id_z", "amdgcn", amdgcn_workgroup_id_z},
      {"__builtin_arm_cdp", "arm", arm_cdp},
      {"__builtin_bpf_preserve_type_info", "bpf", bpf_preserve_type_info},
      {"__builtin_HEXAGON_A2_tfr", "hexagon", hexagon_A2_tfr},
      {"__builtin_lasx_xbz_w", "loongarch", loongarch_lasx_xbz_w},
      {"__builtin_mips_bitrev", "mips", mips_bitrev},
      {"__nvvm_add_rn_d", "nvvm", nvvm_add_rn_d},
      {"__builtin_altivec_dss", "ppc", ppc_altivec_dss},
      {"__builtin_riscv_sha512sum1r", "riscv", riscv_sha512sum1r},
      {"__builtin_tend", "s390", s390_tend},
      {"__builtin_ia32_pause", "x86", x86_sse2_pause},

      {"__does_not_exist", "", not_intrinsic},
      {"__does_not_exist", "arm", not_intrinsic},
      {"__builtin_arm_cdp", "", not_intrinsic},
      {"__builtin_arm_cdp", "x86", not_intrinsic},
  };

  for (const auto &[Builtin, Target, ID] : ClangTests)
    EXPECT_EQ(ID, getIntrinsicForClangBuiltin(Target, Builtin));
}

// Tests to verify getIntrinsicForMSBuiltin.
TEST(IntrinsicNameLookup, MSBuiltinLookup) {
  using namespace Intrinsic;
  static constexpr std::tuple<StringRef, StringRef, ID> MSTests[] = {
      {"__dmb", "aarch64", aarch64_dmb},
      {"__dmb", "arm", arm_dmb},
      {"__dmb", "", not_intrinsic},
      {"__does_not_exist", "", not_intrinsic},
      {"__does_not_exist", "arm", not_intrinsic},
  };
  for (const auto &[Builtin, Target, ID] : MSTests)
    EXPECT_EQ(ID, getIntrinsicForMSBuiltin(Target, Builtin));
}

TEST_F(IntrinsicsTest, InstrProfInheritance) {
  auto isInstrProfInstBase = [](const Instruction &I) {
    return isa<InstrProfInstBase>(I);
  };
#define __ISA(TYPE, PARENT)                                                    \
  auto is##TYPE = [&](const Instruction &I) -> bool {                          \
    return isa<TYPE>(I) && is##PARENT(I);                                      \
  }
  __ISA(InstrProfCntrInstBase, InstrProfInstBase);
  __ISA(InstrProfCoverInst, InstrProfCntrInstBase);
  __ISA(InstrProfIncrementInst, InstrProfCntrInstBase);
  __ISA(InstrProfIncrementInstStep, InstrProfIncrementInst);
  __ISA(InstrProfCallsite, InstrProfCntrInstBase);
  __ISA(InstrProfTimestampInst, InstrProfCntrInstBase);
  __ISA(InstrProfValueProfileInst, InstrProfCntrInstBase);
  __ISA(InstrProfMCDCBitmapInstBase, InstrProfInstBase);
  __ISA(InstrProfMCDCBitmapParameters, InstrProfMCDCBitmapInstBase);
  __ISA(InstrProfMCDCTVBitmapUpdate, InstrProfMCDCBitmapInstBase);
#undef __ISA

  std::vector<
      std::pair<Intrinsic::ID, std::function<bool(const Instruction &)>>>
      LeafIDs = {
          {Intrinsic::instrprof_cover, isInstrProfCoverInst},
          {Intrinsic::instrprof_increment, isInstrProfIncrementInst},
          {Intrinsic::instrprof_increment_step, isInstrProfIncrementInstStep},
          {Intrinsic::instrprof_callsite, isInstrProfCallsite},
          {Intrinsic::instrprof_mcdc_parameters,
           isInstrProfMCDCBitmapParameters},
          {Intrinsic::instrprof_mcdc_tvbitmap_update,
           isInstrProfMCDCTVBitmapUpdate},
          {Intrinsic::instrprof_timestamp, isInstrProfTimestampInst},
          {Intrinsic::instrprof_value_profile, isInstrProfValueProfileInst}};
  for (const auto &[ID, Checker] : LeafIDs) {
    auto *Intr = makeIntrinsic(ID);
    EXPECT_TRUE(Checker(*Intr));
  }
}

// Check that getFnAttributes for intrinsics that do not have any function
// attributes correcty returns an empty set.
TEST(IntrinsicAttributes, TestGetFnAttributesBug) {
  using namespace Intrinsic;
  LLVMContext Context;
  AttributeSet AS = getFnAttributes(Context, experimental_guard);
  EXPECT_FALSE(AS.hasAttributes());
}

// Tests non-overloaded intrinsic declaration.
TEST_F(IntrinsicsTest, NonOverloadedIntrinsic) {
  Type *RetTy = Type::getVoidTy(Context);
  SmallVector<Type *, 1> ArgTys;
  ArgTys.push_back(Type::getInt1Ty(Context));

  Function *F = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::assume,
                                                  RetTy, ArgTys);

  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getIntrinsicID(), Intrinsic::assume);
  EXPECT_EQ(F->getReturnType(), RetTy);
  EXPECT_EQ(F->arg_size(), 1u);
  EXPECT_FALSE(F->isVarArg());
  EXPECT_EQ(F->getName(), "llvm.assume");
}

// Tests overloaded intrinsic with automatic type resolution for scalar types.
TEST_F(IntrinsicsTest, OverloadedIntrinsicScalar) {
  Type *RetTy = Type::getInt32Ty(Context);
  SmallVector<Type *, 2> ArgTys;
  ArgTys.push_back(Type::getInt32Ty(Context));
  ArgTys.push_back(Type::getInt32Ty(Context));

  Function *F = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::umax,
                                                  RetTy, ArgTys);

  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getIntrinsicID(), Intrinsic::umax);
  EXPECT_EQ(F->getReturnType(), RetTy);
  EXPECT_EQ(F->arg_size(), 2u);
  EXPECT_FALSE(F->isVarArg());
  EXPECT_EQ(F->getName(), "llvm.umax.i32");
}

// Tests overloaded intrinsic with automatic type resolution for vector types.
TEST_F(IntrinsicsTest, OverloadedIntrinsicVector) {
  Type *RetTy = FixedVectorType::get(Type::getInt32Ty(Context), 4);
  SmallVector<Type *, 2> ArgTys;
  ArgTys.push_back(RetTy);
  ArgTys.push_back(RetTy);

  Function *F = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::umax,
                                                  RetTy, ArgTys);

  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getIntrinsicID(), Intrinsic::umax);
  EXPECT_EQ(F->getReturnType(), RetTy);
  EXPECT_EQ(F->arg_size(), 2u);
  EXPECT_FALSE(F->isVarArg());
  EXPECT_EQ(F->getName(), "llvm.umax.v4i32");
}

// Tests overloaded intrinsic with automatic type resolution for addrspace.
TEST_F(IntrinsicsTest, OverloadedIntrinsicAddressSpace) {
  Type *RetTy = Type::getVoidTy(Context);
  SmallVector<Type *, 4> ArgTys;
  ArgTys.push_back(PointerType::get(Context, 1)); // ptr addrspace(1)
  ArgTys.push_back(Type::getInt32Ty(Context));    // rw
  ArgTys.push_back(Type::getInt32Ty(Context));    // locality
  ArgTys.push_back(Type::getInt32Ty(Context));    // cache type

  Function *F = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::prefetch,
                                                  RetTy, ArgTys);

  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getIntrinsicID(), Intrinsic::prefetch);
  EXPECT_EQ(F->getReturnType(), RetTy);
  EXPECT_EQ(F->arg_size(), 4u);
  EXPECT_FALSE(F->isVarArg());
  EXPECT_EQ(F->getName(), "llvm.prefetch.p1");
}

// Tests vararg intrinsic declaration.
TEST_F(IntrinsicsTest, VarArgIntrinsicStatepoint) {
  Type *RetTy = Type::getTokenTy(Context);
  SmallVector<Type *, 5> ArgTys;
  ArgTys.push_back(Type::getInt64Ty(Context));    // ID
  ArgTys.push_back(Type::getInt32Ty(Context));    // NumPatchBytes
  ArgTys.push_back(PointerType::get(Context, 0)); // Target
  ArgTys.push_back(Type::getInt32Ty(Context));    // NumCallArgs
  ArgTys.push_back(Type::getInt32Ty(Context));    // Flags

  Function *F = Intrinsic::getOrInsertDeclaration(
      M.get(), Intrinsic::experimental_gc_statepoint, RetTy, ArgTys);

  ASSERT_NE(F, nullptr);
  EXPECT_EQ(F->getIntrinsicID(), Intrinsic::experimental_gc_statepoint);
  EXPECT_EQ(F->getReturnType(), RetTy);
  EXPECT_EQ(F->arg_size(), 5u);
  EXPECT_TRUE(F->isVarArg()) << "experimental_gc_statepoint must be vararg";
  EXPECT_EQ(F->getName(), "llvm.experimental.gc.statepoint.p0");
}

// Tests that different overloads create different declarations.
TEST_F(IntrinsicsTest, DifferentOverloads) {
  // i32 version
  Type *RetTy32 = Type::getInt32Ty(Context);
  SmallVector<Type *, 2> ArgTys32;
  ArgTys32.push_back(Type::getInt32Ty(Context));
  ArgTys32.push_back(Type::getInt32Ty(Context));

  Function *Func32 = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::umax,
                                                       RetTy32, ArgTys32);

  // i64 version
  Type *RetTy64 = Type::getInt64Ty(Context);
  SmallVector<Type *, 2> ArgTys64;
  ArgTys64.push_back(Type::getInt64Ty(Context));
  ArgTys64.push_back(Type::getInt64Ty(Context));

  Function *Func64 = Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::umax,
                                                       RetTy64, ArgTys64);

  EXPECT_NE(Func32, Func64)
      << "Different overloads should be different functions";
  EXPECT_EQ(Func32->getName(), "llvm.umax.i32");
  EXPECT_EQ(Func64->getName(), "llvm.umax.i64");
}

// Tests IRBuilder::CreateIntrinsic with overloaded scalar type.
TEST_F(IntrinsicsTest, IRBuilderCreateIntrinsicScalar) {
  IRBuilder<> Builder(BB);

  Type *RetTy = Type::getInt32Ty(Context);
  SmallVector<Value *, 2> Args;
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 10));
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 20));

  CallInst *CI = Builder.CreateIntrinsic(RetTy, Intrinsic::umax, Args);

  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getIntrinsicID(), Intrinsic::umax);
  EXPECT_EQ(CI->getType(), RetTy);
  EXPECT_EQ(CI->arg_size(), 2u);
  EXPECT_FALSE(CI->getCalledFunction()->isVarArg());
}

// Tests IRBuilder::CreateIntrinsic with overloaded vector type.
TEST_F(IntrinsicsTest, IRBuilderCreateIntrinsicVector) {
  IRBuilder<> Builder(BB);

  Type *RetTy = FixedVectorType::get(Type::getInt32Ty(Context), 4);
  SmallVector<Value *, 2> Args;
  Args.push_back(Constant::getNullValue(RetTy));
  Args.push_back(Constant::getNullValue(RetTy));

  CallInst *CI = Builder.CreateIntrinsic(RetTy, Intrinsic::umax, Args);

  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getIntrinsicID(), Intrinsic::umax);
  EXPECT_EQ(CI->getType(), RetTy);
  EXPECT_EQ(CI->arg_size(), 2u);
  EXPECT_FALSE(CI->getCalledFunction()->isVarArg());
}

// Tests IRBuilder::CreateIntrinsic with overloaded address space.
TEST_F(IntrinsicsTest, IRBuilderCreateIntrinsicAddressSpace) {
  IRBuilder<> Builder(BB);

  Type *RetTy = Type::getVoidTy(Context);
  SmallVector<Value *, 4> Args;
  Args.push_back(Constant::getNullValue(
      PointerType::get(Context, 1))); // ptr addrspace(1) null
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 0)); // rw
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 3)); // locality
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 1)); // cache type

  CallInst *CI = Builder.CreateIntrinsic(RetTy, Intrinsic::prefetch, Args);

  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getIntrinsicID(), Intrinsic::prefetch);
  EXPECT_EQ(CI->getType(), RetTy);
  EXPECT_EQ(CI->arg_size(), 4u);
  EXPECT_FALSE(CI->getCalledFunction()->isVarArg());
  EXPECT_EQ(CI->getCalledFunction()->getName(), "llvm.prefetch.p1");
}

// Tests IRBuilder::CreateIntrinsic with vararg intrinsic.
TEST_F(IntrinsicsTest, IRBuilderCreateIntrinsicVarArg) {
  IRBuilder<> Builder(BB);

  // Create a dummy function to call through statepoint
  FunctionType *DummyFnTy = FunctionType::get(Type::getVoidTy(Context), false);
  Function *DummyFn = Function::Create(DummyFnTy, GlobalValue::ExternalLinkage,
                                       "dummy", M.get());

  Type *RetTy = Type::getTokenTy(Context);
  SmallVector<Value *, 5> Args;
  Args.push_back(ConstantInt::get(Type::getInt64Ty(Context), 0)); // ID
  Args.push_back(
      ConstantInt::get(Type::getInt32Ty(Context), 0)); // NumPatchBytes
  Args.push_back(DummyFn);                             // Target
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 0)); // NumCallArgs
  Args.push_back(ConstantInt::get(Type::getInt32Ty(Context), 0)); // Flags

  CallInst *CI = Builder.CreateIntrinsic(
      RetTy, Intrinsic::experimental_gc_statepoint, Args);

  ASSERT_NE(CI, nullptr);
  EXPECT_EQ(CI->getIntrinsicID(), Intrinsic::experimental_gc_statepoint);
  EXPECT_EQ(CI->getType(), RetTy);
  EXPECT_EQ(CI->arg_size(), 5u);
  EXPECT_TRUE(CI->getCalledFunction()->isVarArg())
      << "experimental_gc_statepoint must be vararg";
}

} // end namespace
