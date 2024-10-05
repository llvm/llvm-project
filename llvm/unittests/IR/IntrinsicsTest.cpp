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
  LLVMContext Context;
  std::unique_ptr<Module> M;
  BasicBlock *BB = nullptr;

  void TearDown() override { M.reset(); }

  void SetUp() override {
    M = std::make_unique<Module>("Test", Context);
    auto F = M->getOrInsertFunction(
        "test", FunctionType::get(Type::getVoidTy(Context), false));
    BB = BasicBlock::Create(Context, "", cast<Function>(F.getCallee()));
    EXPECT_NE(BB, nullptr);
  }

public:
  Instruction *makeIntrinsic(Intrinsic::ID ID) const {
    IRBuilder<> Builder(BB);
    SmallVector<Value *, 4> ProcessedArgs;
    auto *Decl = Intrinsic::getDeclaration(M.get(), ID);
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
  static constexpr const char *const NameTable[] = {
      "llvm.foo", "llvm.foo.a", "llvm.foo.b", "llvm.foo.b.a", "llvm.foo.c",
  };

  static constexpr std::pair<const char *, int> Tests[] = {
      {"llvm.foo", 0},     {"llvm.foo.f64", 0}, {"llvm.foo.b", 2},
      {"llvm.foo.b.a", 3}, {"llvm.foo.c", 4},   {"llvm.foo.c.f64", 4},
      {"llvm.bar", -1},
  };

  for (const auto &[Name, ExpectedIdx] : Tests) {
    int Idx = Intrinsic::lookupLLVMIntrinsicByName(NameTable, Name);
    EXPECT_EQ(ExpectedIdx, Idx);
    if (!StringRef(Name).starts_with("llvm.foo"))
      continue;
    Idx = Intrinsic::lookupLLVMIntrinsicByName(NameTable, Name, "foo");
    EXPECT_EQ(ExpectedIdx, Idx);
  }
}

// Test case to demonstrate potential conflicts between overloaded and non-
// overloaded intrinsics. The name match works by essentially dividing then
// name into . separated components and doing successive search for each
// component. When a search fails, the lowest component of the matching
// range for the previous component is returned.
TEST(IntrinsicNameLookup, OverloadConflict) {
  // Assume possible mangled type strings are just .f32 and .i32.
  static constexpr const char *const NameTable[] = {
      "llvm.foo",
      "llvm.foo.f32",
      "llvm.foo.i32",
  };

  // Here, first we match llvm.foo and our search window is [0,2]. Then we try
  // to match llvm.foo.f64 and there is no match, so it returns the low of the
  // last match. So this lookup works as expected.
  int Idx = Intrinsic::lookupLLVMIntrinsicByName(NameTable, "llvm.foo.f64");
  EXPECT_EQ(Idx, 0);

  // Now imagine if llvm.foo has 2 mangling suffixes, .f32 and .f64. The search
  // will first match llvm.foo to [0, 2] and then llvm.foo.f32 to [1,1] and then
  // not find any match for llvm.foo.f32.f64. So it will return the low of the
  // last match, which is llvm.foo.f32. However, the intent was to match
  // llvm.foo. So the presence of llvm.foo.f32 eliminated the possibility of
  // matching with llvm.foo. So it seems if we have an intrinsic llvm.foo,
  // another one with the same suffix and a single .suffix is not going to
  // cause problems. If there exists another one with 2 or more suffixes,
  // .suffix0 and .suffix1, its possible that the mangling suffix for llvm.foo
  // might match with .suffix0 and then the match will fail to match llvm.foo.
  // .suffix1 won't be a problem because its the last one so the matcher will
  // try an exact match (in which case exact name in the table was searched for,
  // so its expected to match that entry).
  //
  // This example leads the the following rule: if we have an overloaded
  // intrinsic with name `llvm.foo` and another one with same prefix and one or
  // more suffixes, `llvm.foo[.<suffixN>]+`, then the name search will try to
  // first match against suffix0, then suffix1 etc. If suffix0 can match a
  // mangled type, then the search for an `llvm.foo` with a mangling suffix can
  // match against suffix0, preventing a match with `llvm.foo`. If suffix0
  // cannot match a mangled type, then that cannot happen, so we do not need to
  // check for later suffixes. Generalizing, the `llvm.foo[.suffixN]+` will
  // cause a conflict if the first suffix (.suffix0) can match a mangled type
  // (and then we do not need to check later suffixes) and will not cause a
  // conflict if it cannot (and then again, we do not need to check for later
  // suffixes.)
  Idx = Intrinsic::lookupLLVMIntrinsicByName(NameTable, "llvm.foo.f32.f64");
  EXPECT_EQ(Idx, 1);

  // Here .bar and .f33 do not conflict with the manging suffixes, so the search
  // should match against llvm.foo.
  static constexpr const char *const NameTable1[] = {
      "llvm.foo",
      "llvm.foo.bar",
      "llvm.foo.f33",
  };
  Idx = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.f32.f64");
  EXPECT_EQ(Idx, 0);
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

} // end namespace
