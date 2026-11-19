//===- llvm/unittest/IR/CoreBindings.cpp - Tests for C-API bindings -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/Config/llvm-config.h"
#include "gtest/gtest.h"
#include <cstring>
#include <string>

namespace {

TEST(CoreBindings, VersionTest) {
  // Test ability to ignore output parameters
  LLVMGetVersion(nullptr, nullptr, nullptr);

  unsigned Major, Minor, Patch;
  LLVMGetVersion(&Major, &Minor, &Patch);
  EXPECT_EQ(Major, (unsigned)LLVM_VERSION_MAJOR);
  EXPECT_EQ(Minor, (unsigned)LLVM_VERSION_MINOR);
  EXPECT_EQ(Patch, (unsigned)LLVM_VERSION_PATCH);
}

TEST(CoreBindings, IntrinsicGetOverloadTypes) {
  LLVMContextRef C = LLVMContextCreate();
  LLVMModuleRef M = LLVMModuleCreateWithNameInContext("m", C);
  LLVMTypeRef F64 = LLVMDoubleTypeInContext(C);
  LLVMTypeRef I32 = LLVMInt32TypeInContext(C);

  // An overloaded intrinsic, from its base name and the type of a call.
  const char *Powi = "llvm.powi";
  unsigned ID = LLVMLookupIntrinsicID(Powi, strlen(Powi));
  ASSERT_NE(ID, 0u);
  LLVMTypeRef Params[] = {F64, I32};
  LLVMTypeRef FnTy = LLVMFunctionType(F64, Params, 2, false);
  size_t Count = 0;
  EXPECT_TRUE(LLVMIntrinsicGetOverloadTypes(ID, FnTy, nullptr, &Count));
  ASSERT_EQ(Count, 2u);
  LLVMTypeRef Tys[2] = {nullptr, nullptr};
  EXPECT_TRUE(LLVMIntrinsicGetOverloadTypes(ID, FnTy, Tys, &Count));
  EXPECT_EQ(Tys[0], F64);
  EXPECT_EQ(Tys[1], I32);
  LLVMValueRef Decl = LLVMGetIntrinsicDeclaration(M, ID, Tys, Count);
  size_t Len;
  const char *Name = LLVMGetValueName2(Decl, &Len);
  EXPECT_EQ(std::string(Name, Len), "llvm.powi.f64.i32");
  EXPECT_EQ(LLVMGlobalGetValueType(Decl), FnTy);

  // A signature that is not valid for the intrinsic.
  LLVMTypeRef BadTy = LLVMFunctionType(I32, Params, 2, false);
  EXPECT_FALSE(LLVMIntrinsicGetOverloadTypes(ID, BadTy, nullptr, &Count));

  // An intrinsic that is not overloaded has no overload types.
  const char *Trap = "llvm.trap";
  unsigned TrapID = LLVMLookupIntrinsicID(Trap, strlen(Trap));
  LLVMTypeRef TrapTy =
      LLVMFunctionType(LLVMVoidTypeInContext(C), nullptr, 0, false);
  EXPECT_TRUE(LLVMIntrinsicGetOverloadTypes(TrapID, TrapTy, nullptr, &Count));
  EXPECT_EQ(Count, 0u);

  // Not an intrinsic.
  EXPECT_FALSE(LLVMIntrinsicGetOverloadTypes(0, FnTy, nullptr, &Count));

  LLVMDisposeModule(M);
  LLVMContextDispose(C);
}

} // end anonymous namespace
