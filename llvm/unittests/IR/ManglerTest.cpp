//===- llvm/unittest/IR/ManglerTest.cpp - Mangler unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Mangler.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::string mangleStr(StringRef IRName, Mangler &Mang,
                             const DataLayout &DL) {
  std::string Mangled;
  raw_string_ostream SS(Mangled);
  Mang.getNameWithPrefix(SS, IRName, DL);
  SS.flush();
  return Mangled;
}

static std::string mangleFunc(StringRef IRName,
                              GlobalValue::LinkageTypes Linkage,
                              llvm::CallingConv::ID CC, Module &Mod,
                              Mangler &Mang) {
  Type *VoidTy = Type::getVoidTy(Mod.getContext());
  Type *I32Ty = Type::getInt32Ty(Mod.getContext());
  FunctionType *FTy =
      FunctionType::get(VoidTy, {I32Ty, I32Ty, I32Ty}, /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Linkage, IRName, &Mod);
  F->setCallingConv(CC);
  std::string Mangled;
  raw_string_ostream SS(Mangled);
  Mang.getNameWithPrefix(SS, F, false);
  SS.flush();
  F->eraseFromParent();
  return Mangled;
}

namespace {

TEST(ManglerTest, MachO) {
  LLVMContext Ctx;
  DataLayout DL("m:o"); // macho
  Module Mod("test", Ctx);
  Mod.setDataLayout(DL);
  Mangler Mang;
  EXPECT_EQ(mangleStr("foo", Mang, DL), "_foo");
  EXPECT_EQ(mangleStr("\01foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("?foo", Mang, DL), "_?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "_foo");
  EXPECT_EQ(mangleFunc("?foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "_?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::PrivateLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "L_foo");
}

TEST(ManglerTest, WindowsX86) {
  LLVMContext Ctx;
  DataLayout DL("m:x-p:32:32"); // 32-bit windows
  Module Mod("test", Ctx);
  Mod.setDataLayout(DL);
  Mangler Mang;
  EXPECT_EQ(mangleStr("foo", Mang, DL), "_foo");
  EXPECT_EQ(mangleStr("\01foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("?foo", Mang, DL), "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "_foo");
  EXPECT_EQ(mangleFunc("?foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::PrivateLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "L_foo");

  // Test calling conv mangling.
  EXPECT_EQ(mangleFunc("stdcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_StdCall, Mod, Mang),
            "_stdcall@12");
  EXPECT_EQ(mangleFunc("fastcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_FastCall, Mod, Mang),
            "@fastcall@12");
  EXPECT_EQ(mangleFunc("vectorcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_VectorCall, Mod, Mang),
            "vectorcall@@12");

  // Adding a '?' prefix blocks calling convention mangling.
  EXPECT_EQ(mangleFunc("?fastcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_FastCall, Mod, Mang),
            "?fastcall");
}

TEST(ManglerTest, WindowsX64) {
  LLVMContext Ctx;
  DataLayout DL("m:w-p:64:64"); // windows
  Module Mod("test", Ctx);
  Mod.setDataLayout(DL);
  Mangler Mang;
  EXPECT_EQ(mangleStr("foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("\01foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("?foo", Mang, DL), "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "foo");
  EXPECT_EQ(mangleFunc("?foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::PrivateLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            ".Lfoo");

  // Test calling conv mangling.
  EXPECT_EQ(mangleFunc("stdcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_StdCall, Mod, Mang),
            "stdcall");
  EXPECT_EQ(mangleFunc("fastcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_FastCall, Mod, Mang),
            "fastcall");
  EXPECT_EQ(mangleFunc("vectorcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_VectorCall, Mod, Mang),
            "vectorcall@@24");

  // Adding a '?' prefix blocks calling convention mangling.
  EXPECT_EQ(mangleFunc("?vectorcall", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::X86_VectorCall, Mod, Mang),
            "?vectorcall");
}

TEST(ManglerTest, XCOFF) {
  LLVMContext Ctx;
  DataLayout DL("m:a"); // XCOFF/AIX
  Module Mod("test", Ctx);
  Mod.setDataLayout(DL);
  Mangler Mang;
  EXPECT_EQ(mangleStr("foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("\01foo", Mang, DL), "foo");
  EXPECT_EQ(mangleStr("?foo", Mang, DL), "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "foo");
  EXPECT_EQ(mangleFunc("?foo", llvm::GlobalValue::ExternalLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "?foo");
  EXPECT_EQ(mangleFunc("foo", llvm::GlobalValue::PrivateLinkage,
                       llvm::CallingConv::C, Mod, Mang),
            "L..foo");
}

} // end anonymous namespace
