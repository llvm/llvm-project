//===- GraphWriterTest.cpp - GraphWriter unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GraphWriter.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

namespace llvm {
namespace {

class GraphWriterTest : public testing::Test {
protected:
  LLVMContext C;

  std::unique_ptr<Module> makeLLVMModule() {
    const char *ModuleStrig = "define i32 @f(i32 %x) {\n"
                              "bb0:\n"
                              "  %y1 = icmp eq i32 %x, 0 \n"
                              "  br i1 %y1, label %bb1, label %bb2 \n"
                              "bb1:\n"
                              "  br label %bb3\n"
                              "bb2:\n"
                              "  br label %bb3\n"
                              "bb3:\n"
                              "  %y2 = phi i32 [0, %bb1], [1, %bb2] \n"
                              "  ret i32 %y2\n"
                              "}\n";
    SMDiagnostic Err;
    return parseAssemblyString(ModuleStrig, Err, C);
  }
};

static void writeCFGToDotFile(Function &F, std::string Name,
                              bool CFGOnly = false) {
  std::error_code EC;
  llvm::unittest::TempDir Tmp("tmpdir", /*Unique=*/true);
  SmallString<128> FileName(Tmp.path().begin(), Tmp.path().end());
  sys::path::append(FileName, Name + ".dot");
  raw_fd_ostream File(FileName, EC, sys::fs::OpenFlags::OF_Text);

  DOTFuncInfo CFGInfo(&F);

  ASSERT_NO_ERROR(EC);
  // Test intentionally does not pass BPI, WriteGraph should work without it.
  WriteGraph(File, &CFGInfo, CFGOnly);
}

TEST_F(GraphWriterTest, WriteCFGDotFileTest) {
  auto M = makeLLVMModule();
  Function *F = M->getFunction("f");

  writeCFGToDotFile(*F, "test-full");
  writeCFGToDotFile(*F, "test-cfg-only", true);
}

} // end anonymous namespace
} // end namespace llvm
