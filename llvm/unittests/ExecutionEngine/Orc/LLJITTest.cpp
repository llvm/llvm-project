//===----------- LLJITTest.cpp - Unit tests for LLJIT ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "OrcTestCommon.h"
#include "llvm/IR/Module.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(LLJITTest, AddModuleWithIncompatibleTripleErrors) {
  OrcNativeTarget::initialize();

  auto J = LLJITBuilder().create();
  if (!J) {
    consumeError(J.takeError());
    GTEST_SKIP();
  }

  // Build a module whose triple differs from the JIT's by OS, which makes it
  // incompatible with the JIT target. Pick any OS other than the host's.
  Triple ModuleTriple = (*J)->getTargetTriple();
  Triple::OSType IncompatibleOS = Triple::UnknownOS;
  for (Triple::OSType OS : {Triple::Linux, Triple::Win32, Triple::Darwin}) {
    if (OS != ModuleTriple.getOS()) {
      IncompatibleOS = OS;
      break;
    }
  }
  ASSERT_NE(IncompatibleOS, Triple::UnknownOS);

  ModuleTriple.setOS(IncompatibleOS);

  std::unique_ptr<LLVMContext> Ctx(new LLVMContext());
  std::unique_ptr<Module> M(new Module("M", *Ctx));
  M->setTargetTriple(ModuleTriple);

  EXPECT_THAT_ERROR(
      (*J)->addIRModule(ThreadSafeModule(std::move(M), std::move(Ctx))),
      FailedWithMessage(testing::HasSubstr("incompatible triple")));
}

} // namespace
