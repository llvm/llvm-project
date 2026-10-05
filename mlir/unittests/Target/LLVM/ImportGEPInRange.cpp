//===- ImportGEPInRange.cpp -----------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace mlir;

// The textual IR parser builds a GEP's `inrange` at the index width of its
// base pointer, but the C++ API keeps whatever width it is given: clang builds
// vtable address points with a 32-bit range regardless of the index width.
// Importing such a GEP must still produce an `inrange` that verifies.
TEST(ImportGEPInRange, NarrowRangeIsWidenedToIndexWidth) {
  static constexpr const char *ir = R"(
target datalayout = "e-m:e-i64:64-n8:16:32:64-S128"
@vtable = external constant { [7 x ptr] }
define ptr @f() {
  ret ptr getelementptr inbounds inrange(-16, 40) ({ [7 x ptr] }, ptr @vtable, i32 0, i32 0, i32 2)
}
)";
  llvm::LLVMContext llvmContext;
  llvm::SMDiagnostic diag;
  std::unique_ptr<llvm::Module> llvmModule =
      llvm::parseAssemblyString(ir, diag, llvmContext);
  ASSERT_TRUE(llvmModule);

  // Rebuild the GEP with a 32-bit range, as clang's ItaniumCXXABI does.
  auto *ret = llvm::cast<llvm::ReturnInst>(
      llvmModule->getFunction("f")->getEntryBlock().getTerminator());
  auto *gep = llvm::cast<llvm::GEPOperator>(ret->getReturnValue());
  SmallVector<llvm::Constant *> indices;
  for (llvm::Use &index : gep->indices())
    indices.push_back(llvm::cast<llvm::Constant>(index.get()));
  llvm::ConstantRange narrowRange(llvm::APInt(32, -16, /*isSigned=*/true),
                                  llvm::APInt(32, 40, /*isSigned=*/true));
  ret->setOperand(0, llvm::ConstantExpr::getGetElementPtr(
                         gep->getSourceElementType(),
                         llvm::cast<llvm::Constant>(gep->getPointerOperand()),
                         indices, gep->getNoWrapFlags(), narrowRange));

  MLIRContext context;
  context.loadDialect<LLVM::LLVMDialect, DLTIDialect>();
  OwningOpRef<ModuleOp> module =
      translateLLVMIRToModule(std::move(llvmModule), &context);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(verify(*module)));

  unsigned numGEPs = 0;
  module->walk([&](LLVM::GEPOp op) {
    LLVM::ConstantRangeAttr inrange = op.getInrangeAttr();
    if (!inrange)
      return;
    ++numGEPs;
    EXPECT_EQ(inrange.getLower().getBitWidth(), 64u);
    EXPECT_EQ(inrange.getLower().getSExtValue(), -16);
    EXPECT_EQ(inrange.getUpper().getSExtValue(), 40);
  });
  EXPECT_EQ(numGEPs, 1u);
}
