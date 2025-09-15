//===- TestNVGPUTransforms.cpp - Test NVGPU transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nvgpu;

namespace {

struct TestMmaSyncF32ToTF32Patterns
    : public PassWrapper<TestMmaSyncF32ToTF32Patterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMmaSyncF32ToTF32Patterns)

  StringRef getArgument() const final {
    return "test-nvgpu-mmasync-f32-to-tf32-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to convert mma.sync on f32 with tf32 precision";
  }
  TestMmaSyncF32ToTF32Patterns() = default;
  TestMmaSyncF32ToTF32Patterns(const TestMmaSyncF32ToTF32Patterns &pass)
      : PassWrapper(pass) {}

  Option<std::string> precision{
      *this, "precision",
      llvm::cl::desc(
          "Target nvgpu.mma.sync on f32 input with tf32 or tf32x3 precision"),
      llvm::cl::init("tf32")};

  MmaSyncF32Lowering tf32Precision =
      llvm::StringSwitch<MmaSyncF32Lowering>(precision)
          .Case("tf32", MmaSyncF32Lowering::TF32)
          .Case("tf32x3", MmaSyncF32Lowering::TF32x3)
          .Default(MmaSyncF32Lowering::Unkown);

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    populateMmaSyncF32ToTF32Patterns(patterns, tf32Precision);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestNVGPULowerings() {
  PassRegistration<TestMmaSyncF32ToTF32Patterns>();
}

} // namespace test
} // namespace mlir
