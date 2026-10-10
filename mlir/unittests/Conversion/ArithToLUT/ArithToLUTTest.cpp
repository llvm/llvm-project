//===- ArithToLUTTest.cpp - Exhaustive correctness test for the LUT pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For every supported narrow-float format, iterate all 2^N bit patterns, run
// each through the LUT-lowered extf function via the MLIR JIT, and compare the
// result bit-for-bit against the APFloat reference value computed the same
// way buildExtFLUT does at compile time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLUT/ArithToLUT.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/TargetSelect.h"

#include "gmock/gmock.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

// JIT is unavailable on some platforms.
#ifdef __sparc__
#define SKIP_WITHOUT_JIT(x) DISABLED_##x
#else
#define SKIP_WITHOUT_JIT(x) x
#endif

using namespace mlir;

#if !defined(_WIN32) && !defined(_AIX)

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

static LogicalResult lowerToLLVM(ModuleOp module) {
  PassManager pm(module->getName());
  pm.addPass(createConvertArithExtFToLUT());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  return pm.run(module);
}

// Builds a module with function:
// func @test(%arg0: i32) -> f32 { trunci i32 to iN; bitcast iN to fN; extf fN
// to f32 }
static std::string makeModule(const char *floatTypeName, unsigned bitWidth) {
  std::string iN = "i" + std::to_string(bitWidth);
  std::string s;
  s += "func.func @test(%arg0: i32) -> f32 "
       "attributes { llvm.emit_c_interface } {\n";
  s += "  %" + iN + "  = arith.trunci %arg0 : i32 to " + iN + "\n";
  s += "  %fp  = arith.bitcast %" + iN + " : " + iN + " to " + floatTypeName +
       "\n";
  s += std::string("  %f32 = arith.extf %fp : ") + floatTypeName + " to f32\n";
  s += "  return %f32 : f32\n}\n";
  return s;
}

struct NarrowFPFormat {
  const char *mlirName;
  unsigned bitWidth;
  std::function<FloatType(MLIRContext *)> getType;
};

static void runExhaustiveTest(const NarrowFPFormat &fmt) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(makeModule(fmt.mlirName, fmt.bitWidth), &ctx);
  ASSERT_TRUE(!!module) << "parse failed for " << fmt.mlirName;
  ASSERT_TRUE(succeeded(lowerToLLVM(*module)))
      << "lowering failed for " << fmt.mlirName;

  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError) << "JIT creation failed for " << fmt.mlirName;
  auto jit = std::move(jitOrError.get());

  const llvm::fltSemantics &sem = fmt.getType(&ctx).getFloatSemantics();
  unsigned numEntries = 1u << fmt.bitWidth;

  for (unsigned i = 0; i < numEntries; ++i) {
    float got = 0.0f;
    int32_t input = static_cast<int32_t>(i);
    llvm::Error err =
        jit->invoke("test", input, ExecutionEngine::Result<float>(got));
    ASSERT_FALSE(err) << llvm::toString(std::move(err));

    llvm::APFloat ref(sem, llvm::APInt(fmt.bitWidth, static_cast<uint64_t>(i)));
    bool lossy = false;
    ref.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven,
                &lossy);
    float expected = ref.convertToFloat();

    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(got))
          << fmt.mlirName << " pattern " << i << ": expected NaN, got " << got;
    } else {
      uint32_t gotBits = 0, expBits = 0;
      std::memcpy(&gotBits, &got, 4);
      std::memcpy(&expBits, &expected, 4);
      EXPECT_EQ(gotBits, expBits)
          << fmt.mlirName << " pattern " << i << ": expected " << expected
          << ", got " << got;
    }
  }
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF4E2M1FN)) {
  runExhaustiveTest({"f4E2M1FN", 4, [](MLIRContext *ctx) -> FloatType {
                       return Float4E2M1FNType::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E4M3FN)) {
  runExhaustiveTest({"f8E4M3FN", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E4M3FNType::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E5M2)) {
  runExhaustiveTest({"f8E5M2", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E5M2Type::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E4M3FNUZ)) {
  runExhaustiveTest({"f8E4M3FNUZ", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E4M3FNUZType::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E5M2FNUZ)) {
  runExhaustiveTest({"f8E5M2FNUZ", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E5M2FNUZType::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E4M3B11FNUZ)) {
  runExhaustiveTest({"f8E4M3B11FNUZ", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E4M3B11FNUZType::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E3M4)) {
  runExhaustiveTest({"f8E3M4", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E3M4Type::get(ctx);
                     }});
}

TEST(ArithToLUT, SKIP_WITHOUT_JIT(ExtFAllPatternsF8E4M3)) {
  runExhaustiveTest({"f8E4M3", 8, [](MLIRContext *ctx) -> FloatType {
                       return Float8E4M3Type::get(ctx);
                     }});
}

#endif // !_WIN32 && !_AIX
