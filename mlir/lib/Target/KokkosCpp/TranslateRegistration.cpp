//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/KokkosCpp/KokkosCppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// KokkosCpp registration
//===----------------------------------------------------------------------===//

void registerToKokkosTranslation() {
  TranslateFromMLIRRegistration reg1(
      "mlir-to-kokkos", "translate from mlir to Kokkos",
      [](Operation *op, raw_ostream &output) {
        return emitc::translateToKokkosCpp(
            op, output, /* enableSparseSupport */ false);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect,
                        LLVM::LLVMDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        scf::SCFDialect>();
        // clang-format on
      });

  TranslateFromMLIRRegistration reg2(
      "sparse-mlir-to-kokkos", "translate from mlir (with sparse tensors) to Kokkos",
      [](Operation *op, raw_ostream &output) {
        return emitc::translateToKokkosCpp(
            op, output, /* enableSparseSupport */ true);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect,
                        math::MathDialect,
                        LLVM::LLVMDialect,
                        memref::MemRefDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir
