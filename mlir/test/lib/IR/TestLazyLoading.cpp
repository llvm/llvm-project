//===- TestLazyLoading.cpp - Pass to test operation lazy loading  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <list>

using namespace mlir;

namespace {

/// This is a test pass which LazyLoads the current operation recursively.
struct LazyLoadingPass : public PassWrapper<LazyLoadingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyLoadingPass)

  StringRef getArgument() const final { return "test-lazy-loading"; }
  StringRef getDescription() const final { return "Test LazyLoading of op"; }
  LazyLoadingPass() = default;
  LazyLoadingPass(const LazyLoadingPass &) {}

  void runOnOperation() override {
    Operation *op = getOperation();
    std::string bytecode;
    {
      BytecodeWriterConfig config;
      if (version >= 0)
        config.setDesiredBytecodeVersion(version);
      llvm::raw_string_ostream os(bytecode);
      if (failed(writeBytecodeToFile(op, os, config))) {
        op->emitError() << "failed to write bytecode at version "
                        << (int)version;
        signalPassFailure();
        return;
      }
    }
    llvm::MemoryBufferRef buffer(bytecode, "test-lazy-loading");
    Block block;
    ParserConfig config(op->getContext(), /*verifyAfterParse=*/false);
    BytecodeReader reader(buffer, config,
                          /*lazyLoad=*/true);
    std::list<Operation *> toLoadOps;
    if (failed(reader.readTopLevel(&block, [&](Operation *op) {
          toLoadOps.push_back(op);
          return false;
        }))) {
      op->emitError() << "failed to read bytecode";
      return;
    }

    llvm::outs() << "Has " << reader.getNumOpsToMaterialize()
                 << " ops to materialize\n";

    // Recursively print the operations, before and after lazy loading.
    while (!toLoadOps.empty()) {
      Operation *toLoad = toLoadOps.front();
      toLoadOps.pop_front();
      llvm::outs() << "\n\nBefore Materializing...\n\n";
      toLoad->print(llvm::outs());
      llvm::outs() << "\n\nMaterializing...\n\n";
      if (failed(reader.materialize(toLoad, [&](Operation *op) {
            toLoadOps.push_back(op);
            return false;
          }))) {
        toLoad->emitError() << "failed to materialize";
        signalPassFailure();
        return;
      }
      toLoad->print(llvm::outs());
      llvm::outs() << "\n";
      llvm::outs() << "Has " << reader.getNumOpsToMaterialize()
                   << " ops to materialize\n";
    }
  }
  Option<int> version{*this, "bytecode-version",
                      llvm::cl::desc("Specifies the bytecode version to use."),
                      llvm::cl::init(-1)};
};
} // namespace

namespace mlir {
void registerLazyLoadingTestPasses() { PassRegistration<LazyLoadingPass>(); }
} // namespace mlir
