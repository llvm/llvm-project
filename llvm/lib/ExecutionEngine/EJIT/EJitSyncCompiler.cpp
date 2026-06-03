//===-- EJitSyncCompiler.cpp - Synchronous JIT Compiler -------------------===//

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitSyncCompiler.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include <chrono>

using namespace llvm;
using namespace llvm::ejit;

EJitSyncCompiler::Result
EJitSyncCompiler::compile(EJitOrcEngine &engine,
                          const std::string &bitcodeData,
                          const SpecializationContext &ctx) {
  Result result;

  auto start = std::chrono::steady_clock::now();

  engine.setActiveContext(&ctx);

  if (auto Err = engine.loadBitcodeModule(bitcodeData, ctx.cacheKey, ctx.fnName)) {
    engine.setActiveContext(nullptr);
    return result;
  }

  auto addrOrErr = engine.lookup(ctx.cacheKey, ctx.fnName);
  engine.setActiveContext(nullptr);

  if (!addrOrErr)
    return result;

  result.funcPtr = *addrOrErr;
  // NOTE: codeSize is bitcode size, not compiled machine code size.
  result.codeSize = bitcodeData.size();

  auto end = std::chrono::steady_clock::now();
  result.compileTimeMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  return result;
}

#endif // EJIT_FREESTANDING
