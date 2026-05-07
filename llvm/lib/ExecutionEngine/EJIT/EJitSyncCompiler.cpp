//===-- EJitSyncCompiler.cpp - Synchronous JIT Compiler -------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitSyncCompiler.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include <chrono>

using namespace llvm;
using namespace llvm::ejit;

EJitSyncCompiler::Result
EJitSyncCompiler::compile(EJitOrcEngine &engine,
                          const std::string &bitcodeData,
                          const SpecializationContext &ctx,
                          const std::string &cacheKey) {
  Result result;

  auto start = std::chrono::steady_clock::now();

  engine.setActiveContext(&ctx);

  if (auto Err = engine.loadBitcodeModule(bitcodeData, cacheKey, ctx.fnName)) {
    engine.setActiveContext(nullptr);
    return result;
  }

  auto addrOrErr = engine.lookup(cacheKey, ctx.fnName);
  engine.setActiveContext(nullptr);

  if (!addrOrErr)
    return result;

  result.funcPtr = *addrOrErr;

  auto end = std::chrono::steady_clock::now();
  result.compileTimeMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  return result;
}
