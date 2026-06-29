//===-- EJitSyncCompiler.cpp - Synchronous JIT Compiler -------------------===//

#ifndef EJIT_FREESTANDING

#include "llvm/ExecutionEngine/EJIT/EJitSyncCompiler.h"
#include "llvm/ExecutionEngine/EJIT/EJitDiag.h"
#include "llvm/ExecutionEngine/EJIT/EJitOrcEngine.h"
#include <chrono>

using namespace llvm;
using namespace llvm::ejit;

EJitSyncCompiler::Result
EJitSyncCompiler::compile(EJitOrcEngine &engine,
                          const std::string &bitcodeData,
                          const SpecializationContext &ctx) {
  Result result;
  EJIT_DIAG("sync compile begin func=%s key=0x%016lx size=%zu",
            ctx.fnName.c_str(), ctx.cacheKey, bitcodeData.size());

  auto start = std::chrono::steady_clock::now();

  engine.setActiveContext(&ctx);

  if (auto Err = engine.loadBitcodeModule(bitcodeData, ctx.cacheKey, ctx.fnName)) {
    engine.setActiveContext(nullptr);
    EJIT_DIAG("sync compile FAIL func=%s key=0x%016lx: load bitcode failed",
              ctx.fnName.c_str(), ctx.cacheKey);
    consumeError(std::move(Err));
    return result;
  }

  auto addrOrErr = engine.lookup(ctx.cacheKey, ctx.fnName);
  engine.setActiveContext(nullptr);

  if (!addrOrErr) {
    EJIT_DIAG("sync compile FAIL func=%s key=0x%016lx: lookup failed",
              ctx.fnName.c_str(), ctx.cacheKey);
    consumeError(addrOrErr.takeError());
    return result;
  }

  result.funcPtr = *addrOrErr;
  // NOTE: codeSize is bitcode size, not compiled machine code size.
  result.codeSize = bitcodeData.size();

  auto end = std::chrono::steady_clock::now();
  result.compileTimeMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  EJIT_DIAG("sync compile OK func=%s key=0x%016lx pfn=%p time=%llums",
            ctx.fnName.c_str(), ctx.cacheKey, result.funcPtr,
            static_cast<unsigned long long>(result.compileTimeMs));
  return result;
}

#endif // EJIT_FREESTANDING
