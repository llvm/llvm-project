//===------ CompileUtils.cpp - Utilities for compiling IR in the JIT ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>

namespace llvm {
namespace orc {

/// Compile a Module to an ObjectFile.
SimpleCompiler::CompileResult SimpleCompiler::operator()(Module &M) {
  CompileResult CachedObject = tryToLoadFromObjectCache(M);
  if (CachedObject)
    return CachedObject;

  SmallVector<char, 0> ObjBufferSV;

  {
    raw_svector_ostream ObjStream(ObjBufferSV);

    legacy::PassManager PM;
    MCContext *Ctx;
    if (TM.addPassesToEmitMC(PM, Ctx, ObjStream))
      llvm_unreachable("Target does not support MC emission.");
    PM.run(M);
  }

  auto ObjBuffer = std::make_unique<SmallVectorMemoryBuffer>(
      std::move(ObjBufferSV), M.getModuleIdentifier() + "-jitted-objectbuffer");

  auto Obj = object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());

  if (Obj) {
    notifyObjectCompiled(M, *ObjBuffer);
    return std::move(ObjBuffer);
  }

  // TODO: Actually report errors helpfully.
  consumeError(Obj.takeError());
  return nullptr;
}

SimpleCompiler::CompileResult
SimpleCompiler::tryToLoadFromObjectCache(const Module &M) {
  if (!ObjCache)
    return CompileResult();

  return ObjCache->getObject(&M);
}

void SimpleCompiler::notifyObjectCompiled(const Module &M,
                                          const MemoryBuffer &ObjBuffer) {
  if (ObjCache)
    ObjCache->notifyObjectCompiled(&M, ObjBuffer.getMemBufferRef());
}

ConcurrentIRCompiler::ConcurrentIRCompiler(JITTargetMachineBuilder JTMB,
                                           ObjectCache *ObjCache)
    : JTMB(std::move(JTMB)), ObjCache(ObjCache) {}

std::unique_ptr<MemoryBuffer> ConcurrentIRCompiler::operator()(Module &M) {
  auto TM = cantFail(JTMB.createTargetMachine());
  SimpleCompiler C(*TM, ObjCache);
  return C(M);
}

} // end namespace orc
} // end namespace llvm
