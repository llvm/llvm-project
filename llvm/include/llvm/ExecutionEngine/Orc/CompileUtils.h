//===- CompileUtils.h - Utilities for compiling IR in the JIT ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for compiling IR to object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <memory>

namespace llvm {

class MCContext;
class Module;

namespace orc {

/// Simple compile functor: Takes a single IR module and returns an
///        ObjectFile.
class SimpleCompiler {
private:
  class SmallVectorMemoryBuffer : public MemoryBuffer {
  public:
    SmallVectorMemoryBuffer(SmallVector<char, 0> Buffer)
        : Buffer(std::move(Buffer)) {
      init(this->Buffer.data(), this->Buffer.data() + this->Buffer.size(),
           false);
    }

    BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

  private:
    SmallVector<char, 0> Buffer;
  };

public:
  using CompileResult = std::unique_ptr<MemoryBuffer>;

  /// Construct a simple compile functor with the given target.
  SimpleCompiler(TargetMachine &TM, ObjectCache *ObjCache = nullptr)
    : TM(TM), ObjCache(ObjCache) {}

  /// Set an ObjectCache to query before compiling.
  void setObjectCache(ObjectCache *NewCache) { ObjCache = NewCache; }

  /// Compile a Module to an ObjectFile.
  CompileResult operator()(Module &M) {
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

    auto ObjBuffer =
        llvm::make_unique<SmallVectorMemoryBuffer>(std::move(ObjBufferSV));
    auto Obj =
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());

    if (Obj) {
      notifyObjectCompiled(M, *ObjBuffer);
      return std::move(ObjBuffer);
    }

    // TODO: Actually report errors helpfully.
    consumeError(Obj.takeError());
    return nullptr;
  }

private:

  CompileResult tryToLoadFromObjectCache(const Module &M) {
    if (!ObjCache)
      return CompileResult();

    return ObjCache->getObject(&M);
  }

  void notifyObjectCompiled(const Module &M, const MemoryBuffer &ObjBuffer) {
    if (ObjCache)
      ObjCache->notifyObjectCompiled(&M, ObjBuffer.getMemBufferRef());
  }

  TargetMachine &TM;
  ObjectCache *ObjCache = nullptr;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H
