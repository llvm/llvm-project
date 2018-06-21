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
#include "llvm/ExecutionEngine/ObjectMemoryBuffer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <memory>

namespace llvm {

class MCContext;
class Module;

namespace orc {

/// @brief Simple compile functor: Takes a single IR module and returns an
///        ObjectFile.
class SimpleCompiler {
public:

  using CompileResult = object::OwningBinary<object::ObjectFile>;

  /// @brief Construct a simple compile functor with the given target.
  SimpleCompiler(TargetMachine &TM, ObjectCache *ObjCache = nullptr)
    : TM(TM), ObjCache(ObjCache) {}

  /// @brief Set an ObjectCache to query before compiling.
  void setObjectCache(ObjectCache *NewCache) { ObjCache = NewCache; }

  /// @brief Compile a Module to an ObjectFile.
  CompileResult operator()(Module &M) {
    CompileResult CachedObject = tryToLoadFromObjectCache(M);
    if (CachedObject.getBinary())
      return CachedObject;

    SmallVector<char, 0> ObjBufferSV;
    raw_svector_ostream ObjStream(ObjBufferSV);

    legacy::PassManager PM;
    MCContext *Ctx;
    if (TM.addPassesToEmitMC(PM, Ctx, ObjStream))
      llvm_unreachable("Target does not support MC emission.");
    PM.run(M);
    std::unique_ptr<MemoryBuffer> ObjBuffer(
        new ObjectMemoryBuffer(std::move(ObjBufferSV)));
    Expected<std::unique_ptr<object::ObjectFile>> Obj =
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());
    if (Obj) {
      notifyObjectCompiled(M, *ObjBuffer);
      return CompileResult(std::move(*Obj), std::move(ObjBuffer));
    }
    // TODO: Actually report errors helpfully.
    consumeError(Obj.takeError());
    return CompileResult(nullptr, nullptr);
  }

private:

  CompileResult tryToLoadFromObjectCache(const Module &M) {
    if (!ObjCache)
      return CompileResult();

    std::unique_ptr<MemoryBuffer> ObjBuffer = ObjCache->getObject(&M);
    if (!ObjBuffer)
      return CompileResult();

    Expected<std::unique_ptr<object::ObjectFile>> Obj =
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());
    if (!Obj) {
      // TODO: Actually report errors helpfully.
      consumeError(Obj.takeError());
      return CompileResult();
    }

    return CompileResult(std::move(*Obj), std::move(ObjBuffer));
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
