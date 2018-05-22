//===---------------- Layer.h -- Layer interfaces --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Layer interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LAYER_H

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace orc {

class MangleAndInterner {
public:
  MangleAndInterner(ExecutionSession &ES, const DataLayout &DL);
  SymbolStringPtr operator()(StringRef Name);

private:
  ExecutionSession &ES;
  const DataLayout &DL;
};

/// Layer interface.
class IRLayer {
public:
  IRLayer(ExecutionSession &ES);
  virtual ~IRLayer();

  ExecutionSession &getExecutionSession() { return ES; }

  virtual Error add(VSO &V, VModuleKey K, std::unique_ptr<Module> M);
  virtual void emit(MaterializationResponsibility R, VModuleKey K,
                    std::unique_ptr<Module> M) = 0;

private:
  ExecutionSession &ES;
};

class BasicIRLayerMaterializationUnit : public MaterializationUnit {
public:
  BasicIRLayerMaterializationUnit(IRLayer &L, VModuleKey K,
                                  std::unique_ptr<Module> M);

private:
  void materialize(MaterializationResponsibility R) override;
  void discard(const VSO &V, SymbolStringPtr Name) override;

  IRLayer &L;
  VModuleKey K;
  std::unique_ptr<Module> M;
  std::map<SymbolStringPtr, GlobalValue *> Discardable;
};

class ObjectLayer {
public:
  ObjectLayer(ExecutionSession &ES);
  virtual ~ObjectLayer();

  ExecutionSession &getExecutionSession() { return ES; }

  virtual Error add(VSO &V, VModuleKey K, std::unique_ptr<MemoryBuffer> O);
  virtual void emit(MaterializationResponsibility R, VModuleKey K,
                    std::unique_ptr<MemoryBuffer> O) = 0;

private:
  ExecutionSession &ES;
};

/// The MemoryBuffer should represent a valid object file.
/// If there is any chance that the file is invalid it should be validated
/// prior to constructing a BasicObjectLayerMaterializationUnit.
class BasicObjectLayerMaterializationUnit : public MaterializationUnit {
public:
  BasicObjectLayerMaterializationUnit(ObjectLayer &L, VModuleKey K,
                                      std::unique_ptr<MemoryBuffer> O);

private:
  void materialize(MaterializationResponsibility R) override;
  void discard(const VSO &V, SymbolStringPtr Name) override;

  ObjectLayer &L;
  VModuleKey K;
  std::unique_ptr<MemoryBuffer> O;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LAYER_H
