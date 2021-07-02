//===- EPCDebugObjectRegistrar.h - EPC-based debug registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExecutorProcessControl based registration of debug objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCDEBUGOBJECTREGISTRAR_H
#define LLVM_EXECUTIONENGINE_ORC_EPCDEBUGOBJECTREGISTRAR_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {

/// Abstract interface for registering debug objects in the executor process.
class DebugObjectRegistrar {
public:
  virtual Error registerDebugObject(sys::MemoryBlock) = 0;
  virtual ~DebugObjectRegistrar() {}
};

/// Use ExecutorProcessControl to register debug objects locally or in a remote
/// executor process.
class EPCDebugObjectRegistrar : public DebugObjectRegistrar {
public:
  EPCDebugObjectRegistrar(ExecutorProcessControl &EPC,
                          JITTargetAddress RegisterFn)
      : EPC(EPC), RegisterFn(RegisterFn) {}

  Error registerDebugObject(sys::MemoryBlock TargetMem) override {
    return WrapperFunction<void(SPSExecutorAddress, uint64_t)>::call(
        EPCCaller(EPC, RegisterFn), pointerToJITTargetAddress(TargetMem.base()),
        static_cast<uint64_t>(TargetMem.allocatedSize()));
  }

private:
  ExecutorProcessControl &EPC;
  JITTargetAddress RegisterFn;
};

/// Create a ExecutorProcessControl-based DebugObjectRegistrar that emits debug
/// objects to the GDB JIT interface.
Expected<std::unique_ptr<EPCDebugObjectRegistrar>>
createJITLoaderGDBRegistrar(ExecutorProcessControl &EPC);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCDEBUGOBJECTREGISTRAR_H
