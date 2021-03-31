//===- TPCDebugObjectRegistrar.h - TPC-based debug registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TargetProcessControl based registration of debug objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TPCDEBUGOBJECTREGISTRAR_H
#define LLVM_EXECUTIONENGINE_ORC_TPCDEBUGOBJECTREGISTRAR_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Memory.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace llvm {
namespace orc {

/// Abstract interface for registering debug objects in the target process.
class DebugObjectRegistrar {
public:
  virtual Error registerDebugObject(sys::MemoryBlock) = 0;
  virtual ~DebugObjectRegistrar() {}
};

/// Use TargetProcessControl to register debug objects locally or in a remote
/// target process.
class TPCDebugObjectRegistrar : public DebugObjectRegistrar {
public:
  using SerializeBlockInfoFn =
      std::vector<uint8_t> (*)(sys::MemoryBlock TargetMemBlock);

  TPCDebugObjectRegistrar(TargetProcessControl &TPC,
                          JITTargetAddress RegisterFn,
                          SerializeBlockInfoFn SerializeBlockInfo)
      : TPC(TPC), RegisterFn(RegisterFn),
        SerializeBlockInfo(SerializeBlockInfo) {}

  Error registerDebugObject(sys::MemoryBlock TargetMem) override {
    return TPC.runWrapper(RegisterFn, SerializeBlockInfo(TargetMem))
        .takeError();
  }

private:
  TargetProcessControl &TPC;
  JITTargetAddress RegisterFn;
  SerializeBlockInfoFn SerializeBlockInfo;
};

/// Create a TargetProcessControl-based DebugObjectRegistrar that emits debug
/// objects to the GDB JIT interface.
Expected<std::unique_ptr<TPCDebugObjectRegistrar>>
createJITLoaderGDBRegistrar(TargetProcessControl &TPC);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TDEBUGOBJECTREGISTRAR_H
