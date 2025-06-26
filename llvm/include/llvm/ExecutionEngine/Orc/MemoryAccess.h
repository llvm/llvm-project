//===------- MemoryAccess.h - Executor memory access APIs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for accessing memory in the executor processes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

namespace llvm::orc {

/// APIs for manipulating memory in the target process.
class LLVM_ABI MemoryAccess {
public:
  /// Callback function for asynchronous writes.
  using WriteResultFn = unique_function<void(Error)>;

  virtual ~MemoryAccess();

  virtual void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                                WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                                  WriteResultFn OnWriteComplete) = 0;

  Error writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt8sAsync(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }

  Error writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt16sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }

  Error writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt32sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }

  Error writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeUInt64sAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }

  Error writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writeBuffersAsync(Ws,
                      [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }

  Error writePointers(ArrayRef<tpctypes::PointerWrite> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writePointersAsync(Ws,
                       [&](Error Err) { ResultP.set_value(std::move(Err)); });
    return ResultF.get();
  }
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H
