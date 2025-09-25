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

  template <typename T> using ReadUIntsResult = std::vector<T>;
  template <typename T>
  using OnReadUIntsCompleteFn =
      unique_function<void(Expected<ReadUIntsResult<T>>)>;

  using ReadPointersResult = std::vector<ExecutorAddr>;
  using OnReadPointersCompleteFn =
      unique_function<void(Expected<ReadPointersResult>)>;

  using ReadBuffersResult = std::vector<std::vector<uint8_t>>;
  using OnReadBuffersCompleteFn =
      unique_function<void(Expected<ReadBuffersResult>)>;

  using ReadStringsResult = std::vector<std::string>;
  using OnReadStringsCompleteFn =
      unique_function<void(Expected<ReadStringsResult>)>;

  virtual ~MemoryAccess();

  virtual void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                                WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                                  WriteResultFn OnWriteComplete) = 0;

  virtual void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                                 WriteResultFn OnWriteComplete) = 0;

  virtual void readUInt8sAsync(ArrayRef<ExecutorAddr> Rs,
                               OnReadUIntsCompleteFn<uint8_t> OnComplete) = 0;

  virtual void readUInt16sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint16_t> OnComplete) = 0;

  virtual void readUInt32sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint32_t> OnComplete) = 0;

  virtual void readUInt64sAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadUIntsCompleteFn<uint64_t> OnComplete) = 0;

  virtual void readPointersAsync(ArrayRef<ExecutorAddr> Rs,
                                 OnReadPointersCompleteFn OnComplete) = 0;

  virtual void readBuffersAsync(ArrayRef<ExecutorAddrRange> Rs,
                                OnReadBuffersCompleteFn OnComplete) = 0;

  virtual void readStringsAsync(ArrayRef<ExecutorAddr> Rs,
                                OnReadStringsCompleteFn OnComplete) = 0;

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

  Error writePointers(ArrayRef<tpctypes::PointerWrite> Ws) {
    std::promise<MSVCPError> ResultP;
    auto ResultF = ResultP.get_future();
    writePointersAsync(Ws,
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

  Expected<ReadUIntsResult<uint8_t>> readUInt8s(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadUIntsResult<uint8_t>>> P;
    readUInt8sAsync(Rs, [&](Expected<ReadUIntsResult<uint8_t>> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadUIntsResult<uint16_t>> readUInt16s(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadUIntsResult<uint16_t>>> P;
    readUInt16sAsync(Rs, [&](Expected<ReadUIntsResult<uint16_t>> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadUIntsResult<uint32_t>> readUInt32s(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadUIntsResult<uint32_t>>> P;
    readUInt32sAsync(Rs, [&](Expected<ReadUIntsResult<uint32_t>> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadUIntsResult<uint64_t>> readUInt64s(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadUIntsResult<uint64_t>>> P;
    readUInt64sAsync(Rs, [&](Expected<ReadUIntsResult<uint64_t>> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadPointersResult> readPointers(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadPointersResult>> P;
    readPointersAsync(Rs, [&](Expected<ReadPointersResult> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadBuffersResult> readBuffers(ArrayRef<ExecutorAddrRange> Rs) {
    std::promise<MSVCPExpected<ReadBuffersResult>> P;
    readBuffersAsync(Rs, [&](Expected<ReadBuffersResult> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }

  Expected<ReadStringsResult> readStrings(ArrayRef<ExecutorAddr> Rs) {
    std::promise<MSVCPExpected<ReadStringsResult>> P;
    readStringsAsync(Rs, [&](Expected<ReadStringsResult> Result) {
      P.set_value(std::move(Result));
    });
    return P.get_future().get();
  }
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_MEMORYACCESS_H
