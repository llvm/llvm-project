//===------- Calls.h - Runtime-agnostic executor call APIs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime-agnostic interfaces for invoking executor-side operations. These
// abstract over how a call reaches the executor, so clients can be written
// once and used whether the operation is provided by a full ORC runtime or by
// LLVM's own ORC-runtime-lite. Concrete implementations live in subdirectories
// (e.g. RTBridge/SPS).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H
#define LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <cstdint>
#include <future>
#include <string>

namespace llvm::orc::rt {

/// Interface for running a main-like function (int(int argc, char *argv[])) in
/// the executor.
///
/// Given the address of the function to run and an argument vector, invokes the
/// function in the executor and returns its integer result. Calls can be made
/// asynchronously (delivering the result to an OnComplete continuation) or
/// synchronously (blocking until the result is available). Concrete
/// implementations (e.g. rt::sps::MainCaller) determine how the call reaches
/// the executor.
class MainCaller {
public:
  virtual ~MainCaller();

  /// Asynchronously run the main-like function at MainFnAddr with the given
  /// Args, delivering its result (or an error) to OnComplete.
  virtual void operator()(unique_function<void(Expected<int64_t>)> OnComplete,
                          ExecutorAddr MainFnAddr,
                          ArrayRef<std::string> Args) = 0;

  /// Run the main-like function at MainFnAddr with the given Args, blocking
  /// until its result (or an error) is available.
  Expected<int64_t> operator()(ExecutorAddr MainFnAddr,
                               ArrayRef<std::string> Args) {
    std::promise<MSVCPExpected<int64_t>> P;
    auto F = P.get_future();
    this->operator()(
        [P = std::move(P)](Expected<int64_t> R) mutable {
          P.set_value(std::move(R));
        },
        MainFnAddr, Args);
    return F.get();
  }
};

} // namespace llvm::orc::rt

#endif // LLVM_EXECUTIONENGINE_ORC_RTBRIDGE_CALLS_H
