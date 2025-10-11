//===--- ValueResultManager.h - ValueResultManager implementation -----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements ValueResultManager for tracking runtime Values and their handlers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_VALUERESULTMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_VALUERESULTMANAGER_H

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/Value.h"
#include <atomic>
#include <future>
#include <mutex>
#include <unordered_map>

namespace llvm {
namespace orc {

class ValueResultManager {
public:
  using ValueID = uint64_t;
  using ResultCallback = unique_function<void(Value)>;
  using SendResultFn = unique_function<void(Error)>;

  ValueResultManager() = default;

  ValueID createPendingResult(ResultCallback Callback) {
    std::lock_guard<std::mutex> Lock(Mutex);
    ValueID NewID = NextID++;
    Callbacks.emplace(NewID, std::move(Callback));
    return NewID;
  }

  std::pair<ValueID, std::future<Value>> createPendingResult() {
    std::lock_guard<std::mutex> Lock(Mutex);
    ValueID NewID = NextID++;
    std::promise<Value> P;
    auto F = P.get_future();
    Promises.emplace(NewID, std::move(P));
    return {NewID, std::move(F)};
  }

  void notify(SendResultFn SendResult, ValueID ID, Value V) {
    std::lock_guard<std::mutex> Lock(Mutex);
    if (auto It = Callbacks.find(ID); It != Callbacks.end()) {
      It->second(std::move(V));
      Callbacks.erase(It);
      return;
    }

    // if (auto It = Promises.find(ID); It != Promises.end()) {
    //   It->second.set_value(std::move(V));
    //   Promises.erase(It);
    //   return;
    // }
  }

private:
  std::atomic<ValueID> NextID{1};

  std::mutex Mutex;
  std::unordered_map<ValueID, ResultCallback> Callbacks;
  std::unordered_map<ValueID, std::promise<Value>> Promises;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_VALUERESULTMANAGER_H
