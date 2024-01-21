//===-- OpenMP/OMPT/Interface.h - OpenMP Tooling interfaces ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for OpenMP Tool callback dispatchers.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_OMPTINTERFACE_H
#define _OMPTARGET_OMPTINTERFACE_H

// Only provide functionality if target OMPT support is enabled
#ifdef OMPT_SUPPORT
#include <functional>
#include <tuple>

#include "Callback.h"
#include "omp-tools.h"

#include "llvm/Support/ErrorHandling.h"

#define OMPT_IF_BUILT(stmt) stmt
#define OMPT_GET_RETURN_ADDRESS(level) __builtin_return_address(level)

/// Callbacks for target regions require task_data representing the
/// encountering task.
/// Callbacks for target regions and target data ops require
/// target_task_data representing the target task region.
typedef ompt_data_t *(*ompt_get_task_data_t)();
typedef ompt_data_t *(*ompt_get_target_task_data_t)();

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// Function pointers that will be used to track task_data and
/// target_task_data.
static ompt_get_task_data_t ompt_get_task_data_fn;
static ompt_get_target_task_data_t ompt_get_target_task_data_fn;

/// Used to maintain execution state for this thread
class Interface {
public:
  /// Top-level function for invoking callback before device data allocation
  void beginTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                            void **TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after device data allocation
  void endTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                          void **TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before data submit
  void beginTargetDataSubmit(int64_t DeviceId, void *HstPtrBegin,
                             void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after data submit
  void endTargetDataSubmit(int64_t DeviceId, void *HstPtrBegin,
                           void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before device data deallocation
  void beginTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin, void *Code);

  /// Top-level function for invoking callback after device data deallocation
  void endTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin, void *Code);

  /// Top-level function for invoking callback before data retrieve
  void beginTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                               void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback after data retrieve
  void endTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                             void *TgtPtrBegin, size_t Size, void *Code);

  /// Top-level function for invoking callback before kernel dispatch
  void beginTargetSubmit(unsigned int NumTeams = 1);

  /// Top-level function for invoking callback after kernel dispatch
  void endTargetSubmit(unsigned int NumTeams = 1);

  // Target region callbacks

  /// Top-level function for invoking callback before target enter data
  /// construct
  void beginTargetDataEnter(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target enter data
  /// construct
  void endTargetDataEnter(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target exit data
  /// construct
  void beginTargetDataExit(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target exit data
  /// construct
  void endTargetDataExit(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target update construct
  void beginTargetUpdate(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target update construct
  void endTargetUpdate(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback before target construct
  void beginTarget(int64_t DeviceId, void *Code);

  /// Top-level function for invoking callback after target construct
  void endTarget(int64_t DeviceId, void *Code);

  // Callback getter: Target data operations
  template <ompt_target_data_op_t OpType> auto getCallbacks() {
    if constexpr (OpType == ompt_target_data_alloc ||
                  OpType == ompt_target_data_alloc_async)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataAlloc),
                            std::mem_fn(&Interface::endTargetDataAlloc));

    if constexpr (OpType == ompt_target_data_delete ||
                  OpType == ompt_target_data_delete_async)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataDelete),
                            std::mem_fn(&Interface::endTargetDataDelete));

    if constexpr (OpType == ompt_target_data_transfer_to_device ||
                  OpType == ompt_target_data_transfer_to_device_async)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataSubmit),
                            std::mem_fn(&Interface::endTargetDataSubmit));

    if constexpr (OpType == ompt_target_data_transfer_from_device ||
                  OpType == ompt_target_data_transfer_from_device_async)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataRetrieve),
                            std::mem_fn(&Interface::endTargetDataRetrieve));

    llvm_unreachable("Unhandled target data operation type!");
  }

  // Callback getter: Target region operations
  template <ompt_target_t OpType> auto getCallbacks() {
    if constexpr (OpType == ompt_target_enter_data ||
                  OpType == ompt_target_enter_data_nowait)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataEnter),
                            std::mem_fn(&Interface::endTargetDataEnter));

    if constexpr (OpType == ompt_target_exit_data ||
                  OpType == ompt_target_exit_data_nowait)
      return std::make_pair(std::mem_fn(&Interface::beginTargetDataExit),
                            std::mem_fn(&Interface::endTargetDataExit));

    if constexpr (OpType == ompt_target_update ||
                  OpType == ompt_target_update_nowait)
      return std::make_pair(std::mem_fn(&Interface::beginTargetUpdate),
                            std::mem_fn(&Interface::endTargetUpdate));

    if constexpr (OpType == ompt_target || OpType == ompt_target_nowait)
      return std::make_pair(std::mem_fn(&Interface::beginTarget),
                            std::mem_fn(&Interface::endTarget));

    llvm_unreachable("Unknown target region operation type!");
  }

  // Callback getter: Kernel launch operation
  template <ompt_callbacks_t OpType> auto getCallbacks() {
    // We use 'ompt_callbacks_t', because no other enum is currently available
    // to model a kernel launch / target submit operation.
    if constexpr (OpType == ompt_callback_target_submit)
      return std::make_pair(std::mem_fn(&Interface::beginTargetSubmit),
                            std::mem_fn(&Interface::endTargetSubmit));

    llvm_unreachable("Unhandled target operation!");
  }

  /// Setters for target region and target operation correlation ids
  void setTargetDataValue(uint64_t DataValue) { TargetData.value = DataValue; }
  void setTargetDataPtr(void *DataPtr) { TargetData.ptr = DataPtr; }
  void setHostOpId(ompt_id_t OpId) { HostOpId = OpId; }

  /// Getters for target region and target operation correlation ids
  uint64_t getTargetDataValue() { return TargetData.value; }
  void *getTargetDataPtr() { return TargetData.ptr; }
  ompt_id_t getHostOpId() { return HostOpId; }

private:
  /// Target operations id
  ompt_id_t HostOpId = 0;

  /// Target region data
  ompt_data_t TargetData = ompt_data_none;

  /// Task data representing the encountering task
  ompt_data_t *TaskData = nullptr;

  /// Target task data representing the target task region
  ompt_data_t *TargetTaskData = nullptr;

  /// Used for marking begin of a data operation
  void beginTargetDataOperation();

  /// Used for marking end of a data operation
  void endTargetDataOperation();

  /// Used for marking begin of a target region
  void beginTargetRegion();

  /// Used for marking end of a target region
  void endTargetRegion();
};

/// Thread local state for target region and associated metadata
extern thread_local Interface RegionInterface;

template <typename FuncTy, typename ArgsTy, size_t... IndexSeq>
void InvokeInterfaceFunction(FuncTy Func, ArgsTy Args,
                             std::index_sequence<IndexSeq...>) {
  std::invoke(Func, RegionInterface, std::get<IndexSeq>(Args)...);
}

template <typename CallbackPairTy, typename... ArgsTy> class InterfaceRAII {
public:
  InterfaceRAII(CallbackPairTy Callbacks, ArgsTy... Args)
      : Arguments(Args...), beginFunction(std::get<0>(Callbacks)),
        endFunction(std::get<1>(Callbacks)) {
    performIfOmptInitialized(begin());
  }
  ~InterfaceRAII() { performIfOmptInitialized(end()); }

private:
  void begin() {
    auto IndexSequence =
        std::make_index_sequence<std::tuple_size_v<decltype(Arguments)>>{};
    InvokeInterfaceFunction(beginFunction, Arguments, IndexSequence);
  }

  void end() {
    auto IndexSequence =
        std::make_index_sequence<std::tuple_size_v<decltype(Arguments)>>{};
    InvokeInterfaceFunction(endFunction, Arguments, IndexSequence);
  }

  std::tuple<ArgsTy...> Arguments;
  typename CallbackPairTy::first_type beginFunction;
  typename CallbackPairTy::second_type endFunction;
};

// InterfaceRAII's class template argument deduction guide
template <typename CallbackPairTy, typename... ArgsTy>
InterfaceRAII(CallbackPairTy Callbacks, ArgsTy... Args)
    -> InterfaceRAII<CallbackPairTy, ArgsTy...>;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm
#else
#define OMPT_IF_BUILT(stmt)
#endif

#endif // _OMPTARGET_OMPTINTERFACE_H
