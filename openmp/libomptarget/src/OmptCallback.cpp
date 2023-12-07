//===-- OmptCallback.cpp - Target independent OpenMP target RTL --- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OMPT callback interfaces for target independent layer
//
//===----------------------------------------------------------------------===//

#ifdef OMPT_SUPPORT

#include "llvm/Support/DynamicLibrary.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "Debug.h"
#include "OmptCallback.h"
#include "OmptConnector.h"
#include "OmptInterface.h"

#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

using namespace llvm::omp::target::ompt;

// Define OMPT callback functions (bound to actual callbacks later on)
#define defineOmptCallback(Name, Type, Code)                                   \
  Name##_t llvm::omp::target::ompt::Name##_fn = nullptr;
FOREACH_OMPT_NOEMI_EVENT(defineOmptCallback)
FOREACH_OMPT_EMI_EVENT(defineOmptCallback)
#undef defineOmptCallback

/// Forward declaration
class LibomptargetRtlFinalizer;

/// Object that will maintain the RTL finalizer from the plugin
LibomptargetRtlFinalizer *LibraryFinalizer = nullptr;

thread_local Interface llvm::omp::target::ompt::RegionInterface;

bool llvm::omp::target::ompt::Initialized = false;

ompt_get_callback_t llvm::omp::target::ompt::lookupCallbackByCode = nullptr;
ompt_function_lookup_t llvm::omp::target::ompt::lookupCallbackByName = nullptr;
ompt_get_target_task_data_t ompt_get_target_task_data_fn = nullptr;
ompt_get_task_data_t ompt_get_task_data_fn = nullptr;

/// Unique correlation id
static std::atomic<uint64_t> IdCounter(1);

/// Used to create a new correlation id
static uint64_t createId() { return IdCounter.fetch_add(1); }

/// Create a new correlation id and update the operations id
static uint64_t createOpId() {
  uint64_t NewId = createId();
  RegionInterface.setHostOpId(NewId);
  return NewId;
}

/// Create a new correlation id and update the target region id
static uint64_t createRegionId() {
  uint64_t NewId = createId();
  RegionInterface.setTargetDataValue(NewId);
  return NewId;
}

void Interface::beginTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                                     void **TgtPtrBegin, size_t Size,
                                     void *Code) {
  beginTargetDataOperation();
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_begin, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_alloc, HstPtrBegin,
        /* SrcDeviceNum */ omp_get_initial_device(), *TgtPtrBegin,
        /* TgtDeviceNum */ DeviceId, Size, Code);
  } else if (ompt_callback_target_data_op_fn) {
    // HostOpId is set by the runtime
    HostOpId = createOpId();
    // Invoke the tool supplied data op callback
    ompt_callback_target_data_op_fn(
        TargetData.value, HostOpId, ompt_target_data_alloc, HstPtrBegin,
        /* SrcDeviceNum */ omp_get_initial_device(), *TgtPtrBegin,
        /* TgtDeviceNum */ DeviceId, Size, Code);
  }
}

void Interface::endTargetDataAlloc(int64_t DeviceId, void *HstPtrBegin,
                                   void **TgtPtrBegin, size_t Size,
                                   void *Code) {
  // Only EMI callback handles end scope
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_end, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_alloc, HstPtrBegin,
        /* SrcDeviceNum */ omp_get_initial_device(), *TgtPtrBegin,
        /* TgtDeviceNum */ DeviceId, Size, Code);
  }
  endTargetDataOperation();
}

void Interface::beginTargetDataSubmit(int64_t DeviceId, void *TgtPtrBegin,
                                      void *HstPtrBegin, size_t Size,
                                      void *Code) {
  beginTargetDataOperation();
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_begin, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_transfer_to_device, HstPtrBegin,
        /* SrcDeviceNum */ omp_get_initial_device(), TgtPtrBegin, DeviceId,
        Size, Code);
  } else if (ompt_callback_target_data_op_fn) {
    // HostOpId is set by the runtime
    HostOpId = createOpId();
    // Invoke the tool supplied data op callback
    ompt_callback_target_data_op_fn(
        TargetData.value, HostOpId, ompt_target_data_transfer_to_device,
        HstPtrBegin, /* SrcDeviceNum */ omp_get_initial_device(), TgtPtrBegin,
        DeviceId, Size, Code);
  }
}

void Interface::endTargetDataSubmit(int64_t DeviceId, void *TgtPtrBegin,
                                    void *HstPtrBegin, size_t Size,
                                    void *Code) {
  // Only EMI callback handles end scope
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_end, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_transfer_to_device, HstPtrBegin,
        /* SrcDeviceNum */ omp_get_initial_device(), TgtPtrBegin, DeviceId,
        Size, Code);
  }
  endTargetDataOperation();
}

void Interface::beginTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin,
                                      void *Code) {
  beginTargetDataOperation();
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_begin, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_delete, TgtPtrBegin, DeviceId,
        /* TgtPtrBegin */ nullptr, /* TgtDeviceNum */ -1, /* Bytes */ 0, Code);
  } else if (ompt_callback_target_data_op_fn) {
    // HostOpId is set by the runtime
    HostOpId = createOpId();
    // Invoke the tool supplied data op callback
    ompt_callback_target_data_op_fn(TargetData.value, HostOpId,
                                    ompt_target_data_delete, TgtPtrBegin,
                                    DeviceId, /* TgtPtrBegin */ nullptr,
                                    /* TgtDeviceNum */ -1, /* Bytes */ 0, Code);
  }
}

void Interface::endTargetDataDelete(int64_t DeviceId, void *TgtPtrBegin,
                                    void *Code) {
  // Only EMI callback handles end scope
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_end, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_delete, TgtPtrBegin, DeviceId,
        /* TgtPtrBegin */ nullptr, /* TgtDeviceNum */ -1, /* Bytes */ 0, Code);
  }
  endTargetDataOperation();
}

void Interface::beginTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                                        void *TgtPtrBegin, size_t Size,
                                        void *Code) {
  beginTargetDataOperation();
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_begin, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_transfer_from_device, TgtPtrBegin, DeviceId,
        HstPtrBegin,
        /* TgtDeviceNum */ omp_get_initial_device(), Size, Code);
  } else if (ompt_callback_target_data_op_fn) {
    // HostOpId is set by the runtime
    HostOpId = createOpId();
    // Invoke the tool supplied data op callback
    ompt_callback_target_data_op_fn(
        TargetData.value, HostOpId, ompt_target_data_transfer_from_device,
        TgtPtrBegin, DeviceId, HstPtrBegin,
        /* TgtDeviceNum */ omp_get_initial_device(), Size, Code);
  }
}

void Interface::endTargetDataRetrieve(int64_t DeviceId, void *HstPtrBegin,
                                      void *TgtPtrBegin, size_t Size,
                                      void *Code) {
  // Only EMI callback handles end scope
  if (ompt_callback_target_data_op_emi_fn) {
    // HostOpId will be set by the tool. Invoke the tool supplied data op EMI
    // callback
    ompt_callback_target_data_op_emi_fn(
        ompt_scope_end, TargetTaskData, &TargetData, &TargetRegionOpId,
        ompt_target_data_transfer_from_device, TgtPtrBegin, DeviceId,
        HstPtrBegin,
        /* TgtDeviceNum */ omp_get_initial_device(), Size, Code);
  }
  endTargetDataOperation();
}

void Interface::beginTargetSubmit(unsigned int numTeams) {
  if (ompt_callback_target_submit_emi_fn) {
    // HostOpId is set by the tool. Invoke the tool supplied target submit EMI
    // callback
    ompt_callback_target_submit_emi_fn(ompt_scope_begin, &TargetData, &HostOpId,
                                       numTeams);
  } else if (ompt_callback_target_submit_fn) {
    // HostOpId is set by the runtime
    HostOpId = createOpId();
    ompt_callback_target_submit_fn(TargetData.value, HostOpId, numTeams);
  }
}

void Interface::endTargetSubmit(unsigned int numTeams) {
  // Only EMI callback handles end scope
  if (ompt_callback_target_submit_emi_fn) {
    // HostOpId is set by the tool. Invoke the tool supplied target submit EMI
    // callback
    ompt_callback_target_submit_emi_fn(ompt_scope_end, &TargetData, &HostOpId,
                                       numTeams);
  }
}

void Interface::beginTargetDataEnter(int64_t DeviceId, void *Code) {
  beginTargetRegion();
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_enter_data, ompt_scope_begin,
                                DeviceId, TaskData, TargetTaskData, &TargetData,
                                Code);
  } else if (ompt_callback_target_fn) {
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_enter_data, ompt_scope_begin, DeviceId,
                            TaskData, TargetData.value, Code);
  }
}

void Interface::endTargetDataEnter(int64_t DeviceId, void *Code) {
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_enter_data, ompt_scope_end,
                                DeviceId, TaskData, TargetTaskData, &TargetData,
                                Code);
  } else if (ompt_callback_target_fn) {
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_enter_data, ompt_scope_end, DeviceId,
                            TaskData, TargetData.value, Code);
  }
  endTargetRegion();
}

void Interface::beginTargetDataExit(int64_t DeviceId, void *Code) {
  beginTargetRegion();
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_exit_data, ompt_scope_begin,
                                DeviceId, TaskData, TargetTaskData, &TargetData,
                                Code);
  } else if (ompt_callback_target_fn) {
    TargetData.value = createRegionId();
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_exit_data, ompt_scope_begin, DeviceId,
                            TaskData, TargetData.value, Code);
  }
}

void Interface::endTargetDataExit(int64_t DeviceId, void *Code) {
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_exit_data, ompt_scope_end, DeviceId,
                                TaskData, TargetTaskData, &TargetData, Code);
  } else if (ompt_callback_target_fn) {
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_exit_data, ompt_scope_end, DeviceId,
                            TaskData, TargetData.value, Code);
  }
  endTargetRegion();
}

void Interface::beginTargetUpdate(int64_t DeviceId, void *Code) {
  beginTargetRegion();
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_update, ompt_scope_begin, DeviceId,
                                TaskData, TargetTaskData, &TargetData, Code);
  } else if (ompt_callback_target_fn) {
    TargetData.value = createRegionId();
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_update, ompt_scope_begin, DeviceId,
                            TaskData, TargetData.value, Code);
  }
}

void Interface::endTargetUpdate(int64_t DeviceId, void *Code) {
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target_update, ompt_scope_end, DeviceId,
                                TaskData, TargetTaskData, &TargetData, Code);
  } else if (ompt_callback_target_fn) {
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target_update, ompt_scope_end, DeviceId,
                            TaskData, TargetData.value, Code);
  }
  endTargetRegion();
}

void Interface::beginTarget(int64_t DeviceId, void *Code) {
  beginTargetRegion();
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target, ompt_scope_begin, DeviceId,
                                TaskData, TargetTaskData, &TargetData, Code);
  } else if (ompt_callback_target_fn) {
    TargetData.value = createRegionId();
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target, ompt_scope_begin, DeviceId, TaskData,
                            TargetData.value, Code);
  }
}

void Interface::endTarget(int64_t DeviceId, void *Code) {
  if (ompt_callback_target_emi_fn) {
    // Invoke the tool supplied target EMI callback
    ompt_callback_target_emi_fn(ompt_target, ompt_scope_end, DeviceId, TaskData,
                                TargetTaskData, &TargetData, Code);
  } else if (ompt_callback_target_fn) {
    // Invoke the tool supplied target callback
    ompt_callback_target_fn(ompt_target, ompt_scope_end, DeviceId, TaskData,
                            TargetData.value, Code);
  }
  endTargetRegion();
}

void Interface::beginTargetDataOperation() {
  DP("in ompt_target_region_begin (TargetRegionOpId = %lu)\n",
     TargetData.value);
}

void Interface::endTargetDataOperation() {
  DP("in ompt_target_region_end (TargetRegionOpId = %lu)\n", TargetData.value);
}

void Interface::beginTargetRegion() {
  // Set up task state
  assert(ompt_get_task_data_fn && "Calling a null task data function");
  TaskData = ompt_get_task_data_fn();
  // Set up target task state
  assert(ompt_get_target_task_data_fn &&
         "Calling a null target task data function");
  TargetTaskData = ompt_get_target_task_data_fn();
  // Target state will be set later
  TargetData = ompt_data_none;
}

void Interface::endTargetRegion() {
  TaskData = 0;
  TargetTaskData = 0;
  TargetData = ompt_data_none;
}

/// Used to maintain the finalization functions that are received
/// from the plugins during connect.
/// Note: Currently, there are no plugin-specific finalizations, so each plugin
/// will call the same (empty) function.
class LibomptargetRtlFinalizer {
public:
  LibomptargetRtlFinalizer() {}

  void registerRtl(ompt_finalize_t FinalizationFunction) {
    if (FinalizationFunction) {
      RtlFinalizationFunctions.emplace_back(FinalizationFunction);
    }
  }

  void finalize() {
    for (auto FinalizationFunction : RtlFinalizationFunctions)
      FinalizationFunction(/* tool_data */ nullptr);
    RtlFinalizationFunctions.clear();
  }

private:
  llvm::SmallVector<ompt_finalize_t> RtlFinalizationFunctions;
};

int llvm::omp::target::ompt::initializeLibrary(ompt_function_lookup_t lookup,
                                               int initial_device_num,
                                               ompt_data_t *tool_data) {
  DP("Executing initializeLibrary (libomp)\n");
#define bindOmptFunctionName(OmptFunction, DestinationFunction)                \
  DestinationFunction = (OmptFunction##_t)lookup(#OmptFunction);               \
  DP("initializeLibrary (libomp) bound %s=%p\n", #DestinationFunction,         \
     ((void *)(uint64_t)DestinationFunction));

  bindOmptFunctionName(ompt_get_callback, lookupCallbackByCode);
  bindOmptFunctionName(ompt_get_task_data, ompt_get_task_data_fn);
  bindOmptFunctionName(ompt_get_target_task_data, ompt_get_target_task_data_fn);
#undef bindOmptFunctionName

  // Store pointer of 'ompt_libomp_target_fn_lookup' for use by libomptarget
  lookupCallbackByName = lookup;

  assert(lookupCallbackByCode && "lookupCallbackByCode should be non-null");
  assert(lookupCallbackByName && "lookupCallbackByName should be non-null");
  assert(ompt_get_task_data_fn && "ompt_get_task_data_fn should be non-null");
  assert(ompt_get_target_task_data_fn &&
         "ompt_get_target_task_data_fn should be non-null");
  assert(LibraryFinalizer == nullptr &&
         "LibraryFinalizer should not be initialized yet");

  LibraryFinalizer = new LibomptargetRtlFinalizer();

  Initialized = true;

  return 0;
}

void llvm::omp::target::ompt::finalizeLibrary(ompt_data_t *data) {
  DP("Executing finalizeLibrary (libomp)\n");
  // Before disabling OMPT, call the (plugin) finalizations that were registered
  // with this library
  LibraryFinalizer->finalize();
  delete LibraryFinalizer;
  Initialized = false;
}

void llvm::omp::target::ompt::connectLibrary() {
  DP("Entering connectLibrary (libomp)\n");
  // Connect with libomp
  static OmptLibraryConnectorTy LibompConnector("libomp");
  static ompt_start_tool_result_t OmptResult;

  // Initialize OmptResult with the init and fini functions that will be
  // called by the connector
  OmptResult.initialize = ompt::initializeLibrary;
  OmptResult.finalize = ompt::finalizeLibrary;
  OmptResult.tool_data.value = 0;

  // Now call connect that causes the above init/fini functions to be called
  LibompConnector.connect(&OmptResult);

#define bindOmptCallback(Name, Type, Code)                                     \
  if (lookupCallbackByCode)                                                    \
    lookupCallbackByCode(                                                      \
        (ompt_callbacks_t)(Code),                                              \
        (ompt_callback_t *)&(llvm::omp::target::ompt::Name##_fn));
  FOREACH_OMPT_NOEMI_EVENT(bindOmptCallback)
  FOREACH_OMPT_EMI_EVENT(bindOmptCallback)
#undef bindOmptCallback

  DP("Exiting connectLibrary (libomp)\n");
}

extern "C" {
/// Used for connecting libomptarget with a plugin
void ompt_libomptarget_connect(ompt_start_tool_result_t *result) {
  DP("Enter ompt_libomptarget_connect\n");
  if (Initialized && result && LibraryFinalizer) {
    // Cache each fini function, so that they can be invoked on exit
    LibraryFinalizer->registerRtl(result->finalize);
    // Invoke the provided init function with the lookup function maintained
    // in this library so that callbacks maintained by this library are
    // retrieved.
    result->initialize(lookupCallbackByName,
                       /* initial_device_num */ 0, /* tool_data */ nullptr);
  }
  DP("Leave ompt_libomptarget_connect\n");
}
}
#else
extern "C" {
/// Dummy definition when OMPT is disabled
void ompt_libomptarget_connect() {}
}
#endif // OMPT_SUPPORT
