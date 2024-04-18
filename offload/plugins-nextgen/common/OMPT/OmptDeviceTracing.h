//===- OmptDeviceTracing.h - Target independent OMPT callbacks --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface used by target-independent runtimes to coordinate registration and
// invocation of OMPT tracing functionality.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OMPTDEVICETRACING_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OMPTDEVICETRACING_H

#ifdef OMPT_SUPPORT

#include "OmptCommonDefs.h"

#include "llvm/Support/DynamicLibrary.h"

#include <map>
#include <memory>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

extern void setOmptAsyncCopyProfile(bool Enable);
extern void setGlobalOmptKernelProfile(int DeviceId, int Enable);
extern uint64_t getSystemTimestampInNs();

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

// Declare OMPT device tracing function entry points
#define declareOmptTracingFn(Name) extern libomptarget_##Name##_t Name##_fn;
FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(declareOmptTracingFn)
#undef declareOmptTracingFn

// Declare OMPT device tracing function mutexes
#define declareOmptTracingFnMutex(Name) extern std::mutex Name##_mutex;
FOREACH_OMPT_DEVICE_TRACING_FN_IMPLEMENTAIONS(declareOmptTracingFnMutex)
#undef declareOmptTracingFnMutex

extern std::mutex DeviceIdWritingMutex;

/// Activate / deactivate tracing
void setTracingState(bool Enabled);

/// Set 'start' and 'stop' in trace records
void setOmptTimestamp(uint64_t StartTime, uint64_t EndTime);

/// Set the linear function correlation between host and device clocks
void setOmptHostToDeviceRate(double Slope, double Offset);

/// Set / store the number of granted teams in trace records
void setOmptGrantedNumTeams(uint64_t NumTeams);

/// Lookup the given device pointer and return its RTL device ID
int getDeviceId(ompt_device_t *Device);

/// Map the given device pointer to the given DeviceId
void setDeviceId(ompt_device_t *Device, int32_t DeviceId);

/// Rempve the given device pointer from the current mapping
void removeDeviceId(ompt_device_t *Device);

/// Provide name based lookup for the device tracing functions
extern ompt_interface_fn_t
lookupDeviceTracingFn(const char *InterfaceFunctionName);

/// Host to device linear clock correlation
extern double HostToDeviceSlope;

/// Host to device constant clock offset
extern double HostToDeviceOffset;

/// Mapping of device pointers to their corresponding RTL device ID
extern std::map<ompt_device_t *, int32_t> Devices;

// Keep track of enabled tracing event types
extern std::atomic<uint64_t> TracingTypesEnabled;

/// OMPT tracing status; (Re-)Set via 'setTracingState'
extern bool TracingActive;

/// Parent library pointer
extern std::shared_ptr<llvm::sys::DynamicLibrary> ParentLibrary;

/// Get the parent library by pointer. If it is not already set, it will set the
/// parent library pointer.
std::shared_ptr<llvm::sys::DynamicLibrary> getParentLibrary();

/// Set the parent library by filename
void setParentLibrary(const char *Filename);

/// Search for FuncName inside the parent library and assign to FuncPtr.
/// IMPORTANT: This function assumes that the *caller* holds the respective lock
/// for FuncPtr.
template <typename FT>
void ensureFuncPtrLoaded(const std::string &FuncName, FT *FuncPtr) {
  if (*FuncPtr == nullptr) {
    if ((ParentLibrary == nullptr && getParentLibrary() == nullptr) ||
        !ParentLibrary->isValid())
      return;
    void *SymbolPtr = ParentLibrary->getAddressOfSymbol(FuncName.c_str());
    if (SymbolPtr == nullptr)
      return;
    *FuncPtr = reinterpret_cast<FT>(SymbolPtr);
  }
}

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OMPTDEVICETRACING_H
