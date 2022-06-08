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

#include "OmptCommon.h"

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

void compute_parent_dyn_lib(const char *lib_name);

std::shared_ptr<llvm::sys::DynamicLibrary> get_parent_dyn_lib();

/// Search for FuncName inside the OmptDeviceCallbacks object and assign to
/// FuncPtr.
/// IMPORTANT: This function assumes that the *caller* holds the respective lock
/// for FuncPtr.
template <typename FT>
void ensureFuncPtrLoaded(const std::string &FuncName, FT *FuncPtr) {
  if (!(*FuncPtr)) {
    auto libomptarget_dyn_lib = get_parent_dyn_lib();
    if (libomptarget_dyn_lib == nullptr || !libomptarget_dyn_lib->isValid())
      return;
    void *VPtr = libomptarget_dyn_lib->getAddressOfSymbol(FuncName.c_str());
    if (!VPtr)
      return;
    *FuncPtr = reinterpret_cast<FT>(VPtr);
  }
}

double HostToDeviceSlope = .0;
double HostToDeviceOffset = .0;

libomptarget_ompt_set_trace_ompt_t ompt_set_trace_ompt_fn = nullptr;
libomptarget_ompt_start_trace_t ompt_start_trace_fn = nullptr;
libomptarget_ompt_flush_trace_t ompt_flush_trace_fn = nullptr;
libomptarget_ompt_stop_trace_t ompt_stop_trace_fn = nullptr;
libomptarget_ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor_fn =
    nullptr;
libomptarget_ompt_get_record_type_t ompt_get_record_type_fn = nullptr;

/// Libomptarget function that will be used to set timestamps in trace records.
libomptarget_ompt_set_timestamp_t ompt_set_timestamp_fn = nullptr;
/// Libomptarget function that will be used to set num_teams in trace records.
libomptarget_ompt_set_granted_teams_t ompt_set_granted_teams_fn = nullptr;

std::mutex set_trace_mutex;
std::mutex start_trace_mutex;
std::mutex flush_trace_mutex;
std::mutex stop_trace_mutex;
std::mutex advance_buffer_cursor_mutex;
std::mutex get_record_type_mutex;
std::mutex ompt_set_timestamp_mtx;
std::mutex granted_teams_mtx;

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif // OMPT_SUPPORT

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_OMPTDEVICETRACING_H
