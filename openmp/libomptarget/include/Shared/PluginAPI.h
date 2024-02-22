//===-- Shared/PluginAPI.h - Target independent plugin API ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an interface between target independent OpenMP offload
// runtime library libomptarget and target dependent plugin.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_PLUGIN_API_H
#define OMPTARGET_SHARED_PLUGIN_API_H

#include <cstddef>
#include <cstdint>

#include "Shared/APITypes.h"

extern "C" {

// First method called on the plugin
int32_t __tgt_rtl_init_plugin();

// Return the number of available devices of the type supported by the
// target RTL.
int32_t __tgt_rtl_number_of_devices(void);

// Return an integer different from zero if the provided device image can be
// supported by the runtime. The functionality is similar to comparing the
// result of __tgt__rtl__load__binary to NULL. However, this is meant to be a
// lightweight query to determine if the RTL is suitable for an image without
// having to load the library, which can be expensive.
int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image);

// Return an integer other than zero if the data can be exchaned from SrcDevId
// to DstDevId. If it is data exchangable, the device plugin should provide
// function to move data from source device to destination device directly.
int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDevId, int32_t DstDevId);

// Initialize the requires flags for the device.
int64_t __tgt_rtl_init_requires(int64_t RequiresFlags);

// Initialize the specified device. In case of success return 0; otherwise
// return an error code.
int32_t __tgt_rtl_init_device(int32_t ID);

// Pass an executable image section described by image to the specified
// device and prepare an address table of target entities. In case of error,
// return NULL. Otherwise, return a pointer to the built address table.
// Individual entries in the table may also be NULL, when the corresponding
// offload region is not supported on the target device.
int32_t __tgt_rtl_load_binary(int32_t ID, __tgt_device_image *Image,
                              __tgt_device_binary *Binary);

// Look up the device address of the named symbol in the given binary. Returns
// non-zero on failure.
int32_t __tgt_rtl_get_global(__tgt_device_binary Binary, uint64_t Size,
                             const char *Name, void **DevicePtr);

// Look up the device address of the named kernel in the given binary. Returns
// non-zero on failure.
int32_t __tgt_rtl_get_function(__tgt_device_binary Binary, const char *Name,
                               void **DevicePtr);

// Allocate data on the particular target device, of the specified size.
// HostPtr is a address of the host data the allocated target data
// will be associated with (HostPtr may be NULL if it is not known at
// allocation time, like for example it would be for target data that
// is allocated by omp_target_alloc() API). Return address of the
// allocated data on the target that will be used by libomptarget.so to
// initialize the target data mapping structures. These addresses are
// used to generate a table of target variables to pass to
// __tgt_rtl_run_region(). The __tgt_rtl_data_alloc() returns NULL in
// case an error occurred on the target device. Kind dictates what allocator
// to use (e.g. shared, host, device).
void *__tgt_rtl_data_alloc(int32_t ID, int64_t Size, void *HostPtr,
                           int32_t Kind);

// Pass the data content to the target device using the target address. In case
// of success, return zero. Otherwise, return an error code.
int32_t __tgt_rtl_data_submit(int32_t ID, void *TargetPtr, void *HostPtr,
                              int64_t Size);

int32_t __tgt_rtl_data_submit_async(int32_t ID, void *TargetPtr, void *HostPtr,
                                    int64_t Size, __tgt_async_info *AsyncInfo);

// Retrieve the data content from the target device using its address. In case
// of success, return zero. Otherwise, return an error code.
int32_t __tgt_rtl_data_retrieve(int32_t ID, void *HostPtr, void *TargetPtr,
                                int64_t Size);

// Asynchronous version of __tgt_rtl_data_retrieve
int32_t __tgt_rtl_data_retrieve_async(int32_t ID, void *HostPtr,
                                      void *TargetPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfo);

// Copy the data content from one target device to another target device using
// its address. This operation does not need to copy data back to host and then
// from host to another device. In case of success, return zero. Otherwise,
// return an error code.
int32_t __tgt_rtl_data_exchange(int32_t SrcID, void *SrcPtr, int32_t DstID,
                                void *DstPtr, int64_t Size);

// Asynchronous version of __tgt_rtl_data_exchange
int32_t __tgt_rtl_data_exchange_async(int32_t SrcID, void *SrcPtr,
                                      int32_t DesID, void *DstPtr, int64_t Size,
                                      __tgt_async_info *AsyncInfo);

// De-allocate the data referenced by target ptr on the device. In case of
// success, return zero. Otherwise, return an error code. Kind dictates what
// allocator to use (e.g. shared, host, device).
int32_t __tgt_rtl_data_delete(int32_t ID, void *TargetPtr, int32_t Kind);

// Transfer control to the offloaded entry Entry on the target device.
// Args and Offsets are arrays of NumArgs size of target addresses and
// offsets. An offset should be added to the target address before passing it
// to the outlined function on device side. If AsyncInfo is nullptr, it is
// synchronous; otherwise it is asynchronous. However, AsyncInfo may be
// ignored on some platforms, like x86_64. In that case, it is synchronous. In
// case of success, return zero. Otherwise, return an error code.
int32_t __tgt_rtl_run_target_region(int32_t ID, void *Entry, void **Args,
                                    ptrdiff_t *Offsets, int32_t NumArgs);

// Asynchronous version of __tgt_rtl_run_target_region
int32_t __tgt_rtl_run_target_region_async(int32_t ID, void *Entry, void **Args,
                                          ptrdiff_t *Offsets, int32_t NumArgs,
                                          __tgt_async_info *AsyncInfo);

// Similar to __tgt_rtl_run_target_region, but additionally specify the
// number of teams to be created and a number of threads in each team. If
// AsyncInfo is nullptr, it is synchronous; otherwise it is asynchronous.
// However, AsyncInfo may be ignored on some platforms, like x86_64. In that
// case, it is synchronous.
int32_t __tgt_rtl_run_target_team_region(int32_t ID, void *Entry, void **Args,
                                         ptrdiff_t *Offsets, int32_t NumArgs,
                                         int32_t NumTeams, int32_t ThreadLimit,
                                         uint64_t LoopTripcount);

// Asynchronous version of __tgt_rtl_run_target_team_region
int32_t __tgt_rtl_run_target_team_region_async(
    int32_t ID, void *Entry, void **Args, ptrdiff_t *Offsets, int32_t NumArgs,
    int32_t NumTeams, int32_t ThreadLimit, uint64_t LoopTripcount,
    __tgt_async_info *AsyncInfo);

// Device synchronization. In case of success, return zero. Otherwise, return an
// error code.
int32_t __tgt_rtl_synchronize(int32_t ID, __tgt_async_info *AsyncInfo);

// Queries for the completion of asynchronous operations. Instead of blocking
// the calling thread as __tgt_rtl_synchronize, the progress of the operations
// stored in AsyncInfo->Queue is queried in a non-blocking manner, partially
// advancing their execution. If all operations are completed, AsyncInfo->Queue
// is set to nullptr. If there are still pending operations, AsyncInfo->Queue is
// kept as a valid queue. In any case of success (i.e., successful query
// with/without completing all operations), return zero. Otherwise, return an
// error code.
int32_t __tgt_rtl_query_async(int32_t ID, __tgt_async_info *AsyncInfo);

// Set plugin's internal information flag externally.
void __tgt_rtl_set_info_flag(uint32_t);

// Print the device information
void __tgt_rtl_print_device_info(int32_t ID);

// Event related interfaces. It is expected to use the interfaces in the
// following way:
// 1) Create an event on the target device (__tgt_rtl_create_event).
// 2) Record the event based on the status of \p AsyncInfo->Queue at the moment
// of function call to __tgt_rtl_record_event. An event becomes "meaningful"
// once it is recorded, such that others can depend on it.
// 3) Call __tgt_rtl_wait_event to set dependence on the event. Whether the
// operation is blocking or non-blocking depends on the target. It is expected
// to be non-blocking, just set dependence and return.
// 4) Call __tgt_rtl_sync_event to sync the event. It is expected to block the
// thread calling the function.
// 5) Destroy the event (__tgt_rtl_destroy_event).
// {
int32_t __tgt_rtl_create_event(int32_t ID, void **Event);

int32_t __tgt_rtl_record_event(int32_t ID, void *Event,
                               __tgt_async_info *AsyncInfo);

int32_t __tgt_rtl_wait_event(int32_t ID, void *Event,
                             __tgt_async_info *AsyncInfo);

int32_t __tgt_rtl_sync_event(int32_t ID, void *Event);

int32_t __tgt_rtl_destroy_event(int32_t ID, void *Event);
// }

int32_t __tgt_rtl_init_async_info(int32_t ID, __tgt_async_info **AsyncInfoPtr);
int32_t __tgt_rtl_init_device_info(int32_t ID, __tgt_device_info *DeviceInfoPtr,
                                   const char **ErrStr);

// lock/pin host memory
int32_t __tgt_rtl_data_lock(int32_t ID, void *HstPtr, int64_t Size,
                            void **LockedPtr);

// unlock/unpin host memory
int32_t __tgt_rtl_data_unlock(int32_t ID, void *HstPtr);

// Notify the plugin about a new mapping starting at the host address \p HstPtr
// and \p Size bytes. The plugin may lock/pin that buffer to achieve optimal
// memory transfers involving that buffer.
int32_t __tgt_rtl_data_notify_mapped(int32_t ID, void *HstPtr, int64_t Size);

// Notify the plugin about an existing mapping being unmapped, starting at the
// host address \p HstPtr and \p Size bytes.
int32_t __tgt_rtl_data_notify_unmapped(int32_t ID, void *HstPtr);

// Set the global device identifier offset, such that the plugin may determine a
// unique device number.
int32_t __tgt_rtl_set_device_offset(int32_t DeviceIdOffset);

int32_t __tgt_rtl_launch_kernel(int32_t DeviceId, void *TgtEntryPtr,
                                void **TgtArgs, ptrdiff_t *TgtOffsets,
                                KernelArgsTy *KernelArgs,
                                __tgt_async_info *AsyncInfoPtr);

int32_t __tgt_rtl_initialize_record_replay(int32_t DeviceId, int64_t MemorySize,
                                           void *VAddr, bool isRecord,
                                           bool SaveOutput,
                                           uint64_t &ReqPtrArgOffset);

// Returns true if the device \p DeviceId suggests to use auto zero-copy.
int32_t __tgt_rtl_use_auto_zero_copy(int32_t DeviceId);
}

#endif // OMPTARGET_SHARED_PLUGIN_API_H
