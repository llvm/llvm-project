//===-------- omptarget.h - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_H_
#define _OMPTARGET_H_

#include <stdint.h>
#include <stddef.h>

#define OFFLOAD_SUCCESS (0)
#define OFFLOAD_FAIL (~0)

#define OFFLOAD_DEVICE_DEFAULT     -1
#define HOST_DEVICE                -10

/// Data attributes for each data reference used in an OpenMP target region.
enum tgt_map_type {
  // No flags
  OMP_TGT_MAPTYPE_NONE            = 0x000,
  // copy data from host to device
  OMP_TGT_MAPTYPE_TO              = 0x001,
  // copy data from device to host
  OMP_TGT_MAPTYPE_FROM            = 0x002,
  // copy regardless of the reference count
  OMP_TGT_MAPTYPE_ALWAYS          = 0x004,
  // force unmapping of data
  OMP_TGT_MAPTYPE_DELETE          = 0x008,
  // map the pointer as well as the pointee
  OMP_TGT_MAPTYPE_PTR_AND_OBJ     = 0x010,
  // pass device base address to kernel
  OMP_TGT_MAPTYPE_TARGET_PARAM    = 0x020,
  // return base device address of mapped data
  OMP_TGT_MAPTYPE_RETURN_PARAM    = 0x040,
  // private variable - not mapped
  OMP_TGT_MAPTYPE_PRIVATE         = 0x080,
  // copy by value - not mapped
  OMP_TGT_MAPTYPE_LITERAL         = 0x100,
  // mapping is implicit
  OMP_TGT_MAPTYPE_IMPLICIT        = 0x200,
  // copy data to device
  OMP_TGT_MAPTYPE_CLOSE           = 0x400,
  // member of struct, member given by [16 MSBs] - 1
  OMP_TGT_MAPTYPE_MEMBER_OF       = 0xffff000000000000
};

enum OpenMPOffloadingDeclareTargetFlags {
  /// Mark the entry as having a 'link' attribute.
  OMP_DECLARE_TARGET_LINK = 0x01,
  /// Mark the entry as being a global constructor.
  OMP_DECLARE_TARGET_CTOR = 0x02,
  /// Mark the entry as being a global destructor.
  OMP_DECLARE_TARGET_DTOR = 0x04
};

enum OpenMPOffloadingRequiresDirFlags {
  /// flag undefined.
  OMP_REQ_UNDEFINED               = 0x000,
  /// no requires directive present.
  OMP_REQ_NONE                    = 0x001,
  /// reverse_offload clause.
  OMP_REQ_REVERSE_OFFLOAD         = 0x002,
  /// unified_address clause.
  OMP_REQ_UNIFIED_ADDRESS         = 0x004,
  /// unified_shared_memory clause.
  OMP_REQ_UNIFIED_SHARED_MEMORY   = 0x008,
  /// dynamic_allocators clause.
  OMP_REQ_DYNAMIC_ALLOCATORS      = 0x010
};

/// This struct is a record of an entry point or global. For a function
/// entry point the size is expected to be zero
struct __tgt_offload_entry {
  void *addr;   // Pointer to the offload entry info (function or global)
  char *name;   // Name of the function or global
  size_t size;  // Size of the entry info (0 if it is a function)
  int32_t flags; // Flags associated with the entry, e.g. 'link'.
  int32_t reserved; // Reserved, to be used by the runtime library.
};

/// This struct is a record of the device image information
struct __tgt_device_image {
  void *ImageStart;                  // Pointer to the target code start
  void *ImageEnd;                    // Pointer to the target code end
  __tgt_offload_entry *EntriesBegin; // Begin of table with all target entries
  __tgt_offload_entry *EntriesEnd;   // End of table (non inclusive)
};

/// This struct is a record of all the host code that may be offloaded to a
/// target.
struct __tgt_bin_desc {
  int32_t NumDeviceImages;           // Number of device types supported
  __tgt_device_image *DeviceImages;  // Array of device images (1 per dev. type)
  __tgt_offload_entry *HostEntriesBegin; // Begin of table with all host entries
  __tgt_offload_entry *HostEntriesEnd;   // End of table (non inclusive)
};

/// This struct contains the offload entries identified by the target runtime
struct __tgt_target_table {
  __tgt_offload_entry *EntriesBegin; // Begin of the table with all the entries
  __tgt_offload_entry
      *EntriesEnd; // End of the table with all the entries (non inclusive)
};

/// This struct contains information exchanged between different asynchronous
/// operations for device-dependent optimization and potential synchronization
struct __tgt_async_info {
  // A pointer to a queue-like structure where offloading operations are issued.
  // We assume to use this structure to do synchronization. In CUDA backend, it
  // is CUstream.
  void *Queue = nullptr;
};

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_num_devices(void);
int omp_get_initial_device(void);
void *omp_target_alloc(size_t size, int device_num);
void omp_target_free(void *device_ptr, int device_num);
int omp_target_is_present(void *ptr, int device_num);
int omp_target_memcpy(void *dst, void *src, size_t length, size_t dst_offset,
    size_t src_offset, int dst_device, int src_device);
int omp_target_memcpy_rect(void *dst, void *src, size_t element_size,
    int num_dims, const size_t *volume, const size_t *dst_offsets,
    const size_t *src_offsets, const size_t *dst_dimensions,
    const size_t *src_dimensions, int dst_device, int src_device);
int omp_target_associate_ptr(void *host_ptr, void *device_ptr, size_t size,
    size_t device_offset, int device_num);
int omp_target_disassociate_ptr(void *host_ptr, int device_num);

/// add the clauses of the requires directives in a given file
void __tgt_register_requires(int64_t flags);

/// adds a target shared library to the target execution image
void __tgt_register_lib(__tgt_bin_desc *desc);

/// removes a target shared library from the target execution image
void __tgt_unregister_lib(__tgt_bin_desc *desc);

// creates the host to target data mapping, stores it in the
// libomptarget.so internal structure (an entry in a stack of data maps) and
// passes the data to the device;
void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
                             void **args_base, void **args, int64_t *arg_sizes,
                             int64_t *arg_types);
void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int64_t *arg_types,
                                    int32_t depNum, void *depList,
                                    int32_t noAliasDepNum,
                                    void *noAliasDepList);
void __tgt_target_data_begin_mapper(int64_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int64_t *arg_types,
                                    void **arg_mappers);
void __tgt_target_data_begin_nowait_mapper(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList);

// passes data from the target, release target memory and destroys the
// host-target mapping (top entry from the stack of data maps) created by
// the last __tgt_target_data_begin
void __tgt_target_data_end(int64_t device_id, int32_t arg_num, void **args_base,
                           void **args, int64_t *arg_sizes, int64_t *arg_types);
void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int64_t *arg_types,
                                  int32_t depNum, void *depList,
                                  int32_t noAliasDepNum, void *noAliasDepList);
void __tgt_target_data_end_mapper(int64_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int64_t *arg_types,
                                  void **arg_mappers);
void __tgt_target_data_end_nowait_mapper(int64_t device_id, int32_t arg_num,
                                         void **args_base, void **args,
                                         int64_t *arg_sizes, int64_t *arg_types,
                                         void **arg_mappers, int32_t depNum,
                                         void *depList, int32_t noAliasDepNum,
                                         void *noAliasDepList);

/// passes data to/from the target
void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
                              void **args_base, void **args, int64_t *arg_sizes,
                              int64_t *arg_types);
void __tgt_target_data_update_nowait(int64_t device_id, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int64_t *arg_types,
                                     int32_t depNum, void *depList,
                                     int32_t noAliasDepNum,
                                     void *noAliasDepList);
void __tgt_target_data_update_mapper(int64_t device_id, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int64_t *arg_types,
                                     void **arg_mappers);
void __tgt_target_data_update_nowait_mapper(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, void **arg_mappers, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList);

// Performs the same actions as data_begin in case arg_num is non-zero
// and initiates run of offloaded region on target platform; if arg_num
// is non-zero after the region execution is done it also performs the
// same action as data_end above. The following types are used; this
// function returns 0 if it was able to transfer the execution to a
// target and an int different from zero otherwise.
int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
                 void **args_base, void **args, int64_t *arg_sizes,
                 int64_t *arg_types);
int __tgt_target_nowait(int64_t device_id, void *host_ptr, int32_t arg_num,
                        void **args_base, void **args, int64_t *arg_sizes,
                        int64_t *arg_types, int32_t depNum, void *depList,
                        int32_t noAliasDepNum, void *noAliasDepList);
int __tgt_target_mapper(int64_t device_id, void *host_ptr, int32_t arg_num,
                        void **args_base, void **args, int64_t *arg_sizes,
                        int64_t *arg_types, void **arg_mappers);
int __tgt_target_nowait_mapper(int64_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int64_t *arg_types,
                               void **arg_mappers, int32_t depNum,
                               void *depList, int32_t noAliasDepNum,
                               void *noAliasDepList);

int __tgt_target_teams(int64_t device_id, void *host_ptr, int32_t arg_num,
                       void **args_base, void **args, int64_t *arg_sizes,
                       int64_t *arg_types, int32_t num_teams,
                       int32_t thread_limit);
int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int64_t *arg_types,
                              int32_t num_teams, int32_t thread_limit,
                              int32_t depNum, void *depList,
                              int32_t noAliasDepNum, void *noAliasDepList);
int __tgt_target_teams_mapper(int64_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int64_t *arg_types,
                              void **arg_mappers, int32_t num_teams,
                              int32_t thread_limit);
int __tgt_target_teams_nowait_mapper(
    int64_t device_id, void *host_ptr, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types, void **arg_mappers,
    int32_t num_teams, int32_t thread_limit, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList);

void __kmpc_push_target_tripcount(int64_t device_id, uint64_t loop_tripcount);

#ifdef __cplusplus
}
#endif

#ifdef OMPTARGET_DEBUG
#include <stdio.h>
#define DEBUGP(prefix, ...)                                                    \
  {                                                                            \
    fprintf(stderr, "%s --> ", prefix);                                        \
    fprintf(stderr, __VA_ARGS__);                                              \
  }

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <inttypes.h>
#define DPxMOD "0x%0*" PRIxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))

/*
 * To printf a pointer in hex with a fixed width of 16 digits and a leading 0x,
 * use printf("ptr=" DPxMOD "...\n", DPxPTR(ptr));
 *
 * DPxMOD expands to:
 *   "0x%0*" PRIxPTR
 * where PRIxPTR expands to an appropriate modifier for the type uintptr_t on a
 * specific platform, e.g. "lu" if uintptr_t is typedef'd as unsigned long:
 *   "0x%0*lu"
 *
 * Ultimately, the whole statement expands to:
 *   printf("ptr=0x%0*lu...\n",  // the 0* modifier expects an extra argument
 *                               // specifying the width of the output
 *   (int)(2*sizeof(uintptr_t)), // the extra argument specifying the width
 *                               // 8 digits for 32bit systems
 *                               // 16 digits for 64bit
 *   (uintptr_t) ptr);
 */
#else
#define DEBUGP(prefix, ...)                                                    \
  {}
#endif

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

#endif // _OMPTARGET_H_
