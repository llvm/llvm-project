//===- Interface.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __TGT_ACC_TARGET_H__
#define __TGT_ACC_TARGET_H__

#ifdef __cplusplus
#include "flang-rt/runtime/descriptor.h"
using namespace Fortran::ISO;
#else
#include "flang/ISO_Fortran_binding.h"
#endif

#include "Shared/APITypes.h"
#include "Shared/SourceInfo.h"

#include <stddef.h>
#include <stdint.h>

// Portable alignment attribute for C89/C99 compatibility
#if defined(_MSC_VER)
#define ACC_ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__) || defined(__clang__)
#define ACC_ALIGNED(x) __attribute__((aligned(x)))
#else
#define ACC_ALIGNED(x)
#endif

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// OpenACC Target Offload Types and Enums
//===----------------------------------------------------------------------===//

enum {
  // no flags
  TGT_ACC_MAPTYPE_NONE = 0x0,
  // copy data from host to device
  TGT_ACC_MAPTYPE_TO = 0x1, // enter data
  // copy data from device to host
  TGT_ACC_MAPTYPE_FROM = 0x2, // exit data
  // force unmapping of data
  TGT_ACC_MAPTYPE_FINALIZE = 0x8,
  // map the pointer as well as the pointee
  TGT_ACC_MAPTYPE_PTR_AND_OBJ = 0x10,
  // private variable - not mapped
  TGT_ACC_MAPTYPE_PRIVATE = 0x80,
  // copy by value - not mapped
  TGT_ACC_MAPTYPE_LITERAL = 0x100,
  // device pointer - already mapped
  TGT_ACC_MAPTYPE_DEVPTR = 0x400,
  // device pointer
  TGT_ACC_MAPTYPE_MANAGED_DEVPTR = 0x800,
  // present or don't create
  TGT_ACC_MAPTYPE_NO_CREATE = 0x2000,
  // private variable - gang
  TGT_ACC_MAPTYPE_GANG_PRIVATE = 0x4000,
  // private variable - worker
  TGT_ACC_MAPTYPE_WORKER_PRIVATE = 0x8000,
  // private variable - vector
  TGT_ACC_MAPTYPE_VECTOR_PRIVATE = 0x10000,
  // zero modifier
  TGT_ACC_MAPTYPE_INIT_ZERO = 0x20000,
  // device resident memory - persistent allocation
  TGT_ACC_MAPTYPE_DEVICE_RESIDENT = 0x40000,
  // present or not
  TGT_ACC_MAPTYPE_IF_PRESENT = 0x80000,
  // present clause: skip attach/detach to preserve user-managed pointers
  TGT_ACC_MAPTYPE_PRESENT = 0x100000,
};

/// Array descriptor types
enum {
  TGT_ACC_DESC_GENERIC = 0,     // Generic type descriptor.
  TGT_ACC_DESC_F18 = 1,         // Fortran 2018 type descriptor.
  TGT_ACC_DESC_MEMREF = 2,      // MemRef type descriptor.
  TGT_ACC_DESC_OPENACC = 0x1000 // OpenACC descriptor.
};

/// Device pointer type.
typedef uintptr_t tgt_acc_devptr_t;

/// Type descriptor base struct.
typedef struct {
  // Version of the descriptor.
  int32_t Version;
} AccDataDesc;

/// Generic type descriptor.
typedef struct {
  AccDataDesc Base;
} AccDataDescGeneric;

/// F18 type descriptor.
typedef struct {
  AccDataDesc Base;
  CFI_cdesc_t *FortranDescriptor;
} AccDataDescF18;

/// The structure defined by LLVMTypeConverter::getMemRefDescriptorFields.
typedef struct {
  void *allocatedPtr;
  void *alignedPtr;
  uint64_t offset;

  uint64_t sizes[1];
// Below are the real fields in the struct where Rank is a compile-time
// constant. We use offsets from the above sizes to obtain the addresses of
// the sizes and strides arrays.
#if 0
  uint64_t sizes[Rank];
  uint64_t strides[Rank];
#endif
} MemRefDesc;

/// MemRef type descriptor.
typedef struct {
  AccDataDesc Base;
  unsigned char Rank;
  uint64_t ElementSize;
  MemRefDesc *MemRefDescriptor;
} AccDataDescMemRef;

/// OpenACC descriptor.
typedef struct {
  AccDataDesc Base;
  ACC_ALIGNED(8) unsigned char Rank;
  int64_t ElementSize;
  int64_t *LowerBounds;
  int64_t *UpperBounds;
  int64_t *Extents;
  int64_t *StridesInBytes;
  int64_t *StartIndices;
} AccDataDescOpenACC;

/// This struct contains all of the arguments to a target kernel region launch.
typedef struct {
  // Version of this struct for ABI compatibility.
  uint32_t Version;
  // Number of arguments in each input pointer.
  uint32_t ArgNum;
  // Base pointer of each argument (e.g. a struct).
  void **ArgBasePtrs;
  // Pointer to the argument data.
  void **ArgPtrs;
  // Size of the argument data in bytes.
  int64_t *ArgSizes;
  // Type of the data (e.g. to / from).
  int64_t *ArgTypes;
  // Name of the data for debugging, possibly null.
  char **ArgNames;
  // User-defined mappers (e.g. C++ copy ctors), possibly null.
  void **ArgMappers;
  // Type descriptors.
  AccDataDesc **ArgDescs;
  // Loop tripcount.
  uint64_t Tripcount;
  // Values of the num_gangs clause, in three dimensions.
  int64_t NumGangs[3];
  // Value of the num_workers clause.
  int64_t NumWorkers;
  // Value of the vector_length clause.
  int64_t VectorLength;
  // Size of shared memory.
  int64_t SmemSize;
} AccKernelArgsTy;

//===----------------------------------------------------------------------===//
// OpenACC Target Offload Runtime Compiler Interface API
//===----------------------------------------------------------------------===//

/// adds a target shared library to the target execution image
void __tgt_acc_register_lib(__tgt_bin_desc *Desc);

/// removes a target shared library from the target execution image
void __tgt_acc_unregister_lib(__tgt_bin_desc *Desc);

/// 'acc init' directive
void __tgt_acc_init(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                    int64_t DeviceNum);

/// 'acc shutdown' directive
void __tgt_acc_shutdown(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                        int64_t DeviceNum);

/// 'acc declare' directive
void __tgt_acc_declare(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                       uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                       int64_t *ArgSizes, int64_t *ArgTypes, char **ArgNames,
                       void **ArgMappers, AccDataDesc **ArgDescs, int64_t Async,
                       __tgt_bin_desc *Desc);

/// 'acc enter data' directive
void __tgt_acc_data_enter(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                          uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                          int64_t *ArgSizes, int64_t *ArgTypes, char **ArgNames,
                          void **ArgMappers, AccDataDesc **ArgDescs,
                          int64_t Async);

/// 'acc exit data' directive
void __tgt_acc_data_exit(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                         uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                         int64_t *ArgSizes, int64_t *ArgTypes, char **ArgNames,
                         void **ArgMappers, AccDataDesc **ArgDescs,
                         int64_t Async);

/// 'acc update' directive
void __tgt_acc_data_update(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                           uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                           int64_t *ArgSizes, int64_t *ArgTypes,
                           char **ArgNames, void **ArgMappers,
                           AccDataDesc **ArgDescs, int64_t Async);

/// data mapping begin (for `acc data` construct or compute construct)
void __tgt_acc_data_begin(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                          uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                          int64_t *ArgSizes, int64_t *ArgTypes, char **ArgNames,
                          void **ArgMappers, AccDataDesc **ArgDescs,
                          int64_t Async);

/// data mapping end (for `acc data` construct or compute construct)
void __tgt_acc_data_end(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                        uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                        int64_t *ArgSizes, int64_t *ArgTypes, char **ArgNames,
                        void **ArgMappers, AccDataDesc **ArgDescs,
                        int64_t Async);

/// compute construct directive
int __tgt_acc_kernel(ident_t *Loc, void *Kernel, int64_t Flags,
                     int64_t DeviceType, AccKernelArgsTy *Args, int64_t Async,
                     const char *KernelName, __tgt_bin_desc *Desc);

/// 'acc wait' directive
int __tgt_acc_wait(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                   int32_t DeviceNum, uint32_t WaitNum, int64_t *WaitList,
                   int64_t Async);

/// `acc host_data use_device` directive
void *__tgt_acc_get_deviceptr(ident_t *Loc, void *BasePtr, int64_t Flags,
                              void *HostPtr);

/// 'acc set default_async' directive
void __tgt_acc_set_default_async(ident_t *Loc, int64_t Async);

/// 'acc set device_num' directive
void __tgt_acc_set_device_num(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                              int64_t DeviceNum);

/// 'acc set device_type' directive
void __tgt_acc_set_device_type(ident_t *Loc, int64_t Flags, int64_t DeviceType);

/// Mirror allocation for declare action recipes
void __tgt_acc_mirror_alloc(ident_t *Loc, int64_t Flags, int64_t DeviceType,
                            uint32_t ArgNum, void **ArgBasePtrs, void **ArgPtrs,
                            int64_t *ArgSizes, int64_t *ArgTypes,
                            char **ArgNames, void **ArgMappers,
                            AccDataDesc **ArgDescs);

#ifdef __cplusplus
}
#endif

#endif // __TGT_ACC_TARGET_H__
