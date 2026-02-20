//===-- Dynamically loaded offload API ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamically loads the API provided by the LLVMOffload library. We need to do
// this dynamically because this tool is used before it is actually built and
// should be provided even when the user did not specify the offload runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_GPU_LOADER_LLVM_GPU_LOADER_H
#define LLVM_TOOLS_LLVM_GPU_LOADER_LLVM_GPU_LOADER_H

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

typedef enum ol_alloc_type_t {
  OL_ALLOC_TYPE_HOST = 0,
  OL_ALLOC_TYPE_DEVICE = 1,
  OL_ALLOC_TYPE_FORCE_UINT32 = 0x7fffffff
} ol_alloc_type_t;

typedef enum ol_device_info_t {
  OL_DEVICE_INFO_TYPE = 0,
  OL_DEVICE_INFO_PLATFORM = 1,
  OL_DEVICE_INFO_FORCE_UINT32 = 0x7fffffff
} ol_device_info_t;

typedef enum ol_platform_info_t {
  OL_PLATFORM_INFO_NAME = 0,
  OL_PLATFORM_INFO_BACKEND = 3,
  OL_PLATFORM_INFO_FORCE_UINT32 = 0x7fffffff
} ol_platform_info_t;

typedef enum ol_symbol_kind_t {
  OL_SYMBOL_KIND_KERNEL = 0,
  OL_SYMBOL_KIND_GLOBAL_VARIABLE = 1,
  OL_SYMBOL_KIND_FORCE_UINT32 = 0x7fffffff
} ol_symbol_kind_t;

typedef enum ol_errc_t {
  OL_ERRC_SUCCESS = 0,
  OL_ERRC_FORCE_UINT32 = 0x7fffffff
} ol_errc_t;

typedef struct ol_error_struct_t {
  ol_errc_t Code;
  const char *Details;
} ol_error_struct_t;

typedef struct ol_dimensions_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} ol_dimensions_t;

typedef struct ol_kernel_launch_size_args_t {
  size_t Dimensions;
  struct ol_dimensions_t NumGroups;
  struct ol_dimensions_t GroupSize;
  size_t DynSharedMemory;
} ol_kernel_launch_size_args_t;

typedef enum ol_platform_backend_t {
  OL_PLATFORM_BACKEND_UNKNOWN = 0,
  OL_PLATFORM_BACKEND_CUDA = 1,
  OL_PLATFORM_BACKEND_AMDGPU = 2,
  OL_PLATFORM_BACKEND_LEVEL_ZERO = 3,
  OL_PLATFORM_BACKEND_HOST = 4,
  OL_PLATFORM_BACKEND_LAST = 5,
  OL_PLATFORM_BACKEND_FORCE_UINT32 = 0x7fffffff
} ol_platform_backend_t;

typedef enum ol_device_type_t {
  OL_DEVICE_TYPE_DEFAULT = 0,
  OL_DEVICE_TYPE_ALL = 1,
  OL_DEVICE_TYPE_GPU = 2,
  OL_DEVICE_TYPE_CPU = 3,
  OL_DEVICE_TYPE_HOST = 4,
  OL_DEVICE_TYPE_LAST = 5,
  OL_DEVICE_TYPE_FORCE_UINT32 = 0x7fffffff
} ol_device_type_t;

typedef struct ol_init_args_t {
  size_t Size;
  uint32_t NumPlatforms;
  const ol_platform_backend_t *Platforms;
} ol_init_args_t;

#define OL_INIT_ARGS_INIT {sizeof(ol_init_args_t), 0, NULL}

typedef struct ol_device_impl_t *ol_device_handle_t;
typedef struct ol_platform_impl_t *ol_platform_handle_t;
typedef struct ol_program_impl_t *ol_program_handle_t;
typedef struct ol_queue_impl_t *ol_queue_handle_t;
typedef struct ol_symbol_impl_t *ol_symbol_handle_t;
typedef const struct ol_error_struct_t *ol_result_t;

typedef bool (*ol_device_iterate_cb_t)(ol_device_handle_t Device,
                                       void *UserData);

ol_result_t (*olInit)(const ol_init_args_t *);
ol_result_t (*olShutDown)();

ol_result_t (*olIterateDevices)(ol_device_iterate_cb_t Callback,
                                void *UserData);

ol_result_t (*olIsValidBinary)(ol_device_handle_t Device, const void *ProgData,
                               size_t ProgDataSize, bool *Valid);

ol_result_t (*olCreateProgram)(ol_device_handle_t Device, const void *ProgData,
                               size_t ProgDataSize,
                               ol_program_handle_t *Program);

ol_result_t (*olDestroyProgram)(ol_program_handle_t Program);

ol_result_t (*olGetSymbol)(ol_program_handle_t Program, const char *Name,
                           ol_symbol_kind_t Kind, ol_symbol_handle_t *Symbol);

ol_result_t (*olLaunchKernel)(
    ol_queue_handle_t Queue, ol_device_handle_t Device,
    ol_symbol_handle_t Kernel, const void *ArgumentsData, size_t ArgumentsSize,
    const ol_kernel_launch_size_args_t *LaunchSizeArgs);

ol_result_t (*olCreateQueue)(ol_device_handle_t Device,
                             ol_queue_handle_t *Queue);

ol_result_t (*olDestroyQueue)(ol_queue_handle_t Queue);

ol_result_t (*olSyncQueue)(ol_queue_handle_t Queue);

ol_result_t (*olMemAlloc)(ol_device_handle_t Device, ol_alloc_type_t Type,
                          size_t Size, void **AllocationOut);

ol_result_t (*olMemFree)(void *Address);

ol_result_t (*olMemcpy)(ol_queue_handle_t Queue, void *DstPtr,
                        ol_device_handle_t DstDevice, const void *SrcPtr,
                        ol_device_handle_t SrcDevice, size_t Size);

ol_result_t (*olGetDeviceInfo)(ol_device_handle_t Device,
                               ol_device_info_t PropName, size_t PropSize,
                               void *PropValue);

ol_result_t (*olGetPlatformInfo)(ol_platform_handle_t Platform,
                                 ol_platform_info_t PropName, size_t PropSize,
                                 void *PropValue);

llvm::Error loadLLVMOffload() {
  constexpr const char *OffloadLibrary = "libLLVMOffload.so";

  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(OffloadLibrary, &ErrMsg));

  if (!DynlibHandle->isValid())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to dlopen %s: %s", OffloadLibrary,
                                   ErrMsg.c_str());

#define DYNAMIC_INIT(SYM)                                                      \
  do {                                                                         \
    void *Ptr = DynlibHandle->getAddressOfSymbol(#SYM);                        \
    if (!Ptr)                                                                  \
      return llvm::createStringError(                                          \
          llvm::inconvertibleErrorCode(), "Missing symbol '%s' in %s",         \
          reinterpret_cast<const char *>(#SYM), OffloadLibrary);               \
    SYM = reinterpret_cast<decltype(SYM)>(Ptr);                                \
  } while (0)

  DYNAMIC_INIT(olInit);
  DYNAMIC_INIT(olShutDown);
  DYNAMIC_INIT(olIterateDevices);
  DYNAMIC_INIT(olIsValidBinary);
  DYNAMIC_INIT(olCreateProgram);
  DYNAMIC_INIT(olDestroyProgram);
  DYNAMIC_INIT(olGetSymbol);
  DYNAMIC_INIT(olLaunchKernel);
  DYNAMIC_INIT(olCreateQueue);
  DYNAMIC_INIT(olDestroyQueue);
  DYNAMIC_INIT(olSyncQueue);
  DYNAMIC_INIT(olMemAlloc);
  DYNAMIC_INIT(olMemFree);
  DYNAMIC_INIT(olMemcpy);
  DYNAMIC_INIT(olGetDeviceInfo);
  DYNAMIC_INIT(olGetPlatformInfo);
#undef DYNAMIC_INIT

  return llvm::Error::success();
}

#endif // LLVM_TOOLS_LLVM_GPU_LOADER_LLVM_GPU_LOADER_H
