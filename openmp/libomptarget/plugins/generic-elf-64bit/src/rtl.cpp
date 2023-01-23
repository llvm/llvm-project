//===-RTLs/generic-64bit/src/rtl.cpp - Target RTLs Implementation - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for generic 64-bit machine
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ffi.h>
#include <link.h>
#include <list>
#include <string>
#include <vector>

#include "Debug.h"
#include "omptargetplugin.h"

using namespace llvm;
using namespace llvm::sys;

#ifndef TARGET_NAME
#define TARGET_NAME Generic ELF - 64bit
#endif
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME) " RTL"

#ifndef TARGET_ELF_ID
#define TARGET_ELF_ID 0
#endif

#include "elf_common.h"

#define NUMBER_OF_DEVICES 4
#define OFFLOAD_SECTION_NAME "omp_offloading_entries"

/// Array of Dynamic libraries loaded for this target.
struct DynLibTy {
  std::string FileName;
  std::unique_ptr<DynamicLibrary> DynLib;
};

/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  SmallVector<__tgt_offload_entry> Entries;
};

/// Class containing all the device information.
class RTLDeviceInfoTy {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

public:
  std::list<DynLibTy> DynLibs;

  // Record entry point associated with device.
  void createOffloadTable(int32_t DeviceId,
                          SmallVector<__tgt_offload_entry> &&Entries) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncGblEntries[DeviceId].emplace_back();
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    E.Entries = Entries;
    E.Table.EntriesBegin = E.Entries.begin();
    E.Table.EntriesEnd = E.Entries.end();
  }

  // Return true if the entry is associated with device.
  bool findOffloadEntry(int32_t DeviceId, void *Addr) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    for (__tgt_offload_entry *I = E.Table.EntriesBegin,
                             *End = E.Table.EntriesEnd;
         I < End; ++I) {
      if (I->addr == Addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table.
  __tgt_target_table *getOffloadEntriesTable(int32_t DeviceId) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    return &E.Table;
  }

  RTLDeviceInfoTy(int32_t NumDevices) { FuncGblEntries.resize(NumDevices); }

  ~RTLDeviceInfoTy() {
    // Close dynamic libraries
    for (auto &Lib : DynLibs) {
      if (Lib.DynLib->isValid())
        remove(Lib.FileName.c_str());
    }
  }
};

static RTLDeviceInfoTy DeviceInfo(NUMBER_OF_DEVICES);

#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
// If we don't have a valid ELF ID we can just fail.
#if TARGET_ELF_ID < 1
  return 0;
#else
  return elf_check_machine(Image, TARGET_ELF_ID);
#endif
}

int32_t __tgt_rtl_number_of_devices() { return NUMBER_OF_DEVICES; }

int32_t __tgt_rtl_init_device(int32_t DeviceId) { return OFFLOAD_SUCCESS; }

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *Image) {

  DP("Dev %d: load binary from " DPxMOD " image\n", DeviceId,
     DPxPTR(Image->ImageStart));

  assert(DeviceId >= 0 && DeviceId < NUMBER_OF_DEVICES && "bad dev id");

  size_t ImageSize = (size_t)Image->ImageEnd - (size_t)Image->ImageStart;

  // load dynamic library and get the entry points. We use the dl library
  // to do the loading of the library, but we could do it directly to avoid the
  // dump to the temporary file.
  //
  // 1) Create tmp file with the library contents.
  // 2) Use dlopen to load the file and dlsym to retrieve the symbols.
  char TmpName[] = "/tmp/tmpfile_XXXXXX";
  int TmpFd = mkstemp(TmpName);

  if (TmpFd == -1)
    return nullptr;

  FILE *Ftmp = fdopen(TmpFd, "wb");

  if (!Ftmp)
    return nullptr;

  fwrite(Image->ImageStart, ImageSize, 1, Ftmp);
  fclose(Ftmp);

  std::string ErrMsg;
  auto DynLib = std::make_unique<sys::DynamicLibrary>(
      sys::DynamicLibrary::getPermanentLibrary(TmpName, &ErrMsg));
  DynLibTy Lib = {TmpName, std::move(DynLib)};

  if (!Lib.DynLib->isValid()) {
    DP("Target library loading error: %s\n", ErrMsg.c_str());
    return NULL;
  }

  __tgt_offload_entry *HostBegin = Image->EntriesBegin;
  __tgt_offload_entry *HostEnd = Image->EntriesEnd;

  // Create a new offloading entry list using the device symbol address.
  SmallVector<__tgt_offload_entry> Entries;
  for (__tgt_offload_entry *E = HostBegin; E != HostEnd; ++E) {
    if (!E->addr)
      return nullptr;

    __tgt_offload_entry Entry = *E;

    void *DevAddr = Lib.DynLib->getAddressOfSymbol(E->name);
    Entry.addr = DevAddr;

    DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
       DPxPTR(E - HostBegin), E->name, DPxPTR(DevAddr));

    Entries.emplace_back(Entry);
  }

  DeviceInfo.createOffloadTable(DeviceId, std::move(Entries));
  DeviceInfo.DynLibs.emplace_back(std::move(Lib));

  return DeviceInfo.getOffloadEntriesTable(DeviceId);
}

void __tgt_rtl_print_device_info(int32_t DeviceId) {
  printf("    This is a generic-elf-64bit device\n");
}

// Sample implementation of explicit memory allocator. For this plugin all kinds
// are equivalent to each other.
void *__tgt_rtl_data_alloc(int32_t DeviceId, int64_t Size, void *HstPtr,
                           int32_t Kind) {
  void *Ptr = NULL;

  switch (Kind) {
  case TARGET_ALLOC_DEVICE:
  case TARGET_ALLOC_HOST:
  case TARGET_ALLOC_SHARED:
  case TARGET_ALLOC_DEFAULT:
    Ptr = malloc(Size);
    break;
  default:
    REPORT("Invalid target data allocation kind");
  }

  return Ptr;
}

int32_t __tgt_rtl_data_submit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                              int64_t Size) {
  memcpy(TgtPtr, HstPtr, Size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                                int64_t Size) {
  memcpy(HstPtr, TgtPtr, Size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t DeviceId, void *TgtPtr, int32_t) {
  free(TgtPtr);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_launch_kernel(int32_t DeviceId, void *TgtEntryPtr,
                                void **TgtArgs, ptrdiff_t *TgtOffsets,
                                KernelArgsTy *KernelArgs,
                                __tgt_async_info *AsyncInfoPtr) {
  assert(!KernelArgs->NumTeams[1] && !KernelArgs->NumTeams[2] &&
         !KernelArgs->ThreadLimit[1] && !KernelArgs->ThreadLimit[2] &&
         "Only one dimensional kernels supported.");
  // ignore team num and thread limit.

  // Use libffi to launch execution.
  ffi_cif Cif;

  // All args are references.
  std::vector<ffi_type *> ArgsTypes(KernelArgs->NumArgs, &ffi_type_pointer);
  std::vector<void *> Args(KernelArgs->NumArgs);
  std::vector<void *> Ptrs(KernelArgs->NumArgs);

  for (uint32_t I = 0; I < KernelArgs->NumArgs; ++I) {
    Ptrs[I] = (void *)((intptr_t)TgtArgs[I] + TgtOffsets[I]);
    Args[I] = &Ptrs[I];
  }

  ffi_status Status = ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, KernelArgs->NumArgs,
                                   &ffi_type_void, &ArgsTypes[0]);

  assert(Status == FFI_OK && "Unable to prepare target launch!");

  if (Status != FFI_OK)
    return OFFLOAD_FAIL;

  DP("Running entry point at " DPxMOD "...\n", DPxPTR(TgtEntryPtr));

  void (*Entry)(void);
  *((void **)&Entry) = TgtEntryPtr;
  ffi_call(&Cif, Entry, NULL, &Args[0]);
  return OFFLOAD_SUCCESS;
}

#ifdef __cplusplus
}
#endif
