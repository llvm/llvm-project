//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <vector>

#ifdef OMPTARGET_DEBUG
int DebugLevel = 0;
#endif // OMPTARGET_DEBUG



/* All begin addresses for partially mapped structs must be 8-aligned in order
 * to ensure proper alignment of members. E.g.
 *
 * struct S {
 *   int a;   // 4-aligned
 *   int b;   // 4-aligned
 *   int *p;  // 8-aligned
 * } s1;
 * ...
 * #pragma omp target map(tofrom: s1.b, s1.p[0:N])
 * {
 *   s1.b = 5;
 *   for (int i...) s1.p[i] = ...;
 * }
 *
 * Here we are mapping s1 starting from member b, so BaseAddress=&s1=&s1.a and
 * BeginAddress=&s1.b. Let's assume that the struct begins at address 0x100,
 * then &s1.a=0x100, &s1.b=0x104, &s1.p=0x108. Each member obeys the alignment
 * requirements for its type. Now, when we allocate memory on the device, in
 * CUDA's case cuMemAlloc() returns an address which is at least 256-aligned.
 * This means that the chunk of the struct on the device will start at a
 * 256-aligned address, let's say 0x200. Then the address of b will be 0x200 and
 * address of p will be a misaligned 0x204 (on the host there was no need to add
 * padding between b and p, so p comes exactly 4 bytes after b). If the device
 * kernel tries to access s1.p, a misaligned address error occurs (as reported
 * by the CUDA plugin). By padding the begin address down to a multiple of 8 and
 * extending the size of the allocated chuck accordingly, the chuck on the
 * device will start at 0x200 with the padding (4 bytes), then &s1.b=0x204 and
 * &s1.p=0x208, as they should be to satisfy the alignment requirements.
 */
static const int64_t alignment = 8;

/// Map global data and execute pending ctors
static int InitLibrary(DeviceTy& Device) {
  /*
   * Map global data
   */
  int32_t device_id = Device.DeviceID;
  int rc = OFFLOAD_SUCCESS;

  Device.PendingGlobalsMtx.lock();
  TrlTblMtx->lock();
  for (HostEntriesBeginToTransTableTy::iterator
      ii = HostEntriesBeginToTransTable->begin();
      ii != HostEntriesBeginToTransTable->end(); ++ii) {
    TranslationTable *TransTable = &ii->second;
    if (TransTable->HostTable.EntriesBegin ==
        TransTable->HostTable.EntriesEnd) {
      // No host entry so no need to proceed
      continue;
    }
    if (TransTable->TargetsTable[device_id] != 0) {
      // Library entries have already been processed
      continue;
    }

    // 1) get image.
    assert(TransTable->TargetsImages.size() > (size_t)device_id &&
           "Not expecting a device ID outside the table's bounds!");
    __tgt_device_image *img = TransTable->TargetsImages[device_id];
    if (!img) {
      DP("No image loaded for device id %d.\n", device_id);
      rc = OFFLOAD_FAIL;
      break;
    }
    // 2) load image into the target table.
    __tgt_target_table *TargetTable =
        TransTable->TargetsTable[device_id] = Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      DP("Unable to generate entries table for device id %d.\n", device_id);
      TransTable->TargetsImages[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // Verify whether the two table sizes match.
    size_t hsize =
        TransTable->HostTable.EntriesEnd - TransTable->HostTable.EntriesBegin;
    size_t tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

    // Invalid image for these host entries!
    if (hsize != tsize) {
      DP("Host and Target tables mismatch for device id %d [%zx != %zx].\n",
         device_id, hsize, tsize);
      TransTable->TargetsImages[device_id] = 0;
      TransTable->TargetsTable[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // process global data that needs to be mapped.
    Device.DataMapMtx.lock();
    __tgt_target_table *HostTable = &TransTable->HostTable;
    for (__tgt_offload_entry *CurrDeviceEntry = TargetTable->EntriesBegin,
                             *CurrHostEntry = HostTable->EntriesBegin,
                             *EntryDeviceEnd = TargetTable->EntriesEnd;
         CurrDeviceEntry != EntryDeviceEnd;
         CurrDeviceEntry++, CurrHostEntry++) {
      if (CurrDeviceEntry->size != 0) {
        // has data.
        assert(CurrDeviceEntry->size == CurrHostEntry->size &&
               "data size mismatch");

        // Fortran may use multiple weak declarations for the same symbol,
        // therefore we must allow for multiple weak symbols to be loaded from
        // the fat binary. Treat these mappings as any other "regular" mapping.
        // Add entry to map.
        if (Device.getTgtPtrBegin(CurrHostEntry->addr, CurrHostEntry->size))
          continue;
        DP("Add mapping from host " DPxMOD " to device " DPxMOD " with size %zu"
            "\n", DPxPTR(CurrHostEntry->addr), DPxPTR(CurrDeviceEntry->addr),
            CurrDeviceEntry->size);
        Device.HostDataToTargetMap.emplace(
            (uintptr_t)CurrHostEntry->addr /*HstPtrBase*/,
            (uintptr_t)CurrHostEntry->addr /*HstPtrBegin*/,
            (uintptr_t)CurrHostEntry->addr + CurrHostEntry->size /*HstPtrEnd*/,
            (uintptr_t)CurrDeviceEntry->addr /*TgtPtrBegin*/,
            true /*IsRefCountINF*/);
      }
    }
    Device.DataMapMtx.unlock();
  }
  TrlTblMtx->unlock();

  if (rc != OFFLOAD_SUCCESS) {
    Device.PendingGlobalsMtx.unlock();
    return rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    // Call all ctors for all libraries registered so far
    for (auto &lib : Device.PendingCtorsDtors) {
      if (!lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &entry : lib.second.PendingCtors) {
          void *ctor = entry;
          int rc = target(device_id, ctor, 0, NULL, NULL, NULL, NULL, NULL, 1,
              1, true /*team*/);
          if (rc != OFFLOAD_SUCCESS) {
            DP("Running ctor " DPxMOD " failed.\n", DPxPTR(ctor));
            Device.PendingGlobalsMtx.unlock();
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib " DPxMOD "\n", DPxPTR(lib.first));
      }
    }
  }
  Device.HasPendingGlobals = false;
  Device.PendingGlobalsMtx.unlock();

  return OFFLOAD_SUCCESS;
}

// Check whether a device has been initialized, global ctors have been
// executed and global data has been mapped; do so if not already done.
int CheckDeviceAndCtors(int64_t device_id) {
  // Is device ready?
  if (!device_is_ready(device_id)) {
    DP("Device %" PRId64 " is not ready.\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Get device info.
  DeviceTy &Device = Devices[device_id];

  // Check whether global data has been mapped for this device
  Device.PendingGlobalsMtx.lock();
  bool hasPendingGlobals = Device.HasPendingGlobals;
  Device.PendingGlobalsMtx.unlock();
  if (hasPendingGlobals && InitLibrary(Device) != OFFLOAD_SUCCESS) {
    DP("Failed to init globals on device %" PRId64 "\n", device_id);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

static int32_t member_of(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Call the user-defined mapper function followed by the appropriate
// target_data_* function (target_data_{begin,end,update}).
int target_data_mapper(DeviceTy &Device, void *arg_base,
    void *arg, int64_t arg_size, int64_t arg_type, void *arg_mapper,
    TargetDataFuncPtrTy target_data_function) {
  DP("Calling the mapper function " DPxMOD "\n", DPxPTR(arg_mapper));

  // The mapper function fills up Components.
  MapperComponentsTy MapperComponents;
  MapperFuncPtrTy MapperFuncPtr = (MapperFuncPtrTy)(arg_mapper);
  (*MapperFuncPtr)((void *)&MapperComponents, arg_base, arg, arg_size,
      arg_type);

  // Construct new arrays for args_base, args, arg_sizes and arg_types
  // using the information in MapperComponents and call the corresponding
  // target_data_* function using these new arrays.
  std::vector<void *> mapper_args_base;
  std::vector<void *> mapper_args;
  std::vector<int64_t> mapper_arg_sizes;
  std::vector<int64_t> mapper_arg_types;

  for (auto& C : MapperComponents.Components) {
    mapper_args_base.push_back(C.Base);
    mapper_args.push_back(C.Begin);
    mapper_arg_sizes.push_back(C.Size);
    mapper_arg_types.push_back(C.Type);
  }

  int rc = target_data_function(Device, MapperComponents.Components.size(),
      mapper_args_base.data(), mapper_args.data(), mapper_arg_sizes.data(),
      mapper_arg_types.data(), /*arg_mappers*/ nullptr,
      /*__tgt_async_info*/ nullptr);

  return rc;
}

/// Internal function to do the mapping and transfer the data to the device
int target_data_begin(DeviceTy &Device, int32_t arg_num, void **args_base,
                      void **args, int64_t *arg_sizes, int64_t *arg_types,
                      void **arg_mappers, __tgt_async_info *async_info_ptr) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (arg_mappers && arg_mappers[i]) {
      // Instead of executing the regular path of target_data_begin, call the
      // target_data_mapper variant which will call target_data_begin again
      // with new arguments.
      DP("Calling target_data_mapper for the %dth argument\n", i);

      int rc = target_data_mapper(Device, args_base[i], args[i], arg_sizes[i],
          arg_types[i], arg_mappers[i], target_data_begin);

      if (rc != OFFLOAD_SUCCESS) {
        DP("Call to target_data_begin via target_data_mapper for custom mapper"
            " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    int64_t data_size = arg_sizes[i];

    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t padding = 0;
    const int next_i = i+1;
    if (member_of(arg_types[i]) < 0 && next_i < arg_num &&
        member_of(arg_types[next_i]) == i) {
      padding = (int64_t)HstPtrBegin % alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
            "\n", padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *) HstPtrBegin - padding;
        data_size += padding;
      }
    }

    // Address of pointer on the host and device, respectively.
    void *Pointer_HstPtrBegin, *Pointer_TgtPtrBegin;
    bool IsNew, Pointer_IsNew;
    bool IsHostPtr = false;
    bool IsImplicit = arg_types[i] & OMP_TGT_MAPTYPE_IMPLICIT;
    // Force the creation of a device side copy of the data when:
    // a close map modifier was associated with a map that contained a to.
    bool HasCloseModifier = arg_types[i] & OMP_TGT_MAPTYPE_CLOSE;
    // UpdateRef is based on MEMBER_OF instead of TARGET_PARAM because if we
    // have reached this point via __tgt_target_data_begin and not __tgt_target
    // then no argument is marked as TARGET_PARAM ("omp target data map" is not
    // associated with a target region, so there are no target parameters). This
    // may be considered a hack, we could revise the scheme in the future.
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF);
    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Has a pointer entry: \n");
      // base is address of pointer.
      Pointer_TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBase, HstPtrBase,
          sizeof(void *), Pointer_IsNew, IsHostPtr, IsImplicit, UpdateRef,
          HasCloseModifier);
      if (!Pointer_TgtPtrBegin) {
        DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
            "illegal mapping).\n");
        return OFFLOAD_FAIL;
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
          "\n", sizeof(void *), DPxPTR(Pointer_TgtPtrBegin),
          (Pointer_IsNew ? "" : " not"));
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      UpdateRef = true; // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBegin, HstPtrBase,
        data_size, IsNew, IsHostPtr, IsImplicit, UpdateRef, HasCloseModifier);
    if (!TgtPtrBegin && data_size) {
      // If data_size==0, then the argument could be a zero-length pointer to
      // NULL, so getOrAlloc() returning NULL is not an error.
      DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
          "illegal mapping).\n");
      return OFFLOAD_FAIL;
    }
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s new\n", data_size, DPxPTR(TgtPtrBegin),
        (IsNew ? "" : " not"));

    if (arg_types[i] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      uintptr_t Delta = (uintptr_t)HstPtrBegin - (uintptr_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uintptr_t)TgtPtrBegin - Delta);
      DP("Returning device pointer " DPxMOD "\n", DPxPTR(TgtPtrBase));
      args_base[i] = TgtPtrBase;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      bool copy = false;
      if (!(RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
          HasCloseModifier) {
        if (IsNew || (arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS)) {
          copy = true;
        } else if (arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) {
          // Copy data only if the "parent" struct has RefCount==1.
          int32_t parent_idx = member_of(arg_types[i]);
          uint64_t parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
          assert(parent_rc > 0 && "parent struct not found");
          if (parent_rc == 1) {
            copy = true;
          }
        }
      }

      if (copy && !IsHostPtr) {
        DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
           data_size, DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
        int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, data_size,
                                    async_info_ptr);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ && !IsHostPtr) {
      DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
          DPxPTR(Pointer_TgtPtrBegin), DPxPTR(TgtPtrBegin));
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      int rt = Device.data_submit(Pointer_TgtPtrBegin, &TgtPtrBase,
                                  sizeof(void *), async_info_ptr);
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        return OFFLOAD_FAIL;
      }
      // create shadow pointers for this entry
      Device.ShadowMtx.lock();
      Device.ShadowPtrMap[Pointer_HstPtrBegin] = {HstPtrBase,
          Pointer_TgtPtrBegin, TgtPtrBase};
      Device.ShadowMtx.unlock();
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Internal function to undo the mapping and retrieve the data from the device.
int target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
                    void **args, int64_t *arg_sizes, int64_t *arg_types,
                    void **arg_mappers, __tgt_async_info *async_info_ptr) {
  // process each input.
  for (int32_t i = arg_num - 1; i >= 0; --i) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (arg_mappers && arg_mappers[i]) {
      // Instead of executing the regular path of target_data_end, call the
      // target_data_mapper variant which will call target_data_end again
      // with new arguments.
      DP("Calling target_data_mapper for the %dth argument\n", i);

      int rc = target_data_mapper(Device, args_base[i], args[i], arg_sizes[i],
          arg_types[i], arg_mappers[i], target_data_end);

      if (rc != OFFLOAD_SUCCESS) {
        DP("Call to target_data_end via target_data_mapper for custom mapper"
            " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = args[i];
    int64_t data_size = arg_sizes[i];
    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t padding = 0;
    const int next_i = i+1;
    if (member_of(arg_types[i]) < 0 && next_i < arg_num &&
        member_of(arg_types[next_i]) == i) {
      padding = (int64_t)HstPtrBegin % alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
            "\n", padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *) HstPtrBegin - padding;
        data_size += padding;
      }
    }

    bool IsLast, IsHostPtr;
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ);
    bool ForceDelete = arg_types[i] & OMP_TGT_MAPTYPE_DELETE;
    bool HasCloseModifier = arg_types[i] & OMP_TGT_MAPTYPE_CLOSE;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, data_size, IsLast,
        UpdateRef, IsHostPtr);
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s last\n", data_size, DPxPTR(TgtPtrBegin),
        (IsLast ? "" : " not"));

    bool DelEntry = IsLast || ForceDelete;

    if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if ((arg_types[i] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
        bool Always = arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS;
        bool CopyMember = false;
        if (!(RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
            HasCloseModifier) {
          if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
              !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
            // Copy data only if the "parent" struct has RefCount==1.
            int32_t parent_idx = member_of(arg_types[i]);
            uint64_t parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
            assert(parent_rc > 0 && "parent struct not found");
            if (parent_rc == 1) {
              CopyMember = true;
            }
          }
        }

        if ((DelEntry || Always || CopyMember) &&
            !(RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
              TgtPtrBegin == HstPtrBegin)) {
          DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
             data_size, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
          int rt = Device.data_retrieve(HstPtrBegin, TgtPtrBegin, data_size,
                                        async_info_ptr);
          if (rt != OFFLOAD_SUCCESS) {
            DP("Copying data from device failed.\n");
            return OFFLOAD_FAIL;
          }
        }
      }

      // If we copied back to the host a struct/array containing pointers, we
      // need to restore the original host pointer values from their shadow
      // copies. If the struct is going to be deallocated, remove any remaining
      // shadow pointer entries for this struct.
      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + data_size;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
           it != Device.ShadowPtrMap.end();) {
        void **ShadowHstPtrAddr = (void**) it->first;

        // An STL map is sorted on its keys; use this property
        // to quickly determine when to break out of the loop.
        if ((uintptr_t) ShadowHstPtrAddr < lb) {
          ++it;
          continue;
        }
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;

        // If we copied the struct to the host, we need to restore the pointer.
        if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
          DP("Restoring original host pointer value " DPxMOD " for host "
              "pointer " DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
              DPxPTR(ShadowHstPtrAddr));
          *ShadowHstPtrAddr = it->second.HstPtrVal;
        }
        // If the struct is to be deallocated, remove the shadow entry.
        if (DelEntry) {
          DP("Removing shadow pointer " DPxMOD "\n", DPxPTR(ShadowHstPtrAddr));
          it = Device.ShadowPtrMap.erase(it);
        } else {
          ++it;
        }
      }
      Device.ShadowMtx.unlock();

      // Deallocate map
      if (DelEntry) {
        int rt = Device.deallocTgtPtr(HstPtrBegin, data_size, ForceDelete,
                                      HasCloseModifier);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Deallocating data from device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Internal function to pass data to/from the target.
// async_info_ptr is currently unused, added here so target_data_update has the
// same signature as target_data_begin and target_data_end.
int target_data_update(DeviceTy &Device, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    void **arg_mappers, __tgt_async_info *async_info_ptr) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (arg_mappers && arg_mappers[i]) {
      // Instead of executing the regular path of target_data_update, call the
      // target_data_mapper variant which will call target_data_update again
      // with new arguments.
      DP("Calling target_data_mapper for the %dth argument\n", i);

      int rc = target_data_mapper(Device, args_base[i], args[i], arg_sizes[i],
          arg_types[i], arg_mappers[i], target_data_update);

      if (rc != OFFLOAD_SUCCESS) {
        DP("Call to target_data_update via target_data_mapper for custom mapper"
            " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = args[i];
    int64_t MapSize = arg_sizes[i];
    bool IsLast, IsHostPtr;
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, MapSize, IsLast,
        false, IsHostPtr);
    if (!TgtPtrBegin) {
      DP("hst data:" DPxMOD " not found, becomes a noop\n", DPxPTR(HstPtrBegin));
      continue;
    }

    if (RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
        TgtPtrBegin == HstPtrBegin) {
      DP("hst data:" DPxMOD " unified and shared, becomes a noop\n",
         DPxPTR(HstPtrBegin));
      continue;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
      DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
      int rt = Device.data_retrieve(HstPtrBegin, TgtPtrBegin, MapSize, nullptr);
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data from device failed.\n");
        return OFFLOAD_FAIL;
      }

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original host pointer value " DPxMOD " for host pointer "
            DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
            DPxPTR(ShadowHstPtrAddr));
        *ShadowHstPtrAddr = it->second.HstPtrVal;
      }
      Device.ShadowMtx.unlock();
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
      int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, MapSize, nullptr);
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        return OFFLOAD_FAIL;
      }

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original target pointer value " DPxMOD " for target "
            "pointer " DPxMOD "\n", DPxPTR(it->second.TgtPtrVal),
            DPxPTR(it->second.TgtPtrAddr));
        rt = Device.data_submit(it->second.TgtPtrAddr,
            &it->second.TgtPtrVal, sizeof(void *), nullptr);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          Device.ShadowMtx.unlock();
          return OFFLOAD_FAIL;
        }
      }
      Device.ShadowMtx.unlock();
    }
  }
  return OFFLOAD_SUCCESS;
}

static const unsigned LambdaMapping = OMP_TGT_MAPTYPE_PTR_AND_OBJ |
                                      OMP_TGT_MAPTYPE_LITERAL |
                                      OMP_TGT_MAPTYPE_IMPLICIT;
static bool isLambdaMapping(int64_t Mapping) {
  return (Mapping & LambdaMapping) == LambdaMapping;
}

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    void **arg_mappers, int32_t team_num, int32_t thread_limit,
    int IsTeamConstruct) {
  DeviceTy &Device = Devices[device_id];

  // Find the table information in the map or look it up in the translation
  // tables.
  TableMap *TM = 0;
  TblMapMtx->lock();
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap->find(host_ptr);
  if (TableMapIt == HostPtrToTableMap->end()) {
    // We don't have a map. So search all the registered libraries.
    TrlTblMtx->lock();
    for (HostEntriesBeginToTransTableTy::iterator
             ii = HostEntriesBeginToTransTable->begin(),
             ie = HostEntriesBeginToTransTable->end();
         !TM && ii != ie; ++ii) {
      // get the translation table (which contains all the good info).
      TranslationTable *TransTable = &ii->second;
      // iterate over all the host table entries to see if we can locate the
      // host_ptr.
      __tgt_offload_entry *begin = TransTable->HostTable.EntriesBegin;
      __tgt_offload_entry *end = TransTable->HostTable.EntriesEnd;
      __tgt_offload_entry *cur = begin;
      for (uint32_t i = 0; cur < end; ++cur, ++i) {
        if (cur->addr != host_ptr)
          continue;
        // we got a match, now fill the HostPtrToTableMap so that we
        // may avoid this search next time.
        TM = &(*HostPtrToTableMap)[host_ptr];
        TM->Table = TransTable;
        TM->Index = i;
        break;
      }
    }
    TrlTblMtx->unlock();
  } else {
    TM = &TableMapIt->second;
  }
  TblMapMtx->unlock();

  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr " DPxMOD " does not have a matching target pointer.\n",
       DPxPTR(host_ptr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  TrlTblMtx->lock();
  assert(TM->Table->TargetsTable.size() > (size_t)device_id &&
         "Not expecting a device ID outside the table's bounds!");
  __tgt_target_table *TargetTable = TM->Table->TargetsTable[device_id];
  TrlTblMtx->unlock();
  assert(TargetTable && "Global data has not been mapped\n");

  __tgt_async_info AsyncInfo;

  // Move data to device.
  int rc = target_data_begin(Device, arg_num, args_base, args, arg_sizes,
                             arg_types, arg_mappers, &AsyncInfo);
  if (rc != OFFLOAD_SUCCESS) {
    DP("Call to target_data_begin failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  std::vector<void *> tgt_args;
  std::vector<ptrdiff_t> tgt_offsets;

  // List of (first-)private arrays allocated for this target region
  std::vector<void *> fpArrays;
  std::vector<int> tgtArgsPositions(arg_num, -1);

  for (int32_t i = 0; i < arg_num; ++i) {
    if (!(arg_types[i] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into tgt_args.
      // Check for lambda mapping.
      if (isLambdaMapping(arg_types[i])) {
        assert((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
               "PTR_AND_OBJ must be also MEMBER_OF.");
        unsigned idx = member_of(arg_types[i]);
        int tgtIdx = tgtArgsPositions[idx];
        assert(tgtIdx != -1 && "Base address must be translated already.");
        // The parent lambda must be processed already and it must be the last
        // in tgt_args and tgt_offsets arrays.
        void *HstPtrVal = args[i];
        void *HstPtrBegin = args_base[i];
        void *HstPtrBase = args[idx];
        bool IsLast, IsHostPtr; // unused.
        void *TgtPtrBase =
            (void *)((intptr_t)tgt_args[tgtIdx] + tgt_offsets[tgtIdx]);
        DP("Parent lambda base " DPxMOD "\n", DPxPTR(TgtPtrBase));
        uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
        void *TgtPtrBegin = (void *)((uintptr_t)TgtPtrBase + Delta);
        void *Pointer_TgtPtrBegin =
            Device.getTgtPtrBegin(HstPtrVal, arg_sizes[i], IsLast, false,
                                  IsHostPtr);
        if (!Pointer_TgtPtrBegin) {
          DP("No lambda captured variable mapped (" DPxMOD ") - ignored\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        if (RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
            TgtPtrBegin == HstPtrBegin) {
          DP("Unified memory is active, no need to map lambda captured"
             "variable (" DPxMOD ")\n", DPxPTR(HstPtrVal));
          continue;
        }
        DP("Update lambda reference (" DPxMOD ") -> [" DPxMOD "]\n",
           DPxPTR(Pointer_TgtPtrBegin), DPxPTR(TgtPtrBegin));
        int rt = Device.data_submit(TgtPtrBegin, &Pointer_TgtPtrBegin,
                                    sizeof(void *), &AsyncInfo);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      continue;
    }
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *TgtPtrBegin;
    ptrdiff_t TgtBaseOffset;
    bool IsLast, IsHostPtr; // unused.
    if (arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
          DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE) {
      // Allocate memory for (first-)private array
      TgtPtrBegin = Device.RTL->data_alloc(Device.RTLDeviceID,
          arg_sizes[i], HstPtrBegin);
      if (!TgtPtrBegin) {
        DP ("Data allocation for %sprivate array " DPxMOD " failed, "
            "abort target.\n",
            (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
            DPxPTR(HstPtrBegin));
        return OFFLOAD_FAIL;
      }
      fpArrays.push_back(TgtPtrBegin);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD " for "
          "%sprivate array " DPxMOD " - pushing target argument " DPxMOD "\n",
          arg_sizes[i], DPxPTR(TgtPtrBegin),
          (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
          DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBase));
#endif
      // If first-private, copy data from host
      if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
        int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i],
                                    &AsyncInfo);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed, failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBase, sizeof(void *), IsLast,
          false, IsHostPtr);
      TgtBaseOffset = 0; // no offset for ptrs.
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD " to "
         "object " DPxMOD "\n", DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBase),
         DPxPTR(HstPtrBase));
    } else {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
          false, IsHostPtr);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD "\n",
          DPxPTR(TgtPtrBase), DPxPTR(HstPtrBegin));
#endif
    }
    tgtArgsPositions[i] = tgt_args.size();
    tgt_args.push_back(TgtPtrBegin);
    tgt_offsets.push_back(TgtBaseOffset);
  }

  assert(tgt_args.size() == tgt_offsets.size() &&
      "Size mismatch in arguments and offsets");

  // Pop loop trip count
  uint64_t ltc = 0;
  TblMapMtx->lock();
  auto I = Device.LoopTripCnt.find(__kmpc_global_thread_num(NULL));
  if (I != Device.LoopTripCnt.end()) {
    ltc = I->second;
    Device.LoopTripCnt.erase(I);
    DP("loop trip count is %lu.\n", ltc);
  }
  TblMapMtx->unlock();

  // Launch device execution.
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
      TargetTable->EntriesBegin[TM->Index].name,
      DPxPTR(TargetTable->EntriesBegin[TM->Index].addr), TM->Index);
  if (IsTeamConstruct) {
    rc = Device.run_team_region(TargetTable->EntriesBegin[TM->Index].addr,
                                &tgt_args[0], &tgt_offsets[0], tgt_args.size(),
                                team_num, thread_limit, ltc, &AsyncInfo);
  } else {
    rc = Device.run_region(TargetTable->EntriesBegin[TM->Index].addr,
                           &tgt_args[0], &tgt_offsets[0], tgt_args.size(),
                           &AsyncInfo);
  }
  if (rc != OFFLOAD_SUCCESS) {
    DP ("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  // Deallocate (first-)private arrays
  for (auto it : fpArrays) {
    int rt = Device.RTL->data_delete(Device.RTLDeviceID, it);
    if (rt != OFFLOAD_SUCCESS) {
      DP("Deallocation of (first-)private arrays failed.\n");
      return OFFLOAD_FAIL;
    }
  }

  // Move data from device.
  int rt = target_data_end(Device, arg_num, args_base, args, arg_sizes,
                           arg_types, arg_mappers, &AsyncInfo);
  if (rt != OFFLOAD_SUCCESS) {
    DP("Call to target_data_end failed, abort targe.\n");
    return OFFLOAD_FAIL;
  }

  if (Device.RTL->synchronize)
    return Device.RTL->synchronize(device_id, &AsyncInfo);

  return OFFLOAD_SUCCESS;
}
