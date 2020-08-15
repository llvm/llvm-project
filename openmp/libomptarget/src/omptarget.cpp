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
static const int64_t Alignment = 8;

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

static int32_t getParentIndex(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Call the user-defined mapper function followed by the appropriate
// target_data_* function (target_data_{begin,end,update}).
int targetDataMapper(DeviceTy &Device, void *arg_base, void *arg,
                     int64_t arg_size, int64_t arg_type, void *arg_mapper,
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
  std::vector<void *> MapperArgsBase(MapperComponents.Components.size());
  std::vector<void *> MapperArgs(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgSizes(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgTypes(MapperComponents.Components.size());

  for (unsigned I = 0, E = MapperComponents.Components.size(); I < E; ++I) {
    auto &C =
        MapperComponents
            .Components[target_data_function == targetDataEnd ? I : E - I - 1];
    MapperArgsBase[I] = C.Base;
    MapperArgs[I] = C.Begin;
    MapperArgSizes[I] = C.Size;
    MapperArgTypes[I] = C.Type;
  }

  int rc = target_data_function(Device, MapperComponents.Components.size(),
                                MapperArgsBase.data(), MapperArgs.data(),
                                MapperArgSizes.data(), MapperArgTypes.data(),
                                /*arg_mappers*/ nullptr,
                                /*__tgt_async_info*/ nullptr);

  return rc;
}

/// Internal function to do the mapping and transfer the data to the device
int targetDataBegin(DeviceTy &Device, int32_t arg_num, void **args_base,
                    void **args, int64_t *arg_sizes, int64_t *arg_types,
                    void **arg_mappers, __tgt_async_info *async_info_ptr) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (arg_mappers && arg_mappers[i]) {
      // Instead of executing the regular path of targetDataBegin, call the
      // targetDataMapper variant which will call targetDataBegin again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", i);

      int rc = targetDataMapper(Device, args_base[i], args[i], arg_sizes[i],
                                arg_types[i], arg_mappers[i], targetDataBegin);

      if (rc != OFFLOAD_SUCCESS) {
        DP("Call to targetDataBegin via targetDataMapper for custom mapper"
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
    if (getParentIndex(arg_types[i]) < 0 && next_i < arg_num &&
        getParentIndex(arg_types[next_i]) == i) {
      padding = (int64_t)HstPtrBegin % Alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
            "\n", padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *) HstPtrBegin - padding;
        data_size += padding;
      }
    }

    // Address of pointer on the host and device, respectively.
    void *Pointer_HstPtrBegin, *PointerTgtPtrBegin;
    bool IsNew, Pointer_IsNew;
    bool IsHostPtr = false;
    bool IsImplicit = arg_types[i] & OMP_TGT_MAPTYPE_IMPLICIT;
    // Force the creation of a device side copy of the data when:
    // a close map modifier was associated with a map that contained a to.
    bool HasCloseModifier = arg_types[i] & OMP_TGT_MAPTYPE_CLOSE;
    bool HasPresentModifier = arg_types[i] & OMP_TGT_MAPTYPE_PRESENT;
    // UpdateRef is based on MEMBER_OF instead of TARGET_PARAM because if we
    // have reached this point via __tgt_target_data_begin and not __tgt_target
    // then no argument is marked as TARGET_PARAM ("omp target data map" is not
    // associated with a target region, so there are no target parameters). This
    // may be considered a hack, we could revise the scheme in the future.
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF);
    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Has a pointer entry: \n");
      // Base is address of pointer.
      //
      // Usually, the pointer is already allocated by this time.  For example:
      //
      //   #pragma omp target map(s.p[0:N])
      //
      // The map entry for s comes first, and the PTR_AND_OBJ entry comes
      // afterward, so the pointer is already allocated by the time the
      // PTR_AND_OBJ entry is handled below, and PointerTgtPtrBegin is thus
      // non-null.  However, "declare target link" can produce a PTR_AND_OBJ
      // entry for a global that might not already be allocated by the time the
      // PTR_AND_OBJ entry is handled below, and so the allocation might fail
      // when HasPresentModifier.
      PointerTgtPtrBegin = Device.getOrAllocTgtPtr(
          HstPtrBase, HstPtrBase, sizeof(void *), Pointer_IsNew, IsHostPtr,
          IsImplicit, UpdateRef, HasCloseModifier, HasPresentModifier);
      if (!PointerTgtPtrBegin) {
        DP("Call to getOrAllocTgtPtr returned null pointer (%s).\n",
           HasPresentModifier ? "'present' map type modifier"
                              : "device failure or illegal mapping");
        return OFFLOAD_FAIL;
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
          "\n", sizeof(void *), DPxPTR(PointerTgtPtrBegin),
          (Pointer_IsNew ? "" : " not"));
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      UpdateRef = true; // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(
        HstPtrBegin, HstPtrBase, data_size, IsNew, IsHostPtr, IsImplicit,
        UpdateRef, HasCloseModifier, HasPresentModifier);
    // If data_size==0, then the argument could be a zero-length pointer to
    // NULL, so getOrAlloc() returning NULL is not an error.
    if (!TgtPtrBegin && (data_size || HasPresentModifier)) {
      DP("Call to getOrAllocTgtPtr returned null pointer (%s).\n",
         HasPresentModifier ? "'present' map type modifier"
                            : "device failure or illegal mapping");
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
          int32_t parent_idx = getParentIndex(arg_types[i]);
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
        int rt = Device.submitData(TgtPtrBegin, HstPtrBegin, data_size,
                                   async_info_ptr);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ && !IsHostPtr) {
      DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
         DPxPTR(PointerTgtPtrBegin), DPxPTR(TgtPtrBegin));
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      int rt = Device.submitData(PointerTgtPtrBegin, &TgtPtrBase,
                                 sizeof(void *), async_info_ptr);
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        return OFFLOAD_FAIL;
      }
      // create shadow pointers for this entry
      Device.ShadowMtx.lock();
      Device.ShadowPtrMap[Pointer_HstPtrBegin] = {
          HstPtrBase, PointerTgtPtrBegin, TgtPtrBase};
      Device.ShadowMtx.unlock();
    }
  }

  return OFFLOAD_SUCCESS;
}

namespace {
/// This structure contains information to deallocate a target pointer, aka.
/// used to call the function \p DeviceTy::deallocTgtPtr.
struct DeallocTgtPtrInfo {
  /// Host pointer used to look up into the map table
  void *HstPtrBegin;
  /// Size of the data
  int64_t DataSize;
  /// Whether it is forced to be removed from the map table
  bool ForceDelete;
  /// Whether it has \p close modifier
  bool HasCloseModifier;

  DeallocTgtPtrInfo(void *HstPtr, int64_t Size, bool ForceDelete,
                    bool HasCloseModifier)
      : HstPtrBegin(HstPtr), DataSize(Size), ForceDelete(ForceDelete),
        HasCloseModifier(HasCloseModifier) {}
};
} // namespace

/// Internal function to undo the mapping and retrieve the data from the device.
int targetDataEnd(DeviceTy &Device, int32_t ArgNum, void **ArgBases,
                  void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
                  void **ArgMappers, __tgt_async_info *AsyncInfo) {
  int Ret;
  std::vector<DeallocTgtPtrInfo> DeallocTgtPtrs;
  // process each input.
  for (int32_t I = ArgNum - 1; I >= 0; --I) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataEnd, call the
      // targetDataMapper variant which will call targetDataEnd again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      Ret = targetDataMapper(Device, ArgBases[I], Args[I], ArgSizes[I],
                             ArgTypes[I], ArgMappers[I], targetDataEnd);

      if (Ret != OFFLOAD_SUCCESS) {
        DP("Call to targetDataEnd via targetDataMapper for custom mapper"
           " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    int64_t DataSize = ArgSizes[I];
    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    const int NextI = I + 1;
    if (getParentIndex(ArgTypes[I]) < 0 && NextI < ArgNum &&
        getParentIndex(ArgTypes[NextI]) == I) {
      int64_t Padding = (int64_t)HstPtrBegin % Alignment;
      if (Padding) {
        DP("Using a Padding of %" PRId64 " bytes for begin address " DPxMOD
           "\n",
           Padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *)HstPtrBegin - Padding;
        DataSize += Padding;
      }
    }

    bool IsLast, IsHostPtr;
    bool IsImplicit = ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT;
    bool UpdateRef = !(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
                     (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ);
    bool ForceDelete = ArgTypes[I] & OMP_TGT_MAPTYPE_DELETE;
    bool HasCloseModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_CLOSE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin = Device.getTgtPtrBegin(
        HstPtrBegin, DataSize, IsLast, UpdateRef, IsHostPtr, !IsImplicit);
    if (!TgtPtrBegin && (DataSize || HasPresentModifier)) {
      DP("Mapping does not exist (%s)\n",
         (HasPresentModifier ? "'present' map type modifier" : "ignored"));
      if (HasPresentModifier) {
        // This should be an error upon entering an "omp target exit data".  It
        // should not be an error upon exiting an "omp target data" or "omp
        // target".  For "omp target data", Clang thus doesn't include present
        // modifiers for end calls.  For "omp target", we have not found a valid
        // OpenMP program for which the error matters: it appears that, if a
        // program can guarantee that data is present at the beginning of an
        // "omp target" region so that there's no error there, that data is also
        // guaranteed to be present at the end.
        MESSAGE("device mapping required by 'present' map type modifier does "
                "not exist for host address " DPxMOD " (%" PRId64 " bytes)",
                DPxPTR(HstPtrBegin), DataSize);
        return OFFLOAD_FAIL;
      }
    } else {
      DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
         " - is%s last\n",
         DataSize, DPxPTR(TgtPtrBegin), (IsLast ? "" : " not"));
    }

    bool DelEntry = IsLast || ForceDelete;

    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_FROM) {
        bool Always = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
        bool CopyMember = false;
        if (!(RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
            HasCloseModifier) {
          if ((ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
              !(ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
            // Copy data only if the "parent" struct has RefCount==1.
            int32_t ParentIdx = getParentIndex(ArgTypes[I]);
            uint64_t ParentRC = Device.getMapEntryRefCnt(Args[ParentIdx]);
            assert(ParentRC > 0 && "parent struct not found");
            if (ParentRC == 1)
              CopyMember = true;
          }
        }

        if ((DelEntry || Always || CopyMember) &&
            !(RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
              TgtPtrBegin == HstPtrBegin)) {
          DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
             DataSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
          Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize,
                                    AsyncInfo);
          if (Ret != OFFLOAD_SUCCESS) {
            DP("Copying data from device failed.\n");
            return OFFLOAD_FAIL;
          }
        }
      }

      // If we copied back to the host a struct/array containing pointers, we
      // need to restore the original host pointer values from their shadow
      // copies. If the struct is going to be deallocated, remove any remaining
      // shadow pointer entries for this struct.
      uintptr_t LB = (uintptr_t)HstPtrBegin;
      uintptr_t UB = (uintptr_t)HstPtrBegin + DataSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator Itr = Device.ShadowPtrMap.begin();
           Itr != Device.ShadowPtrMap.end();) {
        void **ShadowHstPtrAddr = (void **)Itr->first;

        // An STL map is sorted on its keys; use this property
        // to quickly determine when to break out of the loop.
        if ((uintptr_t)ShadowHstPtrAddr < LB) {
          ++Itr;
          continue;
        }
        if ((uintptr_t)ShadowHstPtrAddr >= UB)
          break;

        // If we copied the struct to the host, we need to restore the pointer.
        if (ArgTypes[I] & OMP_TGT_MAPTYPE_FROM) {
          DP("Restoring original host pointer value " DPxMOD " for host "
             "pointer " DPxMOD "\n",
             DPxPTR(Itr->second.HstPtrVal), DPxPTR(ShadowHstPtrAddr));
          *ShadowHstPtrAddr = Itr->second.HstPtrVal;
        }
        // If the struct is to be deallocated, remove the shadow entry.
        if (DelEntry) {
          DP("Removing shadow pointer " DPxMOD "\n", DPxPTR(ShadowHstPtrAddr));
          Itr = Device.ShadowPtrMap.erase(Itr);
        } else {
          ++Itr;
        }
      }
      Device.ShadowMtx.unlock();

      // Add pointer to the buffer for later deallocation
      if (DelEntry)
        DeallocTgtPtrs.emplace_back(HstPtrBegin, DataSize, ForceDelete,
                                    HasCloseModifier);
    }
  }

  // We need to synchronize before deallocating data.
  // If AsyncInfo is nullptr, the previous data transfer (if has) will be
  // synchronous, so we don't need to synchronize again. If AsyncInfo->Queue is
  // nullptr, there is no data transfer happened because once there is,
  // AsyncInfo->Queue will not be nullptr, so again, we don't need to
  // synchronize.
  if (AsyncInfo && AsyncInfo->Queue) {
    Ret = Device.synchronize(AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      DP("Failed to synchronize device.\n");
      return OFFLOAD_FAIL;
    }
  }

  // Deallocate target pointer
  for (DeallocTgtPtrInfo &Info : DeallocTgtPtrs) {
    Ret = Device.deallocTgtPtr(Info.HstPtrBegin, Info.DataSize,
                               Info.ForceDelete, Info.HasCloseModifier);
    if (Ret != OFFLOAD_SUCCESS) {
      DP("Deallocating data from device failed.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Internal function to pass data to/from the target.
// async_info_ptr is currently unused, added here so target_data_update has the
// same signature as targetDataBegin and targetDataEnd.
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
      // targetDataMapper variant which will call target_data_update again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", i);

      int rc =
          targetDataMapper(Device, args_base[i], args[i], arg_sizes[i],
                           arg_types[i], arg_mappers[i], target_data_update);

      if (rc != OFFLOAD_SUCCESS) {
        DP("Call to target_data_update via targetDataMapper for custom mapper"
           " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = args[i];
    int64_t MapSize = arg_sizes[i];
    bool IsLast, IsHostPtr;
    void *TgtPtrBegin = Device.getTgtPtrBegin(
        HstPtrBegin, MapSize, IsLast, false, IsHostPtr, /*MustContain=*/true);
    if (!TgtPtrBegin) {
      DP("hst data:" DPxMOD " not found, becomes a noop\n", DPxPTR(HstPtrBegin));
      if (arg_types[i] & OMP_TGT_MAPTYPE_PRESENT) {
        MESSAGE("device mapping required by 'present' motion modifier does not "
                "exist for host address " DPxMOD " (%" PRId64 " bytes)",
                DPxPTR(HstPtrBegin), MapSize);
        return OFFLOAD_FAIL;
      }
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
      int rt = Device.retrieveData(HstPtrBegin, TgtPtrBegin, MapSize, nullptr);
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
      int rt = Device.submitData(TgtPtrBegin, HstPtrBegin, MapSize, nullptr);
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        return OFFLOAD_FAIL;
      }

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void **)it->first;
        if ((uintptr_t)ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t)ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original target pointer value " DPxMOD " for target "
           "pointer " DPxMOD "\n",
           DPxPTR(it->second.TgtPtrVal), DPxPTR(it->second.TgtPtrAddr));
        rt = Device.submitData(it->second.TgtPtrAddr, &it->second.TgtPtrVal,
                               sizeof(void *), nullptr);
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

namespace {
/// Find the table information in the map or look it up in the translation
/// tables.
TableMap *getTableMap(void *HostPtr) {
  std::lock_guard<std::mutex> TblMapLock(*TblMapMtx);
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap->find(HostPtr);

  if (TableMapIt != HostPtrToTableMap->end())
    return &TableMapIt->second;

  // We don't have a map. So search all the registered libraries.
  TableMap *TM = nullptr;
  std::lock_guard<std::mutex> TrlTblLock(*TrlTblMtx);
  for (HostEntriesBeginToTransTableTy::iterator Itr =
           HostEntriesBeginToTransTable->begin();
       Itr != HostEntriesBeginToTransTable->end(); ++Itr) {
    // get the translation table (which contains all the good info).
    TranslationTable *TransTable = &Itr->second;
    // iterate over all the host table entries to see if we can locate the
    // host_ptr.
    __tgt_offload_entry *Cur = TransTable->HostTable.EntriesBegin;
    for (uint32_t I = 0; Cur < TransTable->HostTable.EntriesEnd; ++Cur, ++I) {
      if (Cur->addr != HostPtr)
        continue;
      // we got a match, now fill the HostPtrToTableMap so that we
      // may avoid this search next time.
      TM = &(*HostPtrToTableMap)[HostPtr];
      TM->Table = TransTable;
      TM->Index = I;
      return TM;
    }
  }

  return nullptr;
}

/// Get loop trip count
/// FIXME: This function will not work right if calling
/// __kmpc_push_target_tripcount in one thread but doing offloading in another
/// thread, which might occur when we call task yield.
uint64_t getLoopTripCount(int64_t DeviceId) {
  DeviceTy &Device = Devices[DeviceId];
  uint64_t LoopTripCount = 0;

  {
    std::lock_guard<std::mutex> TblMapLock(*TblMapMtx);
    auto I = Device.LoopTripCnt.find(__kmpc_global_thread_num(NULL));
    if (I != Device.LoopTripCnt.end()) {
      LoopTripCount = I->second;
      Device.LoopTripCnt.erase(I);
      DP("loop trip count is %lu.\n", LoopTripCount);
    }
  }

  return LoopTripCount;
}

/// Process data before launching the kernel, including calling targetDataBegin
/// to map and transfer data to target device, transferring (first-)private
/// variables.
int processDataBefore(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                      void **ArgBases, void **Args, int64_t *ArgSizes,
                      int64_t *ArgTypes, void **ArgMappers,
                      std::vector<void *> &TgtArgs,
                      std::vector<ptrdiff_t> &TgtOffsets,
                      std::vector<void *> &FPArrays,
                      __tgt_async_info *AsyncInfo) {
  DeviceTy &Device = Devices[DeviceId];
  int Ret = targetDataBegin(Device, ArgNum, ArgBases, Args, ArgSizes, ArgTypes,
                            ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Call to targetDataBegin failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // List of (first-)private arrays allocated for this target region
  std::vector<int> TgtArgsPositions(ArgNum, -1);

  for (int32_t I = 0; I < ArgNum; ++I) {
    if (!(ArgTypes[I] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into TgtArgs.
      // Check for lambda mapping.
      if (isLambdaMapping(ArgTypes[I])) {
        assert((ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
               "PTR_AND_OBJ must be also MEMBER_OF.");
        unsigned Idx = getParentIndex(ArgTypes[I]);
        int TgtIdx = TgtArgsPositions[Idx];
        assert(TgtIdx != -1 && "Base address must be translated already.");
        // The parent lambda must be processed already and it must be the last
        // in TgtArgs and TgtOffsets arrays.
        void *HstPtrVal = Args[I];
        void *HstPtrBegin = ArgBases[I];
        void *HstPtrBase = Args[Idx];
        bool IsLast, IsHostPtr; // unused.
        void *TgtPtrBase =
            (void *)((intptr_t)TgtArgs[TgtIdx] + TgtOffsets[TgtIdx]);
        DP("Parent lambda base " DPxMOD "\n", DPxPTR(TgtPtrBase));
        uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
        void *TgtPtrBegin = (void *)((uintptr_t)TgtPtrBase + Delta);
        void *PointerTgtPtrBegin = Device.getTgtPtrBegin(
            HstPtrVal, ArgSizes[I], IsLast, false, IsHostPtr);
        if (!PointerTgtPtrBegin) {
          DP("No lambda captured variable mapped (" DPxMOD ") - ignored\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        if (RTLs->RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
            TgtPtrBegin == HstPtrBegin) {
          DP("Unified memory is active, no need to map lambda captured"
             "variable (" DPxMOD ")\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        DP("Update lambda reference (" DPxMOD ") -> [" DPxMOD "]\n",
           DPxPTR(PointerTgtPtrBegin), DPxPTR(TgtPtrBegin));
        Ret = Device.submitData(TgtPtrBegin, &PointerTgtPtrBegin,
                                sizeof(void *), AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      continue;
    }
    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgBases[I];
    void *TgtPtrBegin;
    ptrdiff_t TgtBaseOffset;
    bool IsLast, IsHostPtr; // unused.
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
         DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE) {
      // Allocate memory for (first-)private array
      TgtPtrBegin = Device.allocData(ArgSizes[I], HstPtrBegin);
      if (!TgtPtrBegin) {
        DP("Data allocation for %sprivate array " DPxMOD " failed, "
           "abort target.\n",
           (ArgTypes[I] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
           DPxPTR(HstPtrBegin));
        return OFFLOAD_FAIL;
      }
      FPArrays.push_back(TgtPtrBegin);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD " for "
         "%sprivate array " DPxMOD " - pushing target argument " DPxMOD "\n",
         ArgSizes[I], DPxPTR(TgtPtrBegin),
         (ArgTypes[I] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
         DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBase));
#endif
      // If first-private, copy data from host
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_TO) {
        Ret =
            Device.submitData(TgtPtrBegin, HstPtrBegin, ArgSizes[I], AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed, failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    } else {
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        HstPtrBase = *reinterpret_cast<void **>(HstPtrBase);
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, ArgSizes[I], IsLast,
                                          false, IsHostPtr);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD "\n",
         DPxPTR(TgtPtrBase), DPxPTR(HstPtrBegin));
#endif
    }
    TgtArgsPositions[I] = TgtArgs.size();
    TgtArgs.push_back(TgtPtrBegin);
    TgtOffsets.push_back(TgtBaseOffset);
  }

  assert(TgtArgs.size() == TgtOffsets.size() &&
         "Size mismatch in arguments and offsets");

  return OFFLOAD_SUCCESS;
}

/// Process data after launching the kernel, including transferring data back to
/// host if needed and deallocating target memory of (first-)private variables.
/// FIXME: This function has correctness issue that target memory might be
/// deallocated when they're being used.
int processDataAfter(int64_t DeviceId, void *HostPtr, int32_t ArgNum,
                     void **ArgBases, void **Args, int64_t *ArgSizes,
                     int64_t *ArgTypes, void **ArgMappers,
                     std::vector<void *> &FPArrays,
                     __tgt_async_info *AsyncInfo) {
  DeviceTy &Device = Devices[DeviceId];

  // Move data from device.
  int Ret = targetDataEnd(Device, ArgNum, ArgBases, Args, ArgSizes, ArgTypes,
                          ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Call to targetDataEnd failed, abort targe.\n");
    return OFFLOAD_FAIL;
  }

  // Deallocate (first-)private arrays
  for (void *P : FPArrays) {
    Ret = Device.deleteData(P);
    if (Ret != OFFLOAD_SUCCESS) {
      DP("Deallocation of (first-)private arrays failed.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}
} // namespace

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
int target(int64_t DeviceId, void *HostPtr, int32_t ArgNum, void **ArgBases,
           void **Args, int64_t *ArgSizes, int64_t *ArgTypes, void **ArgMappers,
           int32_t TeamNum, int32_t ThreadLimit, int IsTeamConstruct) {
  DeviceTy &Device = Devices[DeviceId];

  TableMap *TM = getTableMap(HostPtr);
  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr " DPxMOD " does not have a matching target pointer.\n",
       DPxPTR(HostPtr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(*TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  __tgt_async_info AsyncInfo;

  std::vector<void *> TgtArgs;
  std::vector<ptrdiff_t> TgtOffsets;
  std::vector<void *> FPArrays;

  // Process data, such as data mapping, before launching the kernel
  int Ret = processDataBefore(DeviceId, HostPtr, ArgNum, ArgBases, Args,
                              ArgSizes, ArgTypes, ArgMappers, TgtArgs,
                              TgtOffsets, FPArrays, &AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Failed to process data before launching the kernel.\n");
    return OFFLOAD_FAIL;
  }

  // Get loop trip count
  uint64_t LoopTripCount = getLoopTripCount(DeviceId);

  // Launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);

  if (IsTeamConstruct)
    Ret = Device.runTeamRegion(TgtEntryPtr, &TgtArgs[0], &TgtOffsets[0],
                               TgtArgs.size(), TeamNum, ThreadLimit,
                               LoopTripCount, &AsyncInfo);
  else
    Ret = Device.runRegion(TgtEntryPtr, &TgtArgs[0], &TgtOffsets[0],
                           TgtArgs.size(), &AsyncInfo);

  if (Ret != OFFLOAD_SUCCESS) {
    DP("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  // Transfer data back and deallocate target memory for (first-)private
  // variables
  Ret = processDataAfter(DeviceId, HostPtr, ArgNum, ArgBases, Args, ArgSizes,
                         ArgTypes, ArgMappers, FPArrays, &AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Failed to process data after launching the kernel.\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}
