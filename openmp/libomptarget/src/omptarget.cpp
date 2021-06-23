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

#include "omptarget.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <vector>

int AsyncInfoTy::synchronize() {
  int Result = OFFLOAD_SUCCESS;
  if (AsyncInfo.Queue) {
    // If we have a queue we need to synchronize it now.
    Result = Device.synchronize(*this);
    assert(AsyncInfo.Queue == nullptr &&
           "The device plugin should have nulled the queue to indicate there "
           "are no outstanding actions!");
  }
  return Result;
}

void *&AsyncInfoTy::getVoidPtrLocation() {
  BufferLocations.push_back(nullptr);
  return BufferLocations.back();
}

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
static int InitLibrary(DeviceTy &Device) {
  /*
   * Map global data
   */
  int32_t device_id = Device.DeviceID;
  int rc = OFFLOAD_SUCCESS;
  bool supportsEmptyImages = Device.RTL->supports_empty_images &&
                             Device.RTL->supports_empty_images() > 0;

  Device.PendingGlobalsMtx.lock();
  PM->TrlTblMtx.lock();
  for (auto *HostEntriesBegin : PM->HostEntriesBeginRegistrationOrder) {
    TranslationTable *TransTable =
        &PM->HostEntriesBeginToTransTable[HostEntriesBegin];
    if (TransTable->HostTable.EntriesBegin ==
            TransTable->HostTable.EntriesEnd &&
        !supportsEmptyImages) {
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
      REPORT("No image loaded for device id %d.\n", device_id);
      rc = OFFLOAD_FAIL;
      break;
    }
    // 2) load image into the target table.
    __tgt_target_table *TargetTable = TransTable->TargetsTable[device_id] =
        Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      REPORT("Unable to generate entries table for device id %d.\n", device_id);
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
      REPORT("Host and Target tables mismatch for device id %d [%zx != %zx].\n",
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
           "\n",
           DPxPTR(CurrHostEntry->addr), DPxPTR(CurrDeviceEntry->addr),
           CurrDeviceEntry->size);
        Device.HostDataToTargetMap.emplace(
            (uintptr_t)CurrHostEntry->addr /*HstPtrBase*/,
            (uintptr_t)CurrHostEntry->addr /*HstPtrBegin*/,
            (uintptr_t)CurrHostEntry->addr + CurrHostEntry->size /*HstPtrEnd*/,
            (uintptr_t)CurrDeviceEntry->addr /*TgtPtrBegin*/, nullptr,
            true /*IsRefCountINF*/);
      }
    }
    Device.DataMapMtx.unlock();
  }
  PM->TrlTblMtx.unlock();

  if (rc != OFFLOAD_SUCCESS) {
    Device.PendingGlobalsMtx.unlock();
    return rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    AsyncInfoTy AsyncInfo(Device);
    // Call all ctors for all libraries registered so far
    for (auto &lib : Device.PendingCtorsDtors) {
      if (!lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &entry : lib.second.PendingCtors) {
          void *ctor = entry;
          int rc =
              target(nullptr, Device, ctor, 0, nullptr, nullptr, nullptr,
                     nullptr, nullptr, nullptr, 1, 1, true /*team*/, AsyncInfo);
          if (rc != OFFLOAD_SUCCESS) {
            REPORT("Running ctor " DPxMOD " failed.\n", DPxPTR(ctor));
            Device.PendingGlobalsMtx.unlock();
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib " DPxMOD "\n", DPxPTR(lib.first));
      }
    }
    // All constructors have been issued, wait for them now.
    if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;
  }
  Device.HasPendingGlobals = false;
  Device.PendingGlobalsMtx.unlock();

  return OFFLOAD_SUCCESS;
}

void handleTargetOutcome(bool Success, ident_t *Loc) {
  switch (PM->TargetOffloadPolicy) {
  case tgt_disabled:
    if (Success) {
      FATAL_MESSAGE0(1, "expected no offloading while offloading is disabled");
    }
    break;
  case tgt_default:
    FATAL_MESSAGE0(1, "default offloading policy must be switched to "
                      "mandatory or disabled");
    break;
  case tgt_mandatory:
    if (!Success) {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE)
        for (auto &Device : PM->Devices)
          dumpTargetPointerMappings(Loc, Device);
      else
        FAILURE_MESSAGE("Run with LIBOMPTARGET_INFO=%d to dump host-target "
                        "pointer mappings.\n",
                        OMP_INFOTYPE_DUMP_TABLE);

      SourceInfo info(Loc);
      if (info.isAvailible())
        fprintf(stderr, "%s:%d:%d: ", info.getFilename(), info.getLine(),
                info.getColumn());
      else
        FAILURE_MESSAGE("Source location information not present. Compile with "
                        "-g or -gline-tables-only.\n");
      FATAL_MESSAGE0(
          1, "failure of target construct while offloading is mandatory");
    } else {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE)
        for (auto &Device : PM->Devices)
          dumpTargetPointerMappings(Loc, Device);
    }
    break;
  }
}

static void handleDefaultTargetOffload() {
  PM->TargetOffloadMtx.lock();
  if (PM->TargetOffloadPolicy == tgt_default) {
    if (omp_get_num_devices() > 0) {
      DP("Default TARGET OFFLOAD policy is now mandatory "
         "(devices were found)\n");
      PM->TargetOffloadPolicy = tgt_mandatory;
    } else {
      DP("Default TARGET OFFLOAD policy is now disabled "
         "(no devices were found)\n");
      PM->TargetOffloadPolicy = tgt_disabled;
    }
  }
  PM->TargetOffloadMtx.unlock();
}

static bool isOffloadDisabled() {
  if (PM->TargetOffloadPolicy == tgt_default)
    handleDefaultTargetOffload();
  return PM->TargetOffloadPolicy == tgt_disabled;
}

// If offload is enabled, ensure that device DeviceID has been initialized,
// global ctors have been executed, and global data has been mapped.
//
// There are three possible results:
// - Return OFFLOAD_SUCCESS if the device is ready for offload.
// - Return OFFLOAD_FAIL without reporting a runtime error if offload is
//   disabled, perhaps because the initial device was specified.
// - Report a runtime error and return OFFLOAD_FAIL.
//
// If DeviceID == OFFLOAD_DEVICE_DEFAULT, set DeviceID to the default device.
// This step might be skipped if offload is disabled.
int checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc) {
  if (isOffloadDisabled()) {
    DP("Offload is disabled\n");
    return OFFLOAD_FAIL;
  }

  if (DeviceID == OFFLOAD_DEVICE_DEFAULT) {
    DeviceID = omp_get_default_device();
    DP("Use default device id %" PRId64 "\n", DeviceID);
  }

  // Proposed behavior for OpenMP 5.2 in OpenMP spec github issue 2669.
  if (omp_get_num_devices() == 0) {
    DP("omp_get_num_devices() == 0 but offload is manadatory\n");
    handleTargetOutcome(false, Loc);
    return OFFLOAD_FAIL;
  }

  if (DeviceID == omp_get_initial_device()) {
    DP("Device is host (%" PRId64 "), returning as if offload is disabled\n",
       DeviceID);
    return OFFLOAD_FAIL;
  }

  // Is device ready?
  if (!device_is_ready(DeviceID)) {
    REPORT("Device %" PRId64 " is not ready.\n", DeviceID);
    handleTargetOutcome(false, Loc);
    return OFFLOAD_FAIL;
  }

  // Get device info.
  DeviceTy &Device = PM->Devices[DeviceID];

  // Check whether global data has been mapped for this device
  Device.PendingGlobalsMtx.lock();
  bool hasPendingGlobals = Device.HasPendingGlobals;
  Device.PendingGlobalsMtx.unlock();
  if (hasPendingGlobals && InitLibrary(Device) != OFFLOAD_SUCCESS) {
    REPORT("Failed to init globals on device %" PRId64 "\n", DeviceID);
    handleTargetOutcome(false, Loc);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

static int32_t getParentIndex(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

void *targetAllocExplicit(size_t size, int device_num, int kind,
                          const char *name) {
  TIMESCOPE();
  DP("Call to %s for device %d requesting %zu bytes\n", name, device_num, size);

  if (size <= 0) {
    DP("Call to %s with non-positive length\n", name);
    return NULL;
  }

  void *rc = NULL;

  if (device_num == omp_get_initial_device()) {
    rc = malloc(size);
    DP("%s returns host ptr " DPxMOD "\n", name, DPxPTR(rc));
    return rc;
  }

  if (!device_is_ready(device_num)) {
    DP("%s returns NULL ptr\n", name);
    return NULL;
  }

  DeviceTy &Device = PM->Devices[device_num];
  rc = Device.allocData(size, nullptr, kind);
  DP("%s returns device ptr " DPxMOD "\n", name, DPxPTR(rc));
  return rc;
}

/// Call the user-defined mapper function followed by the appropriate
// targetData* function (targetData{Begin,End,Update}).
int targetDataMapper(ident_t *loc, DeviceTy &Device, void *arg_base, void *arg,
                     int64_t arg_size, int64_t arg_type,
                     map_var_info_t arg_names, void *arg_mapper,
                     AsyncInfoTy &AsyncInfo,
                     TargetDataFuncPtrTy target_data_function) {
  TIMESCOPE_WITH_IDENT(loc);
  DP("Calling the mapper function " DPxMOD "\n", DPxPTR(arg_mapper));

  // The mapper function fills up Components.
  MapperComponentsTy MapperComponents;
  MapperFuncPtrTy MapperFuncPtr = (MapperFuncPtrTy)(arg_mapper);
  (*MapperFuncPtr)((void *)&MapperComponents, arg_base, arg, arg_size, arg_type,
                   arg_names);

  // Construct new arrays for args_base, args, arg_sizes and arg_types
  // using the information in MapperComponents and call the corresponding
  // targetData* function using these new arrays.
  std::vector<void *> MapperArgsBase(MapperComponents.Components.size());
  std::vector<void *> MapperArgs(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgSizes(MapperComponents.Components.size());
  std::vector<int64_t> MapperArgTypes(MapperComponents.Components.size());
  std::vector<void *> MapperArgNames(MapperComponents.Components.size());

  for (unsigned I = 0, E = MapperComponents.Components.size(); I < E; ++I) {
    auto &C = MapperComponents.Components[I];
    MapperArgsBase[I] = C.Base;
    MapperArgs[I] = C.Begin;
    MapperArgSizes[I] = C.Size;
    MapperArgTypes[I] = C.Type;
    MapperArgNames[I] = C.Name;
  }

  int rc = target_data_function(loc, Device, MapperComponents.Components.size(),
                                MapperArgsBase.data(), MapperArgs.data(),
                                MapperArgSizes.data(), MapperArgTypes.data(),
                                MapperArgNames.data(), /*arg_mappers*/ nullptr,
                                AsyncInfo, /*FromMapper=*/true);

  return rc;
}

/// Internal function to do the mapping and transfer the data to the device
int targetDataBegin(ident_t *loc, DeviceTy &Device, int32_t arg_num,
                    void **args_base, void **args, int64_t *arg_sizes,
                    int64_t *arg_types, map_var_info_t *arg_names,
                    void **arg_mappers, AsyncInfoTy &AsyncInfo,
                    bool FromMapper) {
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

      map_var_info_t arg_name = (!arg_names) ? nullptr : arg_names[i];
      int rc = targetDataMapper(loc, Device, args_base[i], args[i],
                                arg_sizes[i], arg_types[i], arg_name,
                                arg_mappers[i], AsyncInfo, targetDataBegin);

      if (rc != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataBegin via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    int64_t data_size = arg_sizes[i];
    map_var_info_t HstPtrName = (!arg_names) ? nullptr : arg_names[i];

    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t padding = 0;
    const int next_i = i + 1;
    if (getParentIndex(arg_types[i]) < 0 && next_i < arg_num &&
        getParentIndex(arg_types[next_i]) == i) {
      padding = (int64_t)HstPtrBegin % Alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
           "\n",
           padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *)HstPtrBegin - padding;
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
    bool UpdateRef =
        !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) && !(FromMapper && i == 0);
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
          HstPtrBase, HstPtrBase, sizeof(void *), nullptr, Pointer_IsNew,
          IsHostPtr, IsImplicit, UpdateRef, HasCloseModifier,
          HasPresentModifier);
      if (!PointerTgtPtrBegin) {
        REPORT("Call to getOrAllocTgtPtr returned null pointer (%s).\n",
               HasPresentModifier ? "'present' map type modifier"
                                  : "device failure or illegal mapping");
        return OFFLOAD_FAIL;
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
         "\n",
         sizeof(void *), DPxPTR(PointerTgtPtrBegin),
         (Pointer_IsNew ? "" : " not"));
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      // No need to update pointee ref count for the first element of the
      // subelement that comes from mapper.
      UpdateRef =
          (!FromMapper || i != 0); // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(
        HstPtrBegin, HstPtrBase, data_size, HstPtrName, IsNew, IsHostPtr,
        IsImplicit, UpdateRef, HasCloseModifier, HasPresentModifier);
    // If data_size==0, then the argument could be a zero-length pointer to
    // NULL, so getOrAlloc() returning NULL is not an error.
    if (!TgtPtrBegin && (data_size || HasPresentModifier)) {
      REPORT("Call to getOrAllocTgtPtr returned null pointer (%s).\n",
             HasPresentModifier ? "'present' map type modifier"
                                : "device failure or illegal mapping");
      return OFFLOAD_FAIL;
    }
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
       " - is%s new\n",
       data_size, DPxPTR(TgtPtrBegin), (IsNew ? "" : " not"));

    if (arg_types[i] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      uintptr_t Delta = (uintptr_t)HstPtrBegin - (uintptr_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uintptr_t)TgtPtrBegin - Delta);
      DP("Returning device pointer " DPxMOD "\n", DPxPTR(TgtPtrBase));
      args_base[i] = TgtPtrBase;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      bool copy = false;
      if (!(PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
          HasCloseModifier) {
        if (IsNew || (arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS)) {
          copy = true;
        } else if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
                   !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
          // Copy data only if the "parent" struct has RefCount==1.
          // If this is a PTR_AND_OBJ entry, the OBJ is not part of the struct,
          // so exclude it from this check.
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
        int rt =
            Device.submitData(TgtPtrBegin, HstPtrBegin, data_size, AsyncInfo);
        if (rt != OFFLOAD_SUCCESS) {
          REPORT("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ && !IsHostPtr) {
      DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
         DPxPTR(PointerTgtPtrBegin), DPxPTR(TgtPtrBegin));
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *&TgtPtrBase = AsyncInfo.getVoidPtrLocation();
      TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      int rt = Device.submitData(PointerTgtPtrBegin, &TgtPtrBase,
                                 sizeof(void *), AsyncInfo);
      if (rt != OFFLOAD_SUCCESS) {
        REPORT("Copying data to device failed.\n");
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
  /// Whether it has \p close modifier
  bool HasCloseModifier;

  DeallocTgtPtrInfo(void *HstPtr, int64_t Size, bool HasCloseModifier)
      : HstPtrBegin(HstPtr), DataSize(Size),
        HasCloseModifier(HasCloseModifier) {}
};
} // namespace

/// Internal function to undo the mapping and retrieve the data from the device.
int targetDataEnd(ident_t *loc, DeviceTy &Device, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *ArgNames,
                  void **ArgMappers, AsyncInfoTy &AsyncInfo, bool FromMapper) {
  int Ret;
  std::vector<DeallocTgtPtrInfo> DeallocTgtPtrs;
  void *FromMapperBase = nullptr;
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

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      Ret = targetDataMapper(loc, Device, ArgBases[I], Args[I], ArgSizes[I],
                             ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                             targetDataEnd);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataEnd via targetDataMapper for custom mapper"
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
    bool UpdateRef = (!(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
                      (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) &&
                     !(FromMapper && I == 0);
    bool ForceDelete = ArgTypes[I] & OMP_TGT_MAPTYPE_DELETE;
    bool HasCloseModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_CLOSE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin =
        Device.getTgtPtrBegin(HstPtrBegin, DataSize, IsLast, UpdateRef,
                              IsHostPtr, !IsImplicit, ForceDelete);
    if (!TgtPtrBegin && (DataSize || HasPresentModifier)) {
      DP("Mapping does not exist (%s)\n",
         (HasPresentModifier ? "'present' map type modifier" : "ignored"));
      if (HasPresentModifier) {
        // OpenMP 5.1, sec. 2.21.7.1 "map Clause", p. 350 L10-13:
        // "If a map clause appears on a target, target data, target enter data
        // or target exit data construct with a present map-type-modifier then
        // on entry to the region if the corresponding list item does not appear
        // in the device data environment then an error occurs and the program
        // terminates."
        //
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

    // OpenMP 5.1, sec. 2.21.7.1 "map Clause", p. 351 L14-16:
    // "If the map clause appears on a target, target data, or target exit data
    // construct and a corresponding list item of the original list item is not
    // present in the device data environment on exit from the region then the
    // list item is ignored."
    if (!TgtPtrBegin)
      continue;

    bool DelEntry = IsLast;

    // If the last element from the mapper (for end transfer args comes in
    // reverse order), do not remove the partial entry, the parent struct still
    // exists.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_FROM) {
        bool Always = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
        bool CopyMember = false;
        if (!(PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
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
            !(PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
              TgtPtrBegin == HstPtrBegin)) {
          DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
             DataSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
          Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize,
                                    AsyncInfo);
          if (Ret != OFFLOAD_SUCCESS) {
            REPORT("Copying data from device failed.\n");
            return OFFLOAD_FAIL;
          }
        }
      }
      if (DelEntry && FromMapper && I == 0) {
        DelEntry = false;
        FromMapperBase = HstPtrBegin;
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
        DeallocTgtPtrs.emplace_back(HstPtrBegin, DataSize, HasCloseModifier);
    }
  }

  // TODO: We should not synchronize here but pass the AsyncInfo object to the
  //       allocate/deallocate device APIs.
  //
  // We need to synchronize before deallocating data.
  Ret = AsyncInfo.synchronize();
  if (Ret != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  // Deallocate target pointer
  for (DeallocTgtPtrInfo &Info : DeallocTgtPtrs) {
    if (FromMapperBase && FromMapperBase == Info.HstPtrBegin)
      continue;
    Ret = Device.deallocTgtPtr(Info.HstPtrBegin, Info.DataSize,
                               Info.HasCloseModifier);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Deallocating data from device failed.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

static int targetDataContiguous(ident_t *loc, DeviceTy &Device, void *ArgsBase,
                                void *HstPtrBegin, int64_t ArgSize,
                                int64_t ArgType, AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_IDENT(loc);
  bool IsLast, IsHostPtr;
  void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, ArgSize, IsLast, false,
                                            IsHostPtr, /*MustContain=*/true);
  if (!TgtPtrBegin) {
    DP("hst data:" DPxMOD " not found, becomes a noop\n", DPxPTR(HstPtrBegin));
    if (ArgType & OMP_TGT_MAPTYPE_PRESENT) {
      MESSAGE("device mapping required by 'present' motion modifier does not "
              "exist for host address " DPxMOD " (%" PRId64 " bytes)",
              DPxPTR(HstPtrBegin), ArgSize);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  if (PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
      TgtPtrBegin == HstPtrBegin) {
    DP("hst data:" DPxMOD " unified and shared, becomes a noop\n",
       DPxPTR(HstPtrBegin));
    return OFFLOAD_SUCCESS;
  }

  if (ArgType & OMP_TGT_MAPTYPE_FROM) {
    DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
       ArgSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
    int Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, ArgSize, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data from device failed.\n");
      return OFFLOAD_FAIL;
    }

    uintptr_t LB = (uintptr_t)HstPtrBegin;
    uintptr_t UB = (uintptr_t)HstPtrBegin + ArgSize;
    Device.ShadowMtx.lock();
    for (ShadowPtrListTy::iterator IT = Device.ShadowPtrMap.begin();
         IT != Device.ShadowPtrMap.end(); ++IT) {
      void **ShadowHstPtrAddr = (void **)IT->first;
      if ((uintptr_t)ShadowHstPtrAddr < LB)
        continue;
      if ((uintptr_t)ShadowHstPtrAddr >= UB)
        break;
      DP("Restoring original host pointer value " DPxMOD
         " for host pointer " DPxMOD "\n",
         DPxPTR(IT->second.HstPtrVal), DPxPTR(ShadowHstPtrAddr));
      *ShadowHstPtrAddr = IT->second.HstPtrVal;
    }
    Device.ShadowMtx.unlock();
  }

  if (ArgType & OMP_TGT_MAPTYPE_TO) {
    DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
       ArgSize, DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
    int Ret = Device.submitData(TgtPtrBegin, HstPtrBegin, ArgSize, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      return OFFLOAD_FAIL;
    }

    uintptr_t LB = (uintptr_t)HstPtrBegin;
    uintptr_t UB = (uintptr_t)HstPtrBegin + ArgSize;
    Device.ShadowMtx.lock();
    for (ShadowPtrListTy::iterator IT = Device.ShadowPtrMap.begin();
         IT != Device.ShadowPtrMap.end(); ++IT) {
      void **ShadowHstPtrAddr = (void **)IT->first;
      if ((uintptr_t)ShadowHstPtrAddr < LB)
        continue;
      if ((uintptr_t)ShadowHstPtrAddr >= UB)
        break;
      DP("Restoring original target pointer value " DPxMOD " for target "
         "pointer " DPxMOD "\n",
         DPxPTR(IT->second.TgtPtrVal), DPxPTR(IT->second.TgtPtrAddr));
      Ret = Device.submitData(IT->second.TgtPtrAddr, &IT->second.TgtPtrVal,
                              sizeof(void *), AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Copying data to device failed.\n");
        Device.ShadowMtx.unlock();
        return OFFLOAD_FAIL;
      }
    }
    Device.ShadowMtx.unlock();
  }
  return OFFLOAD_SUCCESS;
}

static int targetDataNonContiguous(ident_t *loc, DeviceTy &Device,
                                   void *ArgsBase,
                                   __tgt_target_non_contig *NonContig,
                                   uint64_t Size, int64_t ArgType,
                                   int CurrentDim, int DimSize, uint64_t Offset,
                                   AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_IDENT(loc);
  int Ret = OFFLOAD_SUCCESS;
  if (CurrentDim < DimSize) {
    for (unsigned int I = 0; I < NonContig[CurrentDim].Count; ++I) {
      uint64_t CurOffset =
          (NonContig[CurrentDim].Offset + I) * NonContig[CurrentDim].Stride;
      // we only need to transfer the first element for the last dimension
      // since we've already got a contiguous piece.
      if (CurrentDim != DimSize - 1 || I == 0) {
        Ret = targetDataNonContiguous(loc, Device, ArgsBase, NonContig, Size,
                                      ArgType, CurrentDim + 1, DimSize,
                                      Offset + CurOffset, AsyncInfo);
        // Stop the whole process if any contiguous piece returns anything
        // other than OFFLOAD_SUCCESS.
        if (Ret != OFFLOAD_SUCCESS)
          return Ret;
      }
    }
  } else {
    char *Ptr = (char *)ArgsBase + Offset;
    DP("Transfer of non-contiguous : host ptr " DPxMOD " offset %" PRIu64
       " len %" PRIu64 "\n",
       DPxPTR(Ptr), Offset, Size);
    Ret = targetDataContiguous(loc, Device, ArgsBase, Ptr, Size, ArgType,
                               AsyncInfo);
  }
  return Ret;
}

static int getNonContigMergedDimension(__tgt_target_non_contig *NonContig,
                                       int32_t DimSize) {
  int RemovedDim = 0;
  for (int I = DimSize - 1; I > 0; --I) {
    if (NonContig[I].Count * NonContig[I].Stride == NonContig[I - 1].Stride)
      RemovedDim++;
  }
  return RemovedDim;
}

/// Internal function to pass data to/from the target.
int targetDataUpdate(ident_t *loc, DeviceTy &Device, int32_t ArgNum,
                     void **ArgsBase, void **Args, int64_t *ArgSizes,
                     int64_t *ArgTypes, map_var_info_t *ArgNames,
                     void **ArgMappers, AsyncInfoTy &AsyncInfo, bool) {
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataUpdate, call the
      // targetDataMapper variant which will call targetDataUpdate again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Ret = targetDataMapper(loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                 ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                 targetDataUpdate);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataUpdate via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    int Ret = OFFLOAD_SUCCESS;

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_NON_CONTIG) {
      __tgt_target_non_contig *NonContig = (__tgt_target_non_contig *)Args[I];
      int32_t DimSize = ArgSizes[I];
      uint64_t Size =
          NonContig[DimSize - 1].Count * NonContig[DimSize - 1].Stride;
      int32_t MergedDim = getNonContigMergedDimension(NonContig, DimSize);
      Ret = targetDataNonContiguous(
          loc, Device, ArgsBase[I], NonContig, Size, ArgTypes[I],
          /*current_dim=*/0, DimSize - MergedDim, /*offset=*/0, AsyncInfo);
    } else {
      Ret = targetDataContiguous(loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                 ArgTypes[I], AsyncInfo);
    }
    if (Ret == OFFLOAD_FAIL)
      return OFFLOAD_FAIL;
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
  std::lock_guard<std::mutex> TblMapLock(PM->TblMapMtx);
  HostPtrToTableMapTy::iterator TableMapIt =
      PM->HostPtrToTableMap.find(HostPtr);

  if (TableMapIt != PM->HostPtrToTableMap.end())
    return &TableMapIt->second;

  // We don't have a map. So search all the registered libraries.
  TableMap *TM = nullptr;
  std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
  for (HostEntriesBeginToTransTableTy::iterator Itr =
           PM->HostEntriesBeginToTransTable.begin();
       Itr != PM->HostEntriesBeginToTransTable.end(); ++Itr) {
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
      TM = &(PM->HostPtrToTableMap)[HostPtr];
      TM->Table = TransTable;
      TM->Index = I;
      return TM;
    }
  }

  return nullptr;
}

/// Get loop trip count
/// FIXME: This function will not work right if calling
/// __kmpc_push_target_tripcount_mapper in one thread but doing offloading in
/// another thread, which might occur when we call task yield.
uint64_t getLoopTripCount(int64_t DeviceId) {
  DeviceTy &Device = PM->Devices[DeviceId];
  uint64_t LoopTripCount = 0;

  {
    std::lock_guard<std::mutex> TblMapLock(PM->TblMapMtx);
    auto I = Device.LoopTripCnt.find(__kmpc_global_thread_num(NULL));
    if (I != Device.LoopTripCnt.end()) {
      LoopTripCount = I->second;
      Device.LoopTripCnt.erase(I);
      DP("loop trip count is %" PRIu64 ".\n", LoopTripCount);
    }
  }

  return LoopTripCount;
}

/// A class manages private arguments in a target region.
class PrivateArgumentManagerTy {
  /// A data structure for the information of first-private arguments. We can
  /// use this information to optimize data transfer by packing all
  /// first-private arguments and transfer them all at once.
  struct FirstPrivateArgInfoTy {
    /// The index of the element in \p TgtArgs corresponding to the argument
    const int Index;
    /// Host pointer begin
    const char *HstPtrBegin;
    /// Host pointer end
    const char *HstPtrEnd;
    /// Aligned size
    const int64_t AlignedSize;
    /// Host pointer name
    const map_var_info_t HstPtrName = nullptr;

    FirstPrivateArgInfoTy(int Index, const void *HstPtr, int64_t Size,
                          const map_var_info_t HstPtrName = nullptr)
        : Index(Index), HstPtrBegin(reinterpret_cast<const char *>(HstPtr)),
          HstPtrEnd(HstPtrBegin + Size), AlignedSize(Size + Size % Alignment),
          HstPtrName(HstPtrName) {}
  };

  /// A vector of target pointers for all private arguments
  std::vector<void *> TgtPtrs;

  /// A vector of information of all first-private arguments to be packed
  std::vector<FirstPrivateArgInfoTy> FirstPrivateArgInfo;
  /// Host buffer for all arguments to be packed
  std::vector<char> FirstPrivateArgBuffer;
  /// The total size of all arguments to be packed
  int64_t FirstPrivateArgSize = 0;

  /// A reference to the \p DeviceTy object
  DeviceTy &Device;
  /// A pointer to a \p AsyncInfoTy object
  AsyncInfoTy &AsyncInfo;

  // TODO: What would be the best value here? Should we make it configurable?
  // If the size is larger than this threshold, we will allocate and transfer it
  // immediately instead of packing it.
  static constexpr const int64_t FirstPrivateArgSizeThreshold = 1024;

public:
  /// Constructor
  PrivateArgumentManagerTy(DeviceTy &Dev, AsyncInfoTy &AsyncInfo)
      : Device(Dev), AsyncInfo(AsyncInfo) {}

  /// Add a private argument
  int addArg(void *HstPtr, int64_t ArgSize, int64_t ArgOffset,
             bool IsFirstPrivate, void *&TgtPtr, int TgtArgsIndex,
             const map_var_info_t HstPtrName = nullptr,
             const bool AllocImmediately = false) {
    // If the argument is not first-private, or its size is greater than a
    // predefined threshold, we will allocate memory and issue the transfer
    // immediately.
    if (ArgSize > FirstPrivateArgSizeThreshold || !IsFirstPrivate ||
        AllocImmediately) {
      TgtPtr = Device.allocData(ArgSize, HstPtr);
      if (!TgtPtr) {
        DP("Data allocation for %sprivate array " DPxMOD " failed.\n",
           (IsFirstPrivate ? "first-" : ""), DPxPTR(HstPtr));
        return OFFLOAD_FAIL;
      }
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtr + ArgOffset);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD
         " for %sprivate array " DPxMOD " - pushing target argument " DPxMOD
         "\n",
         ArgSize, DPxPTR(TgtPtr), (IsFirstPrivate ? "first-" : ""),
         DPxPTR(HstPtr), DPxPTR(TgtPtrBase));
#endif
      // If first-private, copy data from host
      if (IsFirstPrivate) {
        DP("Submitting firstprivate data to the device.\n");
        int Ret = Device.submitData(TgtPtr, HstPtr, ArgSize, AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed, failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      TgtPtrs.push_back(TgtPtr);
    } else {
      DP("Firstprivate array " DPxMOD " of size %" PRId64 " will be packed\n",
         DPxPTR(HstPtr), ArgSize);
      // When reach this point, the argument must meet all following
      // requirements:
      // 1. Its size does not exceed the threshold (see the comment for
      // FirstPrivateArgSizeThreshold);
      // 2. It must be first-private (needs to be mapped to target device).
      // We will pack all this kind of arguments to transfer them all at once
      // to reduce the number of data transfer. We will not take
      // non-first-private arguments, aka. private arguments that doesn't need
      // to be mapped to target device, into account because data allocation
      // can be very efficient with memory manager.

      // Placeholder value
      TgtPtr = nullptr;
      FirstPrivateArgInfo.emplace_back(TgtArgsIndex, HstPtr, ArgSize,
                                       HstPtrName);
      FirstPrivateArgSize += FirstPrivateArgInfo.back().AlignedSize;
    }

    return OFFLOAD_SUCCESS;
  }

  /// Pack first-private arguments, replace place holder pointers in \p TgtArgs,
  /// and start the transfer.
  int packAndTransfer(std::vector<void *> &TgtArgs) {
    if (!FirstPrivateArgInfo.empty()) {
      assert(FirstPrivateArgSize != 0 &&
             "FirstPrivateArgSize is 0 but FirstPrivateArgInfo is empty");
      FirstPrivateArgBuffer.resize(FirstPrivateArgSize, 0);
      auto Itr = FirstPrivateArgBuffer.begin();
      // Copy all host data to this buffer
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        std::copy(Info.HstPtrBegin, Info.HstPtrEnd, Itr);
        Itr = std::next(Itr, Info.AlignedSize);
      }
      // Allocate target memory
      void *TgtPtr =
          Device.allocData(FirstPrivateArgSize, FirstPrivateArgBuffer.data());
      if (TgtPtr == nullptr) {
        DP("Failed to allocate target memory for private arguments.\n");
        return OFFLOAD_FAIL;
      }
      TgtPtrs.push_back(TgtPtr);
      DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD "\n",
         FirstPrivateArgSize, DPxPTR(TgtPtr));
      // Transfer data to target device
      int Ret = Device.submitData(TgtPtr, FirstPrivateArgBuffer.data(),
                                  FirstPrivateArgSize, AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        DP("Failed to submit data of private arguments.\n");
        return OFFLOAD_FAIL;
      }
      // Fill in all placeholder pointers
      auto TP = reinterpret_cast<uintptr_t>(TgtPtr);
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        void *&Ptr = TgtArgs[Info.Index];
        assert(Ptr == nullptr && "Target pointer is already set by mistaken");
        Ptr = reinterpret_cast<void *>(TP);
        TP += Info.AlignedSize;
        DP("Firstprivate array " DPxMOD " of size %" PRId64 " mapped to " DPxMOD
           "\n",
           DPxPTR(Info.HstPtrBegin), Info.HstPtrEnd - Info.HstPtrBegin,
           DPxPTR(Ptr));
      }
    }

    return OFFLOAD_SUCCESS;
  }

  /// Free all target memory allocated for private arguments
  int free() {
    for (void *P : TgtPtrs) {
      int Ret = Device.deleteData(P);
      if (Ret != OFFLOAD_SUCCESS) {
        DP("Deallocation of (first-)private arrays failed.\n");
        return OFFLOAD_FAIL;
      }
    }

    TgtPtrs.clear();

    return OFFLOAD_SUCCESS;
  }
};

/// Process data before launching the kernel, including calling targetDataBegin
/// to map and transfer data to target device, transferring (first-)private
/// variables.
static int processDataBefore(ident_t *loc, int64_t DeviceId, void *HostPtr,
                             int32_t ArgNum, void **ArgBases, void **Args,
                             int64_t *ArgSizes, int64_t *ArgTypes,
                             map_var_info_t *ArgNames, void **ArgMappers,
                             std::vector<void *> &TgtArgs,
                             std::vector<ptrdiff_t> &TgtOffsets,
                             PrivateArgumentManagerTy &PrivateArgumentManager,
                             AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_NAME_AND_IDENT("mappingBeforeTargetRegion", loc);
  DeviceTy &Device = PM->Devices[DeviceId];
  int Ret = targetDataBegin(loc, Device, ArgNum, ArgBases, Args, ArgSizes,
                            ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataBegin failed, abort target.\n");
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
        void *&PointerTgtPtrBegin = AsyncInfo.getVoidPtrLocation();
        PointerTgtPtrBegin = Device.getTgtPtrBegin(HstPtrVal, ArgSizes[I],
                                                   IsLast, false, IsHostPtr);
        if (!PointerTgtPtrBegin) {
          DP("No lambda captured variable mapped (" DPxMOD ") - ignored\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        if (PM->RTLs.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
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
          REPORT("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
      }
      continue;
    }
    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgBases[I];
    void *TgtPtrBegin;
    map_var_info_t HstPtrName = (!ArgNames) ? nullptr : ArgNames[I];
    ptrdiff_t TgtBaseOffset;
    bool IsLast, IsHostPtr; // unused.
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
         DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE) {
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
      const bool IsFirstPrivate = (ArgTypes[I] & OMP_TGT_MAPTYPE_TO);
      // If there is a next argument and it depends on the current one, we need
      // to allocate the private memory immediately. If this is not the case,
      // then the argument can be marked for optimization and packed with the
      // other privates.
      const bool AllocImmediately =
          (I < ArgNum - 1 && (ArgTypes[I + 1] & OMP_TGT_MAPTYPE_MEMBER_OF));
      Ret = PrivateArgumentManager.addArg(
          HstPtrBegin, ArgSizes[I], TgtBaseOffset, IsFirstPrivate, TgtPtrBegin,
          TgtArgs.size(), HstPtrName, AllocImmediately);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Failed to process %sprivate argument " DPxMOD "\n",
               (IsFirstPrivate ? "first-" : ""), DPxPTR(HstPtrBegin));
        return OFFLOAD_FAIL;
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

  // Pack and transfer first-private arguments
  Ret = PrivateArgumentManager.packAndTransfer(TgtArgs);
  if (Ret != OFFLOAD_SUCCESS) {
    DP("Failed to pack and transfer first private arguments\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

/// Process data after launching the kernel, including transferring data back to
/// host if needed and deallocating target memory of (first-)private variables.
static int processDataAfter(ident_t *loc, int64_t DeviceId, void *HostPtr,
                            int32_t ArgNum, void **ArgBases, void **Args,
                            int64_t *ArgSizes, int64_t *ArgTypes,
                            map_var_info_t *ArgNames, void **ArgMappers,
                            PrivateArgumentManagerTy &PrivateArgumentManager,
                            AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_NAME_AND_IDENT("mappingAfterTargetRegion", loc);
  DeviceTy &Device = PM->Devices[DeviceId];

  // Move data from device.
  int Ret = targetDataEnd(loc, Device, ArgNum, ArgBases, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataEnd failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // Free target memory for private arguments
  Ret = PrivateArgumentManager.free();
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Failed to deallocate target memory for private args\n");
    return OFFLOAD_FAIL;
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
int target(ident_t *loc, DeviceTy &Device, void *HostPtr, int32_t ArgNum,
           void **ArgBases, void **Args, int64_t *ArgSizes, int64_t *ArgTypes,
           map_var_info_t *ArgNames, void **ArgMappers, int32_t TeamNum,
           int32_t ThreadLimit, int IsTeamConstruct, AsyncInfoTy &AsyncInfo) {
  int32_t DeviceId = Device.DeviceID;

  TableMap *TM = getTableMap(HostPtr);
  // No map for this host pointer found!
  if (!TM) {
    REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
           DPxPTR(HostPtr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  std::vector<void *> TgtArgs;
  std::vector<ptrdiff_t> TgtOffsets;

  PrivateArgumentManagerTy PrivateArgumentManager(Device, AsyncInfo);

  int Ret;
  if (ArgNum) {
    // Process data, such as data mapping, before launching the kernel
    Ret = processDataBefore(loc, DeviceId, HostPtr, ArgNum, ArgBases, Args,
                            ArgSizes, ArgTypes, ArgNames, ArgMappers, TgtArgs,
                            TgtOffsets, PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data before launching the kernel.\n");
      return OFFLOAD_FAIL;
    }
  }

  // Get loop trip count
  uint64_t LoopTripCount = getLoopTripCount(DeviceId);

  // Launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);

  {
    TIMESCOPE_WITH_NAME_AND_IDENT(
        IsTeamConstruct ? "runTargetTeamRegion" : "runTargetRegion", loc);
    if (IsTeamConstruct)
      Ret = Device.runTeamRegion(TgtEntryPtr, &TgtArgs[0], &TgtOffsets[0],
                                 TgtArgs.size(), TeamNum, ThreadLimit,
                                 LoopTripCount, AsyncInfo);
    else
      Ret = Device.runRegion(TgtEntryPtr, &TgtArgs[0], &TgtOffsets[0],
                             TgtArgs.size(), AsyncInfo);
  }

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  if (ArgNum) {
    // Transfer data back and deallocate target memory for (first-)private
    // variables
    Ret = processDataAfter(loc, DeviceId, HostPtr, ArgNum, ArgBases, Args,
                           ArgSizes, ArgTypes, ArgNames, ArgMappers,
                           PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data after launching the kernel.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}
