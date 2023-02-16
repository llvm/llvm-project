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
#include <cstdint>
#include <vector>

using llvm::SmallVector;

int AsyncInfoTy::synchronize() {
  int Result = OFFLOAD_SUCCESS;
  if (!isQueueEmpty()) {
    switch (SyncType) {
    case SyncTy::BLOCKING:
      // If we have a queue we need to synchronize it now.
      Result = Device.synchronize(*this);
      assert(AsyncInfo.Queue == nullptr &&
             "The device plugin should have nulled the queue to indicate there "
             "are no outstanding actions!");
      break;
    case SyncTy::NON_BLOCKING:
      Result = Device.queryAsync(*this);
      break;
    }
  }

  // Run any pending post-processing function registered on this async object.
  if (Result == OFFLOAD_SUCCESS && isQueueEmpty())
    Result = runPostProcessing();

  return Result;
}

void *&AsyncInfoTy::getVoidPtrLocation() {
  BufferLocations.push_back(nullptr);
  return BufferLocations.back();
}

std::optional<bool> AsyncInfoTy::isDone() {
  if (int Result = synchronize())
    return std::nullopt;

  // The async info operations are completed when the internal queue is empty.
  return isQueueEmpty();
}

int32_t AsyncInfoTy::runPostProcessing() {
  size_t Size = PostProcessingFunctions.size();
  for (size_t I = 0; I < Size; ++I) {
    const int Result = PostProcessingFunctions[I]();
    if (Result != OFFLOAD_SUCCESS)
      return Result;
  }

  // Clear the vector up until the last known function, since post-processing
  // procedures might add new procedures themselves.
  const auto PrevBegin = PostProcessingFunctions.begin();
  PostProcessingFunctions.erase(PrevBegin, PrevBegin + Size);

  return OFFLOAD_SUCCESS;
}

bool AsyncInfoTy::isQueueEmpty() const { return AsyncInfo.Queue == nullptr; }

/* All begin addresses for partially mapped structs must be aligned, up to 16,
 * in order to ensure proper alignment of members. E.g.
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
static const int64_t MaxAlignment = 16;

/// Return the alignment requirement of partially mapped structs, see
/// MaxAlignment above.
static int64_t getPartialStructRequiredAlignment(void *HstPtrBase) {
  auto BaseAlignment = reinterpret_cast<uintptr_t>(HstPtrBase) % MaxAlignment;
  return BaseAlignment == 0 ? MaxAlignment : BaseAlignment;
}

/// Map global data and execute pending ctors
static int initLibrary(DeviceTy &Device) {
  /*
   * Map global data
   */
  int32_t DeviceId = Device.DeviceID;
  int Rc = OFFLOAD_SUCCESS;
  bool SupportsEmptyImages = Device.RTL->supports_empty_images &&
                             Device.RTL->supports_empty_images() > 0;
  {
    std::lock_guard<decltype(PM->TrlTblMtx)> LG(PM->TrlTblMtx);
    for (auto *HostEntriesBegin : PM->HostEntriesBeginRegistrationOrder) {
      TranslationTable *TransTable =
          &PM->HostEntriesBeginToTransTable[HostEntriesBegin];
      if (TransTable->HostTable.EntriesBegin ==
              TransTable->HostTable.EntriesEnd &&
          !SupportsEmptyImages) {
        // No host entry so no need to proceed
        continue;
      }

      if (TransTable->TargetsTable[DeviceId] != 0) {
        // Library entries have already been processed
        continue;
      }

      // 1) get image.
      assert(TransTable->TargetsImages.size() > (size_t)DeviceId &&
             "Not expecting a device ID outside the table's bounds!");
      __tgt_device_image *Img = TransTable->TargetsImages[DeviceId];
      if (!Img) {
        REPORT("No image loaded for device id %d.\n", DeviceId);
        Rc = OFFLOAD_FAIL;
        break;
      }
      // 2) load image into the target table.
      __tgt_target_table *TargetTable = TransTable->TargetsTable[DeviceId] =
          Device.loadBinary(Img);
      // Unable to get table for this image: invalidate image and fail.
      if (!TargetTable) {
        REPORT("Unable to generate entries table for device id %d.\n",
               DeviceId);
        TransTable->TargetsImages[DeviceId] = 0;
        Rc = OFFLOAD_FAIL;
        break;
      }

      // Verify whether the two table sizes match.
      size_t Hsize =
          TransTable->HostTable.EntriesEnd - TransTable->HostTable.EntriesBegin;
      size_t Tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

      // Invalid image for these host entries!
      if (Hsize != Tsize) {
        REPORT(
            "Host and Target tables mismatch for device id %d [%zx != %zx].\n",
            DeviceId, Hsize, Tsize);
        TransTable->TargetsImages[DeviceId] = 0;
        TransTable->TargetsTable[DeviceId] = 0;
        Rc = OFFLOAD_FAIL;
        break;
      }

      DeviceTy::HDTTMapAccessorTy HDTTMap =
          Device.HostDataToTargetMap.getExclusiveAccessor();

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
          // the fat binary. Treat these mappings as any other "regular"
          // mapping. Add entry to map.
          if (Device.getTgtPtrBegin(HDTTMap, CurrHostEntry->addr,
                                    CurrHostEntry->size))
            continue;

          DP("Add mapping from host " DPxMOD " to device " DPxMOD
             " with size %zu"
             "\n",
             DPxPTR(CurrHostEntry->addr), DPxPTR(CurrDeviceEntry->addr),
             CurrDeviceEntry->size);
          HDTTMap->emplace(new HostDataToTargetTy(
              (uintptr_t)CurrHostEntry->addr /*HstPtrBase*/,
              (uintptr_t)CurrHostEntry->addr /*HstPtrBegin*/,
              (uintptr_t)CurrHostEntry->addr +
                  CurrHostEntry->size /*HstPtrEnd*/,
              (uintptr_t)CurrDeviceEntry->addr /*TgtPtrBegin*/,
              false /*UseHoldRefCount*/, CurrHostEntry->name,
              true /*IsRefCountINF*/));

          // Notify about the new mapping.
          if (Device.notifyDataMapped(CurrHostEntry->addr, CurrHostEntry->size))
            return OFFLOAD_FAIL;
        }
      }
    }
  }

  if (Rc != OFFLOAD_SUCCESS) {
    return Rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    AsyncInfoTy AsyncInfo(Device);
    // Call all ctors for all libraries registered so far
    for (auto &Lib : Device.PendingCtorsDtors) {
      if (!Lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &Entry : Lib.second.PendingCtors) {
          void *Ctor = Entry;
          int Rc = target(nullptr, Device, Ctor, CTorDTorKernelArgs, AsyncInfo);
          if (Rc != OFFLOAD_SUCCESS) {
            REPORT("Running ctor " DPxMOD " failed.\n", DPxPTR(Ctor));
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        Lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib " DPxMOD "\n", DPxPTR(Lib.first));
      }
    }
    // All constructors have been issued, wait for them now.
    if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;
  }
  Device.HasPendingGlobals = false;

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
          dumpTargetPointerMappings(Loc, *Device);
      else
        FAILURE_MESSAGE("Consult https://openmp.llvm.org/design/Runtimes.html "
                        "for debugging options.\n");

      SourceInfo Info(Loc);
      if (Info.isAvailible())
        fprintf(stderr, "%s:%d:%d: ", Info.getFilename(), Info.getLine(),
                Info.getColumn());
      else
        FAILURE_MESSAGE("Source location information not present. Compile with "
                        "-g or -gline-tables-only.\n");
      FATAL_MESSAGE0(
          1, "failure of target construct while offloading is mandatory");
    } else {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE)
        for (auto &Device : PM->Devices)
          dumpTargetPointerMappings(Loc, *Device);
    }
    break;
  }
}

static void handleDefaultTargetOffload() {
  std::lock_guard<decltype(PM->TargetOffloadMtx)> LG(PM->TargetOffloadMtx);
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
}

static bool isOffloadDisabled() {
  if (PM->TargetOffloadPolicy == tgt_default)
    handleDefaultTargetOffload();
  return PM->TargetOffloadPolicy == tgt_disabled;
}

// If offload is enabled, ensure that device DeviceID has been initialized,
// global ctors have been executed, and global data has been mapped.
//
// The return bool indicates if the offload is to the host device
// There are three possible results:
// - Return false if the taregt device is ready for offload
// - Return true without reporting a runtime error if offload is
//   disabled, perhaps because the initial device was specified.
// - Report a runtime error and return true.
//
// If DeviceID == OFFLOAD_DEVICE_DEFAULT, set DeviceID to the default device.
// This step might be skipped if offload is disabled.
bool checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc) {
  if (isOffloadDisabled()) {
    DP("Offload is disabled\n");
    return true;
  }

  if (DeviceID == OFFLOAD_DEVICE_DEFAULT) {
    DeviceID = omp_get_default_device();
    DP("Use default device id %" PRId64 "\n", DeviceID);
  }

  // Proposed behavior for OpenMP 5.2 in OpenMP spec github issue 2669.
  if (omp_get_num_devices() == 0) {
    DP("omp_get_num_devices() == 0 but offload is manadatory\n");
    handleTargetOutcome(false, Loc);
    return true;
  }

  if (DeviceID == omp_get_initial_device()) {
    DP("Device is host (%" PRId64 "), returning as if offload is disabled\n",
       DeviceID);
    return true;
  }

  // Is device ready?
  if (!deviceIsReady(DeviceID)) {
    REPORT("Device %" PRId64 " is not ready.\n", DeviceID);
    handleTargetOutcome(false, Loc);
    return true;
  }

  // Get device info.
  DeviceTy &Device = *PM->Devices[DeviceID];

  // Check whether global data has been mapped for this device
  {
    std::lock_guard<decltype(Device.PendingGlobalsMtx)> LG(
        Device.PendingGlobalsMtx);
    if (Device.HasPendingGlobals && initLibrary(Device) != OFFLOAD_SUCCESS) {
      REPORT("Failed to init globals on device %" PRId64 "\n", DeviceID);
      handleTargetOutcome(false, Loc);
      return true;
    }
  }

  return false;
}

static int32_t getParentIndex(int64_t Type) {
  return ((Type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

void *targetAllocExplicit(size_t Size, int DeviceNum, int Kind,
                          const char *Name) {
  TIMESCOPE();
  DP("Call to %s for device %d requesting %zu bytes\n", Name, DeviceNum, Size);

  if (Size <= 0) {
    DP("Call to %s with non-positive length\n", Name);
    return NULL;
  }

  void *Rc = NULL;

  if (DeviceNum == omp_get_initial_device()) {
    Rc = malloc(Size);
    DP("%s returns host ptr " DPxMOD "\n", Name, DPxPTR(Rc));
    return Rc;
  }

  if (!deviceIsReady(DeviceNum)) {
    DP("%s returns NULL ptr\n", Name);
    return NULL;
  }

  DeviceTy &Device = *PM->Devices[DeviceNum];
  Rc = Device.allocData(Size, nullptr, Kind);
  DP("%s returns device ptr " DPxMOD "\n", Name, DPxPTR(Rc));
  return Rc;
}

void targetFreeExplicit(void *DevicePtr, int DeviceNum, int Kind,
                        const char *Name) {
  TIMESCOPE();
  DP("Call to %s for device %d and address " DPxMOD "\n", Name, DeviceNum,
     DPxPTR(DevicePtr));

  if (!DevicePtr) {
    DP("Call to %s with NULL ptr\n", Name);
    return;
  }

  if (DeviceNum == omp_get_initial_device()) {
    free(DevicePtr);
    DP("%s deallocated host ptr\n", Name);
    return;
  }

  if (!deviceIsReady(DeviceNum)) {
    DP("%s returns, nothing to do\n", Name);
    return;
  }

  PM->Devices[DeviceNum]->deleteData(DevicePtr, Kind);
  DP("omp_target_free deallocated device ptr\n");
}

void *targetLockExplicit(void *HostPtr, size_t Size, int DeviceNum,
                         const char *Name) {
  TIMESCOPE();
  DP("Call to %s for device %d locking %zu bytes\n", Name, DeviceNum, Size);

  if (Size <= 0) {
    DP("Call to %s with non-positive length\n", Name);
    return NULL;
  }

  void *rc = NULL;

  if (!deviceIsReady(DeviceNum)) {
    DP("%s returns NULL ptr\n", Name);
    return NULL;
  }

  DeviceTy *DevicePtr = nullptr;
  {
    std::lock_guard<decltype(PM->RTLsMtx)> LG(PM->RTLsMtx);

    if (!PM->Devices[DeviceNum]) {
      DP("%s returns, device %d not available\n", Name, DeviceNum);
      return nullptr;
    }

    DevicePtr = PM->Devices[DeviceNum].get();
  }

  int32_t err = 0;
  if (DevicePtr->RTL->data_lock) {
    err = DevicePtr->RTL->data_lock(DeviceNum, HostPtr, Size, &rc);
    if (err) {
      DP("Could not lock ptr %p\n", HostPtr);
      return nullptr;
    }
  }
  DP("%s returns device ptr " DPxMOD "\n", Name, DPxPTR(rc));
  return rc;
}

void targetUnlockExplicit(void *HostPtr, int DeviceNum, const char *Name) {
  TIMESCOPE();
  DP("Call to %s for device %d unlocking\n", Name, DeviceNum);

  DeviceTy *DevicePtr = nullptr;
  {
    std::lock_guard<decltype(PM->RTLsMtx)> LG(PM->RTLsMtx);

    // Don't check deviceIsReady as it can initialize the device if needed.
    // Just check if DeviceNum exists as targetUnlockExplicit can be called
    // during process exit/free (and it may have been already destroyed) and
    // targetAllocExplicit will have already checked deviceIsReady anyway.
    size_t DevicesSize = PM->Devices.size();

    if (DevicesSize <= (size_t)DeviceNum) {
      DP("Device ID  %d does not have a matching RTL\n", DeviceNum);
      return;
    }

    if (!PM->Devices[DeviceNum]) {
      DP("%s returns, device %d not available\n", Name, DeviceNum);
      return;
    }

    DevicePtr = PM->Devices[DeviceNum].get();
  } // unlock RTLsMtx

  if (DevicePtr->RTL->data_unlock)
    DevicePtr->RTL->data_unlock(DeviceNum, HostPtr);

  DP("%s returns\n", Name);
}

/// Call the user-defined mapper function followed by the appropriate
// targetData* function (targetData{Begin,End,Update}).
int targetDataMapper(ident_t *Loc, DeviceTy &Device, void *ArgBase, void *Arg,
                     int64_t ArgSize, int64_t ArgType, map_var_info_t ArgNames,
                     void *ArgMapper, AsyncInfoTy &AsyncInfo,
                     TargetDataFuncPtrTy TargetDataFunction) {
  TIMESCOPE_WITH_IDENT(Loc);
  DP("Calling the mapper function " DPxMOD "\n", DPxPTR(ArgMapper));

  // The mapper function fills up Components.
  MapperComponentsTy MapperComponents;
  MapperFuncPtrTy MapperFuncPtr = (MapperFuncPtrTy)(ArgMapper);
  (*MapperFuncPtr)((void *)&MapperComponents, ArgBase, Arg, ArgSize, ArgType,
                   ArgNames);

  // Construct new arrays for args_base, args, arg_sizes and arg_types
  // using the information in MapperComponents and call the corresponding
  // targetData* function using these new arrays.
  SmallVector<void *> MapperArgsBase(MapperComponents.Components.size());
  SmallVector<void *> MapperArgs(MapperComponents.Components.size());
  SmallVector<int64_t> MapperArgSizes(MapperComponents.Components.size());
  SmallVector<int64_t> MapperArgTypes(MapperComponents.Components.size());
  SmallVector<void *> MapperArgNames(MapperComponents.Components.size());

  for (unsigned I = 0, E = MapperComponents.Components.size(); I < E; ++I) {
    auto &C = MapperComponents.Components[I];
    MapperArgsBase[I] = C.Base;
    MapperArgs[I] = C.Begin;
    MapperArgSizes[I] = C.Size;
    MapperArgTypes[I] = C.Type;
    MapperArgNames[I] = C.Name;
  }

  int Rc = TargetDataFunction(Loc, Device, MapperComponents.Components.size(),
                              MapperArgsBase.data(), MapperArgs.data(),
                              MapperArgSizes.data(), MapperArgTypes.data(),
                              MapperArgNames.data(), /*arg_mappers*/ nullptr,
                              AsyncInfo, /*FromMapper=*/true);

  return Rc;
}

/// Internal function to do the mapping and transfer the data to the device
int targetDataBegin(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                    void **ArgsBase, void **Args, int64_t *ArgSizes,
                    int64_t *ArgTypes, map_var_info_t *ArgNames,
                    void **ArgMappers, AsyncInfoTy &AsyncInfo,
                    bool FromMapper) {
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataBegin, call the
      // targetDataMapper variant which will call targetDataBegin again
      // with new arguments.
      DP("Calling targetDataMapper for the %dth argument\n", I);

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Rc = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                targetDataBegin);

      if (Rc != OFFLOAD_SUCCESS) {
        REPORT("Call to targetDataBegin via targetDataMapper for custom mapper"
               " failed.\n");
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgsBase[I];
    int64_t DataSize = ArgSizes[I];
    map_var_info_t HstPtrName = (!ArgNames) ? nullptr : ArgNames[I];

    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t Padding = 0;
    const int NextI = I + 1;
    if (getParentIndex(ArgTypes[I]) < 0 && NextI < ArgNum &&
        getParentIndex(ArgTypes[NextI]) == I) {
      int64_t Alignment = getPartialStructRequiredAlignment(HstPtrBase);
      Padding = (int64_t)HstPtrBegin % Alignment;
      if (Padding) {
        DP("Using a padding of %" PRId64 " bytes for begin address " DPxMOD
           "\n",
           Padding, DPxPTR(HstPtrBegin));
        HstPtrBegin = (char *)HstPtrBegin - Padding;
        DataSize += Padding;
      }
    }

    // Address of pointer on the host and device, respectively.
    void *PointerHstPtrBegin, *PointerTgtPtrBegin;
    TargetPointerResultTy PointerTpr;
    bool IsHostPtr = false;
    bool IsImplicit = ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT;
    // Force the creation of a device side copy of the data when:
    // a close map modifier was associated with a map that contained a to.
    bool HasCloseModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_CLOSE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;
    bool HasHoldModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_OMPX_HOLD;
    // UpdateRef is based on MEMBER_OF instead of TARGET_PARAM because if we
    // have reached this point via __tgt_target_data_begin and not __tgt_target
    // then no argument is marked as TARGET_PARAM ("omp target data map" is not
    // associated with a target region, so there are no target parameters). This
    // may be considered a hack, we could revise the scheme in the future.
    bool UpdateRef =
        !(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) && !(FromMapper && I == 0);
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
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
      PointerTpr = Device.getTargetPointer(
          HstPtrBase, HstPtrBase, sizeof(void *), /*HstPtrName=*/nullptr,
          /*HasFlagTo=*/false, /*HasFlagAlways=*/false, IsImplicit, UpdateRef,
          HasCloseModifier, HasPresentModifier, HasHoldModifier, AsyncInfo);
      PointerTgtPtrBegin = PointerTpr.TargetPointer;
      IsHostPtr = PointerTpr.Flags.IsHostPointer;
      if (!PointerTgtPtrBegin) {
        REPORT("Call to getTargetPointer returned null pointer (%s).\n",
               HasPresentModifier ? "'present' map type modifier"
                                  : "device failure or illegal mapping");
        return OFFLOAD_FAIL;
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
         "\n",
         sizeof(void *), DPxPTR(PointerTgtPtrBegin),
         (PointerTpr.Flags.IsNewEntry ? "" : " not"));
      PointerHstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      // No need to update pointee ref count for the first element of the
      // subelement that comes from mapper.
      UpdateRef =
          (!FromMapper || I != 0); // subsequently update ref count of pointee
    }

    const bool HasFlagTo = ArgTypes[I] & OMP_TGT_MAPTYPE_TO;
    const bool HasFlagAlways = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
    auto TPR = Device.getTargetPointer(
        HstPtrBegin, HstPtrBase, DataSize, HstPtrName, HasFlagTo, HasFlagAlways,
        IsImplicit, UpdateRef, HasCloseModifier, HasPresentModifier,
        HasHoldModifier, AsyncInfo);
    void *TgtPtrBegin = TPR.TargetPointer;
    IsHostPtr = TPR.Flags.IsHostPointer;
    // If data_size==0, then the argument could be a zero-length pointer to
    // NULL, so getOrAlloc() returning NULL is not an error.
    if (!TgtPtrBegin && (DataSize || HasPresentModifier)) {
      REPORT("Call to getTargetPointer returned null pointer (%s).\n",
             HasPresentModifier ? "'present' map type modifier"
                                : "device failure or illegal mapping");
      return OFFLOAD_FAIL;
    }
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
       " - is%s new\n",
       DataSize, DPxPTR(TgtPtrBegin), (TPR.Flags.IsNewEntry ? "" : " not"));

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      uintptr_t Delta = (uintptr_t)HstPtrBegin - (uintptr_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uintptr_t)TgtPtrBegin - Delta);
      DP("Returning device pointer " DPxMOD "\n", DPxPTR(TgtPtrBase));
      ArgsBase[I] = TgtPtrBase;
    }

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ && !IsHostPtr) {
      // Check whether we need to update the pointer on the device
      bool UpdateDevPtr = false;

      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *ExpectedTgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);

      Device.ShadowMtx.lock();
      auto Entry = Device.ShadowPtrMap.find(PointerHstPtrBegin);
      // If this pointer is not in the map we need to insert it. If the map
      // contains a stale entry, we need to update it (e.g. if the pointee was
      // deallocated and later on is reallocated at another device address). The
      // latter scenario is the subject of LIT test env/base_ptr_ref_count.c. An
      // entry is removed from ShadowPtrMap only when the PTR of a PTR_AND_OBJ
      // pair is deallocated, not when the OBJ is deallocated. In
      // env/base_ptr_ref_count.c the PTR is a global "declare target" pointer,
      // so it stays in the map for the lifetime of the application. When the
      // OBJ is deallocated and later on allocated again (at a different device
      // address), ShadowPtrMap still contains an entry for Pointer_HstPtrBegin
      // which is stale, pointing to the old ExpectedTgtPtrBase of the OBJ.
      if (Entry == Device.ShadowPtrMap.end() ||
          Entry->second.TgtPtrVal != ExpectedTgtPtrBase) {
        // create or update shadow pointers for this entry
        Device.ShadowPtrMap[PointerHstPtrBegin] = {
            HstPtrBase, PointerTgtPtrBegin, ExpectedTgtPtrBase};
        PointerTpr.Entry->setMayContainAttachedPointers();
        UpdateDevPtr = true;
      }

      if (UpdateDevPtr) {
        std::lock_guard<decltype(*PointerTpr.Entry)> LG(*PointerTpr.Entry);
        Device.ShadowMtx.unlock();

        DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
           DPxPTR(PointerTgtPtrBegin), DPxPTR(TgtPtrBegin));

        void *&TgtPtrBase = AsyncInfo.getVoidPtrLocation();
        TgtPtrBase = ExpectedTgtPtrBase;

        int Ret = Device.submitData(PointerTgtPtrBegin, &TgtPtrBase,
                                    sizeof(void *), AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          REPORT("Copying data to device failed.\n");
          return OFFLOAD_FAIL;
        }
        if (PointerTpr.Entry->addEventIfNecessary(Device, AsyncInfo) !=
            OFFLOAD_SUCCESS)
          return OFFLOAD_FAIL;
      } else
        Device.ShadowMtx.unlock();
    }
  }

  return OFFLOAD_SUCCESS;
}

namespace {
/// This structure contains information to deallocate a target pointer, aka.
/// used to fix up the shadow map and potentially delete the entry from the
/// mapping table via \p DeviceTy::deallocTgtPtr.
struct PostProcessingInfo {
  /// Host pointer used to look up into the map table
  void *HstPtrBegin;

  /// Size of the data
  int64_t DataSize;

  /// The mapping type (bitfield).
  int64_t ArgType;

  /// Index of the argument in the data mapping scheme.
  int32_t ArgIndex;

  /// The target pointer information.
  TargetPointerResultTy TPR;

  PostProcessingInfo(void *HstPtr, int64_t Size, int64_t ArgType,
                     int32_t ArgIndex, TargetPointerResultTy TPR)
      : HstPtrBegin(HstPtr), DataSize(Size), ArgType(ArgType),
        ArgIndex(ArgIndex), TPR(TPR) {}
};

/// Apply \p CB to the shadow map pointer entries in the range \p Begin, to
/// \p Begin + \p Size. \p CB is called with a locked shadow pointer map and the
/// passed iterator can be updated. If the callback returns OFFLOAD_FAIL the
/// rest of the map is not checked anymore.
template <typename CBTy>
static void applyToShadowMapEntries(DeviceTy &Device, CBTy CB, void *Begin,
                                    uintptr_t Size,
                                    const TargetPointerResultTy &TPR) {
  // If we have an object that is too small to hold a pointer subobject, no need
  // to do any checking.
  if (Size < sizeof(void *))
    return;

  // If the map entry for the object was never marked as containing attached
  // pointers, no need to do any checking.
  if (!TPR.Entry || !TPR.Entry->getMayContainAttachedPointers())
    return;

  uintptr_t LB = (uintptr_t)Begin;
  uintptr_t UB = LB + Size;
  // Now we are looking into the shadow map so we need to lock it.
  std::lock_guard<decltype(Device.ShadowMtx)> LG(Device.ShadowMtx);
  for (ShadowPtrListTy::iterator Itr = Device.ShadowPtrMap.begin();
       Itr != Device.ShadowPtrMap.end();) {
    uintptr_t ShadowHstPtrAddr = (uintptr_t)Itr->first;

    // An STL map is sorted on its keys; use this property
    // to quickly determine when to break out of the loop.
    if (ShadowHstPtrAddr < LB) {
      ++Itr;
      continue;
    }
    if (ShadowHstPtrAddr >= UB)
      break;

    if (CB(Itr) == OFFLOAD_FAIL)
      break;
  }
}

} // namespace

/// Applies the necessary post-processing procedures to entries listed in \p
/// EntriesInfo after the execution of all device side operations from a target
/// data end. This includes the update of pointers at the host and removal of
/// device buffer when needed. It returns OFFLOAD_FAIL or OFFLOAD_SUCCESS
/// according to the successfulness of the operations.
[[nodiscard]] static int
postProcessingTargetDataEnd(DeviceTy *Device,
                            SmallVector<PostProcessingInfo> EntriesInfo,
                            bool FromMapper) {
  int Ret = OFFLOAD_SUCCESS;
  void *FromMapperBase = nullptr;

  for (auto &[HstPtrBegin, DataSize, ArgType, ArgIndex, TPR] : EntriesInfo) {
    bool DelEntry = !TPR.isHostPointer();

    // If the last element from the mapper (for end transfer args comes in
    // reverse order), do not remove the partial entry, the parent struct still
    // exists.
    if ((ArgType & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(ArgType & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if (DelEntry && FromMapper && ArgIndex == 0) {
      DelEntry = false;
      FromMapperBase = HstPtrBegin;
    }

    // If we marked the entry to be deleted we need to verify no other
    // thread reused it by now. If deletion is still supposed to happen by
    // this thread LR will be set and exclusive access to the HDTT map
    // will avoid another thread reusing the entry now. Note that we do
    // not request (exclusive) access to the HDTT map if DelEntry is
    // not set.
    DeviceTy::HDTTMapAccessorTy HDTTMap =
        Device->HostDataToTargetMap.getExclusiveAccessor(!DelEntry);

    const bool IsNotLastUser = TPR.Entry->decDataEndThreadCount() != 0;
    if (DelEntry && (TPR.Entry->getTotalRefCount() != 0 || IsNotLastUser)) {
      // The thread is not in charge of deletion anymore. Give up access
      // to the HDTT map and unset the deletion flag.
      HDTTMap.destroy();
      DelEntry = false;
    }

    // If we copied back to the host a struct/array containing pointers,
    // we need to restore the original host pointer values from their
    // shadow copies. If the struct is going to be deallocated, remove any
    // remaining shadow pointer entries for this struct.
    const bool HasFrom = ArgType & OMP_TGT_MAPTYPE_FROM;
    auto CB = [&](ShadowPtrListTy::iterator &Itr) {
      // If we copied the struct to the host, we need to restore the
      // pointer.
      if (HasFrom) {
        void **ShadowHstPtrAddr = (void **)Itr->first;
        *ShadowHstPtrAddr = Itr->second.HstPtrVal;
        DP("Restoring original host pointer value " DPxMOD " for host "
           "pointer " DPxMOD "\n",
           DPxPTR(Itr->second.HstPtrVal), DPxPTR(ShadowHstPtrAddr));
      }
      // If the struct is to be deallocated, remove the shadow entry.
      if (DelEntry) {
        DP("Removing shadow pointer " DPxMOD "\n", DPxPTR((void **)Itr->first));
        auto OldItr = Itr;
        Itr++;
        Device->ShadowPtrMap.erase(OldItr);
      } else {
        ++Itr;
      }
      return OFFLOAD_SUCCESS;
    };
    applyToShadowMapEntries(*Device, CB, HstPtrBegin, DataSize, TPR);

    if (!DelEntry || (FromMapperBase && FromMapperBase == HstPtrBegin))
      continue;

    // If we are deleting the entry the DataMapMtx is locked and we own
    // the entry.
    Ret = Device->eraseMapEntry(HDTTMap, TPR.Entry, DataSize);
    // Entry is already remove from the map, we can unlock it now.
    HDTTMap.destroy();
    Ret |= Device->deallocTgtPtrAndEntry(TPR.Entry, DataSize);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Deallocating data from device failed.\n");
      break;
    }
  }

  return Ret;
}

/// Internal function to undo the mapping and retrieve the data from the device.
int targetDataEnd(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *ArgNames,
                  void **ArgMappers, AsyncInfoTy &AsyncInfo, bool FromMapper) {
  int Ret = OFFLOAD_SUCCESS;
  SmallVector<PostProcessingInfo> PostProcessingPtrs;
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
      Ret = targetDataMapper(Loc, Device, ArgBases[I], Args[I], ArgSizes[I],
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
    void *HstPtrBase = ArgBases[I];
    int64_t DataSize = ArgSizes[I];
    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    const int NextI = I + 1;
    if (getParentIndex(ArgTypes[I]) < 0 && NextI < ArgNum &&
        getParentIndex(ArgTypes[NextI]) == I) {
      int64_t Alignment = getPartialStructRequiredAlignment(HstPtrBase);
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
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;
    bool HasHoldModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_OMPX_HOLD;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    TargetPointerResultTy TPR = Device.getTgtPtrBegin(
        HstPtrBegin, DataSize, IsLast, UpdateRef, HasHoldModifier, IsHostPtr,
        !IsImplicit, ForceDelete, /*FromDataEnd=*/true);
    void *TgtPtrBegin = TPR.TargetPointer;
    if (!TPR.isPresent() && !TPR.isHostPointer() &&
        (DataSize || HasPresentModifier)) {
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
    if (!TPR.isPresent())
      continue;

    // Move data back to the host
    const bool HasAlways = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
    const bool HasFrom = ArgTypes[I] & OMP_TGT_MAPTYPE_FROM;
    if (HasFrom && (HasAlways || IsLast) && !IsHostPtr) {
      DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
         DataSize, DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));

      std::lock_guard<decltype(*TPR.Entry)> LG(*TPR.Entry);
      // Wait for any previous transfer if an event is present.
      if (void *Event = TPR.Entry->getEvent()) {
        if (Device.waitEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS) {
          REPORT("Failed to wait for event " DPxMOD ".\n", DPxPTR(Event));
          return OFFLOAD_FAIL;
        }
      }

      Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize, AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT("Copying data from device failed.\n");
        return OFFLOAD_FAIL;
      }

      // As we are expecting to delete the entry the d2h copy might race
      // with another one that also tries to delete the entry. This happens
      // as the entry can be reused and the reuse might happen after the
      // copy-back was issued but before it completed. Since the reuse might
      // also copy-back a value we would race.
      if (IsLast) {
        if (TPR.Entry->addEventIfNecessary(Device, AsyncInfo) !=
            OFFLOAD_SUCCESS)
          return OFFLOAD_FAIL;
      }
    }

    // Add pointer to the buffer for post-synchronize processing.
    PostProcessingPtrs.emplace_back(HstPtrBegin, DataSize, ArgTypes[I], I, TPR);
  }

  // Add post-processing functions
  // TODO: We might want to remove `mutable` in the future by not changing the
  // captured variables somehow.
  AsyncInfo.addPostProcessingFunction(
      [=, Device = &Device,
       PostProcessingPtrs = std::move(PostProcessingPtrs)]() mutable -> int {
        return postProcessingTargetDataEnd(Device, PostProcessingPtrs,
                                           FromMapperBase);
      });

  return Ret;
}

static int targetDataContiguous(ident_t *Loc, DeviceTy &Device, void *ArgsBase,
                                void *HstPtrBegin, int64_t ArgSize,
                                int64_t ArgType, AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_IDENT(Loc);
  bool IsLast, IsHostPtr;
  TargetPointerResultTy TPR = Device.getTgtPtrBegin(
      HstPtrBegin, ArgSize, IsLast, /*UpdateRefCount=*/false,
      /*UseHoldRefCount=*/false, IsHostPtr, /*MustContain=*/true);
  void *TgtPtrBegin = TPR.TargetPointer;
  if (!TPR.isPresent()) {
    DP("hst data:" DPxMOD " not found, becomes a noop\n", DPxPTR(HstPtrBegin));
    if (ArgType & OMP_TGT_MAPTYPE_PRESENT) {
      MESSAGE("device mapping required by 'present' motion modifier does not "
              "exist for host address " DPxMOD " (%" PRId64 " bytes)",
              DPxPTR(HstPtrBegin), ArgSize);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  if (IsHostPtr) {
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

    // Wait for device-to-host memcopies for whole struct to complete,
    // before restoring the correct host pointer.
    AsyncInfo.addPostProcessingFunction([=, Device = &Device]() -> int {
      auto CB = [&](ShadowPtrListTy::iterator &Itr) {
        void **ShadowHstPtrAddr = (void **)Itr->first;
        *ShadowHstPtrAddr = Itr->second.HstPtrVal;
        DP("Restoring original host pointer value " DPxMOD
           " for host pointer " DPxMOD "\n",
           DPxPTR(Itr->second.HstPtrVal), DPxPTR(ShadowHstPtrAddr));
        ++Itr;
        return OFFLOAD_SUCCESS;
      };
      applyToShadowMapEntries(*Device, CB, HstPtrBegin, ArgSize, TPR);

      return OFFLOAD_SUCCESS;
    });
  }

  if (ArgType & OMP_TGT_MAPTYPE_TO) {
    DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
       ArgSize, DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
    int Ret = Device.submitData(TgtPtrBegin, HstPtrBegin, ArgSize, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Copying data to device failed.\n");
      return OFFLOAD_FAIL;
    }

    auto CB = [&](ShadowPtrListTy::iterator &Itr) {
      DP("Restoring original target pointer value " DPxMOD " for target "
         "pointer " DPxMOD "\n",
         DPxPTR(Itr->second.TgtPtrVal), DPxPTR(Itr->second.TgtPtrAddr));
      Ret = Device.submitData(Itr->second.TgtPtrAddr, &Itr->second.TgtPtrVal,
                              sizeof(void *), AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS)
        REPORT("Copying data to device failed.\n");
      ++Itr;
      return Ret;
    };
    applyToShadowMapEntries(Device, CB, HstPtrBegin, ArgSize, TPR);
  }
  return OFFLOAD_SUCCESS;
}

static int targetDataNonContiguous(ident_t *Loc, DeviceTy &Device,
                                   void *ArgsBase,
                                   __tgt_target_non_contig *NonContig,
                                   uint64_t Size, int64_t ArgType,
                                   int CurrentDim, int DimSize, uint64_t Offset,
                                   AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_IDENT(Loc);
  int Ret = OFFLOAD_SUCCESS;
  if (CurrentDim < DimSize) {
    for (unsigned int I = 0; I < NonContig[CurrentDim].Count; ++I) {
      uint64_t CurOffset =
          (NonContig[CurrentDim].Offset + I) * NonContig[CurrentDim].Stride;
      // we only need to transfer the first element for the last dimension
      // since we've already got a contiguous piece.
      if (CurrentDim != DimSize - 1 || I == 0) {
        Ret = targetDataNonContiguous(Loc, Device, ArgsBase, NonContig, Size,
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
    Ret = targetDataContiguous(Loc, Device, ArgsBase, Ptr, Size, ArgType,
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
int targetDataUpdate(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
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
      int Ret = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
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
          Loc, Device, ArgsBase[I], NonContig, Size, ArgTypes[I],
          /*current_dim=*/0, DimSize - MergedDim, /*offset=*/0, AsyncInfo);
    } else {
      Ret = targetDataContiguous(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
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

/// A class manages private arguments in a target region.
class PrivateArgumentManagerTy {
  /// A data structure for the information of first-private arguments. We can
  /// use this information to optimize data transfer by packing all
  /// first-private arguments and transfer them all at once.
  struct FirstPrivateArgInfoTy {
    /// The index of the element in \p TgtArgs corresponding to the argument
    int Index;
    /// Host pointer begin
    char *HstPtrBegin;
    /// Host pointer end
    char *HstPtrEnd;
    /// Aligned size
    int64_t AlignedSize;
    /// Host pointer name
    map_var_info_t HstPtrName = nullptr;

    FirstPrivateArgInfoTy(int Index, void *HstPtr, int64_t Size,
                          const map_var_info_t HstPtrName = nullptr)
        : Index(Index), HstPtrBegin(reinterpret_cast<char *>(HstPtr)),
          HstPtrEnd(HstPtrBegin + Size),
          AlignedSize(Size + Size % MaxAlignment), HstPtrName(HstPtrName) {}
  };

  /// A vector of target pointers for all private arguments
  SmallVector<void *> TgtPtrs;

  /// A vector of information of all first-private arguments to be packed
  SmallVector<FirstPrivateArgInfoTy> FirstPrivateArgInfo;
  /// Host buffer for all arguments to be packed
  SmallVector<char> FirstPrivateArgBuffer;
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
  int packAndTransfer(SmallVector<void *> &TgtArgs) {
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
static int processDataBefore(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                             int32_t ArgNum, void **ArgBases, void **Args,
                             int64_t *ArgSizes, int64_t *ArgTypes,
                             map_var_info_t *ArgNames, void **ArgMappers,
                             SmallVector<void *> &TgtArgs,
                             SmallVector<ptrdiff_t> &TgtOffsets,
                             PrivateArgumentManagerTy &PrivateArgumentManager,
                             AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_NAME_AND_IDENT("mappingBeforeTargetRegion", Loc);
  DeviceTy &Device = *PM->Devices[DeviceId];
  int Ret = targetDataBegin(Loc, Device, ArgNum, ArgBases, Args, ArgSizes,
                            ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataBegin failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // List of (first-)private arrays allocated for this target region
  SmallVector<int> TgtArgsPositions(ArgNum, -1);

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
        bool IsLast, IsHostPtr; // IsLast is unused.
        void *TgtPtrBase =
            (void *)((intptr_t)TgtArgs[TgtIdx] + TgtOffsets[TgtIdx]);
        DP("Parent lambda base " DPxMOD "\n", DPxPTR(TgtPtrBase));
        uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
        void *TgtPtrBegin = (void *)((uintptr_t)TgtPtrBase + Delta);
        void *&PointerTgtPtrBegin = AsyncInfo.getVoidPtrLocation();
        TargetPointerResultTy TPR = Device.getTgtPtrBegin(
            HstPtrVal, ArgSizes[I], IsLast, /*UpdateRefCount=*/false,
            /*UseHoldRefCount=*/false, IsHostPtr);
        PointerTgtPtrBegin = TPR.TargetPointer;
        if (!TPR.isPresent()) {
          DP("No lambda captured variable mapped (" DPxMOD ") - ignored\n",
             DPxPTR(HstPtrVal));
          continue;
        }
        if (IsHostPtr) {
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
    TargetPointerResultTy TPR;
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
      TPR = Device.getTgtPtrBegin(HstPtrBegin, ArgSizes[I], IsLast,
                                  /*UpdateRefCount=*/false,
                                  /*UseHoldRefCount=*/false, IsHostPtr);
      TgtPtrBegin = TPR.TargetPointer;
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
static int processDataAfter(ident_t *Loc, int64_t DeviceId, void *HostPtr,
                            int32_t ArgNum, void **ArgBases, void **Args,
                            int64_t *ArgSizes, int64_t *ArgTypes,
                            map_var_info_t *ArgNames, void **ArgMappers,
                            PrivateArgumentManagerTy &PrivateArgumentManager,
                            AsyncInfoTy &AsyncInfo) {
  TIMESCOPE_WITH_NAME_AND_IDENT("mappingAfterTargetRegion", Loc);
  DeviceTy &Device = *PM->Devices[DeviceId];

  // Move data from device.
  int Ret = targetDataEnd(Loc, Device, ArgNum, ArgBases, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Call to targetDataEnd failed, abort target.\n");
    return OFFLOAD_FAIL;
  }

  // Free target memory for private arguments after synchronization.
  // TODO: We might want to remove `mutable` in the future by not changing the
  // captured variables somehow.
  AsyncInfo.addPostProcessingFunction(
      [PrivateArgumentManager =
           std::move(PrivateArgumentManager)]() mutable -> int {
        int Ret = PrivateArgumentManager.free();
        if (Ret != OFFLOAD_SUCCESS) {
          REPORT("Failed to deallocate target memory for private args\n");
          return OFFLOAD_FAIL;
        }
        return Ret;
      });

  return OFFLOAD_SUCCESS;
}
} // namespace

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo) {
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

  DP("loop trip count is %" PRIu64 ".\n", KernelArgs.Tripcount);

  // We need to keep bases and offsets separate. Sometimes (e.g. in OpenCL) we
  // need to manifest base pointers prior to launching a kernel. Even if we have
  // mapped an object only partially, e.g. A[N:M], although the kernel is
  // expected to access elements starting at address &A[N] and beyond, we still
  // need to manifest the base of the array &A[0]. In other cases, e.g. the COI
  // API, we need the begin address itself, i.e. &A[N], as the API operates on
  // begin addresses, not bases. That's why we pass args and offsets as two
  // separate entities so that each plugin can do what it needs. This behavior
  // was introdued via https://reviews.llvm.org/D33028 and commit 1546d319244c.
  SmallVector<void *> TgtArgs;
  SmallVector<ptrdiff_t> TgtOffsets;

  PrivateArgumentManagerTy PrivateArgumentManager(Device, AsyncInfo);

  int NumClangLaunchArgs = KernelArgs.NumArgs;
  int Ret = OFFLOAD_SUCCESS;
  if (NumClangLaunchArgs) {
    // Process data, such as data mapping, before launching the kernel
    Ret = processDataBefore(Loc, DeviceId, HostPtr, NumClangLaunchArgs,
                            KernelArgs.ArgBasePtrs, KernelArgs.ArgPtrs,
                            KernelArgs.ArgSizes, KernelArgs.ArgTypes,
                            KernelArgs.ArgNames, KernelArgs.ArgMappers, TgtArgs,
                            TgtOffsets, PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data before launching the kernel.\n");
      return OFFLOAD_FAIL;
    }

    // Clang might pass more values via the ArgPtrs to the runtime that we pass
    // on to the kernel.
    // TOOD: Next time we adjust the KernelArgsTy we should introduce a new
    // NumKernelArgs field.
    KernelArgs.NumArgs = TgtArgs.size();
  }

  // Launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);

  {
    assert(KernelArgs.NumArgs == TgtArgs.size() && "Argument count mismatch!");
    TIMESCOPE_WITH_NAME_AND_IDENT("Initiate Kernel Launch", Loc);
    Ret = Device.launchKernel(TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(),
                              KernelArgs, AsyncInfo);
  }

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  if (NumClangLaunchArgs) {
    // Transfer data back and deallocate target memory for (first-)private
    // variables
    Ret = processDataAfter(Loc, DeviceId, HostPtr, NumClangLaunchArgs,
                           KernelArgs.ArgBasePtrs, KernelArgs.ArgPtrs,
                           KernelArgs.ArgSizes, KernelArgs.ArgTypes,
                           KernelArgs.ArgNames, KernelArgs.ArgMappers,
                           PrivateArgumentManager, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT("Failed to process data after launching the kernel.\n");
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Executes a kernel using pre-recorded information for loading to
/// device memory to launch the target kernel with the pre-recorded
/// configuration.
int target_replay(ident_t *Loc, DeviceTy &Device, void *HostPtr,
                  void *DeviceMemory, int64_t DeviceMemorySize, void **TgtArgs,
                  ptrdiff_t *TgtOffsets, int32_t NumArgs, int32_t NumTeams,
                  int32_t ThreadLimit, uint64_t LoopTripCount,
                  AsyncInfoTy &AsyncInfo) {
  int32_t DeviceId = Device.DeviceID;
  TableMap *TM = getTableMap(HostPtr);
  // Fail if the table map fails to find the target kernel pointer for the
  // provided host pointer.
  if (!TM) {
    REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
           DPxPTR(HostPtr));
    return OFFLOAD_FAIL;
  }

  // Retrieve the target table of offloading entries.
  __tgt_target_table *TargetTable = nullptr;
  {
    std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
    assert(TM->Table->TargetsTable.size() > (size_t)DeviceId &&
           "Not expecting a device ID outside the table's bounds!");
    TargetTable = TM->Table->TargetsTable[DeviceId];
  }
  assert(TargetTable && "Global data has not been mapped\n");

  // Retrieve the target kernel pointer, allocate and store the recorded device
  // memory data, and launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
  DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
     TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);

  void *TgtPtr = Device.allocData(DeviceMemorySize, /* HstPtr */ nullptr,
                                  TARGET_ALLOC_DEFAULT);
  Device.submitData(TgtPtr, DeviceMemory, DeviceMemorySize, AsyncInfo);

  KernelArgsTy KernelArgs = {0};
  KernelArgs.Version = 2;
  KernelArgs.NumArgs = NumArgs;
  KernelArgs.Tripcount = LoopTripCount;
  KernelArgs.NumTeams[0] = NumTeams;
  KernelArgs.ThreadLimit[0] = ThreadLimit;

  int Ret = Device.launchKernel(TgtEntryPtr, TgtArgs, TgtOffsets, KernelArgs,
                                AsyncInfo);

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT("Executing target region abort target.\n");
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}
