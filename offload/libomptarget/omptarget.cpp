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
#include "OffloadPolicy.h"
#include "OpenMP/OMPT/Callback.h"
#include "OpenMP/OMPT/Interface.h"
#include "PluginManager.h"
#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include "Shared/Profile.h"

#include "OpenMP/Mapping.h"
#include "OpenMP/omp.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Object/ObjectFile.h"

#include <cassert>
#include <cstdint>
#include <vector>

using llvm::SmallVector;
#ifdef OMPT_SUPPORT
using namespace llvm::omp::target::ompt;
#endif
using namespace llvm::omp::target::debug;

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

bool AsyncInfoTy::isDone() const { return isQueueEmpty(); }

int32_t AsyncInfoTy::runPostProcessing() {
  size_t Size = PostProcessingFunctions.size();
  for (size_t I = 0; I < Size; ++I) {
    const int Result = PostProcessingFunctions[I]();
    if (Result != OFFLOAD_SUCCESS)
      return Result;
  }

  // Clear the vector up until the last known function, since post-processing
  // procedures might add new procedures themselves.
  const auto *PrevBegin = PostProcessingFunctions.begin();
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
static uint64_t getPartialStructRequiredAlignment(void *HstPtrBase) {
  int LowestOneBit = __builtin_ffsl(reinterpret_cast<uintptr_t>(HstPtrBase));
  uint64_t BaseAlignment = 1 << (LowestOneBit - 1);
  return MaxAlignment < BaseAlignment ? MaxAlignment : BaseAlignment;
}

void handleTargetOutcome(bool Success, ident_t *Loc) {
  switch (OffloadPolicy::get(*PM).Kind) {
  case OffloadPolicy::DISABLED:
    if (Success) {
      FATAL_MESSAGE0(1, "expected no offloading while offloading is disabled");
    }
    break;
  case OffloadPolicy::MANDATORY:
    if (!Success) {
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE) {
        auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
        for (auto &Device : PM->devices(ExclusiveDevicesAccessor))
          dumpTargetPointerMappings(Loc, Device);
      } else
        FAILURE_MESSAGE("Consult https://openmp.llvm.org/design/Runtimes.html "
                        "for debugging options.\n");

      if (!PM->getNumActivePlugins()) {
        FAILURE_MESSAGE(
            "No images found compatible with the installed hardware. ");

        llvm::SmallVector<llvm::StringRef> Archs;
        for (auto &Image : PM->deviceImages()) {
          const char *Start = reinterpret_cast<const char *>(
              Image.getExecutableImage().ImageStart);
          uint64_t Length =
              utils::getPtrDiff(Start, Image.getExecutableImage().ImageEnd);
          llvm::MemoryBufferRef Buffer(llvm::StringRef(Start, Length),
                                       /*Identifier=*/"");

          auto ObjectOrErr = llvm::object::ObjectFile::createObjectFile(Buffer);
          if (auto Err = ObjectOrErr.takeError()) {
            llvm::consumeError(std::move(Err));
            continue;
          }

          if (auto CPU = (*ObjectOrErr)->tryGetCPUName())
            Archs.push_back(*CPU);
        }
        fprintf(stderr, "Found %zu image(s): (%s)\n", Archs.size(),
                llvm::join(Archs, ",").c_str());
      }

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
      if (getInfoLevel() & OMP_INFOTYPE_DUMP_TABLE) {
        auto ExclusiveDevicesAccessor = PM->getExclusiveDevicesAccessor();
        for (auto &Device : PM->devices(ExclusiveDevicesAccessor))
          dumpTargetPointerMappings(Loc, Device);
      }
    }
    break;
  }
}

static int32_t getParentIndex(int64_t Type) {
  return ((Type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

void *targetAllocExplicit(size_t Size, int DeviceNum, int Kind,
                          const char *Name) {
  ODBG(ODT_Interface) << "Call to " << Name << " for device " << DeviceNum
                      << " requesting " << Size << " bytes";

  if (Size <= 0) {
    ODBG(ODT_Interface) << "Call to " << Name << " with non-positive length";
    return NULL;
  }

  void *Rc = NULL;

  if (DeviceNum == omp_get_initial_device()) {
    Rc = malloc(Size);
    ODBG(ODT_Interface) << Name << " returns host ptr " << Rc;
    return Rc;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  Rc = DeviceOrErr->allocData(Size, nullptr, Kind);
  ODBG(ODT_Interface) << Name << " returns device ptr " << Rc;
  return Rc;
}

void targetFreeExplicit(void *DevicePtr, int DeviceNum, int Kind,
                        const char *Name) {
  ODBG(ODT_Interface) << "Call to " << Name << " for device " << DeviceNum
                      << " and address " << DevicePtr;

  if (!DevicePtr) {
    ODBG(ODT_Interface) << "Call to " << Name << " with NULL ptr";
    return;
  }

  if (DeviceNum == omp_get_initial_device()) {
    free(DevicePtr);
    ODBG(ODT_Interface) << Name << " deallocated host ptr";
    return;
  }

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  if (DeviceOrErr->deleteData(DevicePtr, Kind) == OFFLOAD_FAIL)
    FATAL_MESSAGE(DeviceNum, "%s",
                  "Failed to deallocate device ptr. Set "
                  "OFFLOAD_TRACK_ALLOCATION_TRACES=1 to track allocations.");

  ODBG(ODT_Interface) << "omp_target_free deallocated device ptr";
}

void *targetLockExplicit(void *HostPtr, size_t Size, int DeviceNum,
                         const char *Name) {
  ODBG(ODT_Interface) << "Call to " << Name << " for device " << DeviceNum
                      << " locking " << Size << " bytes";

  if (Size <= 0) {
    ODBG(ODT_Interface) << "Call to " << Name << " with non-positive length";
    return NULL;
  }

  void *RC = NULL;

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  int32_t Err = 0;
  Err = DeviceOrErr->RTL->data_lock(DeviceNum, HostPtr, Size, &RC);
  if (Err) {
    ODBG(ODT_Interface) << "Could not lock ptr " << HostPtr;
    return nullptr;
  }
  ODBG(ODT_Interface) << Name << " returns device ptr " << RC;
  return RC;
}

void targetUnlockExplicit(void *HostPtr, int DeviceNum, const char *Name) {
  ODBG(ODT_Interface) << "Call to " << Name << " for device " << DeviceNum
                      << " unlocking";

  auto DeviceOrErr = PM->getDevice(DeviceNum);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceNum, "%s", toString(DeviceOrErr.takeError()).c_str());

  DeviceOrErr->RTL->data_unlock(DeviceNum, HostPtr);
  ODBG(ODT_Interface) << Name << " returns";
}

/// Call the user-defined mapper function followed by the appropriate
// targetData* function (targetData{Begin,End,Update}).
int targetDataMapper(ident_t *Loc, DeviceTy &Device, void *ArgBase, void *Arg,
                     int64_t ArgSize, int64_t ArgType, map_var_info_t ArgNames,
                     void *ArgMapper, AsyncInfoTy &AsyncInfo,
                     TargetDataFuncPtrTy TargetDataFunction,
                     AttachInfoTy *AttachInfo = nullptr) {
  ODBG(ODT_Interface) << "Calling the mapper function " << ArgMapper;

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
                              AsyncInfo, AttachInfo, /*FromMapper=*/true);

  return Rc;
}

/// Returns a buffer of the requested \p Size, to be used as the source for
/// `submitData`.
///
/// For small buffers (`Size <= sizeof(void*)`), uses \p AsyncInfo's
/// getVoidPtrLocation().
/// For larger buffers, creates a dynamic buffer which will be eventually
/// deleted by \p AsyncInfo's post-processing callback.
static char *getOrCreateSourceBufferForSubmitData(AsyncInfoTy &AsyncInfo,
                                                  int64_t Size) {
  constexpr int64_t VoidPtrSize = sizeof(void *);

  if (Size <= VoidPtrSize) {
    void *&BufferElement = AsyncInfo.getVoidPtrLocation();
    return reinterpret_cast<char *>(&BufferElement);
  }

  // Create a dynamic buffer for larger data and schedule its deletion.
  char *DataBuffer = new char[Size];
  AsyncInfo.addPostProcessingFunction([DataBuffer]() {
    delete[] DataBuffer;
    return OFFLOAD_SUCCESS;
  });
  return DataBuffer;
}

/// Calculates the target pointee base by applying the host
/// pointee begin/base delta to the target pointee begin.
///
/// ```
/// TgtPteeBase = TgtPteeBegin - (HstPteeBegin - HstPteeBase)
/// ```
static void *calculateTargetPointeeBase(void *HstPteeBase, void *HstPteeBegin,
                                        void *TgtPteeBegin) {
  uint64_t Delta = reinterpret_cast<uint64_t>(HstPteeBegin) -
                   reinterpret_cast<uint64_t>(HstPteeBase);
  void *TgtPteeBase = reinterpret_cast<void *>(
      reinterpret_cast<uint64_t>(TgtPteeBegin) - Delta);

  ODBG(ODT_Mapping) << "HstPteeBase: " << HstPteeBase
                    << ", HstPteeBegin: " << HstPteeBegin
                    << ", Delta (HstPteeBegin - HstPteeBase): " << Delta << "\n"
                    << "TgtPteeBase (TgtPteeBegin - Delta): " << TgtPteeBase
                    << ", TgtPteeBegin: " << TgtPteeBegin;

  return TgtPteeBase;
}

/// Utility function to perform a pointer attachment operation.
///
/// For something like:
/// ```cpp
///  int *p;
///  ...
///  #pragma omp target enter data map(to:p[10:10])
/// ```
///
/// for which the attachment operation gets represented using:
/// ```
///   &p, &p[10], sizeof(p), ATTACH
/// ```
///
/// (Hst|Tgt)PtrAddr   represents &p
/// (Hst|Tgt)PteeBase  represents &p[0]
/// (Hst|Tgt)PteeBegin represents &p[10]
///
/// This function first computes the expected TgtPteeBase using:
///   `<Select>TgtPteeBase = TgtPteeBegin - (HstPteeBegin - HstPteeBase)`
///
/// and then attaches TgtPteeBase to TgtPtrAddr.
///
/// \p HstPtrSize represents the size of the pointer p. For C/C++, this
/// should be same as "sizeof(void*)" (say 8).
///
/// However, for Fortran, pointers/allocatables, which are also eligible for
/// "pointer-attachment", may be implemented using descriptors that contain the
/// address of the pointee in the first 8 bytes, but also contain other
/// information such as lower-bound/upper-bound etc in their subsequent fields.
///
/// For example, for the following:
/// ```fortran
///   integer, allocatable :: x(:)
///   integer, pointer :: p(:)
///   ...
///   p => x(10: 19)
///   ...
///   !$omp target enter data map(to:p(:))
/// ```
///
/// The map should trigger a pointer-attachment (assuming the pointer-attachment
/// conditions as noted on processAttachEntries are met) between the descriptor
/// for p, and its pointee data.
///
/// Since only the first 8 bytes of the descriptor contain the address of the
/// pointee, an attachment operation on device descriptors involves:
/// * Setting the first 8 bytes of the device descriptor to point the device
/// address of the pointee.
/// * Copying the remaining information about bounds/offset etc. from the host
/// descriptor to the device descriptor.
///
/// The function also handles pointer-attachment portion of PTR_AND_OBJ maps,
/// like:
/// ```
///   &p, &p[10], 10 * sizeof(p[10]), PTR_AND_OBJ
/// ```
/// by using `sizeof(void*)` as \p HstPtrSize.
static int performPointerAttachment(DeviceTy &Device, AsyncInfoTy &AsyncInfo,
                                    void **HstPtrAddr, void *HstPteeBase,
                                    void *HstPteeBegin, void **TgtPtrAddr,
                                    void *TgtPteeBegin, int64_t HstPtrSize,
                                    TargetPointerResultTy &PtrTPR) {
  assert(PtrTPR.getEntry() &&
         "Need a valid pointer entry to perform pointer-attachment");

  constexpr int64_t VoidPtrSize = sizeof(void *);
  assert(HstPtrSize >= VoidPtrSize && "PointerSize is too small");

  void *TgtPteeBase =
      calculateTargetPointeeBase(HstPteeBase, HstPteeBegin, TgtPteeBegin);

  // Add shadow pointer tracking
  if (!PtrTPR.getEntry()->addShadowPointer(
          ShadowPtrInfoTy{HstPtrAddr, TgtPtrAddr, TgtPteeBase, HstPtrSize})) {
    ODBG(ODT_Mapping) << "Pointer " << TgtPtrAddr << " is already attached to "
                      << TgtPteeBase;
    return OFFLOAD_SUCCESS;
  }

  ODBG(ODT_Mapping) << "Update pointer (" << TgtPtrAddr << ") -> ["
                    << TgtPteeBase << "]\n";

  // Lambda to handle submitData result and perform final steps.
  auto HandleSubmitResult = [&](int SubmitResult) -> int {
    if (SubmitResult != OFFLOAD_SUCCESS) {
      REPORT() << "Failed to update pointer on device.";
      return OFFLOAD_FAIL;
    }

    if (PtrTPR.getEntry()->addEventIfNecessary(Device, AsyncInfo) !=
        OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;

    return OFFLOAD_SUCCESS;
  };

  // Get a buffer to be used as the source for data submission.
  char *SrcBuffer = getOrCreateSourceBufferForSubmitData(AsyncInfo, HstPtrSize);

  // The pointee's address should occupy the first VoidPtrSize bytes
  // irrespective of HstPtrSize.
  std::memcpy(SrcBuffer, &TgtPteeBase, VoidPtrSize);

  // For larger "pointers" (e.g., Fortran descriptors), copy remaining
  // descriptor fields from the host descriptor into the buffer.
  if (HstPtrSize > VoidPtrSize) {
    uint64_t HstDescriptorFieldsSize = HstPtrSize - VoidPtrSize;
    void *HstDescriptorFieldsAddr =
        reinterpret_cast<char *>(HstPtrAddr) + VoidPtrSize;
    std::memcpy(SrcBuffer + VoidPtrSize, HstDescriptorFieldsAddr,
                HstDescriptorFieldsSize);

    ODBG(ODT_Mapping) << "Updating " << HstPtrSize << " bytes of descriptor ("
                      << TgtPtrAddr << ") (pointer + "
                      << HstDescriptorFieldsSize
                      << " additional bytes from host descriptor "
                      << HstDescriptorFieldsAddr << ")";
  }

  // Submit the populated source buffer to device.
  int SubmitResult = Device.submitData(TgtPtrAddr, SrcBuffer, HstPtrSize,
                                       AsyncInfo, PtrTPR.getEntry());
  return HandleSubmitResult(SubmitResult);
}

/// Internal function to do the mapping and transfer the data to the device
int targetDataBegin(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                    void **ArgsBase, void **Args, int64_t *ArgSizes,
                    int64_t *ArgTypes, map_var_info_t *ArgNames,
                    void **ArgMappers, AsyncInfoTy &AsyncInfo,
                    AttachInfoTy *AttachInfo, bool FromMapper) {
  assert(AttachInfo && "AttachInfo must be available for targetDataBegin for "
                       "handling ATTACH map-types.");
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;
    TIMESCOPE_WITH_DETAILS_AND_IDENT(
        "HostToDev", "Size=" + std::to_string(ArgSizes[I]) + "B", Loc);
    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataBegin, call the
      // targetDataMapper variant which will call targetDataBegin again
      // with new arguments.
      ODBG(ODT_Mapping) << "Calling targetDataMapper for the " << I
                        << "th argument";

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Rc = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                targetDataBegin, AttachInfo);

      if (Rc != OFFLOAD_SUCCESS) {
        REPORT() << "Call to targetDataBegin via targetDataMapper for custom "
                    "mapper failed";
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    void *HstPtrBase = ArgsBase[I];
    int64_t DataSize = ArgSizes[I];
    map_var_info_t HstPtrName = (!ArgNames) ? nullptr : ArgNames[I];

    // ATTACH map-types are supposed to be handled after all mapping for the
    // construct is done. Defer their processing.
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_ATTACH) {
      const bool IsCorrespondingPointerInit =
          (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE);
      // We don't need to keep track of PRIVATE | ATTACH entries. They
      // represent corresponding-pointer-initialization, and are handled
      // similar to firstprivate (PRIVATE | TO) entries by
      // PrivateArgumentManager.
      if (!IsCorrespondingPointerInit)
        AttachInfo->AttachEntries.emplace_back(
            /*PointerBase=*/HstPtrBase, /*PointeeBegin=*/HstPtrBegin,
            /*PointerSize=*/DataSize, /*MapType=*/ArgTypes[I],
            /*PointeeName=*/HstPtrName);

      ODBG(ODT_Mapping) << "Deferring ATTACH map-type processing for argument "
                        << I;
      continue;
    }

    // Adjust for proper alignment if this is a combined entry (for structs).
    // Look at the next argument - if that is MEMBER_OF this one, then this one
    // is a combined entry.
    int64_t TgtPadding = 0;
    const int NextI = I + 1;
    if (getParentIndex(ArgTypes[I]) < 0 && NextI < ArgNum &&
        getParentIndex(ArgTypes[NextI]) == I) {
      int64_t Alignment = getPartialStructRequiredAlignment(HstPtrBase);
      TgtPadding = (int64_t)HstPtrBegin % Alignment;
      if (TgtPadding) {
        ODBG(ODT_Mapping) << "Using a padding of " << TgtPadding
                          << " bytes for begin address " << HstPtrBegin;
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

    MappingInfoTy::HDTTMapAccessorTy HDTTMap =
        Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      ODBG(ODT_Mapping) << "Has a pointer entry";
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
      PointerTpr = Device.getMappingInfo().getTargetPointer(
          HDTTMap, HstPtrBase, HstPtrBase, /*TgtPadding=*/0, sizeof(void *),
          /*HstPtrName=*/nullptr,
          /*HasFlagTo=*/false, /*HasFlagAlways=*/false, IsImplicit, UpdateRef,
          HasCloseModifier, HasPresentModifier, HasHoldModifier, AsyncInfo,
          /*OwnedTPR=*/nullptr, /*ReleaseHDTTMap=*/false);
      PointerTgtPtrBegin = PointerTpr.TargetPointer;
      IsHostPtr = PointerTpr.Flags.IsHostPointer;
      if (!PointerTgtPtrBegin) {
        REPORT() << "Call to getTargetPointer returned null pointer ("
                 << (HasPresentModifier ? "'present' map type modifier"
                                        : "device failure or illegal mapping")
                 << ")";
        return OFFLOAD_FAIL;
      }

      // Track new allocation, for eventual use in attachment decision-making.
      if (PointerTpr.Flags.IsNewEntry && !IsHostPtr)
        AttachInfo->NewAllocations[HstPtrBase] = sizeof(void *);

      ODBG(ODT_Mapping) << "There are " << sizeof(void *)
                        << " bytes allocated at target address "
                        << PointerTgtPtrBegin << " - is"
                        << (PointerTpr.Flags.IsNewEntry ? "" : " not")
                        << " new";
      PointerHstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *reinterpret_cast<void **>(HstPtrBase);
      // No need to update pointee ref count for the first element of the
      // subelement that comes from mapper.
      UpdateRef =
          (!FromMapper || I != 0); // subsequently update ref count of pointee
    }

    const bool HasFlagTo = ArgTypes[I] & OMP_TGT_MAPTYPE_TO;
    const bool HasFlagAlways = ArgTypes[I] & OMP_TGT_MAPTYPE_ALWAYS;
    // Note that HDTTMap will be released in getTargetPointer.
    auto TPR = Device.getMappingInfo().getTargetPointer(
        HDTTMap, HstPtrBegin, HstPtrBase, TgtPadding, DataSize, HstPtrName,
        HasFlagTo, HasFlagAlways, IsImplicit, UpdateRef, HasCloseModifier,
        HasPresentModifier, HasHoldModifier, AsyncInfo, PointerTpr.getEntry());
    void *TgtPtrBegin = TPR.TargetPointer;
    IsHostPtr = TPR.Flags.IsHostPointer;
    // If data_size==0, then the argument could be a zero-length pointer to
    // NULL, so getOrAlloc() returning NULL is not an error.
    if (!TgtPtrBegin && (DataSize || HasPresentModifier)) {
      REPORT() << "Call to getTargetPointer returned null pointer ("
               << (HasPresentModifier ? "'present' map type modifier"
                                      : "device failure or illegal mapping")
               << ").";
      return OFFLOAD_FAIL;
    }

    // Track new allocation, for eventual use in attachment decision-making.
    if (TPR.Flags.IsNewEntry && !IsHostPtr && TgtPtrBegin)
      AttachInfo->NewAllocations[HstPtrBegin] = DataSize;

    ODBG(ODT_Mapping) << "There are " << DataSize
                      << " bytes allocated at target address " << TgtPtrBegin
                      << " - is" << (TPR.Flags.IsNewEntry ? "" : " not")
                      << " new";

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      uintptr_t Delta = reinterpret_cast<uintptr_t>(HstPtrBegin) -
                        reinterpret_cast<uintptr_t>(HstPtrBase);
      void *TgtPtrBase;
      if (TgtPtrBegin) {
        // Lookup succeeded, return device pointer adjusted by delta
        TgtPtrBase = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(TgtPtrBegin) - Delta);
        ODBG(ODT_Mapping) << "Returning device pointer " << TgtPtrBase;
      } else {
        // Lookup failed. So we have to decide what to do based on the
        // requested fallback behavior.
        //
        // Treat "preserve" as the default fallback behavior, since as per
        // OpenMP 5.1, for use_device_ptr/addr, when there's no corresponding
        // device pointer to translate into, it's the user's responsibility to
        // ensure that the host address is device-accessible.
        //
        // OpenMP 5.1, sec 2.14.2, target data construct, p 188, l26-31:
        // If a list item that appears in a use_device_ptr clause ... does not
        // point to a mapped object, it must contain a valid device address for
        // the target device, and the list item references are instead converted
        // to references to a local device pointer that refers to this device
        // address.
        //
        // OpenMP 6.1's `fb_nullify` fallback behavior: when the FB_NULLIFY bit
        // is set by the compiler, e.g. for `use/need_device_ptr(fb_nullify)`),
        // return `nullptr - Delta` when lookup fails.
        if (ArgTypes[I] & OMP_TGT_MAPTYPE_FB_NULLIFY) {
          TgtPtrBase = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(nullptr) - Delta);
          ODBG(ODT_Mapping) << "Returning offsetted null pointer " << TgtPtrBase
                            << " as fallback (lookup failed)";
        } else {
          TgtPtrBase = reinterpret_cast<void *>(
              reinterpret_cast<uintptr_t>(HstPtrBegin) - Delta);
          ODBG(ODT_Mapping) << "Returning host pointer " << TgtPtrBase
                            << " as fallback (lookup failed)";
        }
      }
      ArgsBase[I] = TgtPtrBase;
    }

    if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ && !IsHostPtr) {
      int Ret = performPointerAttachment(
          Device, AsyncInfo, reinterpret_cast<void **>(PointerHstPtrBegin),
          HstPtrBase, HstPtrBegin,
          reinterpret_cast<void **>(PointerTgtPtrBegin), TgtPtrBegin,
          sizeof(void *), PointerTpr);
      if (Ret != OFFLOAD_SUCCESS)
        return OFFLOAD_FAIL;
    }

    // Check if variable can be used on the device:
    bool IsStructMember = ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF;
    if (getInfoLevel() & OMP_INFOTYPE_EMPTY_MAPPING && ArgTypes[I] != 0 &&
        !IsStructMember && !IsImplicit && !TPR.isPresent() &&
        !TPR.isContained() && !TPR.isHostPointer())
      INFO(OMP_INFOTYPE_EMPTY_MAPPING, Device.DeviceID,
           "variable %s does not have a valid device counterpart\n",
           (HstPtrName) ? getNameFromMapping(HstPtrName).c_str() : "unknown");
  }

  return OFFLOAD_SUCCESS;
}

/// Process deferred ATTACH map entries collected during targetDataBegin.
///
/// From OpenMP's perspective, when mapping something that has a base pointer,
/// such as:
/// ```cpp
///   int *p;
///   #pragma omp enter target data map(to: p[10:20])
/// ```
///
/// a pointer-attachment between p and &p[10] should occur if both p and
/// p[10] are present on the device after doing all allocations for all maps
/// on the construct, and one of the following is true:
///
/// * The pointer p was newly allocated while handling the construct
/// * The pointee p[10:20] was newly allocated while handling the construct
/// * attach(always) map-type modifier was specified (OpenMP 6.1)
///
/// That's why we collect all attach entries and new memory allocations during
/// targetDataBegin, and use that information to make the decision of whether
/// to perform a pointer-attachment or not here, after maps have been handled.
///
/// Additionally, once we decide that a pointer-attachment should be performed,
/// we need to make sure that it happens after any previously submitted data
/// transfers have completed, to avoid the possibility of the pending transfers
/// clobbering the attachment. For example:
///
/// ```cpp
///   int *p = ...;
///   int **pp = &p;
///   map(to: pp[0], p[0])
/// ```
///
/// Which would be represented by:
/// ```
/// &pp[0], &pp[0], sizeof(pp[0]), TO (1)
/// &p[0], &p[0], sizeof(p[0]), TO    (2)
///
/// &pp, &pp[0], sizeof(pp), ATTACH   (3)
/// &p, &p[0], sizeof(p), ATTACH      (4)
/// ```
///
/// (4) and (1) are both trying to modify the device memory corresponding to
/// `&p`. So, if we decide that (4) should do an attachment, we also need to
/// ensure that (4) happens after (1) is complete.
///
/// For this purpose, we insert a data_fence before the first
/// pointer-attachment, (3), to ensure that all pending transfers finish first.
int processAttachEntries(DeviceTy &Device, AttachInfoTy &AttachInfo,
                         AsyncInfoTy &AsyncInfo) {
  // Report all tracked allocations from both main loop and ATTACH processing
  if (!AttachInfo.NewAllocations.empty()) {
    ODBG_OS(ODT_Mapping, [&](llvm::raw_ostream &OS) {
      OS << "Tracked " << AttachInfo.NewAllocations.size()
         << " total new allocations:";
      for (const auto &Alloc : AttachInfo.NewAllocations) {
        OS << "  Host ptr: " << Alloc.first << ", Size: " << Alloc.second
           << " bytes";
      }
    });
  }

  if (AttachInfo.AttachEntries.empty())
    return OFFLOAD_SUCCESS;

  ODBG(ODT_Mapping) << "Processing " << AttachInfo.AttachEntries.size()
                    << " deferred ATTACH map entries";

  bool TreatAttachAutoAsAlways = MappingConfig::get().TreatAttachAutoAsAlways;
  if (TreatAttachAutoAsAlways)
    ODBG(ODT_Mapping) << "Treating ATTACH(auto) as ATTACH(always) because "
                      << "LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS is true";

  int Ret = OFFLOAD_SUCCESS;
  bool IsFirstPointerAttachment = true;
  for (size_t EntryIdx = 0; EntryIdx < AttachInfo.AttachEntries.size();
       ++EntryIdx) {
    const auto &AttachEntry = AttachInfo.AttachEntries[EntryIdx];

    void **HstPtr = reinterpret_cast<void **>(AttachEntry.PointerBase);

    void *HstPteeBase = *HstPtr;
    void *HstPteeBegin = AttachEntry.PointeeBegin;

    int64_t PtrSize = AttachEntry.PointerSize;
    int64_t MapType = AttachEntry.MapType;

    ODBG(ODT_Mapping) << "Processing ATTACH entry " << EntryIdx
                      << ": HstPtr=" << HstPtr
                      << ", HstPteeBegin=" << HstPteeBegin
                      << ", PtrSize=" << PtrSize << ", MapType=0x"
                      << llvm::utohexstr(MapType);

    bool IsAttachAlways =
        (MapType & OMP_TGT_MAPTYPE_ALWAYS) || TreatAttachAutoAsAlways;

    // Lambda to check if a pointer was newly allocated
    auto WasNewlyAllocated = [&](void *Ptr, const char *PtrName) {
      bool IsNewlyAllocated =
          llvm::any_of(AttachInfo.NewAllocations, [&](const auto &Alloc) {
            void *AllocPtr = Alloc.first;
            int64_t AllocSize = Alloc.second;
            return Ptr >= AllocPtr &&
                   Ptr < reinterpret_cast<void *>(
                             reinterpret_cast<char *>(AllocPtr) + AllocSize);
          });
      ODBG(ODT_Mapping) << "Attach " << PtrName << " " << Ptr
                        << " was newly allocated: "
                        << (IsNewlyAllocated ? "yes" : "no");
      return IsNewlyAllocated;
    };

    // Only process ATTACH if either the pointee or the pointer was newly
    // allocated, or the ALWAYS flag is set.
    if (!IsAttachAlways && !WasNewlyAllocated(HstPteeBegin, "pointee") &&
        !WasNewlyAllocated(HstPtr, "pointer")) {
      ODBG(ODT_Mapping) << "Skipping ATTACH entry " << EntryIdx
                        << ": neither pointer nor pointee was newly "
                           "allocated and no ALWAYS flag";
      continue;
    }

    // Lambda to perform target pointer lookup and validation
    auto LookupTargetPointer =
        [&](void *Ptr, int64_t Size,
            const char *PtrType) -> std::optional<TargetPointerResultTy> {
      // ATTACH map-type does not change ref-count, or do any allocation
      // We just need to do a lookup for the pointer/pointee.
      TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
          Ptr, Size, /*UpdateRefCount=*/false,
          /*UseHoldRefCount=*/false, /*MustContain=*/true);

      ODBG(ODT_Mapping) << "Attach " << PtrType << " lookup - IsPresent="
                        << (TPR.isPresent() ? "yes" : "no") << ", IsHostPtr="
                        << (TPR.Flags.IsHostPointer ? "yes" : "no");

      if (!TPR.isPresent()) {
        ODBG(ODT_Mapping) << "Skipping ATTACH entry " << EntryIdx << ": "
                          << PtrType << " not present on device";
        return std::nullopt;
      }
      if (TPR.Flags.IsHostPointer) {
        ODBG(ODT_Mapping) << "Skipping ATTACH entry " << EntryIdx
                          << ": device version of the " << PtrType
                          << " is a host pointer.";
        return std::nullopt;
      }

      return TPR;
    };

    // Get device version of the pointee (e.g., &p[10]) first, as we can
    // release its TPR after extracting the pointer value.
    void *TgtPteeBegin = [&]() -> void * {
      if (auto PteeTPROpt = LookupTargetPointer(HstPteeBegin, 0, "pointee"))
        return PteeTPROpt->TargetPointer;
      return nullptr;
    }();

    if (!TgtPteeBegin)
      continue;

    // Get device version of the pointer (e.g., &p) next. We need to keep its
    // TPR for use in shadow-pointer handling during pointer-attachment.
    auto PtrTPROpt = LookupTargetPointer(HstPtr, PtrSize, "pointer");
    if (!PtrTPROpt)
      continue;
    TargetPointerResultTy &PtrTPR = *PtrTPROpt;
    void **TgtPtrBase = reinterpret_cast<void **>(PtrTPR.TargetPointer);

    // Insert a data-fence before the first pointer-attachment.
    if (IsFirstPointerAttachment) {
      IsFirstPointerAttachment = false;
      ODBG(ODT_Mapping)
          << "Inserting a data fence before the first pointer attachment.";
      Ret = Device.dataFence(AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT() << "Failed to insert data fence.";
        return OFFLOAD_FAIL;
      }
    }

    // Do the pointer-attachment, i.e. update the device pointer to point to
    // device pointee.
    Ret = performPointerAttachment(Device, AsyncInfo, HstPtr, HstPteeBase,
                                   HstPteeBegin, TgtPtrBase, TgtPteeBegin,
                                   PtrSize, PtrTPR);
    if (Ret != OFFLOAD_SUCCESS)
      return OFFLOAD_FAIL;

    ODBG(ODT_Mapping) << "ATTACH entry " << EntryIdx
                      << " processed successfully";
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

  /// The target pointer information.
  TargetPointerResultTy TPR;

  PostProcessingInfo(void *HstPtr, int64_t Size, int64_t ArgType,
                     TargetPointerResultTy &&TPR)
      : HstPtrBegin(HstPtr), DataSize(Size), ArgType(ArgType),
        TPR(std::move(TPR)) {}
};

} // namespace

/// Applies the necessary post-processing procedures to entries listed in \p
/// EntriesInfo after the execution of all device side operations from a target
/// data end. This includes the update of pointers at the host and removal of
/// device buffer when needed. It returns OFFLOAD_FAIL or OFFLOAD_SUCCESS
/// according to the successfulness of the operations.
[[nodiscard]] static int
postProcessingTargetDataEnd(DeviceTy *Device,
                            SmallVector<PostProcessingInfo> &EntriesInfo) {
  int Ret = OFFLOAD_SUCCESS;

  for (auto &[HstPtrBegin, DataSize, ArgType, TPR] : EntriesInfo) {
    bool DelEntry = !TPR.isHostPointer();

    // If the last element from the mapper (for end transfer args comes in
    // reverse order), do not remove the partial entry, the parent struct still
    // exists.
    if ((ArgType & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(ArgType & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    // If we marked the entry to be deleted we need to verify no other
    // thread reused it by now. If deletion is still supposed to happen by
    // this thread LR will be set and exclusive access to the HDTT map
    // will avoid another thread reusing the entry now. Note that we do
    // not request (exclusive) access to the HDTT map if DelEntry is
    // not set.
    MappingInfoTy::HDTTMapAccessorTy HDTTMap =
        Device->getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();

    // We cannot use a lock guard because we may end up delete the mutex.
    // We also explicitly unlocked the entry after it was put in the EntriesInfo
    // so it can be reused.
    TPR.getEntry()->lock();
    auto *Entry = TPR.getEntry();

    const bool IsNotLastUser = Entry->decDataEndThreadCount() != 0;
    if (DelEntry && (Entry->getTotalRefCount() != 0 || IsNotLastUser)) {
      // The thread is not in charge of deletion anymore. Give up access
      // to the HDTT map and unset the deletion flag.
      HDTTMap.destroy();
      DelEntry = false;
    }

    // If we copied back to the host a struct/array containing pointers, or
    // Fortran descriptors (which are larger than a "void *"), we need to
    // restore the original host pointer/descriptor values from their shadow
    // copies. If the struct is going to be deallocated, remove any remaining
    // shadow pointer entries for this struct.
    const bool HasFrom = ArgType & OMP_TGT_MAPTYPE_FROM;
    if (HasFrom) {
      Entry->foreachShadowPointerInfo([&](const ShadowPtrInfoTy &ShadowPtr) {
        constexpr int64_t VoidPtrSize = sizeof(void *);
        if (ShadowPtr.PtrSize > VoidPtrSize) {
          ODBG(ODT_Mapping)
              << "Restoring host descriptor " << (void *)ShadowPtr.HstPtrAddr
              << " to its original content (" << ShadowPtr.PtrSize
              << " bytes), containing pointee address "
              << (void *)ShadowPtr.HstPtrContent.data();
        } else {
          ODBG(ODT_Mapping)
              << "Restoring host pointer " << (void *)ShadowPtr.HstPtrAddr
              << " to its original value "
              << (void *)ShadowPtr.HstPtrContent.data();
        }
        std::memcpy(ShadowPtr.HstPtrAddr, ShadowPtr.HstPtrContent.data(),
                    ShadowPtr.PtrSize);
        return OFFLOAD_SUCCESS;
      });
    }

    // Give up the lock as we either don't need it anymore (e.g., done with
    // TPR), or erase TPR.
    TPR.setEntry(nullptr);

    if (!DelEntry)
      continue;

    Ret = Device->getMappingInfo().eraseMapEntry(HDTTMap, Entry, DataSize);
    // Entry is already remove from the map, we can unlock it now.
    HDTTMap.destroy();
    Ret |= Device->getMappingInfo().deallocTgtPtrAndEntry(Entry, DataSize);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT() << "Deallocating data from device failed.";
      break;
    }
  }

  delete &EntriesInfo;
  return Ret;
}

/// Internal function to undo the mapping and retrieve the data from the device.
int targetDataEnd(ident_t *Loc, DeviceTy &Device, int32_t ArgNum,
                  void **ArgBases, void **Args, int64_t *ArgSizes,
                  int64_t *ArgTypes, map_var_info_t *ArgNames,
                  void **ArgMappers, AsyncInfoTy &AsyncInfo,
                  AttachInfoTy *AttachInfo, bool FromMapper) {
  int Ret = OFFLOAD_SUCCESS;
  auto *PostProcessingPtrs = new SmallVector<PostProcessingInfo>();
  // process each input.
  for (int32_t I = ArgNum - 1; I >= 0; --I) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    // Ignore ATTACH entries - they should only be honored on map-entering
    // directives. They may be encountered here while handling the "end" part of
    // "#pragma omp target".
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_ATTACH) {
      ODBG(ODT_Mapping) << "Ignoring ATTACH entry " << I << " in targetDataEnd";
      continue;
    }

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataEnd, call the
      // targetDataMapper variant which will call targetDataEnd again
      // with new arguments.
      ODBG(ODT_Mapping) << "Calling targetDataMapper for the " << I
                        << "th argument";

      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      Ret = targetDataMapper(Loc, Device, ArgBases[I], Args[I], ArgSizes[I],
                             ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                             targetDataEnd);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT() << "Call to targetDataEnd via targetDataMapper for custom "
                    "mapper failed.";
        return OFFLOAD_FAIL;
      }

      // Skip the rest of this function, continue to the next argument.
      continue;
    }

    void *HstPtrBegin = Args[I];
    int64_t DataSize = ArgSizes[I];
    bool IsImplicit = ArgTypes[I] & OMP_TGT_MAPTYPE_IMPLICIT;
    bool UpdateRef = (!(ArgTypes[I] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
                      (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) &&
                     !(FromMapper && I == 0);
    bool ForceDelete = ArgTypes[I] & OMP_TGT_MAPTYPE_DELETE;
    bool HasPresentModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_PRESENT;
    bool HasHoldModifier = ArgTypes[I] & OMP_TGT_MAPTYPE_OMPX_HOLD;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
        HstPtrBegin, DataSize, UpdateRef, HasHoldModifier, !IsImplicit,
        ForceDelete, /*FromDataEnd=*/true);
    void *TgtPtrBegin = TPR.TargetPointer;
    if (!TPR.isPresent() && !TPR.isHostPointer() &&
        (DataSize || HasPresentModifier)) {
      ODBG(ODT_Mapping) << "Mapping does not exist ("
                        << (HasPresentModifier ? "'present' map type modifier"
                                               : "ignored")
                        << ")";
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
      ODBG(ODT_Mapping) << "There are " << DataSize
                        << " bytes allocated at target address " << TgtPtrBegin
                        << " - is" << (TPR.Flags.IsLast ? "" : " not")
                        << " last";
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
    if (HasFrom && (HasAlways || TPR.Flags.IsLast) &&
        !TPR.Flags.IsHostPointer && DataSize != 0) {
      ODBG(ODT_Mapping) << "Moving " << DataSize
                        << " bytes (tgt:" << TgtPtrBegin
                        << ") -> (hst:" << HstPtrBegin << ")";
      TIMESCOPE_WITH_DETAILS_AND_IDENT(
          "DevToHost", "Size=" + std::to_string(DataSize) + "B", Loc);
      // Wait for any previous transfer if an event is present.
      if (void *Event = TPR.getEntry()->getEvent()) {
        if (Device.waitEvent(Event, AsyncInfo) != OFFLOAD_SUCCESS) {
          REPORT() << "Failed to wait for event " << Event << ".";
          return OFFLOAD_FAIL;
        }
      }

      Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, DataSize, AsyncInfo,
                                TPR.getEntry());
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT() << "Copying data from device failed.";
        return OFFLOAD_FAIL;
      }

      // As we are expecting to delete the entry the d2h copy might race
      // with another one that also tries to delete the entry. This happens
      // as the entry can be reused and the reuse might happen after the
      // copy-back was issued but before it completed. Since the reuse might
      // also copy-back a value we would race.
      if (TPR.Flags.IsLast) {
        if (TPR.getEntry()->addEventIfNecessary(Device, AsyncInfo) !=
            OFFLOAD_SUCCESS)
          return OFFLOAD_FAIL;
      }
    }

    // Add pointer to the buffer for post-synchronize processing.
    PostProcessingPtrs->emplace_back(HstPtrBegin, DataSize, ArgTypes[I],
                                     std::move(TPR));
    PostProcessingPtrs->back().TPR.getEntry()->unlock();
  }

  // Add post-processing functions
  // TODO: We might want to remove `mutable` in the future by not changing the
  // captured variables somehow.
  AsyncInfo.addPostProcessingFunction([=, Device = &Device]() mutable -> int {
    return postProcessingTargetDataEnd(Device, *PostProcessingPtrs);
  });

  return Ret;
}

static int targetDataContiguous(ident_t *Loc, DeviceTy &Device, void *ArgsBase,
                                void *HstPtrBegin, int64_t ArgSize,
                                int64_t ArgType, AsyncInfoTy &AsyncInfo) {
  TargetPointerResultTy TPR = Device.getMappingInfo().getTgtPtrBegin(
      HstPtrBegin, ArgSize, /*UpdateRefCount=*/false,
      /*UseHoldRefCount=*/false, /*MustContain=*/true);
  void *TgtPtrBegin = TPR.TargetPointer;
  if (!TPR.isPresent()) {
    ODBG(ODT_Mapping) << "hst data:" << HstPtrBegin
                      << " not found, becomes a noop";
    if (ArgType & OMP_TGT_MAPTYPE_PRESENT) {
      MESSAGE("device mapping required by 'present' motion modifier does not "
              "exist for host address " DPxMOD " (%" PRId64 " bytes)",
              DPxPTR(HstPtrBegin), ArgSize);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  if (TPR.Flags.IsHostPointer) {
    ODBG(ODT_Mapping) << "hst data:" << HstPtrBegin
                      << " unified and shared, becomes a noop";
    return OFFLOAD_SUCCESS;
  }

  if (ArgType & OMP_TGT_MAPTYPE_TO) {
    ODBG(ODT_Mapping) << "Moving " << ArgSize << " bytes (hst:" << HstPtrBegin
                      << ") -> (tgt:" << TgtPtrBegin << ")";
    int Ret = Device.submitData(TgtPtrBegin, HstPtrBegin, ArgSize, AsyncInfo,
                                TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT() << "Copying data to device failed.";
      return OFFLOAD_FAIL;
    }
    if (TPR.getEntry()) {
      int Ret = TPR.getEntry()->foreachShadowPointerInfo(
          [&](ShadowPtrInfoTy &ShadowPtr) {
            constexpr int64_t VoidPtrSize = sizeof(void *);
            if (ShadowPtr.PtrSize > VoidPtrSize) {
              ODBG(ODT_Mapping)
                  << "Restoring target descriptor " << ShadowPtr.TgtPtrAddr
                  << " to its original content (" << ShadowPtr.PtrSize
                  << " bytes), containing pointee address "
                  << ShadowPtr.TgtPtrContent.data();
            } else {
              ODBG(ODT_Mapping)
                  << "Restoring target pointer " << ShadowPtr.TgtPtrAddr
                  << " to its original value "
                  << ShadowPtr.TgtPtrContent.data();
            }
            Ret = Device.submitData(ShadowPtr.TgtPtrAddr,
                                    ShadowPtr.TgtPtrContent.data(),
                                    ShadowPtr.PtrSize, AsyncInfo);
            if (Ret != OFFLOAD_SUCCESS) {
              REPORT() << "Copying data to device failed.";
              return OFFLOAD_FAIL;
            }
            return OFFLOAD_SUCCESS;
          });
      if (Ret != OFFLOAD_SUCCESS) {
        ODBG(ODT_Mapping) << "Updating shadow map failed";
        return Ret;
      }
    }
  }

  if (ArgType & OMP_TGT_MAPTYPE_FROM) {
    ODBG(ODT_Mapping) << "Moving " << ArgSize << " bytes (tgt:" << TgtPtrBegin
                      << ") -> (hst:" << HstPtrBegin << ")";
    int Ret = Device.retrieveData(HstPtrBegin, TgtPtrBegin, ArgSize, AsyncInfo,
                                  TPR.getEntry());
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT() << "Copying data from device failed.";
      return OFFLOAD_FAIL;
    }

    // Wait for device-to-host memcopies for whole struct to complete,
    // before restoring the correct host pointer/descriptor.
    if (auto *Entry = TPR.getEntry()) {
      AsyncInfo.addPostProcessingFunction([=]() -> int {
        int Ret = Entry->foreachShadowPointerInfo(
            [&](const ShadowPtrInfoTy &ShadowPtr) {
              constexpr int64_t VoidPtrSize = sizeof(void *);
              if (ShadowPtr.PtrSize > VoidPtrSize) {
                ODBG(ODT_Mapping)
                    << "Restoring host descriptor " << ShadowPtr.HstPtrAddr
                    << " to its original content (" << ShadowPtr.PtrSize
                    << " bytes), containing pointee address "
                    << ShadowPtr.HstPtrContent.data();
              } else {
                ODBG(ODT_Mapping)
                    << "Restoring host pointer " << ShadowPtr.HstPtrAddr
                    << " to its original value "
                    << ShadowPtr.HstPtrContent.data();
              }
              std::memcpy(ShadowPtr.HstPtrAddr, ShadowPtr.HstPtrContent.data(),
                          ShadowPtr.PtrSize);
              return OFFLOAD_SUCCESS;
            });
        Entry->unlock();
        if (Ret != OFFLOAD_SUCCESS) {
          ODBG(ODT_Mapping) << "Updating shadow map failed";
          return Ret;
        }
        return OFFLOAD_SUCCESS;
      });
    }
  }

  return OFFLOAD_SUCCESS;
}

static int targetDataNonContiguous(ident_t *Loc, DeviceTy &Device,
                                   void *ArgsBase,
                                   __tgt_target_non_contig *NonContig,
                                   uint64_t Size, int64_t ArgType,
                                   int CurrentDim, int DimSize, uint64_t Offset,
                                   AsyncInfoTy &AsyncInfo) {
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
    void *Ptr = reinterpret_cast<void *>((char *)ArgsBase + Offset);
    ODBG(ODT_Mapping) << "Transfer of non-contiguous : host ptr " << Ptr
                      << " offset " << Offset << " len " << Size;
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
                     void **ArgMappers, AsyncInfoTy &AsyncInfo,
                     AttachInfoTy *AttachInfo, bool FromMapper) {
  // process each input.
  for (int32_t I = 0; I < ArgNum; ++I) {
    if ((ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) ||
        (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    if (ArgMappers && ArgMappers[I]) {
      // Instead of executing the regular path of targetDataUpdate, call the
      // targetDataMapper variant which will call targetDataUpdate again
      // with new arguments.
      ODBG(ODT_Mapping) << "Calling targetDataMapper for the " << I
                        << "th argument";
      map_var_info_t ArgName = (!ArgNames) ? nullptr : ArgNames[I];
      int Ret = targetDataMapper(Loc, Device, ArgsBase[I], Args[I], ArgSizes[I],
                                 ArgTypes[I], ArgName, ArgMappers[I], AsyncInfo,
                                 targetDataUpdate);

      if (Ret != OFFLOAD_SUCCESS) {
        REPORT() << "Call to targetDataUpdate via targetDataMapper for custom "
                    "mapper failed.";
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
    llvm::offloading::EntryTy *Cur = TransTable->HostTable.EntriesBegin;
    for (uint32_t I = 0; Cur < TransTable->HostTable.EntriesEnd; ++Cur, ++I) {
      if (Cur->Address != HostPtr)
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
    /// Host pointer begin
    char *HstPtrBegin;
    /// Host pointer end
    char *HstPtrEnd;
    /// The index of the element in \p TgtArgs corresponding to the argument
    int Index;
    /// Alignment of the entry (base of the entry, not after the entry).
    uint32_t Alignment;
    /// Size (without alignment, see padding)
    uint32_t Size;
    /// Padding used to align this argument entry, if necessary.
    uint32_t Padding;
    /// Host pointer name
    map_var_info_t HstPtrName = nullptr;
    /// For corresponding-pointer-initialization: host pointee base address.
    void *HstPteeBase = nullptr;
    /// For corresponding-pointer-initialization: host pointee begin address.
    void *HstPteeBegin = nullptr;
    /// Whether this argument needs corresponding-pointer-initialization.
    bool IsCorrespondingPointerInit = false;

    FirstPrivateArgInfoTy(int Index, void *HstPtr, uint32_t Size,
                          uint32_t Alignment, uint32_t Padding,
                          map_var_info_t HstPtrName = nullptr,
                          void *HstPteeBase = nullptr,
                          void *HstPteeBegin = nullptr,
                          bool IsCorrespondingPointerInit = false)
        : HstPtrBegin(reinterpret_cast<char *>(HstPtr)),
          HstPtrEnd(HstPtrBegin + Size), Index(Index), Alignment(Alignment),
          Size(Size), Padding(Padding), HstPtrName(HstPtrName),
          HstPteeBase(HstPteeBase), HstPteeBegin(HstPteeBegin),
          IsCorrespondingPointerInit(IsCorrespondingPointerInit) {}
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

  /// \returns the value of the target pointee's base to be used for
  /// corresponding-pointer-initialization.
  void *getTargetPointeeBaseForCorrespondingPointerInitialization(
      void *HstPteeBase, void *HstPteeBegin) {
    // See if the pointee's begin address has corresponding storage on device.
    void *TgtPteeBegin = [&]() -> void * {
      if (!HstPteeBegin) {
        ODBG(ODT_Mapping)
            << "Corresponding-pointer-initialization: pointee begin address is "
               "null";
        return nullptr;
      }

      return Device.getMappingInfo()
          .getTgtPtrBegin(HstPteeBegin, /*Size=*/0, /*UpdateRefCount=*/false,
                          /*UseHoldRefCount=*/false)
          .TargetPointer;
    }();

    // If it does, we calculate target pointee base using it, and return it.
    // Otherwise, we retain the host pointee's base as the target pointee base
    // of the initialized pointer. It's the user's responsibility to ensure
    // that if a lookup fails, the host pointee is accessible on the device.
    return TgtPteeBegin ? calculateTargetPointeeBase(HstPteeBase, HstPteeBegin,
                                                     TgtPteeBegin)
                        : HstPteeBase;
  }

  /// Initialize the source buffer for corresponding-pointer-initialization.
  ///
  /// It computes and stores the target pointee base address (or the host
  /// pointee's base address, if lookup of target pointee fails) to the first
  /// `sizeof(void*)` bytes of \p Buffer, and for larger pointers
  /// (Fortran descriptors), the remaining fields of the host descriptor
  /// \p HstPtr after those `sizeof(void*)` bytes.
  ///
  /// Corresponding-pointer-initialization represents the initialization of the
  /// private version of a base-pointer/referring-pointer on a target construct.
  ///
  /// For example, for the following test:
  /// ```cpp
  ///   int x[10];
  ///   int *px = &x[0];
  ///   ...
  ///   #pragma omp target data map(tofrom:px)
  ///   {
  ///     int **ppx = omp_get_mapped_ptr(&px, omp_get_default_device());
  ///     #pragma omp target map(tofrom:px[1]) is_device_ptr(ppx)
  ///     {
  ///        foo(px, ppx);
  ///     }
  ///   }
  /// ```
  /// The following shows a possible way to implement the mapping of `px`,
  /// which is pre-determined firstprivate and should get initialized
  /// via corresponding-pointer-initialization:
  ///
  /// (A) Possible way to implement the above with PRIVATE | ATTACH:
  /// ```llvm
  ///  ; maps for px:
  ///  ; &px[0], &px[1], sizeof(px[1]), TO | FROM                // (1)
  ///  ; &px,    &px[1], sizeof(px),    ATTACH                   // (2)
  ///  ; &px,    &px[1], sizeof(px),    PRIVATE | ATTACH | PARAM // (3)
  ///  call... @__omp_outlined...(ptr %px, ptr %ppx)
  ///  define ... @__omp_outlined(ptr %px, ptr %ppx) {...
  ///    foo(%px, %ppx)
  ///  ...}
  /// ```
  /// `(1)` maps the pointee `px[1].
  /// `(2)` attaches it to the mapped version of `px`. It can be controlled by
  /// the user based on the `attach(auto/always/never)` map-type modifier.
  /// `(3)` privatizes and initializes the private pointer `px`, and passes it
  /// into the kernel as the argument `%px`. Can be skipped if `px` is not
  /// referenced in the target construct.
  ///
  /// While this method is not too beneficial compared to just doing the
  /// initialization in the body of the kernel, like:
  /// (B) Possible way to implement the above without PRIVATE | ATTACH:
  /// ```llvm
  ///  ; maps for px:
  ///  ; &px[0], &px[1], sizeof(px[1]), TO | FROM | PARAM        // (4)
  ///  ; &px,    &px[1], sizeof(px),    ATTACH                   // (5)
  ///  call... @__omp_outlined...(ptr %px0, ptr %ppx)
  ///  define ... __omp_outlined...(ptr %px0, ptr %ppx) {
  ///    %px = alloca ptr;
  ///    store ptr %px0, ptr %px
  ///    foo(%px, %ppx)
  ///  }
  /// ```
  ///
  /// (B) is not so convenient for Fortran descriptors, because in
  /// addition to the lookup, the remaining fields of the descriptor have
  /// to be passed into the kernel to initialize the private copy, which
  /// makes (A) a cleaner option for them. e.g.
  /// ```f90
  /// integer, pointer :: p(:)
  /// !$omp target map(p(1))
  /// ```
  ///
  /// (C) Possible mapping for the above Fortran test using PRIVATE | ATTACH:
  /// ```llvm
  ///  ; maps for p:
  ///  ; &p(1),       &p(1), sizeof(p(1)),       TO | FROM
  ///  ; &ref_ptr(p), &p(1), sizeof(ref_ptr(p)), ATTACH
  ///  ; &ref_ptr(p), &p(1), sizeof(ref_ptr(p)), PRIVATE | ATTACH | PARAM
  ///  call... @__omp_outlined...(ptr %ref_ptr_of_p)
  void initBufferForCorrespondingPointerInitialization(char *Buffer,
                                                       void *HstPtr,
                                                       int64_t HstPtrSize,
                                                       void *HstPteeBase,
                                                       void *HstPteeBegin) {
    constexpr int64_t VoidPtrSize = sizeof(void *);
    assert(HstPtrSize >= VoidPtrSize &&
           "corresponding-pointer-initialization: pointer size is too small");

    void *TgtPteeBase =
        getTargetPointeeBaseForCorrespondingPointerInitialization(HstPteeBase,
                                                                  HstPteeBegin);

    // Store the target pointee base address to the first VoidPtrSize bytes
    ODBG(ODT_Mapping)
        << "Corresponding-pointer-initialization: setting target pointee base "
           "for "
        << HstPtr << ", with pointee base " << TgtPteeBase;
    std::memcpy(Buffer, &TgtPteeBase, VoidPtrSize);
    if (HstPtrSize <= VoidPtrSize)
      return;

    // For Fortran descriptors, copy the remaining descriptor fields from host
    uint64_t HstDescriptorFieldsSize = HstPtrSize - VoidPtrSize;
    void *HstDescriptorFieldsAddr = static_cast<char *>(HstPtr) + VoidPtrSize;
    ODBG(ODT_Mapping) << "Corresponding-pointer-initialization: copying "
                      << HstDescriptorFieldsSize
                      << " bytes of descriptor fields into buffer at offset "
                      << VoidPtrSize << ", from " << HstDescriptorFieldsAddr;
    std::memcpy(Buffer + VoidPtrSize, HstDescriptorFieldsAddr,
                HstDescriptorFieldsSize);
  }

  /// Helper function to create and initialize a buffer to be used as the source
  /// for corresponding-pointer-initialization.
  void *createAndInitSourceBufferForCorrespondingPointerInitialization(
      void *HstPtr, int64_t HstPtrSize, void *HstPteeBase, void *HstPteeBegin) {
    char *Buffer = getOrCreateSourceBufferForSubmitData(AsyncInfo, HstPtrSize);
    initBufferForCorrespondingPointerInitialization(Buffer, HstPtr, HstPtrSize,
                                                    HstPteeBase, HstPteeBegin);
    return Buffer;
  }

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
             map_var_info_t HstPtrName = nullptr,
             const bool AllocImmediately = false, void *HstPteeBase = nullptr,
             void *HstPteeBegin = nullptr,
             bool IsCorrespondingPointerInit = false) {
    // If the argument is not first-private, or its size is greater than a
    // predefined threshold, we will allocate memory and issue the transfer
    // immediately.
    if (ArgSize > FirstPrivateArgSizeThreshold || !IsFirstPrivate ||
        AllocImmediately) {
      TgtPtr = Device.allocData(ArgSize, HstPtr);
      if (!TgtPtr) {
        ODBG(ODT_Alloc) << "Data allocation for "
                        << (IsFirstPrivate ? "first-" : "") << "private array "
                        << HstPtr << " failed.";
        return OFFLOAD_FAIL;
      }

      ODBG(ODT_Alloc) << "Allocated " << ArgSize
                      << " bytes of target memory at " << TgtPtr << " for "
                      << (IsFirstPrivate ? "first-" : "") << "private array "
                      << HstPtr << " - pushing target argument "
                      << (void *)((intptr_t)TgtPtr + ArgOffset);

      // If first-private, copy data from host
      if (IsFirstPrivate) {
        ODBG(ODT_Mapping) << "Submitting firstprivate data to the device.";

        // The source value used for corresponding-pointer-initialization
        // is different vs regular firstprivates.
        void *DataSource =
            IsCorrespondingPointerInit
                ? createAndInitSourceBufferForCorrespondingPointerInitialization(
                      HstPtr, ArgSize, HstPteeBase, HstPteeBegin)
                : HstPtr;
        int Ret = Device.submitData(TgtPtr, DataSource, ArgSize, AsyncInfo);
        if (Ret != OFFLOAD_SUCCESS) {
          ODBG(ODT_Mapping) << "Copying "
                            << (IsCorrespondingPointerInit
                                    ? "corresponding-pointer-initialization"
                                    : "firstprivate")
                            << " data to device failed.";
          return OFFLOAD_FAIL;
        }
      }
      TgtPtrs.push_back(TgtPtr);
    } else {
      ODBG(ODT_Mapping) << "Firstprivate array " << HstPtr << " of size "
                        << ArgSize << " will be packed";
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
      auto *LastFPArgInfo =
          FirstPrivateArgInfo.empty() ? nullptr : &FirstPrivateArgInfo.back();

      // Compute the start alignment of this entry, add padding if necessary.
      // TODO: Consider sorting instead.
      uint32_t Padding = 0;
      uint32_t StartAlignment =
          LastFPArgInfo ? LastFPArgInfo->Alignment : MaxAlignment;
      if (LastFPArgInfo) {
        // Check if we keep the start alignment or if it is shrunk due to the
        // size of the last element.
        uint32_t Offset = LastFPArgInfo->Size % StartAlignment;
        if (Offset)
          StartAlignment = Offset;
        // We only need as much alignment as the host pointer had (since we
        // don't know the alignment information from the source we might end up
        // overaligning accesses but not too much).
        uint32_t RequiredAlignment =
            llvm::bit_floor(getPartialStructRequiredAlignment(HstPtr));
        if (RequiredAlignment > StartAlignment) {
          Padding = RequiredAlignment - StartAlignment;
          StartAlignment = RequiredAlignment;
        }
      }

      FirstPrivateArgInfo.emplace_back(
          TgtArgsIndex, HstPtr, ArgSize, StartAlignment, Padding, HstPtrName,
          HstPteeBase, HstPteeBegin, IsCorrespondingPointerInit);

      FirstPrivateArgSize += Padding + ArgSize;
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
      auto *Itr = FirstPrivateArgBuffer.begin();
      // Copy all host data to this buffer
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        // First pad the pointer as we (have to) pad it on the device too.
        Itr = std::next(Itr, Info.Padding);

        if (Info.IsCorrespondingPointerInit)
          initBufferForCorrespondingPointerInitialization(
              &*Itr, Info.HstPtrBegin, Info.Size, Info.HstPteeBase,
              Info.HstPteeBegin);
        else
          std::copy(Info.HstPtrBegin, Info.HstPtrEnd, Itr);
        Itr = std::next(Itr, Info.Size);
      }
      // Allocate target memory
      void *TgtPtr =
          Device.allocData(FirstPrivateArgSize, FirstPrivateArgBuffer.data());
      if (TgtPtr == nullptr) {
        ODBG(ODT_Alloc)
            << "Failed to allocate target memory for private arguments.";
        return OFFLOAD_FAIL;
      }
      TgtPtrs.push_back(TgtPtr);
      ODBG(ODT_Alloc) << "Allocated " << FirstPrivateArgSize
                      << " bytes of target memory at " << TgtPtr;
      // Transfer data to target device
      int Ret = Device.submitData(TgtPtr, FirstPrivateArgBuffer.data(),
                                  FirstPrivateArgSize, AsyncInfo);
      if (Ret != OFFLOAD_SUCCESS) {
        ODBG(ODT_DataTransfer) << "Failed to submit data of private arguments.";
        return OFFLOAD_FAIL;
      }
      // Fill in all placeholder pointers
      auto TP = reinterpret_cast<uintptr_t>(TgtPtr);
      for (FirstPrivateArgInfoTy &Info : FirstPrivateArgInfo) {
        void *&Ptr = TgtArgs[Info.Index];
        assert(Ptr == nullptr && "Target pointer is already set by mistaken");
        // Pad the device pointer to get the right alignment.
        TP += Info.Padding;
        Ptr = reinterpret_cast<void *>(TP);
        TP += Info.Size;
        ODBG(ODT_Mapping) << "Firstprivate array " << Info.HstPtrBegin
                          << " of size " << (Info.HstPtrEnd - Info.HstPtrBegin)
                          << " mapped to " << Ptr;
      }
    }

    return OFFLOAD_SUCCESS;
  }

  /// Free all target memory allocated for private arguments
  int free() {
    for (void *P : TgtPtrs) {
      int Ret = Device.deleteData(P);
      if (Ret != OFFLOAD_SUCCESS) {
        ODBG(ODT_Alloc) << "Deallocation of (first-)private arrays failed.";
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

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  // Create AttachInfo for tracking any ATTACH entries, or new-allocations
  // when handling the "begin" mapping for a target constructs.
  AttachInfoTy AttachInfo;

  int Ret = targetDataBegin(Loc, *DeviceOrErr, ArgNum, ArgBases, Args, ArgSizes,
                            ArgTypes, ArgNames, ArgMappers, AsyncInfo,
                            &AttachInfo, false /*FromMapper=*/);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT() << "Call to targetDataBegin failed, abort target.";
    return OFFLOAD_FAIL;
  }

  // Process collected ATTACH entries
  if (!AttachInfo.AttachEntries.empty()) {
    Ret = processAttachEntries(*DeviceOrErr, AttachInfo, AsyncInfo);
    if (Ret != OFFLOAD_SUCCESS) {
      REPORT() << "Failed to process ATTACH entries.";
      return OFFLOAD_FAIL;
    }
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
        void *TgtPtrBase =
            (void *)((intptr_t)TgtArgs[TgtIdx] + TgtOffsets[TgtIdx]);
        ODBG(ODT_Mapping) << "Parent lambda base " << TgtPtrBase;
        uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
        void *TgtPtrBegin = (void *)((uintptr_t)TgtPtrBase + Delta);
        void *&PointerTgtPtrBegin = AsyncInfo.getVoidPtrLocation();
        TargetPointerResultTy TPR =
            DeviceOrErr->getMappingInfo().getTgtPtrBegin(
                HstPtrVal, ArgSizes[I], /*UpdateRefCount=*/false,
                /*UseHoldRefCount=*/false);
        PointerTgtPtrBegin = TPR.TargetPointer;
        if (!TPR.isPresent()) {
          ODBG(ODT_Mapping) << "No lambda captured variable mapped "
                            << HstPtrVal << " - ignored";
          continue;
        }
        if (TPR.Flags.IsHostPointer) {
          ODBG(ODT_Mapping)
              << "Unified memory is active, no need to map lambda captured"
                 "variable ("
              << HstPtrVal << ")";
          continue;
        }
        ODBG(ODT_Mapping) << "Update lambda reference (" << PointerTgtPtrBegin
                          << ") -> [" << TgtPtrBegin << "]";
        Ret =
            DeviceOrErr->submitData(TgtPtrBegin, &PointerTgtPtrBegin,
                                    sizeof(void *), AsyncInfo, TPR.getEntry());
        if (Ret != OFFLOAD_SUCCESS) {
          REPORT() << "Copying data to device failed.";
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
    TargetPointerResultTy TPR;
    if (ArgTypes[I] & OMP_TGT_MAPTYPE_LITERAL) {
      ODBG(ODT_Mapping) << "Forwarding first-private value " << HstPtrBase
                        << " to the target construct";
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (ArgTypes[I] & OMP_TGT_MAPTYPE_PRIVATE) {
      // For cases like:
      // ```
      // int *p = ...;
      // #pragma omp target map(p[0:10])
      // ```
      // `p` is predetermined firstprivate on the target construct, and the
      // method to determine the initial value of the private copy on the
      // device is called "corresponding-pointer-initialization".
      //
      // Such firstprivate pointers that need
      // corresponding-pointer-initialization are represented using the
      // `PRIVATE | ATTACH` map-types, in contrast to regular firstprivate
      // entries, which use `PRIVATE | TO`. The structure of these
      // `PRIVATE | ATTACH` entries is the same as the non-private
      // `ATTACH` entries used to represent pointer-attachments, i.e.:
      // ```
      //  &hst_ptr_base/begin, &hst_ptee_begin, sizeof(hst_ptr)
      // ```
      const bool IsAttach = (ArgTypes[I] & OMP_TGT_MAPTYPE_ATTACH);
      void *HstPteeBase = nullptr;
      void *HstPteeBegin = nullptr;
      if (IsAttach) {
        // For corresponding-pointer-initialization, Args[I] is HstPteeBegin,
        // and ArgBases[I] is both HstPtrBase/HstPtrBegin.
        HstPteeBase = *reinterpret_cast<void **>(HstPtrBase);
        HstPteeBegin = Args[I];
        HstPtrBegin = ArgBases[I];
      }
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
      // Corresponding-pointer-initialization is a special case of firstprivate,
      // since it also involves initializing the private pointer.
      const bool IsFirstPrivate =
          (ArgTypes[I] & OMP_TGT_MAPTYPE_TO) || IsAttach;

      // If there is a next argument and it depends on the current one, we need
      // to allocate the private memory immediately. If this is not the case,
      // then the argument can be marked for optimization and packed with the
      // other privates.
      const bool AllocImmediately =
          (I < ArgNum - 1 && (ArgTypes[I + 1] & OMP_TGT_MAPTYPE_MEMBER_OF));
      Ret = PrivateArgumentManager.addArg(
          HstPtrBegin, ArgSizes[I], TgtBaseOffset, IsFirstPrivate, TgtPtrBegin,
          /*TgtArgsIndex=*/TgtArgs.size(), HstPtrName, AllocImmediately,
          HstPteeBase, HstPteeBegin, /*IsCorrespondingPointerInit=*/IsAttach);
      if (Ret != OFFLOAD_SUCCESS) {
        REPORT() << "Failed to process "
                 << (IsAttach ? "corresponding-pointer-initialization " : "")
                 << (IsFirstPrivate ? "first-" : "") << "private argument "
                 << HstPtrBegin << ".";
        return OFFLOAD_FAIL;
      }
    } else {
      if (ArgTypes[I] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        HstPtrBase = *reinterpret_cast<void **>(HstPtrBase);
      TPR = DeviceOrErr->getMappingInfo().getTgtPtrBegin(
          HstPtrBegin, ArgSizes[I],
          /*UpdateRefCount=*/false,
          /*UseHoldRefCount=*/false);
      TgtPtrBegin = TPR.TargetPointer;
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      ODBG(ODT_Mapping) << "Obtained target argument " << TgtPtrBase
                        << " from host pointer " << HstPtrBegin;
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
    ODBG(ODT_Mapping) << "Failed to pack and transfer first private arguments";
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

  auto DeviceOrErr = PM->getDevice(DeviceId);
  if (!DeviceOrErr)
    FATAL_MESSAGE(DeviceId, "%s", toString(DeviceOrErr.takeError()).c_str());

  // Move data from device.
  int Ret = targetDataEnd(Loc, *DeviceOrErr, ArgNum, ArgBases, Args, ArgSizes,
                          ArgTypes, ArgNames, ArgMappers, AsyncInfo);
  if (Ret != OFFLOAD_SUCCESS) {
    REPORT() << "Call to targetDataEnd failed, abort target.";
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
          REPORT() << "Failed to deallocate target memory for private args";
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
    REPORT() << "Host ptr " << HostPtr
             << " does not have a matching target pointer.";
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

  ODBG(ODT_Kernel) << "loop trip count is " << KernelArgs.Tripcount;

  // We need to keep bases and offsets separate. Sometimes (e.g. in OpenCL) we
  // need to manifest base pointers prior to launching a kernel. Even if we have
  // mapped an object only partially, e.g. A[N:M], although the kernel is
  // expected to access elements starting at address &A[N] and beyond, we still
  // need to manifest the base of the array &A[0]. In other cases, e.g. the COI
  // API, we need the begin address itself, i.e. &A[N], as the API operates on
  // begin addresses, not bases. That's why we pass args and offsets as two
  // separate entities so that each plugin can do what it needs. This behavior
  // was introduced via https://reviews.llvm.org/D33028 and commit 1546d319244c.
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
      REPORT() << "Failed to process data before launching the kernel.";
      return OFFLOAD_FAIL;
    }

    // Clang might pass more values via the ArgPtrs to the runtime that we pass
    // on to the kernel.
    // TODO: Next time we adjust the KernelArgsTy we should introduce a new
    // NumKernelArgs field.
    KernelArgs.NumArgs = TgtArgs.size();
  }

  // Launch device execution.
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
  ODBG(ODT_Kernel) << "Launching target execution "
                   << TargetTable->EntriesBegin[TM->Index].SymbolName
                   << " with pointer " << TgtEntryPtr << " (index=" << TM->Index
                   << ").";

  {
    assert(KernelArgs.NumArgs == TgtArgs.size() && "Argument count mismatch!");
    TIMESCOPE_WITH_DETAILS_AND_IDENT(
        "Kernel Target",
        "NumArguments=" + std::to_string(KernelArgs.NumArgs) +
            ";NumTeams=" + std::to_string(KernelArgs.NumTeams[0]) +
            ";TripCount=" + std::to_string(KernelArgs.Tripcount),
        Loc);

#ifdef OMPT_SUPPORT
    /// RAII to establish tool anchors before and after kernel launch
    int32_t NumTeams = KernelArgs.NumTeams[0];
    // No need to guard this with OMPT_IF_BUILT
    InterfaceRAII TargetSubmitRAII(
        RegionInterface.getCallbacks<ompt_callback_target_submit>(), NumTeams);
#endif

    Ret = Device.launchKernel(TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(),
                              KernelArgs, AsyncInfo);
  }

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT() << "Executing target region abort target.";
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
      REPORT() << "Failed to process data after launching the kernel.";
      return OFFLOAD_FAIL;
    }
  }

  return OFFLOAD_SUCCESS;
}

/// Enables the record replay mechanism by pre-allocating MemorySize
/// and informing the record-replayer of whether to store the output
/// in some file.
int target_activate_rr(DeviceTy &Device, uint64_t MemorySize, void *VAddr,
                       bool IsRecord, bool SaveOutput,
                       uint64_t &ReqPtrArgOffset) {
  return Device.RTL->initialize_record_replay(Device.DeviceID, MemorySize,
                                              VAddr, IsRecord, SaveOutput,
                                              ReqPtrArgOffset);
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
    REPORT() << "Host ptr " << HostPtr
             << " does not have a matching target pointer.";
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
  void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].Address;
  ODBG(ODT_Kernel) << "Launching target execution "
                   << TargetTable->EntriesBegin[TM->Index].SymbolName
                   << " with pointer " << TgtEntryPtr << " (index=" << TM->Index
                   << ").";

  void *TgtPtr = Device.allocData(DeviceMemorySize, /*HstPtr=*/nullptr,
                                  TARGET_ALLOC_DEFAULT);
  Device.submitData(TgtPtr, DeviceMemory, DeviceMemorySize, AsyncInfo);

  KernelArgsTy KernelArgs{};
  KernelArgs.Version = OMP_KERNEL_ARG_VERSION;
  KernelArgs.NumArgs = NumArgs;
  KernelArgs.Tripcount = LoopTripCount;
  KernelArgs.NumTeams[0] = NumTeams;
  KernelArgs.ThreadLimit[0] = ThreadLimit;

  int Ret = Device.launchKernel(TgtEntryPtr, TgtArgs, TgtOffsets, KernelArgs,
                                AsyncInfo);

  if (Ret != OFFLOAD_SUCCESS) {
    REPORT() << "Executing target region abort target.";
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}
