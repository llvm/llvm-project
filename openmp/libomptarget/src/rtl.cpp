//===----------- rtl.cpp - Target independent OpenMP target RTL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/OffloadBinary.h"
#include "llvm/OffloadArch/OffloadArch.h"

#include "OpenMP/OMPT/Callback.h"
#include "PluginManager.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include "Shared/Debug.h"
#include "Shared/Profile.h"
#include "Shared/Utils.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <linux/limits.h>

#include <mutex>
#include <string>

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::omp::target;

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* AMDGPU target        */ "libomptarget.rtl.amdgpu",
    /* CUDA target          */ "libomptarget.rtl.cuda",
    /* x86_64 target        */ "libomptarget.rtl.x86_64",
    /* PowerPC target       */ "libomptarget.rtl.ppc64",
    /* AArch64 target       */ "libomptarget.rtl.aarch64",
};

#ifdef OMPT_SUPPORT
extern void ompt::connectLibrary();
#endif

__attribute__((constructor(101))) void init() {
  DP("Init target library!\n");

  PM = new PluginManager();

#ifdef OMPT_SUPPORT
  // Initialize OMPT first
  ompt::connectLibrary();
#endif

  Profiler::get();
  PM->RTLs.loadRTLs();
  PM->registerDelayedLibraries();
}

__attribute__((destructor(101))) void deinit() {
  DP("Deinit target library!\n");
  delete PM;
}

void PluginAdaptorManagerTy::loadRTLs() {
  // FIXME this is amdgcn specific.
  // Propogate HIP_VISIBLE_DEVICES if set to ROCR_VISIBLE_DEVICES.
  if (char *hipVisDevs = getenv("HIP_VISIBLE_DEVICES")) {
    if (char *rocrVisDevs = getenv("ROCR_VISIBLE_DEVICES")) {
      if (strcmp(hipVisDevs, rocrVisDevs) != 0)
        fprintf(stderr,
                "Warning both HIP_VISIBLE_DEVICES %s "
                "and ROCR_VISIBLE_DEVICES %s set\n",
                hipVisDevs, rocrVisDevs);
    }
  }

  // Parse environment variable OMP_TARGET_OFFLOAD (if set)
  PM->TargetOffloadPolicy =
      (kmp_target_offload_kind_t)__kmpc_get_target_offload();
  if (PM->TargetOffloadPolicy == tgt_disabled) {
    return;
  }

  DP("Loading RTLs...\n");
  BoolEnvar UseFirstGoodRTL("LIBOMPTARGET_USE_FIRST_GOOD_RTL", false);

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (const char *Name : RTLNames) {
    AllRTLs.emplace_back();

    PluginAdaptorTy &RTL = AllRTLs.back();

    const std::string BaseRTLName(Name);

    if (!attemptLoadRTL(BaseRTLName + ".so", RTL))
      AllRTLs.pop_back();
  }

  DP("RTLs loaded!\n");
}

bool PluginAdaptorManagerTy::attemptLoadRTL(const std::string &RTLName, PluginAdaptorTy &RTL) {
  const char *Name = RTLName.c_str();

  DP("Loading library '%s'...\n", Name);

  std::string ErrMsg;
  auto DynLibrary = std::make_unique<sys::DynamicLibrary>(
      sys::DynamicLibrary::getPermanentLibrary(Name, &ErrMsg));

  if (!DynLibrary->isValid()) {
    // Library does not exist or cannot be found.
    DP("Unable to load library '%s': %s!\n", Name, ErrMsg.c_str());
    return false;
  }

  DP("Successfully loaded library '%s'!\n", Name);

#define PLUGIN_API_HANDLE(NAME, MANDATORY)                                     \
  *((void **)&RTL.NAME) =                                                      \
      DynLibrary->getAddressOfSymbol(GETNAME(__tgt_rtl_##NAME));               \
  if (MANDATORY && !RTL.NAME) {                                                \
    DP("Invalid plugin as necessary interface is not found %s\n", #NAME);      \
    return false;                                                              \
  }

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  // Remove plugin on failure to call optional init_plugin
  int32_t Rc = RTL.init_plugin();
  if (Rc != OFFLOAD_SUCCESS) {
    DP("Unable to initialize library '%s': %u!\n", Name, Rc);
    return false;
  }

  // No devices are supported by this RTL?
  if (!(RTL.NumberOfDevices = RTL.number_of_devices())) {
    // The RTL is invalid! Will pop the object from the RTLs list.
    DP("No devices supported in this RTL\n");
    return false;
  }

#ifdef OMPTARGET_DEBUG
  RTL.RTLName = Name;
#endif

  DP("Registering RTL %s supporting %d devices!\n", Name, RTL.NumberOfDevices);

  RTL.LibraryHandler = std::move(DynLibrary);

  // Successfully loaded
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering libs

static void registerImageIntoTranslationTable(TranslationTable &TT,
                                              PluginAdaptorTy &RTL,
                                              __tgt_device_image *Image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.Idx + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t I = 0; I < RTL.NumberOfDevices; ++I) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.Idx + I] != Image) {
      TT.TargetsImages[RTL.Idx + I] = Image;
      TT.TargetsTable[RTL.Idx + I] = 0; // lazy initialization of target table.
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering Ctors/Dtors

static void registerGlobalCtorsDtorsForImage(__tgt_bin_desc *Desc,
                                             __tgt_device_image *Img,
                                             PluginAdaptorTy *RTL) {

  for (int32_t I = 0; I < RTL->NumberOfDevices; ++I) {
    DeviceTy &Device = *PM->Devices[RTL->Idx + I];
    Device.PendingGlobalsMtx.lock();
    Device.HasPendingGlobals = true;
    for (__tgt_offload_entry *Entry = Img->EntriesBegin;
         Entry != Img->EntriesEnd; ++Entry) {
      // Globals are not callable and use a different set of flags.
      if (Entry->size != 0)
        continue;

      if (Entry->flags & OMP_DECLARE_TARGET_CTOR) {
        DP("Adding ctor " DPxMOD " to the pending list.\n",
           DPxPTR(Entry->addr));
        Device.PendingCtorsDtors[Desc].PendingCtors.push_back(Entry->addr);
        MESSAGE("WARNING: Calling deprecated constructor for entry %s will be "
                "removed in a future release \n",
                Entry->name);
      } else if (Entry->flags & OMP_DECLARE_TARGET_DTOR) {
        // Dtors are pushed in reverse order so they are executed from end
        // to beginning when unregistering the library!
        DP("Adding dtor " DPxMOD " to the pending list.\n",
           DPxPTR(Entry->addr));
        Device.PendingCtorsDtors[Desc].PendingDtors.push_front(Entry->addr);
        MESSAGE("WARNING: Calling deprecated destructor for entry %s will be "
                "removed in a future release \n",
                Entry->name);
      }

      if (Entry->flags & OMP_DECLARE_TARGET_LINK) {
        DP("The \"link\" attribute is not yet supported!\n");
      }
    }
    Device.PendingGlobalsMtx.unlock();
  }
}

static __tgt_device_image getExecutableImage(__tgt_device_image *Image) {
  StringRef ImageStr(static_cast<char *>(Image->ImageStart),
                     static_cast<char *>(Image->ImageEnd) -
                         static_cast<char *>(Image->ImageStart));
  auto BinaryOrErr =
      object::OffloadBinary::create(MemoryBufferRef(ImageStr, ""));
  if (!BinaryOrErr) {
    consumeError(BinaryOrErr.takeError());
    return *Image;
  }

  void *Begin = const_cast<void *>(
      static_cast<const void *>((*BinaryOrErr)->getImage().bytes_begin()));
  void *End = const_cast<void *>(
      static_cast<const void *>((*BinaryOrErr)->getImage().bytes_end()));

  return {Begin, End, Image->EntriesBegin, Image->EntriesEnd};
}

static __tgt_image_info getImageInfo(__tgt_device_image *Image) {
  StringRef ImageStr(static_cast<char *>(Image->ImageStart),
                     static_cast<char *>(Image->ImageEnd) -
                         static_cast<char *>(Image->ImageStart));
  auto BinaryOrErr =
      object::OffloadBinary::create(MemoryBufferRef(ImageStr, ""));
  if (!BinaryOrErr) {
    consumeError(BinaryOrErr.takeError());
    return __tgt_image_info{};
  }
  return __tgt_image_info{(*BinaryOrErr)->getArch().data()};
}

void PluginAdaptorManagerTy::registerRequires(int64_t Flags) {
  // TODO: add more elaborate check.
  // Minimal check: only set requires flags if previous value
  // is undefined. This ensures that only the first call to this
  // function will set the requires flags. All subsequent calls
  // will be checked for compatibility.
  assert(Flags != OMP_REQ_UNDEFINED &&
         "illegal undefined flag for requires directive!");
  if (RequiresFlags == OMP_REQ_UNDEFINED) {
    RequiresFlags = Flags;
    return;
  }

  // If multiple compilation units are present enforce
  // consistency across all of them for require clauses:
  //  - reverse_offload
  //  - unified_address
  //  - unified_shared_memory
  if ((RequiresFlags & OMP_REQ_REVERSE_OFFLOAD) !=
      (Flags & OMP_REQ_REVERSE_OFFLOAD)) {
    FATAL_MESSAGE0(
        1, "'#pragma omp requires reverse_offload' not used consistently!");
  }
  if ((RequiresFlags & OMP_REQ_UNIFIED_ADDRESS) !=
      (Flags & OMP_REQ_UNIFIED_ADDRESS)) {
    FATAL_MESSAGE0(
        1, "'#pragma omp requires unified_address' not used consistently!");
  }
  if ((RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) !=
      (Flags & OMP_REQ_UNIFIED_SHARED_MEMORY)) {
    FATAL_MESSAGE0(
        1,
        "'#pragma omp requires unified_shared_memory' not used consistently!");
  }

  // TODO: insert any other missing checks

  DP("New requires flags %" PRId64 " compatible with existing %" PRId64 "!\n",
     Flags, RequiresFlags);
}

void PluginAdaptorManagerTy::initRTLonce(PluginAdaptorTy &R) {
  // If this RTL is not already in use, initialize it.
  if (R.IsUsed || !R.NumberOfDevices)
    return;

  // Initialize the device information for the RTL we are about to use.
  const size_t Start = PM->Devices.size();
  PM->Devices.reserve(Start + R.NumberOfDevices);
  for (int32_t DeviceId = 0; DeviceId < R.NumberOfDevices; DeviceId++) {
    PM->Devices.push_back(std::make_unique<DeviceTy>(&R));
    // global device ID
    PM->Devices[Start + DeviceId]->DeviceID = Start + DeviceId;
    // RTL local device ID
    PM->Devices[Start + DeviceId]->RTLDeviceID = DeviceId;
  }

  // Initialize the index of this RTL and save it in the used RTLs.
  R.Idx = Start;
  R.IsUsed = true;
  UsedRTLs.push_back(&R);

  // If possible, set the device identifier offset
  if (R.set_device_offset)
    R.set_device_offset(Start);

  DP("RTL " DPxMOD " has index %d!\n", DPxPTR(R.LibraryHandler.get()), R.Idx);
}

void PluginAdaptorManagerTy::initAllRTLs() {
  for (auto &R : AllRTLs)
    initRTLonce(R);
}

void PluginAdaptorManagerTy::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Extract the exectuable image and extra information if availible.
  std::list<std::pair<__tgt_device_image, __tgt_image_info> *> RemainingImages;

  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i) {
    PM->Images.emplace_back(getExecutableImage(&Desc->DeviceImages[i]),
                            getImageInfo(&Desc->DeviceImages[i]));
    RemainingImages.push_back(&PM->Images.back());
  }

  // Register the images with the RTLs that understand them, if any.
  for (auto &ImageAndInfo : AllRTLs) {
    // Obtain the image and information that was previously extracted.
    PluginAdaptorTy *FoundRTL = nullptr;

    std::list<std::pair<__tgt_device_image, __tgt_image_info> *>
        AvailableImages;

    if (ImageAndInfo.exists_valid_binary_for_RTL(&RemainingImages, &AvailableImages)) {
      // Obtain the image and information that was previously extracted.
      for (auto AvailImage : AvailableImages) {
        __tgt_device_image *Img = &AvailImage->first;

        DP("Image " DPxMOD " is compatible with RTL %s!\n",
           DPxPTR(Img->ImageStart), ImageAndInfo.RTLName.c_str());

        initRTLonce(ImageAndInfo);

        // Initialize (if necessary) translation table for this library.
        PM->TrlTblMtx.lock();
        if (!PM->HostEntriesBeginToTransTable.count(Desc->HostEntriesBegin)) {
          PM->HostEntriesBeginRegistrationOrder.push_back(
              Desc->HostEntriesBegin);
          TranslationTable &TransTable =
              (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];
          TransTable.HostTable.EntriesBegin = Desc->HostEntriesBegin;
          TransTable.HostTable.EntriesEnd = Desc->HostEntriesEnd;
        }

        // Retrieve translation table for this library.
        TranslationTable &TransTable =
            (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];

        DP("Registering image " DPxMOD " with RTL %s!\n",
           DPxPTR(Img->ImageStart), ImageAndInfo.RTLName.c_str());
        registerImageIntoTranslationTable(TransTable, ImageAndInfo, Img);
        ImageAndInfo.UsedImages.insert(Img);

        PM->TrlTblMtx.unlock();
        FoundRTL = &ImageAndInfo;

        // Load ctors/dtors for static objects
        registerGlobalCtorsDtorsForImage(Desc, Img, FoundRTL);
      }

      if (RemainingImages.empty())
        break;
    }

#ifdef OMPTARGET_DEBUG
    for (auto Img : RemainingImages)
      DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
         DPxPTR(Img->first.ImageStart), ImageAndInfo.RTLName.c_str());
#endif
  }

#ifdef OMPTARGET_DEBUG
  for (auto Img : RemainingImages)
    DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->first.ImageStart));
#endif

  PM->RTLsMtx.unlock();

  DP("Done registering entries!\n");
}

void PluginAdaptorManagerTy::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.

  for (auto &ImageAndInfo : PM->Images) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &ImageAndInfo.first;

    PluginAdaptorTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto *R : UsedRTLs) {

      assert(R->IsUsed && "Expecting used RTLs.");

      // Ensure that we do not use any unused images associated with this RTL.
      if (!R->UsedImages.contains(Img))
        continue;

      FoundRTL = R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t I = 0; I < FoundRTL->NumberOfDevices; ++I) {
        DeviceTy &Device = *PM->Devices[FoundRTL->Idx + I];
        Device.PendingGlobalsMtx.lock();
        if (Device.PendingCtorsDtors[Desc].PendingCtors.empty()) {
          AsyncInfoTy AsyncInfo(Device);
          for (auto &Dtor : Device.PendingCtorsDtors[Desc].PendingDtors) {
            int Rc =
                target(nullptr, Device, Dtor, CTorDTorKernelArgs, AsyncInfo);
            if (Rc != OFFLOAD_SUCCESS) {
              DP("Running destructor " DPxMOD " failed.\n", DPxPTR(Dtor));
            }
          }
          // Remove this library's entry from PendingCtorsDtors
          Device.PendingCtorsDtors.erase(Desc);
          // All constructors have been issued, wait for them now.
          if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
            DP("Failed synchronizing destructors kernels.\n");
        }
        Device.PendingGlobalsMtx.unlock();
      }

      DP("Unregistered image " DPxMOD " from RTL " DPxMOD "!\n",
         DPxPTR(Img->ImageStart), DPxPTR(R->LibraryHandler.get()));

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL) {
      DP("No RTLs in use support the image " DPxMOD "!\n",
         DPxPTR(Img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from PM->HostPtrToTableMap
  PM->TblMapMtx.lock();
  for (__tgt_offload_entry *Cur = Desc->HostEntriesBegin;
       Cur < Desc->HostEntriesEnd; ++Cur) {
    PM->HostPtrToTableMap.erase(Cur->addr);
  }

  // Remove translation table for this descriptor.
  auto TransTable =
      PM->HostEntriesBeginToTransTable.find(Desc->HostEntriesBegin);
  if (TransTable != PM->HostEntriesBeginToTransTable.end()) {
    DP("Removing translation table for descriptor " DPxMOD "\n",
       DPxPTR(Desc->HostEntriesBegin));
    PM->HostEntriesBeginToTransTable.erase(TransTable);
  } else {
    DP("Translation table for descriptor " DPxMOD " cannot be found, probably "
       "it has been already removed.\n",
       DPxPTR(Desc->HostEntriesBegin));
  }

  PM->TblMapMtx.unlock();

  DP("Done unregistering library!\n");
}

// HACK: These depricated device stubs still needs host versions for fallback
// FIXME: Deprecate upstream, change test cases to use malloc & free directly
extern "C" char *global_allocate(uint32_t sz) { return (char *)malloc(sz); }
extern "C" int global_free(void *ptr) {
  free(ptr);
  return 0;
}
