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

#include "OmptCallback.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

#include "Utilities.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::omp::target;

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target       */ "libomptarget.rtl.ppc64",
    /* x86_64 target        */ "libomptarget.rtl.x86_64",
    /* CUDA target          */ "libomptarget.rtl.cuda",
    /* AArch64 target       */ "libomptarget.rtl.aarch64",
    /* AMDGPU target        */ "libomptarget.rtl.amdgpu",
};

PluginManager *PM;

static char *ProfileTraceFile = nullptr;

#ifdef OMPT_SUPPORT
extern void ompt::connectLibrary();
#endif

__attribute__((constructor(101))) void init() {
  DP("Init target library!\n");

  bool UseEventsForAtomicTransfers = true;
  if (const char *ForceAtomicMap = getenv("LIBOMPTARGET_MAP_FORCE_ATOMIC")) {
    std::string ForceAtomicMapStr(ForceAtomicMap);
    if (ForceAtomicMapStr == "false" || ForceAtomicMapStr == "FALSE")
      UseEventsForAtomicTransfers = false;
    else if (ForceAtomicMapStr != "true" && ForceAtomicMapStr != "TRUE")
      fprintf(stderr,
              "Warning: 'LIBOMPTARGET_MAP_FORCE_ATOMIC' accepts only "
              "'true'/'TRUE' or 'false'/'FALSE' as options, '%s' ignored\n",
              ForceAtomicMap);
  }

  PM = new PluginManager(UseEventsForAtomicTransfers);

  ProfileTraceFile = getenv("LIBOMPTARGET_PROFILE");
  // TODO: add a configuration option for time granularity
  if (ProfileTraceFile)
    timeTraceProfilerInitialize(500 /* us */, "libomptarget");

#ifdef OMPT_SUPPORT
  // Initialize OMPT first
  ompt::connectLibrary();
#endif

  PM->RTLs.loadRTLs();
  PM->registerDelayedLibraries();
}

__attribute__((destructor(101))) void deinit() {
  DP("Deinit target library!\n");
  delete PM;

  if (ProfileTraceFile) {
    // TODO: add env var for file output
    if (auto E = timeTraceProfilerWrite(ProfileTraceFile, "-"))
      fprintf(stderr, "Error writing out the time trace\n");

    timeTraceProfilerCleanup();
  }
}

void RTLsTy::loadRTLs() {
  // Parse environment variable OMP_TARGET_OFFLOAD (if set)
  PM->TargetOffloadPolicy =
      (kmp_target_offload_kind_t)__kmpc_get_target_offload();
  if (PM->TargetOffloadPolicy == tgt_disabled) {
    return;
  }

  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (const char *Name : RTLNames) {
    AllRTLs.emplace_back();

    RTLInfoTy &RTL = AllRTLs.back();

    const std::string BaseRTLName(Name);
    if (!attemptLoadRTL(BaseRTLName + ".so", RTL))
      AllRTLs.pop_back();
  }

  DP("RTLs loaded!\n");
}

bool RTLsTy::attemptLoadRTL(const std::string &RTLName, RTLInfoTy &RTL) {
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

  // Remove plugin on failure to call optional init_plugin
  *((void **)&RTL.init_plugin) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_init_plugin");
  if (RTL.init_plugin) {
    int32_t Rc = RTL.init_plugin();
    if (Rc != OFFLOAD_SUCCESS) {
      DP("Unable to initialize library '%s': %u!\n", Name, Rc);
      return false;
    }
  }

  bool ValidPlugin = true;

  if (!(*((void **)&RTL.is_valid_binary) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_is_valid_binary")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.number_of_devices) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_number_of_devices")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.init_device) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_init_device")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.load_binary) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_load_binary")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.data_alloc) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_data_alloc")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.data_submit) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_data_submit")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.data_retrieve) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_data_retrieve")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.data_delete) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_data_delete")))
    ValidPlugin = false;
  if (!(*((void **)&RTL.launch_kernel) =
            DynLibrary->getAddressOfSymbol("__tgt_rtl_launch_kernel")))
    ValidPlugin = false;

  // Invalid plugin
  if (!ValidPlugin) {
    DP("Invalid plugin as necessary interface is not found.\n");
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

  // Optional functions
  *((void **)&RTL.deinit_plugin) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_deinit_plugin");
  *((void **)&RTL.is_valid_binary_info) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_is_valid_binary_info");
  *((void **)&RTL.deinit_device) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_deinit_device");
  *((void **)&RTL.init_requires) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_init_requires");
  *((void **)&RTL.data_submit_async) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_submit_async");
  *((void **)&RTL.data_retrieve_async) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_retrieve_async");
  *((void **)&RTL.synchronize) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_synchronize");
  *((void **)&RTL.query_async) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_query_async");
  *((void **)&RTL.data_exchange) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_exchange");
  *((void **)&RTL.data_exchange_async) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_exchange_async");
  *((void **)&RTL.is_data_exchangable) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_is_data_exchangable");
  *((void **)&RTL.register_lib) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_register_lib");
  *((void **)&RTL.unregister_lib) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_unregister_lib");
  *((void **)&RTL.supports_empty_images) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_supports_empty_images");
  *((void **)&RTL.set_info_flag) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_set_info_flag");
  *((void **)&RTL.print_device_info) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_print_device_info");
  *((void **)&RTL.create_event) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_create_event");
  *((void **)&RTL.record_event) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_record_event");
  *((void **)&RTL.wait_event) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_wait_event");
  *((void **)&RTL.sync_event) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_sync_event");
  *((void **)&RTL.destroy_event) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_destroy_event");
  *((void **)&RTL.release_async_info) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_release_async_info");
  *((void **)&RTL.init_async_info) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_init_async_info");
  *((void **)&RTL.init_device_info) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_init_device_info");
  *((void **)&RTL.data_lock) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_lock");
  *((void **)&RTL.data_unlock) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_unlock");
  *((void **)&RTL.data_notify_mapped) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_notify_mapped");
  *((void **)&RTL.data_notify_unmapped) =
      DynLibrary->getAddressOfSymbol("__tgt_rtl_data_notify_unmapped");

  RTL.LibraryHandler = std::move(DynLibrary);

  // Successfully loaded
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering libs

static void registerImageIntoTranslationTable(TranslationTable &TT,
                                              RTLInfoTy &RTL,
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
                                             RTLInfoTy *RTL) {

  for (int32_t I = 0; I < RTL->NumberOfDevices; ++I) {
    DeviceTy &Device = *PM->Devices[RTL->Idx + I];
    Device.PendingGlobalsMtx.lock();
    Device.HasPendingGlobals = true;
    for (__tgt_offload_entry *Entry = Img->EntriesBegin;
         Entry != Img->EntriesEnd; ++Entry) {
      if (Entry->flags & OMP_DECLARE_TARGET_CTOR) {
        DP("Adding ctor " DPxMOD " to the pending list.\n",
           DPxPTR(Entry->addr));
        Device.PendingCtorsDtors[Desc].PendingCtors.push_back(Entry->addr);
      } else if (Entry->flags & OMP_DECLARE_TARGET_DTOR) {
        // Dtors are pushed in reverse order so they are executed from end
        // to beginning when unregistering the library!
        DP("Adding dtor " DPxMOD " to the pending list.\n",
           DPxPTR(Entry->addr));
        Device.PendingCtorsDtors[Desc].PendingDtors.push_front(Entry->addr);
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

void RTLsTy::registerRequires(int64_t Flags) {
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

void RTLsTy::initRTLonce(RTLInfoTy &R) {
  // If this RTL is not already in use, initialize it.
  if (!R.IsUsed && R.NumberOfDevices != 0) {
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
    R.Idx = (UsedRTLs.empty())
                ? 0
                : UsedRTLs.back()->Idx + UsedRTLs.back()->NumberOfDevices;
    assert((size_t)R.Idx == Start &&
           "RTL index should equal the number of devices used so far.");
    R.IsUsed = true;
    UsedRTLs.push_back(&R);

    DP("RTL " DPxMOD " has index %d!\n", DPxPTR(R.LibraryHandler.get()), R.Idx);
  }
}

void RTLsTy::initAllRTLs() {
  for (auto &R : AllRTLs)
    initRTLonce(R);
}

void RTLsTy::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Extract the exectuable image and extra information if availible.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->Images.emplace_back(getExecutableImage(&Desc->DeviceImages[i]),
                            getImageInfo(&Desc->DeviceImages[i]));

  // Register the images with the RTLs that understand them, if any.
  for (auto &ImageAndInfo : PM->Images) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &ImageAndInfo.first;
    __tgt_image_info *Info = &ImageAndInfo.second;

    RTLInfoTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : AllRTLs) {
      if (R.is_valid_binary_info) {
        if (!R.is_valid_binary_info(Img, Info)) {
          DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
             DPxPTR(Img->ImageStart), R.RTLName.c_str());
          continue;
        }
      } else if (!R.is_valid_binary(Img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
           DPxPTR(Img->ImageStart), R.RTLName.c_str());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
         DPxPTR(Img->ImageStart), R.RTLName.c_str());

      initRTLonce(R);

      // Initialize (if necessary) translation table for this library.
      PM->TrlTblMtx.lock();
      if (!PM->HostEntriesBeginToTransTable.count(Desc->HostEntriesBegin)) {
        PM->HostEntriesBeginRegistrationOrder.push_back(Desc->HostEntriesBegin);
        TranslationTable &TransTable =
            (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];
        TransTable.HostTable.EntriesBegin = Desc->HostEntriesBegin;
        TransTable.HostTable.EntriesEnd = Desc->HostEntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];

      DP("Registering image " DPxMOD " with RTL %s!\n", DPxPTR(Img->ImageStart),
         R.RTLName.c_str());
      registerImageIntoTranslationTable(TransTable, R, Img);
      R.UsedImages.insert(Img);

      PM->TrlTblMtx.unlock();
      FoundRTL = &R;

      // Load ctors/dtors for static objects
      registerGlobalCtorsDtorsForImage(Desc, Img, FoundRTL);

      // if an RTL was found we are done - proceed to register the next image
      break;
    }

    if (!FoundRTL) {
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();

  DP("Done registering entries!\n");
}

void RTLsTy::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (auto &ImageAndInfo : PM->Images) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &ImageAndInfo.first;

    RTLInfoTy *FoundRTL = NULL;

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
            int Rc = target(nullptr, Device, Dtor, CTorDTorKernelArgs, AsyncInfo);
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

  // TODO: Write some RTL->unload_image(...) function?
  for (auto *R : UsedRTLs) {
    if (R->deinit_plugin) {
      if (R->deinit_plugin() != OFFLOAD_SUCCESS) {
        DP("Failure deinitializing RTL %s!\n", R->RTLName.c_str());
      }
    }
  }

  DP("Done unregistering library!\n");
}
