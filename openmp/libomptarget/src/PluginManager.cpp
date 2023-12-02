//===-- PluginManager.cpp - Plugin loading and communication API ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling plugins.
//
//===----------------------------------------------------------------------===//

#include "PluginManager.h"

using namespace llvm;
using namespace llvm::sys;

PluginManager *PM;

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target       */ "libomptarget.rtl.ppc64",
    /* x86_64 target        */ "libomptarget.rtl.x86_64",
    /* CUDA target          */ "libomptarget.rtl.cuda",
    /* AArch64 target       */ "libomptarget.rtl.aarch64",
    /* AMDGPU target        */ "libomptarget.rtl.amdgpu",
};

PluginAdaptorTy::PluginAdaptorTy(const std::string &Name) : Name(Name) {
  DP("Attempting to load library '%s'...\n", Name.c_str());

  std::string ErrMsg;
  LibraryHandler = std::make_unique<DynamicLibrary>(
      DynamicLibrary::getPermanentLibrary(Name.c_str(), &ErrMsg));

  if (!LibraryHandler->isValid()) {
    // Library does not exist or cannot be found.
    DP("Unable to load library '%s': %s!\n", Name.c_str(), ErrMsg.c_str());
    return;
  }

  DP("Successfully loaded library '%s'!\n", Name.c_str());

#define PLUGIN_API_HANDLE(NAME, MANDATORY)                                     \
  NAME = reinterpret_cast<decltype(NAME)>(                                     \
      LibraryHandler->getAddressOfSymbol(GETNAME(__tgt_rtl_##NAME)));          \
  if (MANDATORY && !NAME) {                                                    \
    DP("Invalid plugin as necessary interface is not found.\n");               \
    return;                                                                    \
  }

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  // Remove plugin on failure to call optional init_plugin
  int32_t Rc = init_plugin();
  if (Rc != OFFLOAD_SUCCESS) {
    DP("Unable to initialize library '%s': %u!\n", Name.c_str(), Rc);
    return;
  }

  // No devices are supported by this RTL?
  NumberOfDevices = number_of_devices();
  if (!NumberOfDevices) {
    DP("No devices supported in this RTL\n");
    return;
  }

  DP("Registered '%s' with %d devices!\n", Name.c_str(), NumberOfDevices);
}

void PluginAdaptorTy::addOffloadEntries(DeviceImageTy &DI) {
  for (int32_t I = 0; I < NumberOfDevices; ++I) {
    DeviceTy &Device = *PM->Devices[DeviceOffset + I];
    for (OffloadEntryTy &Entry : DI.entries())
      Device.addOffloadEntry(Entry);
  }
}

void PluginManager::init() {
  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (const char *Name : RTLNames) {
    PluginAdaptors.emplace_back(std::string(Name) + ".so");
    if (PluginAdaptors.back().getNumDevices() <= 0)
      PluginAdaptors.pop_back();
  }

  DP("RTLs loaded!\n");
}

void PluginManager::initPlugin(PluginAdaptorTy &Plugin) {
  // If this RTL is not already in use, initialize it.
  if (Plugin.isUsed() || !Plugin.NumberOfDevices)
    return;

  // Initialize the device information for the RTL we are about to use.
  const size_t Start = Devices.size();
  Devices.reserve(Start + Plugin.NumberOfDevices);
  for (int32_t DeviceId = 0; DeviceId < Plugin.NumberOfDevices; DeviceId++) {
    Devices.push_back(std::make_unique<DeviceTy>(&Plugin));
    // global device ID
    Devices[Start + DeviceId]->DeviceID = Start + DeviceId;
    // RTL local device ID
    Devices[Start + DeviceId]->RTLDeviceID = DeviceId;
  }

  // Initialize the index of this RTL and save it in the used RTLs.
  Plugin.DeviceOffset = Start;

  // If possible, set the device identifier offset in the plugin.
  if (Plugin.set_device_offset)
    Plugin.set_device_offset(Start);

  DP("RTL " DPxMOD " has index %d!\n", DPxPTR(Plugin.LibraryHandler.get()),
     Plugin.DeviceOffset);
}

void PluginManager::initAllPlugins() {
  for (auto &R : PluginAdaptors)
    initPlugin(R);
}

static void registerImageIntoTranslationTable(TranslationTable &TT,
                                              PluginAdaptorTy &RTL,
                                              __tgt_device_image *Image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.DeviceOffset + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t I = 0; I < RTL.NumberOfDevices; ++I) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.DeviceOffset + I] != Image) {
      TT.TargetsImages[RTL.DeviceOffset + I] = Image;
      TT.TargetsTable[RTL.DeviceOffset + I] =
          0; // lazy initialization of target table.
    }
  }
}

void PluginManager::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Extract the exectuable image and extra information if availible.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->addDeviceImage(*Desc, Desc->DeviceImages[i]);

  // Register the images with the RTLs that understand them, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();
    __tgt_image_info *Info = &DI.getImageInfo();

    PluginAdaptorTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : PM->pluginAdaptors()) {
      if (R.is_valid_binary_info) {
        if (!R.is_valid_binary_info(Img, Info)) {
          DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
             DPxPTR(Img->ImageStart), R.Name.c_str());
          continue;
        }
      } else if (!R.is_valid_binary(Img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
           DPxPTR(Img->ImageStart), R.Name.c_str());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
         DPxPTR(Img->ImageStart), R.Name.c_str());

      PM->initPlugin(R);

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
         R.Name.c_str());
      registerImageIntoTranslationTable(TransTable, R, Img);
      R.UsedImages.insert(Img);

      PM->TrlTblMtx.unlock();
      FoundRTL = &R;

      // Register all offload entries with the devices handled by the plugin.
      R.addOffloadEntries(DI);

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

// Temporary forward declaration, old style CTor/DTor handling is going away.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

void PluginManager::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    PluginAdaptorTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto &R : PM->pluginAdaptors()) {
      if (!R.isUsed())
        continue;

      // Ensure that we do not use any unused images associated with this RTL.
      if (!R.UsedImages.contains(Img))
        continue;

      FoundRTL = &R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t I = 0; I < FoundRTL->NumberOfDevices; ++I) {
        DeviceTy &Device = *PM->Devices[FoundRTL->DeviceOffset + I];
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
         DPxPTR(Img->ImageStart), DPxPTR(R.LibraryHandler.get()));

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
