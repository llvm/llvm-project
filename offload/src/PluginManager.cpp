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
#include "Shared/Debug.h"
#include "Shared/Profile.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

using namespace llvm;
using namespace llvm::sys;

PluginManager *PM = nullptr;

// Every plugin exports this method to create an instance of the plugin type.
#define PLUGIN_TARGET(Name) extern "C" GenericPluginTy *createPlugin_##Name();
#include "Shared/Targets.def"

void PluginManager::init() {
  TIMESCOPE();
  DP("Loading RTLs...\n");

  // Attempt to create an instance of each supported plugin.
#define PLUGIN_TARGET(Name)                                                    \
  do {                                                                         \
    Plugins.emplace_back(                                                      \
        std::unique_ptr<GenericPluginTy>(createPlugin_##Name()));              \
  } while (false);
#include "Shared/Targets.def"

  DP("RTLs loaded!\n");
}

void PluginManager::deinit() {
  TIMESCOPE();
  DP("Unloading RTLs...\n");

  for (auto &Plugin : Plugins) {
    if (!Plugin->is_initialized())
      continue;

    if (auto Err = Plugin->deinit()) {
      [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
      DP("Failed to deinit plugin: %s\n", InfoMsg.c_str());
    }
    Plugin.release();
  }

  DP("RTLs unloaded!\n");
}

void PluginManager::initAllPlugins() {
  for (auto &R : plugins()) {
    if (auto Err = R.init()) {
      [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
      DP("Failed to init plugin: %s\n", InfoMsg.c_str());
      continue;
    }
    DP("Registered plugin %s with %d visible device(s)\n", R.getName(),
       R.number_of_devices());
  }
}

void PluginManager::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Add in all the OpenMP requirements associated with this binary.
  for (__tgt_offload_entry &Entry :
       llvm::make_range(Desc->HostEntriesBegin, Desc->HostEntriesEnd))
    if (Entry.flags == OMP_REGISTER_REQUIRES)
      PM->addRequirements(Entry.data);

  // Extract the exectuable image and extra information if availible.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->addDeviceImage(*Desc, Desc->DeviceImages[i]);

  // Register the images with the RTLs that understand them, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    GenericPluginTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : PM->plugins()) {
      if (!R.is_plugin_compatible(Img))
        continue;

      if (!R.is_initialized()) {
        if (auto Err = R.init()) {
          [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
          DP("Failed to init plugin: %s\n", InfoMsg.c_str());
          continue;
        }
        DP("Registered plugin %s with %d visible device(s)\n", R.getName(),
           R.number_of_devices());
      }

      if (!R.number_of_devices()) {
        DP("Skipping plugin %s with no visible devices\n", R.getName());
        continue;
      }

      for (int32_t DeviceId = 0; DeviceId < R.number_of_devices(); ++DeviceId) {
        if (!R.is_device_compatible(DeviceId, Img))
          continue;

        DP("Image " DPxMOD " is compatible with RTL %s device %d!\n",
           DPxPTR(Img->ImageStart), R.getName(), DeviceId);

        if (!R.is_device_initialized(DeviceId)) {
          // Initialize the device information for the RTL we are about to use.
          auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();

          int32_t UserId = ExclusiveDevicesAccessor->size();

          // Set the device identifier offset in the plugin.
#ifdef OMPT_SUPPORT
          R.set_device_identifier(UserId, DeviceId);
#endif

          auto Device = std::make_unique<DeviceTy>(&R, UserId, DeviceId);
          if (auto Err = Device->init()) {
            [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
            DP("Failed to init device %d: %s\n", DeviceId, InfoMsg.c_str());
            continue;
          }

          ExclusiveDevicesAccessor->push_back(std::move(Device));

          // We need to map between the plugin's device identifier and the one
          // that OpenMP will use.
          PM->DeviceIds[std::make_pair(&R, DeviceId)] = UserId;
        }

        // Initialize (if necessary) translation table for this library.
        PM->TrlTblMtx.lock();
        if (!PM->HostEntriesBeginToTransTable.count(Desc->HostEntriesBegin)) {
          PM->HostEntriesBeginRegistrationOrder.push_back(
              Desc->HostEntriesBegin);
          TranslationTable &TT =
              (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];
          TT.HostTable.EntriesBegin = Desc->HostEntriesBegin;
          TT.HostTable.EntriesEnd = Desc->HostEntriesEnd;
        }

        // Retrieve translation table for this library.
        TranslationTable &TT =
            (PM->HostEntriesBeginToTransTable)[Desc->HostEntriesBegin];

        DP("Registering image " DPxMOD " with RTL %s!\n",
           DPxPTR(Img->ImageStart), R.getName());

        auto UserId = PM->DeviceIds[std::make_pair(&R, DeviceId)];
        if (TT.TargetsTable.size() < static_cast<size_t>(UserId + 1)) {
          TT.DeviceTables.resize(UserId + 1, {});
          TT.TargetsImages.resize(UserId + 1, nullptr);
          TT.TargetsEntries.resize(UserId + 1, {});
          TT.TargetsTable.resize(UserId + 1, nullptr);
        }

        // Register the image for this target type and invalidate the table.
        TT.TargetsImages[UserId] = Img;
        TT.TargetsTable[UserId] = nullptr;

        PM->UsedImages.insert(Img);
        FoundRTL = &R;

        PM->TrlTblMtx.unlock();
      }
    }
    if (!FoundRTL)
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->ImageStart));
  }
  PM->RTLsMtx.unlock();

  bool UseAutoZeroCopy = Plugins.size() > 0;

  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  for (const auto &Device : *ExclusiveDevicesAccessor)
    UseAutoZeroCopy &= Device->useAutoZeroCopy();

  // Auto Zero-Copy can only be currently triggered when the system is an
  // homogeneous APU architecture without attached discrete GPUs.
  // If all devices suggest to use it, change requirment flags to trigger
  // zero-copy behavior when mapping memory.
  if (UseAutoZeroCopy)
    addRequirements(OMPX_REQ_AUTO_ZERO_COPY);

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

    GenericPluginTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto &R : PM->plugins()) {
      if (R.is_initialized())
        continue;

      // Ensure that we do not use any unused images associated with this RTL.
      if (!UsedImages.contains(Img))
        continue;

      FoundRTL = &R;

      DP("Unregistered image " DPxMOD " from RTL\n", DPxPTR(Img->ImageStart));

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

Expected<DeviceTy &> PluginManager::getDevice(uint32_t DeviceNo) {
  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  if (DeviceNo >= ExclusiveDevicesAccessor->size())
    return createStringError(
        inconvertibleErrorCode(),
        "Device number '%i' out of range, only %i devices available", DeviceNo,
        ExclusiveDevicesAccessor->size());

  return *(*ExclusiveDevicesAccessor)[DeviceNo];
}
