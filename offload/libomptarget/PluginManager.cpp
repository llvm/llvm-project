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
#include "OffloadPolicy.h"
#include "Shared/Debug.h"
#include "Shared/Profile.h"
#include "device.h"

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
  if (OffloadPolicy::isOffloadDisabled()) {
    DP("Offload is disabled. Skipping plugin initialization\n");
    return;
  }

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

bool PluginManager::initializePlugin(GenericPluginTy &Plugin) {
  if (Plugin.is_initialized())
    return true;

  if (auto Err = Plugin.init()) {
    [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
    DP("Failed to init plugin: %s\n", InfoMsg.c_str());
    return false;
  }

  DP("Registered plugin %s with %d visible device(s)\n", Plugin.getName(),
     Plugin.number_of_devices());
  return true;
}

bool PluginManager::initializeDevice(GenericPluginTy &Plugin,
                                     int32_t DeviceId) {
  if (Plugin.is_device_initialized(DeviceId)) {
    auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
    (*ExclusiveDevicesAccessor)[PM->DeviceIds[std::make_pair(&Plugin,
                                                             DeviceId)]]
        ->setHasPendingImages(true);
    return true;
  }

  // Initialize the device information for the RTL we are about to use.
  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();

  int32_t UserId = ExclusiveDevicesAccessor->size();

  // Set the device identifier offset in the plugin.
#ifdef OMPT_SUPPORT
  Plugin.set_device_identifier(UserId, DeviceId);
#endif

  auto Device = std::make_unique<DeviceTy>(&Plugin, UserId, DeviceId);
  if (auto Err = Device->init()) {
    [[maybe_unused]] std::string InfoMsg = toString(std::move(Err));
    DP("Failed to init device %d: %s\n", DeviceId, InfoMsg.c_str());
    return false;
  }

  ExclusiveDevicesAccessor->push_back(std::move(Device));

  // We need to map between the plugin's device identifier and the one
  // that OpenMP will use.
  PM->DeviceIds[std::make_pair(&Plugin, DeviceId)] = UserId;

  return true;
}

void PluginManager::initializeAllDevices() {
  for (auto &Plugin : plugins()) {
    if (!initializePlugin(Plugin))
      continue;

    for (int32_t DeviceId = 0; DeviceId < Plugin.number_of_devices();
         ++DeviceId) {
      initializeDevice(Plugin, DeviceId);
    }
  }
  // After all plugins are initialized, register atExit cleanup handlers
  std::atexit([]() {
    // Interop cleanup should be done before the plugins are deinitialized as
    // the backend libraries may be already unloaded.
    if (PM)
      PM->InteropTbl.clear();
  });
}

// Returns a pointer to the binary descriptor, upgrading from a legacy format if
// necessary.
__tgt_bin_desc *PluginManager::upgradeLegacyEntries(__tgt_bin_desc *Desc) {
  struct LegacyEntryTy {
    void *Address;
    char *SymbolName;
    size_t Size;
    int32_t Flags;
    int32_t Data;
  };

  if (UpgradedDescriptors.contains(Desc))
    return &UpgradedDescriptors[Desc];

  if (Desc->HostEntriesBegin == Desc->HostEntriesEnd ||
      Desc->HostEntriesBegin->Reserved == 0)
    return Desc;

  // The new format mandates that each entry starts with eight bytes of zeroes.
  // This allows us to detect the old format as this is a null pointer.
  llvm::SmallVector<llvm::offloading::EntryTy, 0> &NewEntries =
      LegacyEntries.emplace_back();
  for (LegacyEntryTy &Entry : llvm::make_range(
           reinterpret_cast<LegacyEntryTy *>(Desc->HostEntriesBegin),
           reinterpret_cast<LegacyEntryTy *>(Desc->HostEntriesEnd))) {
    llvm::offloading::EntryTy &NewEntry = NewEntries.emplace_back();

    NewEntry.Address = Entry.Address;
    NewEntry.Flags = Entry.Flags;
    NewEntry.Data = Entry.Data;
    NewEntry.Size = Entry.Size;
    NewEntry.SymbolName = Entry.SymbolName;
    NewEntry.Kind = object::OffloadKind::OFK_OpenMP;
  }

  // Create a new image struct so we can update the entries list.
  llvm::SmallVector<__tgt_device_image, 0> &NewImages =
      LegacyImages.emplace_back();
  for (int32_t Image = 0; Image < Desc->NumDeviceImages; ++Image)
    NewImages.emplace_back(
        __tgt_device_image{Desc->DeviceImages[Image].ImageStart,
                           Desc->DeviceImages[Image].ImageEnd,
                           NewEntries.begin(), NewEntries.end()});

  // Create the new binary descriptor containing the newly created memory.
  __tgt_bin_desc &NewDesc = UpgradedDescriptors[Desc];
  NewDesc.DeviceImages = NewImages.begin();
  NewDesc.NumDeviceImages = Desc->NumDeviceImages;
  NewDesc.HostEntriesBegin = NewEntries.begin();
  NewDesc.HostEntriesEnd = NewEntries.end();

  return &NewDesc;
}

void PluginManager::registerLib(__tgt_bin_desc *Desc) {
  PM->RTLsMtx.lock();

  // Upgrade the entries from the legacy implementation if necessary.
  Desc = upgradeLegacyEntries(Desc);

  // Add in all the OpenMP requirements associated with this binary.
  for (llvm::offloading::EntryTy &Entry :
       llvm::make_range(Desc->HostEntriesBegin, Desc->HostEntriesEnd))
    if (Entry.Kind == object::OffloadKind::OFK_OpenMP &&
        Entry.Flags == OMP_REGISTER_REQUIRES)
      PM->addRequirements(Entry.Data);

  // Extract the executable image and extra information if available.
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i)
    PM->addDeviceImage(*Desc, Desc->DeviceImages[i]);

  // Register the images with the RTLs that understand them, if any.
  llvm::DenseMap<GenericPluginTy *, llvm::DenseSet<int32_t>> UsedDevices;
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &Desc->DeviceImages[i];

    GenericPluginTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : plugins()) {
      if (!R.is_plugin_compatible(Img))
        continue;

      if (!initializePlugin(R))
        continue;

      if (!R.number_of_devices()) {
        DP("Skipping plugin %s with no visible devices\n", R.getName());
        continue;
      }

      for (int32_t DeviceId = 0; DeviceId < R.number_of_devices(); ++DeviceId) {
        // We only want a single matching image to be registered for each binary
        // descriptor. This prevents multiple of the same image from being
        // registered for the same device in the case that they are mutually
        // compatible, such as sm_80 and sm_89.
        if (UsedDevices[&R].contains(DeviceId)) {
          DP("Image " DPxMOD
             " is a duplicate, not loaded on RTL %s device %d!\n",
             DPxPTR(Img->ImageStart), R.getName(), DeviceId);
          continue;
        }

        if (!R.is_device_compatible(DeviceId, Img))
          continue;

        DP("Image " DPxMOD " is compatible with RTL %s device %d!\n",
           DPxPTR(Img->ImageStart), R.getName(), DeviceId);

        if (!initializeDevice(R, DeviceId))
          continue;

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

        UsedDevices[&R].insert(DeviceId);
        PM->UsedImages.insert(Img);
        FoundRTL = &R;

        PM->TrlTblMtx.unlock();
      }
    }
    if (!FoundRTL)
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(Img->ImageStart));
  }
  PM->RTLsMtx.unlock();

  bool UseAutoZeroCopy = false;

  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  // APUs are homogeneous set of GPUs. Check the first device for
  // configuring Auto Zero-Copy.
  if (ExclusiveDevicesAccessor->size() > 0) {
    auto &Device = *(*ExclusiveDevicesAccessor)[0];
    UseAutoZeroCopy = Device.useAutoZeroCopy();
  }

  if (UseAutoZeroCopy)
    addRequirements(OMPX_REQ_AUTO_ZERO_COPY);

  DP("Done registering entries!\n");
}

// Temporary forward declaration, old style CTor/DTor handling is going away.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

void PluginManager::unregisterLib(__tgt_bin_desc *Desc) {
  DP("Unloading target library!\n");

  Desc = upgradeLegacyEntries(Desc);

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (DeviceImageTy &DI : PM->deviceImages()) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &DI.getExecutableImage();

    GenericPluginTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto &R : plugins()) {
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
  for (llvm::offloading::EntryTy *Cur = Desc->HostEntriesBegin;
       Cur < Desc->HostEntriesEnd; ++Cur) {
    if (Cur->Kind == object::OffloadKind::OFK_OpenMP)
      PM->HostPtrToTableMap.erase(Cur->Address);
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

/// Map global data and execute pending ctors
static int loadImagesOntoDevice(DeviceTy &Device) {
  /*
   * Map global data
   */
  int32_t DeviceId = Device.DeviceID;
  int Rc = OFFLOAD_SUCCESS;
  {
    std::lock_guard<decltype(PM->TrlTblMtx)> LG(PM->TrlTblMtx);
    for (auto *HostEntriesBegin : PM->HostEntriesBeginRegistrationOrder) {
      TranslationTable *TransTable =
          &PM->HostEntriesBeginToTransTable[HostEntriesBegin];
      DP("Trans table %p : %p\n", TransTable->HostTable.EntriesBegin,
         TransTable->HostTable.EntriesEnd);
      if (TransTable->HostTable.EntriesBegin ==
          TransTable->HostTable.EntriesEnd) {
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

      // 2) Load the image onto the given device.
      auto BinaryOrErr = Device.loadBinary(Img);
      if (llvm::Error Err = BinaryOrErr.takeError()) {
        REPORT("Failed to load image %s\n",
               llvm::toString(std::move(Err)).c_str());
        Rc = OFFLOAD_FAIL;
        break;
      }

      // 3) Create the translation table.
      llvm::SmallVector<llvm::offloading::EntryTy> &DeviceEntries =
          TransTable->TargetsEntries[DeviceId];
      for (llvm::offloading::EntryTy &Entry :
           llvm::make_range(Img->EntriesBegin, Img->EntriesEnd)) {
        if (Entry.Kind != object::OffloadKind::OFK_OpenMP)
          continue;

        __tgt_device_binary &Binary = *BinaryOrErr;

        llvm::offloading::EntryTy DeviceEntry = Entry;
        if (Entry.Size) {
          if (Device.RTL->get_global(Binary, Entry.Size, Entry.SymbolName,
                                     &DeviceEntry.Address) != OFFLOAD_SUCCESS)
            REPORT("Failed to load symbol %s\n", Entry.SymbolName);

          // If unified memory is active, the corresponding global is a device
          // reference to the host global. We need to initialize the pointer on
          // the device to point to the memory on the host.
          if ((PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
              (PM->getRequirements() & OMPX_REQ_AUTO_ZERO_COPY)) {
            if (Device.RTL->data_submit(DeviceId, DeviceEntry.Address,
                                        Entry.Address,
                                        Entry.Size) != OFFLOAD_SUCCESS)
              REPORT("Failed to write symbol for USM %s\n", Entry.SymbolName);
          }
        } else if (Entry.Address) {
          if (Device.RTL->get_function(Binary, Entry.SymbolName,
                                       &DeviceEntry.Address) != OFFLOAD_SUCCESS)
            REPORT("Failed to load kernel %s\n", Entry.SymbolName);
        }
        DP("Entry point " DPxMOD " maps to%s %s (" DPxMOD ")\n",
           DPxPTR(Entry.Address), (Entry.Size) ? " global" : "",
           Entry.SymbolName, DPxPTR(DeviceEntry.Address));

        DeviceEntries.emplace_back(DeviceEntry);
      }

      // Set the storage for the table and get a pointer to it.
      __tgt_target_table DeviceTable{&DeviceEntries[0],
                                     &DeviceEntries[0] + DeviceEntries.size()};
      TransTable->DeviceTables[DeviceId] = DeviceTable;
      __tgt_target_table *TargetTable = TransTable->TargetsTable[DeviceId] =
          &TransTable->DeviceTables[DeviceId];

      MappingInfoTy::HDTTMapAccessorTy HDTTMap =
          Device.getMappingInfo().HostDataToTargetMap.getExclusiveAccessor();

      __tgt_target_table *HostTable = &TransTable->HostTable;
      for (llvm::offloading::EntryTy *
               CurrDeviceEntry = TargetTable->EntriesBegin,
              *CurrHostEntry = HostTable->EntriesBegin,
              *EntryDeviceEnd = TargetTable->EntriesEnd;
           CurrDeviceEntry != EntryDeviceEnd;
           CurrDeviceEntry++, CurrHostEntry++) {
        if (CurrDeviceEntry->Size == 0 ||
            CurrDeviceEntry->Kind != object::OffloadKind::OFK_OpenMP)
          continue;

        assert(CurrDeviceEntry->Size == CurrHostEntry->Size &&
               "data size mismatch");

        // Fortran may use multiple weak declarations for the same symbol,
        // therefore we must allow for multiple weak symbols to be loaded from
        // the fat binary. Treat these mappings as any other "regular"
        // mapping. Add entry to map.
        if (Device.getMappingInfo().getTgtPtrBegin(
                HDTTMap, CurrHostEntry->Address, CurrHostEntry->Size))
          continue;

        void *CurrDeviceEntryAddr = CurrDeviceEntry->Address;

        // For indirect mapping, follow the indirection and map the actual
        // target.
        if (CurrDeviceEntry->Flags & OMP_DECLARE_TARGET_INDIRECT) {
          AsyncInfoTy AsyncInfo(Device);
          void *DevPtr;
          Device.retrieveData(&DevPtr, CurrDeviceEntryAddr, sizeof(void *),
                              AsyncInfo, /*Entry=*/nullptr, &HDTTMap);
          if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
            return OFFLOAD_FAIL;
          CurrDeviceEntryAddr = DevPtr;
        }

        DP("Add mapping from host " DPxMOD " to device " DPxMOD " with size %zu"
           ", name \"%s\"\n",
           DPxPTR(CurrHostEntry->Address), DPxPTR(CurrDeviceEntry->Address),
           CurrDeviceEntry->Size, CurrDeviceEntry->SymbolName);
        HDTTMap->emplace(new HostDataToTargetTy(
            (uintptr_t)CurrHostEntry->Address /*HstPtrBase*/,
            (uintptr_t)CurrHostEntry->Address /*HstPtrBegin*/,
            (uintptr_t)CurrHostEntry->Address +
                CurrHostEntry->Size /*HstPtrEnd*/,
            (uintptr_t)CurrDeviceEntryAddr /*TgtAllocBegin*/,
            (uintptr_t)CurrDeviceEntryAddr /*TgtPtrBegin*/,
            false /*UseHoldRefCount*/, CurrHostEntry->SymbolName,
            true /*IsRefCountINF*/));

        // Notify about the new mapping.
        if (Device.notifyDataMapped(CurrHostEntry->Address,
                                    CurrHostEntry->Size))
          return OFFLOAD_FAIL;
      }
    }
    Device.setHasPendingImages(false);
  }

  if (Rc != OFFLOAD_SUCCESS)
    return Rc;

  static Int32Envar DumpOffloadEntries =
      Int32Envar("OMPTARGET_DUMP_OFFLOAD_ENTRIES", -1);
  if (DumpOffloadEntries.get() == DeviceId)
    Device.dumpOffloadEntries();

  return OFFLOAD_SUCCESS;
}

Expected<DeviceTy &> PluginManager::getDevice(uint32_t DeviceNo) {
  DeviceTy *DevicePtr;
  {
    auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
    if (DeviceNo >= ExclusiveDevicesAccessor->size())
      return error::createOffloadError(
          error::ErrorCode::INVALID_VALUE,
          "device number '%i' out of range, only %i devices available",
          DeviceNo, ExclusiveDevicesAccessor->size());

    DevicePtr = &*(*ExclusiveDevicesAccessor)[DeviceNo];
  }

  // Check whether global data has been mapped for this device
  if (DevicePtr->hasPendingImages())
    if (loadImagesOntoDevice(*DevicePtr) != OFFLOAD_SUCCESS)
      return error::createOffloadError(error::ErrorCode::BACKEND_FAILURE,
                                       "failed to load images on device '%i'",
                                       DeviceNo);
  return *DevicePtr;
}
