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
using namespace llvm::omp::target::debug;

PluginManager *PM = nullptr;

extern "C" GenericPluginTy *
__ol_tgt_GetPluginFromPlatform(ol_platform_handle_t Platform);
extern "C" int32_t __ol_tgt_GetPluginDeviceId(ol_device_handle_t Device);

void PluginManager::init() {
  TIMESCOPE();
  if (OffloadPolicy::isOffloadDisabled()) {
    ODBG(ODT_Init) << "Offload is disabled. Skipping plugin initialization";
    return;
  }

  ODBG(ODT_Init) << "Loading RTLs";
  if (ol_result_t Res = olInit(nullptr))
    REPORT() << "Failed to initialize liboffload: " << Res->Details;

  if (ol_result_t Res = olIteratePlatforms(
          [](ol_platform_handle_t Platform, void *Data) {
            auto *PM = static_cast<PluginManager *>(Data);
            auto *Plugin = __ol_tgt_GetPluginFromPlatform(Platform);
            ODBG(ODT_Init) << "Adding plugin " << Plugin->getName()
                           << " from liboffload";
            PM->Plugins.push_back(Plugin);
            ol_platform_backend_t Backend;
            olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                              sizeof(Backend), &Backend);
            if (Backend == OL_PLATFORM_BACKEND_HOST)
              PM->HostPlatform = Platform;
            return true;
          },
          this))
    REPORT() << "Failed to iterate platforms: " << Res->Details;

  ODBG(ODT_Init) << "RTLs loaded!";
}

void PluginManager::deinit() {
  TIMESCOPE();
  ODBG(ODT_Deinit) << "Unloading RTLs...";

  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  for (auto &Device : *ExclusiveDevicesAccessor)
    if (auto Err = Device->deinit())
      REPORT() << "Failed to deinitialize device " << Device->DeviceID
               << ": " << toString(std::move(Err));

  Plugins.clear();
  if (auto Res = olShutDown())
    REPORT() << "Failed to deinitialize liboffload: " << Res->Details;

  ODBG(ODT_Deinit) << "RTLs unloaded!";
}

ol_device_handle_t PluginManager::getHostDevice() {
  if (!HostDevice) {
    olIterateDevices(
        [](ol_device_handle_t D, void *Data) {
          ol_platform_handle_t Platform;
          olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                          &Platform);
          ol_platform_backend_t Backend;
          olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                            &Backend);

          if (Backend == OL_PLATFORM_BACKEND_HOST) {
            GenericPluginTy *HostPlugin =
                __ol_tgt_GetPluginFromPlatform(Platform);
            HostPlugin->set_device_identifier(omp_initial_device, 0);
            *(static_cast<ol_device_handle_t *>(Data)) = D;
            return false;
          }

          return true;
        },
        &HostDevice);
  }

  return HostDevice;
}

bool PluginManager::initializeDevice(ol_device_handle_t DeviceHandle) {
  if (PM->DeviceIds.find(DeviceHandle) != PM->DeviceIds.end()) {
    auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
    (*ExclusiveDevicesAccessor)[PM->DeviceIds[DeviceHandle]]
        ->setHasPendingImages(true);
    return true;
  }

  ol_platform_handle_t PlatformHandle;
  if (auto Ret = olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                                 sizeof(PlatformHandle), &PlatformHandle);
      Ret != OL_SUCCESS) {
    REPORT() << "Failed to get platform while initializing device "
             << DeviceHandle;
    return false;
  }

  GenericPluginTy &Plugin = *__ol_tgt_GetPluginFromPlatform(PlatformHandle);
  int32_t DeviceId = __ol_tgt_GetPluginDeviceId(DeviceHandle);

  auto ExclusiveDevicesAccessor = getExclusiveDevicesAccessor();
  // Initialize the device information for the RTL we are about to use.
  int32_t UserId = ExclusiveDevicesAccessor->size();

  // Set the device identifier offset in the plugin.
#ifdef OMPT_SUPPORT
  Plugin.set_device_identifier(UserId, DeviceId);
#endif

  auto Device =
      std::make_unique<DeviceTy>(&Plugin, UserId, DeviceId, DeviceHandle);
  if (auto Err = Device->init()) {
    std::string InfoMsg = toString(std::move(Err));
    ODBG(ODT_Init) << "Failed to init device " << DeviceId << ": " << InfoMsg;
    return false;
  }

  ExclusiveDevicesAccessor->push_back(std::move(Device));

  // We need to map between the plugin's device identifier and the one
  // that OpenMP will use.
  PM->DeviceIds[DeviceHandle] = UserId;

  return true;
}

void PluginManager::initializeAllDevices() {
  olIterateDevices(
      [](ol_device_handle_t Device, void *UserData) {
        PM->initializeDevice(Device);
        return true;
      },
      nullptr);
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
  llvm::SmallVector<ol_device_handle_t> UsedDevices;
  for (int32_t i = 0; i < Desc->NumDeviceImages; ++i) {
    // Obtain the image and information that was previously extracted.
    __tgt_device_image *Img = &Desc->DeviceImages[i];

    struct RegisterImageState {
      __tgt_bin_desc *Desc;
      __tgt_device_image *Img;
      llvm::SmallVector<ol_device_handle_t> &UsedDevices;
      bool FoundRTL = false;
    } State{Desc, Img, UsedDevices, false};

    if (ol_result_t Res = olIterateCompatibleDevices(
            Img->ImageStart, utils::getPtrDiff(Img->ImageEnd, Img->ImageStart),
            [](ol_device_handle_t DeviceHandle, void *Data) {
              auto &State = *static_cast<RegisterImageState *>(Data);

              ol_platform_handle_t PlatformHandle;
              if (auto Res =
                      olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                                      sizeof(PlatformHandle), &PlatformHandle);
                  Res != OL_SUCCESS) {
                REPORT() << "Failed to get platform info for device "
                         << DeviceHandle << ":" << Res->Details;
                PlatformHandle = nullptr;
              }

              llvm::SmallString<256> PlatformName("Unknown");
              if (PlatformHandle) {
                size_t PlatformNameSize = 0;
                if (auto Res = olGetPlatformInfoSize(PlatformHandle,
                                                     OL_PLATFORM_INFO_NAME,
                                                     &PlatformNameSize);
                    Res != OL_SUCCESS)
                  PlatformNameSize = 0;

                PlatformName.resize(PlatformNameSize);
                if (PlatformNameSize > 0) {
                  if (auto Res = olGetPlatformInfo(
                          PlatformHandle, OL_PLATFORM_INFO_NAME,
                          PlatformNameSize, PlatformName.data());
                      Res != OL_SUCCESS)
                    PlatformName = "Unknown";
                } else
                  PlatformName = "Unknown";
              }

              // We only want a single matching image to be registered for each
              // binary descriptor. This prevents multiple of the same image
              // from being registered for the same device in the case that
              // they are mutually compatible, such as sm_80 and sm_89.
              if (llvm::is_contained(State.UsedDevices, DeviceHandle)) {
                ODBG(ODT_Init) << "Image " << State.Img->ImageStart
                               << " is a duplicate, not loaded on RTL "
                               << PlatformName << " on device " << DeviceHandle;
                return true;
              }

              ODBG(ODT_Init)
                  << "Image " << State.Img->ImageStart << " with RTL "
                  << PlatformName << " on device " << DeviceHandle;

              PM->initializeDevice(DeviceHandle);

              // Initialize (if necessary) translation table for this library.
              PM->TrlTblMtx.lock();
              if (!PM->HostEntriesBeginToTransTable.count(
                      State.Desc->HostEntriesBegin)) {
                PM->HostEntriesBeginRegistrationOrder.push_back(
                    State.Desc->HostEntriesBegin);
                TranslationTable &TT =
                    (PM->HostEntriesBeginToTransTable)[State.Desc
                                                           ->HostEntriesBegin];
                TT.HostTable.EntriesBegin = State.Desc->HostEntriesBegin;
                TT.HostTable.EntriesEnd = State.Desc->HostEntriesEnd;
              }

              // Retrieve translation table for this library.
              TranslationTable &TT =
                  (PM->HostEntriesBeginToTransTable)[State.Desc
                                                         ->HostEntriesBegin];

              ODBG(ODT_Init) << "Registering image " << State.Img->ImageStart
                             << " with RTL " << PlatformName;

              auto UserId = PM->DeviceIds[DeviceHandle];
              if (TT.TargetsTable.size() < static_cast<size_t>(UserId + 1)) {
                TT.DeviceTables.resize(UserId + 1, {});
                TT.TargetsImages.resize(UserId + 1, nullptr);
                TT.TargetsEntries.resize(UserId + 1, {});
                TT.TargetsTable.resize(UserId + 1, nullptr);
              }

              // Register the image for this target type and invalidate the
              // table.
              TT.TargetsImages[UserId] = State.Img;
              TT.TargetsTable[UserId] = nullptr;

              State.UsedDevices.push_back(DeviceHandle);
              PM->UsedImages.insert(State.Img);
              State.FoundRTL = true;

              PM->TrlTblMtx.unlock();
              return true;
            },
            &State))
      REPORT() << "Failed to iterate compatible devices: " << Res->Details;

    if (!State.FoundRTL)
      ODBG(ODT_Init) << "No RTL found for image " << Img->ImageStart << "!";
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

  ODBG(ODT_Init) << "Done registering entries!";
}

// Temporary forward declaration, old style CTor/DTor handling is going away.
int target(ident_t *Loc, DeviceTy &Device, void *HostPtr,
           KernelArgsTy &KernelArgs, AsyncInfoTy &AsyncInfo);

void PluginManager::unregisterLib(__tgt_bin_desc *Desc) {
  ODBG(ODT_Deinit) << "Unloading target library!";

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

      ODBG(ODT_Deinit) << "Unregistered image " << Img->ImageStart
                       << " from RTL";

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL) {
      ODBG(ODT_Deinit) << "No RTLs in use support the image "
                       << Img->ImageStart;
    }
  }
  PM->RTLsMtx.unlock();
  ODBG(ODT_Deinit) << "Done unregistering images!";

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
    ODBG(ODT_Deinit) << "Removing translation table for descriptor "
                     << Desc->HostEntriesBegin;
    PM->HostEntriesBeginToTransTable.erase(TransTable);
  } else {
    ODBG(ODT_Deinit) << "Translation table for descriptor "
                     << Desc->HostEntriesBegin << " cannot be found, probably "
                     << "it has been already removed.";
  }

  PM->TblMapMtx.unlock();

  ODBG(ODT_Deinit) << "Done unregistering library!";
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
      ODBG(ODT_Init) << "Trans table " << TransTable->HostTable.EntriesBegin
                     << " : " << TransTable->HostTable.EntriesEnd;
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
        REPORT() << "No image loaded for device id " << DeviceId << ".";
        Rc = OFFLOAD_FAIL;
        break;
      }

      // 2) Load the image onto the given device.
      auto BinaryOrErr = Device.loadBinary(Img);
      if (llvm::Error Err = BinaryOrErr.takeError()) {
        REPORT() << "Failed to load image " << llvm::toString(std::move(Err));
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
          if (!(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE))
            if (Device.RTL->get_global(Binary, Entry.Size, Entry.SymbolName,
                                       &DeviceEntry.Address) != OFFLOAD_SUCCESS)
              REPORT() << "Failed to load symbol " << Entry.SymbolName;

          // If unified memory is active, the corresponding global is a device
          // reference to the host global. We need to initialize the pointer on
          // the device to point to the memory on the host.
          if (!(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT_VTABLE) &&
              !(Entry.Flags & OMP_DECLARE_TARGET_INDIRECT) &&
              ((PM->getRequirements() & OMP_REQ_UNIFIED_SHARED_MEMORY) ||
               (PM->getRequirements() & OMPX_REQ_AUTO_ZERO_COPY))) {
            AsyncInfoTy AsyncInfo(Device);
            if (Device.submitData(DeviceEntry.Address,
                                  Entry.Address,
                                  Entry.Size,
                                  AsyncInfo) != OFFLOAD_SUCCESS)
              REPORT() << "Failed to write symbol for USM " << Entry.SymbolName;
           }
        } else if (Entry.Address) {
          if (Device.RTL->get_function(Binary, Entry.SymbolName,
                                       &DeviceEntry.Address) != OFFLOAD_SUCCESS)
            REPORT() << "Failed to load kernel " << Entry.SymbolName;
        }
        ODBG(ODT_Mapping) << "Entry point " << Entry.Address << " maps to"
                          << (Entry.Size ? " global" : "") << " "
                          << Entry.SymbolName << " (" << DeviceEntry.Address
                          << ")";

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

        ODBG(ODT_Mapping) << "Add mapping from host " << CurrHostEntry->Address
                          << " to device " << CurrDeviceEntry->Address
                          << " with size " << CurrDeviceEntry->Size
                          << ", name \"" << CurrDeviceEntry->SymbolName << "\"";
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
