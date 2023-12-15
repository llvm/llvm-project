//===- GlobalHandler.cpp - Target independent global & env. var handling --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target independent global handler and environment manager.
//
//===----------------------------------------------------------------------===//

#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "Utils/ELF.h"

#include "Shared/Utils.h"

#include "llvm/Support/Error.h"

#include <cstring>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;

const ELF64LEObjectFile *
GenericGlobalHandlerTy::getOrCreateELFObjectFile(const GenericDeviceTy &Device,
                                                 DeviceImageTy &Image) {

  auto Search = ELFObjectFiles.find(Image.getId());
  if (Search != ELFObjectFiles.end())
    // The ELF object file was already there.
    return &Search->second;

  // The ELF object file we are checking is not created yet.
  Expected<ELF64LEObjectFile> ElfOrErr =
      ELF64LEObjectFile::create(Image.getMemoryBuffer());
  if (!ElfOrErr) {
    consumeError(ElfOrErr.takeError());
    return nullptr;
  }

  auto Result =
      ELFObjectFiles.try_emplace(Image.getId(), std::move(ElfOrErr.get()));
  assert(Result.second && "Map insertion failed");
  assert(Result.first != ELFObjectFiles.end() && "Map insertion failed");

  return &Result.first->second;
}

Error GenericGlobalHandlerTy::moveGlobalBetweenDeviceAndHost(
    GenericDeviceTy &Device, DeviceImageTy &Image, const GlobalTy &HostGlobal,
    bool Device2Host) {

  GlobalTy DeviceGlobal(HostGlobal.getName(), HostGlobal.getSize());

  // Get the metadata from the global on the device.
  if (auto Err = getGlobalMetadataFromDevice(Device, Image, DeviceGlobal))
    return Err;

  // Perform the actual transfer.
  return moveGlobalBetweenDeviceAndHost(Device, HostGlobal, DeviceGlobal,
                                        Device2Host);
}

/// Actually move memory between host and device. See readGlobalFromDevice and
/// writeGlobalToDevice for the interface description.
Error GenericGlobalHandlerTy::moveGlobalBetweenDeviceAndHost(
    GenericDeviceTy &Device, const GlobalTy &HostGlobal,
    const GlobalTy &DeviceGlobal, bool Device2Host) {

  // Transfer the data from the source to the destination.
  if (Device2Host) {
    if (auto Err =
            Device.dataRetrieve(HostGlobal.getPtr(), DeviceGlobal.getPtr(),
                                HostGlobal.getSize(), nullptr))
      return Err;
  } else {
    if (auto Err = Device.dataSubmit(DeviceGlobal.getPtr(), HostGlobal.getPtr(),
                                     HostGlobal.getSize(), nullptr))
      return Err;
  }

  DP("Succesfully %s %u bytes associated with global symbol '%s' %s the device "
     "(%p -> %p).\n",
     Device2Host ? "read" : "write", HostGlobal.getSize(),
     HostGlobal.getName().data(), Device2Host ? "from" : "to",
     DeviceGlobal.getPtr(), HostGlobal.getPtr());

  return Plugin::success();
}

bool GenericGlobalHandlerTy::isSymbolInImage(GenericDeviceTy &Device,
                                             DeviceImageTy &Image,
                                             StringRef SymName) {
  // Get the ELF object file for the image. Notice the ELF object may already
  // be created in previous calls, so we can reuse it. If this is unsuccessful
  // just return false as we couldn't find it.
  const ELF64LEObjectFile *ELFObj = getOrCreateELFObjectFile(Device, Image);
  if (!ELFObj)
    return false;

  // Search the ELF symbol using the symbol name.
  auto SymOrErr = utils::elf::getSymbol(*ELFObj, SymName);
  if (!SymOrErr) {
    consumeError(SymOrErr.takeError());
    return false;
  }

  return *SymOrErr;
}

Error GenericGlobalHandlerTy::getGlobalMetadataFromImage(
    GenericDeviceTy &Device, DeviceImageTy &Image, GlobalTy &ImageGlobal) {

  // Get the ELF object file for the image. Notice the ELF object may already
  // be created in previous calls, so we can reuse it.
  const ELF64LEObjectFile *ELFObj = getOrCreateELFObjectFile(Device, Image);
  if (!ELFObj)
    return Plugin::error("Unable to create ELF object for image %p",
                         Image.getStart());

  // Search the ELF symbol using the symbol name.
  auto SymOrErr = utils::elf::getSymbol(*ELFObj, ImageGlobal.getName());
  if (!SymOrErr)
    return Plugin::error("Failed ELF lookup of global '%s': %s",
                         ImageGlobal.getName().data(),
                         toString(SymOrErr.takeError()).data());

  if (!*SymOrErr)
    return Plugin::error("Failed to find global symbol '%s' in the ELF image",
                         ImageGlobal.getName().data());

  auto AddrOrErr = utils::elf::getSymbolAddress(*ELFObj, **SymOrErr);
  // Get the section to which the symbol belongs.
  if (!AddrOrErr)
    return Plugin::error("Failed to get ELF symbol from global '%s': %s",
                         ImageGlobal.getName().data(),
                         toString(AddrOrErr.takeError()).data());

  // Setup the global symbol's address and size.
  ImageGlobal.setPtr(const_cast<void *>(*AddrOrErr));
  ImageGlobal.setSize((*SymOrErr)->st_size);

  return Plugin::success();
}

Error GenericGlobalHandlerTy::readGlobalFromImage(GenericDeviceTy &Device,
                                                  DeviceImageTy &Image,
                                                  const GlobalTy &HostGlobal) {

  GlobalTy ImageGlobal(HostGlobal.getName(), -1);
  if (auto Err = getGlobalMetadataFromImage(Device, Image, ImageGlobal))
    return Err;

  if (ImageGlobal.getSize() != HostGlobal.getSize())
    return Plugin::error("Transfer failed because global symbol '%s' has "
                         "%u bytes in the ELF image but %u bytes on the host",
                         HostGlobal.getName().data(), ImageGlobal.getSize(),
                         HostGlobal.getSize());

  DP("Global symbol '%s' was found in the ELF image and %u bytes will copied "
     "from %p to %p.\n",
     HostGlobal.getName().data(), HostGlobal.getSize(), ImageGlobal.getPtr(),
     HostGlobal.getPtr());

  assert(Image.getStart() <= ImageGlobal.getPtr() &&
         advanceVoidPtr(ImageGlobal.getPtr(), ImageGlobal.getSize()) <
             advanceVoidPtr(Image.getStart(), Image.getSize()) &&
         "Attempting to read outside the image!");

  // Perform the copy from the image to the host memory.
  std::memcpy(HostGlobal.getPtr(), ImageGlobal.getPtr(), HostGlobal.getSize());

  return Plugin::success();
}
