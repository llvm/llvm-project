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

Expected<ELF64LEObjectFile>
GenericGlobalHandlerTy::getELFObjectFile(DeviceImageTy &Image) {
  assert(utils::elf::isELF(Image.getMemoryBuffer().getBuffer()) &&
         "Input is not an ELF file");

  Expected<ELF64LEObjectFile> ElfOrErr =
      ELF64LEObjectFile::create(Image.getMemoryBuffer());
  return ElfOrErr;
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

  DP("Succesfully %s %u bytes associated with global symbol '%s' %s the "
     "device "
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
  auto ELFObjOrErr = getELFObjectFile(Image);
  if (!ELFObjOrErr) {
    consumeError(ELFObjOrErr.takeError());
    return false;
  }

  // Search the ELF symbol using the symbol name.
  auto SymOrErr = utils::elf::getSymbol(*ELFObjOrErr, SymName);
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
  auto ELFObj = getELFObjectFile(Image);
  if (!ELFObj)
    return ELFObj.takeError();

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

bool GenericGlobalHandlerTy::hasProfilingGlobals(GenericDeviceTy &Device,
                                                 DeviceImageTy &Image) {
  GlobalTy global(getInstrProfNamesVarName().str(), 0);
  if (auto Err = getGlobalMetadataFromImage(Device, Image, global)) {
    consumeError(std::move(Err));
    return false;
  }
  return true;
}

Expected<GPUProfGlobals>
GenericGlobalHandlerTy::readProfilingGlobals(GenericDeviceTy &Device,
                                             DeviceImageTy &Image) {
  GPUProfGlobals profdata;
  const auto *elf = getOrCreateELFObjectFile(Device, Image);
  profdata.targetTriple = elf->makeTriple();
  // Iterate through
  for (auto &sym : elf->symbols()) {
    if (auto name = sym.getName()) {
      // Check if given current global is a profiling global based
      // on name
      if (name->equals(getInstrProfNamesVarName())) {
        // Read in profiled function names
        std::vector<char> chars(sym.getSize() / sizeof(char), ' ');
        GlobalTy NamesGlobal(name->str(), sym.getSize(), chars.data());
        if (auto Err = readGlobalFromDevice(Device, Image, NamesGlobal))
          return Err;
        std::string names(chars.begin(), chars.end());
        profdata.names = std::move(names);
      } else if (name->starts_with(getInstrProfCountersVarPrefix())) {
        // Read global variable profiling counts
        std::vector<int64_t> counts(sym.getSize() / sizeof(int64_t), 0);
        GlobalTy CountGlobal(name->str(), sym.getSize(), counts.data());
        if (auto Err = readGlobalFromDevice(Device, Image, CountGlobal))
          return Err;
        profdata.counts.push_back(std::move(counts));
      } else if (name->starts_with(getInstrProfDataVarPrefix())) {
        // Read profiling data for this global variable
        __llvm_profile_data data{};
        GlobalTy DataGlobal(name->str(), sym.getSize(), &data);
        if (auto Err = readGlobalFromDevice(Device, Image, DataGlobal))
          return Err;
        profdata.data.push_back(std::move(data));
      }
    }
  }
  return profdata;
}

void GPUProfGlobals::dump() const {
  llvm::outs() << "======= GPU Profile =======\nTarget: " << targetTriple.str()
               << "\n";

  llvm::outs() << "======== Counters =========\n";
  for (const auto &count : counts) {
    llvm::outs() << "[";
    for (size_t i = 0; i < count.size(); i++) {
      if (i == 0)
        llvm::outs() << " ";
      llvm::outs() << count[i] << " ";
    }
    llvm::outs() << "]\n";
  }

  llvm::outs() << "========== Data ===========\n";
  for (const auto &d : data) {
    llvm::outs() << "{ ";
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer)                     \
  llvm::outs() << d.Name << " ";
#include "llvm/ProfileData/InstrProfData.inc"
    llvm::outs() << " }\n";
  }

  llvm::outs() << "======== Functions ========\n";
  InstrProfSymtab symtab;
  if (Error Err = symtab.create(StringRef(names))) {
    consumeError(std::move(Err));
  }
  symtab.dumpNames(llvm::outs());
  llvm::outs() << "===========================\n";
}
