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

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfData.inc"
#include "llvm/Support/Error.h"

#include <cstring>
#include <string>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;
using namespace error;
using namespace llvm::offload::debug;

Expected<std::unique_ptr<ObjectFile>>
GenericGlobalHandlerTy::getELFObjectFile(DeviceImageTy &Image) {
  assert(utils::elf::isELF(Image.getMemoryBuffer().getBuffer()) &&
         "Input is not an ELF file");

  auto Expected =
      ELFObjectFileBase::createELFObjectFile(Image.getMemoryBuffer());
  if (!Expected) {
    return Plugin::error(ErrorCode::INVALID_BINARY, Expected.takeError(),
                         "error parsing binary");
  }
  return Expected;
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

  ODBG(OLDT_DataTransfer) << "Successfully " << (Device2Host ? "read" : "write")
                          << " " << HostGlobal.getSize()
                          << " bytes associated with global symbol '"
                          << HostGlobal.getName() << "' "
                          << (Device2Host ? "from" : "to") << " the device ("
                          << DeviceGlobal.getPtr() << " -> "
                          << HostGlobal.getPtr() << ").";

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
  auto SymOrErr = utils::elf::getSymbol(**ELFObjOrErr, SymName);
  if (!SymOrErr) {
    consumeError(SymOrErr.takeError());
    return false;
  }

  return SymOrErr->has_value();
}

Error GenericGlobalHandlerTy::getGlobalMetadataFromImage(
    GenericDeviceTy &Device, DeviceImageTy &Image, GlobalTy &ImageGlobal) {

  // Get the ELF object file for the image. Notice the ELF object may already
  // be created in previous calls, so we can reuse it.
  auto ELFObj = getELFObjectFile(Image);
  if (!ELFObj)
    return ELFObj.takeError();

  // Search the ELF symbol using the symbol name.
  auto SymOrErr = utils::elf::getSymbol(**ELFObj, ImageGlobal.getName());
  if (!SymOrErr)
    return Plugin::error(
        ErrorCode::NOT_FOUND, "failed ELF lookup of global '%s': %s",
        ImageGlobal.getName().data(), toString(SymOrErr.takeError()).data());

  if (!SymOrErr->has_value())
    return Plugin::error(ErrorCode::NOT_FOUND,
                         "failed to find global symbol '%s' in the ELF image",
                         ImageGlobal.getName().data());

  auto AddrOrErr = utils::elf::getSymbolAddress(**SymOrErr);
  // Get the section to which the symbol belongs.
  if (!AddrOrErr)
    return Plugin::error(
        ErrorCode::NOT_FOUND, "failed to get ELF symbol from global '%s': %s",
        ImageGlobal.getName().data(), toString(AddrOrErr.takeError()).data());

  // Setup the global symbol's address and size.
  ImageGlobal.setPtr(const_cast<void *>(*AddrOrErr));
  ImageGlobal.setSize((*SymOrErr)->getSize());

  return Plugin::success();
}

Error GenericGlobalHandlerTy::readGlobalFromImage(GenericDeviceTy &Device,
                                                  DeviceImageTy &Image,
                                                  const GlobalTy &HostGlobal) {

  GlobalTy ImageGlobal(HostGlobal.getName(), -1);
  if (auto Err = getGlobalMetadataFromImage(Device, Image, ImageGlobal))
    return Err;

  if (ImageGlobal.getSize() != HostGlobal.getSize())
    return Plugin::error(ErrorCode::INVALID_BINARY,
                         "transfer failed because global symbol '%s' has "
                         "%u bytes in the ELF image but %u bytes on the host",
                         HostGlobal.getName().data(), ImageGlobal.getSize(),
                         HostGlobal.getSize());

  ODBG(OLDT_DataTransfer) << "Global symbol '" << HostGlobal.getName()
                          << "' was found in the ELF image and "
                          << HostGlobal.getSize() << " bytes will copied from "
                          << ImageGlobal.getPtr() << " to "
                          << HostGlobal.getPtr() << ".";

  assert(Image.getStart() <= ImageGlobal.getPtr() &&
         utils::advancePtr(ImageGlobal.getPtr(), ImageGlobal.getSize()) <
             utils::advancePtr(Image.getStart(), Image.getSize()) &&
         "Attempting to read outside the image!");

  // Perform the copy from the image to the host memory.
  std::memcpy(HostGlobal.getPtr(), ImageGlobal.getPtr(), HostGlobal.getSize());

  return Plugin::success();
}

Expected<GPUProfGlobals>
GenericGlobalHandlerTy::readProfilingGlobals(GenericDeviceTy &Device,
                                             DeviceImageTy &Image) {
  const char *TableName = INSTR_PROF_QUOTE(INSTR_PROF_SECT_BOUNDS_TABLE);
  if (!isSymbolInImage(Device, Image, TableName))
    return GPUProfGlobals{};

  GPUProfGlobals ProfData;
  auto ObjFile = getELFObjectFile(Image);
  if (!ObjFile)
    return ObjFile.takeError();

  std::unique_ptr<ELFObjectFileBase> ELFObj(
      static_cast<ELFObjectFileBase *>(ObjFile->release()));
  ProfData.TargetTriple = ELFObj->makeTriple();

  __llvm_profile_gpu_sections Table = {};
  GlobalTy TableGlobal(TableName, sizeof(Table), &Table);
  if (auto Err = readGlobalFromDevice(Device, Image, TableGlobal))
    return Err;

  // Read the contiguous data from one of the profiling sections on the device.
  auto ReadSection = [&](const void *Start, const void *Stop,
                         SmallVector<char> &Out) -> Error {
    uintptr_t Begin = reinterpret_cast<uintptr_t>(Start);
    uintptr_t End = reinterpret_cast<uintptr_t>(Stop);
    size_t Size = End - Begin;
    Out.resize_for_overwrite(Size);
    return Size ? Device.dataRetrieve(Out.data(), Start, Size,
                                      /*AsyncInfo=*/nullptr)
                : Error::success();
  };

  if (auto Err =
          ReadSection(Table.NamesStart, Table.NamesStop, ProfData.NamesSection))
    return Err;
  if (auto Err = ReadSection(Table.CountersStart, Table.CountersStop,
                             ProfData.CountersSection))
    return Err;
  if (auto Err =
          ReadSection(Table.DataStart, Table.DataStop, ProfData.DataSection))
    return Err;

  ProfData.DeviceCountersDelta =
      reinterpret_cast<intptr_t>(Table.CountersStart) -
      reinterpret_cast<intptr_t>(Table.DataStart);

  // Get the profiling version from the device.
  if (auto Err = Device.dataRetrieve(&ProfData.Version, Table.VersionVar,
                                     sizeof(uint64_t),
                                     /*AsyncInfo=*/nullptr))
    return Err;

  return ProfData;
}

void GPUProfGlobals::dump() const {
  outs() << "======= GPU Profile =======\nTarget: " << TargetTriple.str()
         << "\n";

  size_t NumCounters = CountersSection.size() / sizeof(int64_t);
  outs() << "======== Counters (" << NumCounters << ") =========\n";
  auto *Counts = reinterpret_cast<const int64_t *>(CountersSection.data());
  for (size_t i = 0; i < NumCounters; i++) {
    if (i > 0 && i % 10 == 0)
      outs() << "\n";
    else if (i != 0)
      outs() << " ";
    outs() << Counts[i];
  }
  outs() << "\n";

  size_t NumDataEntries = DataSection.size() / sizeof(__llvm_profile_data);
  outs() << "========== Data (" << NumDataEntries << ") ===========\n";

  outs() << "======== Functions ========\n";
  InstrProfSymtab Symtab;
  if (Error Err =
          Symtab.create(StringRef(NamesSection.data(), NamesSection.size())))
    consumeError(std::move(Err));
  Symtab.dumpNames(outs());
  outs() << "===========================\n";
}

Error GPUProfGlobals::write() const {
  if (!__llvm_write_custom_profile)
    return Plugin::error(ErrorCode::INVALID_BINARY,
                         "could not find symbol __llvm_write_custom_profile. "
                         "The compiler-rt profiling library must be linked for "
                         "GPU PGO to work.");

  // Lay out as [Data][Counters][Names] to match the raw profile format order.
  // TODO: Move this interface to compiler-rt.
  SmallVector<char> Buffer(DataSection.size() + CountersSection.size() +
                           NamesSection.size());
  char *DataBegin = Buffer.data();
  char *CountersBegin = DataBegin + DataSection.size();
  char *NamesBegin = CountersBegin + CountersSection.size();

  memcpy(DataBegin, DataSection.data(), DataSection.size());
  memcpy(CountersBegin, CountersSection.data(), CountersSection.size());
  memcpy(NamesBegin, NamesSection.data(), NamesSection.size());

  // Adjust CounterPtr values so they are consistent with the host layout rather
  // than the device layout.
  intptr_t HostDelta = CountersBegin - DataBegin;
  intptr_t Adjustment = HostDelta - DeviceCountersDelta;
  auto *Records = reinterpret_cast<__llvm_profile_data *>(DataBegin);
  size_t NumRecords = DataSection.size() / sizeof(__llvm_profile_data);
  for (size_t I = 0; I < NumRecords; I++)
    Records[I].CounterPtr = reinterpret_cast<void *>(
        reinterpret_cast<intptr_t>(Records[I].CounterPtr) + Adjustment);

  int Result = __llvm_write_custom_profile(
      TargetTriple.str().c_str(),
      reinterpret_cast<const __llvm_profile_data *>(DataBegin),
      reinterpret_cast<const __llvm_profile_data *>(DataBegin +
                                                    DataSection.size()),
      CountersBegin, CountersBegin + CountersSection.size(), NamesBegin,
      NamesBegin + NamesSection.size(), &Version);
  if (Result != 0)
    return Plugin::error(ErrorCode::HOST_IO,
                         "error writing GPU PGO data to file");

  return Plugin::success();
}

bool GPUProfGlobals::empty() const {
  return CountersSection.empty() && DataSection.empty() && NamesSection.empty();
}
