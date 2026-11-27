//===-- SubprocessMemory.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubprocessMemory.h"
#include "Error.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cerrno>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace llvm {
namespace exegesis {

#if defined(__linux__) && !defined(__ANDROID__)

Error SubprocessMemory::initializeSubprocessMemory() {
  AuxiliaryMemoryFD = memfd_create("auxmem", MFD_CLOEXEC);
  if (AuxiliaryMemoryFD == -1)
    return make_error<Failure>(
        "Failed to create shared memory object for auxiliary memory: " +
        Twine(strerror(errno)));
  if (ftruncate(AuxiliaryMemoryFD, AuxiliaryMemorySize) != 0)
    return make_error<Failure>("Truncating the auxiliary memory failed: " +
                               Twine(strerror(errno)));
  return Error::success();
}

Error SubprocessMemory::addMemoryDefinition(
    StringMap<MemoryValue> MemoryDefinitions) {
  MemoryDefinitionFDs.reserve(MemoryDefinitions.size());
  for (const auto &E : MemoryDefinitions) {
    const MemoryValue &MemVal = E.second;
    std::string SharedMemoryName = formatv("memdef{0}", MemVal.Index);
    int SharedMemoryFD = memfd_create(SharedMemoryName.c_str(), MFD_CLOEXEC);
    if (SharedMemoryFD == -1)
      return make_error<Failure>(
          "Failed to create shared memory object for memory definition: " +
          Twine(strerror(errno)));
    MemoryDefinitionFDs.push_back({MemVal.Index, SharedMemoryFD});
    if (ftruncate(SharedMemoryFD, MemVal.SizeBytes) != 0)
      return make_error<Failure>("Truncating a memory definition failed: " +
                                 Twine(strerror(errno)));

    char *SharedMemoryMapping = static_cast<char *>(
        mmap(nullptr, MemVal.SizeBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
             SharedMemoryFD, 0));
    if (SharedMemoryMapping == MAP_FAILED)
      return make_error<Failure>(
          "Mapping a memory definition in the parent failed: " +
          Twine(strerror(errno)));
    // fill the buffer with the specified value
    size_t CurrentByte = 0;
    const size_t ValueWidthBytes = MemVal.Value.getBitWidth() / 8;
    while (CurrentByte < MemVal.SizeBytes - ValueWidthBytes) {
      memcpy(SharedMemoryMapping + CurrentByte, MemVal.Value.getRawData(),
             ValueWidthBytes);
      CurrentByte += ValueWidthBytes;
    }
    // fill the last section
    memcpy(SharedMemoryMapping + CurrentByte, MemVal.Value.getRawData(),
           MemVal.SizeBytes - CurrentByte);
    if (munmap(SharedMemoryMapping, MemVal.SizeBytes) != 0)
      return make_error<Failure>(
          "Unmapping a memory definition in the parent failed: " +
          Twine(strerror(errno)));
  }
  return Error::success();
}

Expected<int> SubprocessMemory::setupAuxiliaryMemoryInSubprocess(
    int CounterFileDescriptor) const {
  // set up memory value file descriptors
  int *AuxiliaryMemoryMapping = static_cast<int *>(
      mmap(nullptr, AuxiliaryMemorySize, PROT_READ | PROT_WRITE, MAP_SHARED,
           AuxiliaryMemoryFD, 0));
  if (AuxiliaryMemoryMapping == MAP_FAILED)
    return make_error<Failure>("Mapping auxiliary memory failed");
  AuxiliaryMemoryMapping[0] = CounterFileDescriptor;
  for (const MemoryDefinitionFD &MemDef : MemoryDefinitionFDs)
    AuxiliaryMemoryMapping[AuxiliaryMemoryOffset + MemDef.Index] = MemDef.FD;
  if (munmap(AuxiliaryMemoryMapping, AuxiliaryMemorySize) == -1)
    return make_error<Failure>("Unmapping auxiliary memory failed");
  return AuxiliaryMemoryFD;
}

SubprocessMemory::~SubprocessMemory() {
  if (AuxiliaryMemoryFD != -1) {
    if (close(AuxiliaryMemoryFD) != 0)
      errs() << "Failed to close auxiliary memory FD: " << strerror(errno)
             << "\n";
  }
  for (const MemoryDefinitionFD &MemDef : MemoryDefinitionFDs) {
    if (close(MemDef.FD) != 0)
      errs() << "Failed to close shared memory FD: " << strerror(errno) << "\n";
  }
}

#else

Error SubprocessMemory::initializeSubprocessMemory() {
  return make_error<Failure>(
      "initializeSubprocessMemory is only supported on Linux");
}

Error SubprocessMemory::addMemoryDefinition(
    StringMap<MemoryValue> MemoryDefinitions) {
  return make_error<Failure>("addMemoryDefinitions is only supported on Linux");
}

Expected<int> SubprocessMemory::setupAuxiliaryMemoryInSubprocess(
    int CounterFileDescriptor) const {
  return make_error<Failure>(
      "setupAuxiliaryMemoryInSubprocess is only supported on Linux");
}

SubprocessMemory::~SubprocessMemory() {}

#endif // defined(__linux__) && !defined(__ANDROID__)

Expected<int>
SubprocessMemory::getMemoryDefinitionFD(size_t DefinitionIndex) const {
  for (const MemoryDefinitionFD &MemDef : MemoryDefinitionFDs) {
    if (MemDef.Index == DefinitionIndex)
      return MemDef.FD;
  }
  return make_error<Failure>("Memory definition index not found");
}

} // namespace exegesis
} // namespace llvm
