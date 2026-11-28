//===-- SubprocessMemory.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines a class that automatically handles auxiliary memory and the
/// underlying shared memory backings for memory definitions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_SUBPROCESSMEMORY_H
#define LLVM_TOOLS_LLVM_EXEGESIS_SUBPROCESSMEMORY_H

#include "BenchmarkResult.h"
#include "llvm/ADT/StringMap.h"
#include <vector>

namespace llvm {
namespace exegesis {

class SubprocessMemory {
public:
  static constexpr size_t AuxiliaryMemoryOffset = 1;
  static constexpr size_t AuxiliaryMemorySize = 4096;

  SubprocessMemory() = default;
  SubprocessMemory(const SubprocessMemory &) = delete;
  SubprocessMemory &operator=(const SubprocessMemory &) = delete;

  Error initializeSubprocessMemory();

  // The following function sets up memory definitions. It creates
  // shared memory objects for the definitions and fills them with the
  // specified values.
  Error addMemoryDefinition(StringMap<MemoryValue> MemoryDefinitions);

  // The following function sets up the auxiliary memory by writing the counter
  // file descriptor and memory definition file descriptors into the auxiliary
  // memory section.
  Expected<int>
  setupAuxiliaryMemoryInSubprocess(int CounterFileDescriptor) const;

  Expected<int> getMemoryDefinitionFD(size_t DefinitionIndex) const;

  ~SubprocessMemory();

private:
  struct MemoryDefinitionFD {
    size_t Index;
    int FD;
  };

  int AuxiliaryMemoryFD = -1;
  std::vector<MemoryDefinitionFD> MemoryDefinitionFDs;
};

} // namespace exegesis
} // namespace llvm

#endif
