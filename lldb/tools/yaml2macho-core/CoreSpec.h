//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CoreSpec holds the internal representation of the data that will be
/// written into the corefile.  Theads, register sets within threads, registers
/// within register sets.  Block of memory.  Metadata about the CPU or binaries
/// that were present.
//===----------------------------------------------------------------------===//

#ifndef YAML2MACHOCOREFILE_CORESPEC_H
#define YAML2MACHOCOREFILE_CORESPEC_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct RegisterNameAndValue {
  std::string name;
  uint64_t value;
};

enum RegisterFlavor { GPR = 0, FPR, EXC };

struct RegisterSet {
  RegisterFlavor flavor;
  std::vector<RegisterNameAndValue> registers;
};

struct Thread {
  std::vector<RegisterSet> regsets;
};

enum MemoryType { UInt8 = 0, UInt32, UInt64 };

struct MemoryRegion {
  uint64_t addr;
  MemoryType type;
  uint32_t size;
  // One of the following formats.
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> words;
  std::vector<uint64_t> doublewords;
};

struct AddressableBits {
  std::optional<int> lowmem_bits;
  std::optional<int> highmem_bits;
};

struct Binary {
  std::string name;
  std::string uuid;
  bool value_is_slide;
  uint64_t value;
};

struct CoreSpec {
  uint32_t cputype;
  uint32_t cpusubtype;
  int wordsize;

  std::vector<Thread> threads;
  std::vector<MemoryRegion> memory_regions;

  std::optional<AddressableBits> addressable_bits;
  std::vector<Binary> binaries;

  CoreSpec() : cputype(0), cpusubtype(0), wordsize(0) {}
};

CoreSpec from_yaml(char *buf, size_t len);

#endif
