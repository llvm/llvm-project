//===-- CoreSpec.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoreSpec.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/YAMLTraits.h"
#include <stdio.h>
#include <string>

using llvm::yaml::Input;
using llvm::yaml::IO;
using llvm::yaml::MappingTraits;

template <> struct llvm::yaml::MappingTraits<RegisterNameAndValue> {
  static void mapping(IO &io, RegisterNameAndValue &name_value) {
    io.mapRequired("name", name_value.name);
    io.mapRequired("value", name_value.value);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(RegisterNameAndValue)

template <> struct llvm::yaml::ScalarEnumerationTraits<RegisterFlavor> {
  static void enumeration(IO &io, RegisterFlavor &flavor) {
    io.enumCase(flavor, "gpr", RegisterFlavor::GPR);
    io.enumCase(flavor, "fpr", RegisterFlavor::FPR);
    io.enumCase(flavor, "exc", RegisterFlavor::EXC);
  }
};

template <> struct llvm::yaml::MappingTraits<RegisterSet> {
  static void mapping(IO &io, RegisterSet &regset) {
    io.mapRequired("flavor", regset.flavor);
    io.mapRequired("registers", regset.registers);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(RegisterSet)

template <> struct llvm::yaml::MappingTraits<Thread> {
  static void mapping(IO &io, Thread &thread) {
    io.mapRequired("regsets", thread.regsets);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(Thread)

template <> struct llvm::yaml::MappingTraits<MemoryRegion> {
  static void mapping(IO &io, MemoryRegion &memory) {
    io.mapRequired("addr", memory.addr);
    io.mapOptional("UInt8", memory.bytes);
    io.mapOptional("UInt32", memory.words);
    io.mapOptional("UInt64", memory.doublewords);

    if (memory.bytes.size()) {
      memory.type = MemoryType::UInt8;
      memory.size = memory.bytes.size();
    } else if (memory.words.size()) {
      memory.type = MemoryType::UInt32;
      memory.size = memory.words.size() * 4;
    } else if (memory.doublewords.size()) {
      memory.type = MemoryType::UInt64;
      memory.size = memory.doublewords.size() * 8;
    }
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(MemoryRegion)

template <> struct llvm::yaml::MappingTraits<Binary> {
  static void mapping(IO &io, Binary &binary) {
    io.mapOptional("name", binary.name);
    io.mapRequired("uuid", binary.uuid);
    std::optional<uint64_t> va, slide;
    io.mapOptional("virtual-address", va);
    io.mapOptional("slide", slide);
    if (va && *va != UINT64_MAX) {
      binary.value_is_slide = false;
      binary.value = *va;
    } else if (slide && *slide != UINT64_MAX) {
      binary.value_is_slide = true;
      binary.value = *slide;
    } else {
      fprintf(stderr,
              "No virtual-address or slide specified for binary %s, aborting\n",
              binary.uuid.c_str());
      exit(1);
    }
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(Binary)

template <> struct llvm::yaml::MappingTraits<AddressableBits> {
  static void mapping(IO &io, AddressableBits &addr_bits) {
    std::optional<int> addressable_bits;
    io.mapOptional("num-bits", addressable_bits);
    if (addressable_bits) {
      addr_bits.lowmem_bits = *addressable_bits;
      addr_bits.highmem_bits = *addressable_bits;
    } else {
      io.mapOptional("lowmem-num-bits", addr_bits.lowmem_bits);
      io.mapOptional("highmem-num-bits", addr_bits.highmem_bits);
    }
  }
};

template <> struct llvm::yaml::MappingTraits<CoreSpec> {
  static void mapping(IO &io, CoreSpec &corespec) {
    std::string cpuname;
    io.mapRequired("cpu", cpuname);
    if (cpuname == "armv7m") {
      corespec.cputype = llvm::MachO::CPU_TYPE_ARM;
      corespec.cpusubtype = llvm::MachO::CPU_SUBTYPE_ARM_V7M;
    } else if (cpuname == "armv7") {
      corespec.cputype = llvm::MachO::CPU_TYPE_ARM;
      corespec.cpusubtype = llvm::MachO::CPU_SUBTYPE_ARM_ALL;
    } else if (cpuname == "riscv") {
      corespec.cputype = llvm::MachO::CPU_TYPE_RISCV;
      corespec.cpusubtype = llvm::MachO::CPU_SUBTYPE_RISCV_ALL;
    } else if (cpuname == "arm64") {
      corespec.cputype = llvm::MachO::CPU_TYPE_ARM64;
      corespec.cpusubtype = llvm::MachO::CPU_SUBTYPE_ARM64_ALL;
    } else {
      fprintf(stderr, "Unrecognized cpu name %s, exiting.\n", cpuname.c_str());
      exit(1);
    }
    io.mapOptional("threads", corespec.threads);
    io.mapOptional("memory-regions", corespec.memory_regions);
    if (corespec.cputype == llvm::MachO::CPU_TYPE_ARM ||
        corespec.cputype == llvm::MachO::CPU_TYPE_RISCV)
      corespec.wordsize = 4;
    else if (corespec.cputype == llvm::MachO::CPU_TYPE_ARM64)
      corespec.wordsize = 8;
    else {
      fprintf(stderr,
              "Unrecognized cputype, could not set wordsize, exiting.\n");
      exit(1);
    }
    io.mapOptional("addressable-bits", corespec.addressable_bits);
    io.mapOptional("binaries", corespec.binaries);
    if (corespec.addressable_bits) {
      if (!corespec.addressable_bits->lowmem_bits)
        corespec.addressable_bits->lowmem_bits = corespec.wordsize * 8;
      if (!corespec.addressable_bits->highmem_bits)
        corespec.addressable_bits->highmem_bits = corespec.wordsize * 8;
    }
  }
};

CoreSpec from_yaml(char *buf, size_t len) {
  llvm::StringRef file_corespec_strref(buf, len);

  Input yin(file_corespec_strref);

  CoreSpec v;
  yin >> v;

  if (yin.error()) {
    fprintf(stderr, "Unable to parse YAML, exiting\n");
    exit(1);
  }

  return v;
}
