//===- unittest/BinaryFormat/MachOTest.cpp - MachO support tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::MachO;

#if BYTE_ORDER == BIG_ENDIAN
// As discussed in Issue #86793, this test cannot work on a strict-alignment
// targets like SPARC.  Besides, it's undefined behaviour on big-endian hosts.
#define MAYBE_UnalignedLC DISABLED_UnalignedLC
#else
#define MAYBE_UnalignedLC UnalignedLC
#endif

TEST(MachOTest, MAYBE_UnalignedLC) {
  unsigned char Valid32BitMachO[] = {
      0xCE, 0xFA, 0xED, 0xFE, 0x07, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00,
      0x85, 0x80, 0x21, 0x01, 0x01, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
      0x5F, 0x5F, 0x50, 0x41, 0x47, 0x45, 0x5A, 0x45, 0x52, 0x4F, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x5F, 0x5F, 0x4C, 0x49,
      0x4E, 0x4B, 0x45, 0x44, 0x49, 0x54, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x40, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00,
      0x8C, 0x0B, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  mach_header *Header =
      reinterpret_cast<mach_header *>(Valid32BitMachO);
  if (!sys::IsLittleEndianHost)
    swapStruct(*Header);
  ASSERT_EQ(Header->magic, MH_MAGIC);
  unsigned char *Current = Valid32BitMachO + sizeof(mach_header);
  unsigned char *BufferEnd =
      Valid32BitMachO + sizeof(mach_header) + Header->sizeofcmds;
  while (Current < BufferEnd) {
    macho_load_command *LC =
        reinterpret_cast<macho_load_command *>(Current);
    if (!sys::IsLittleEndianHost)
      swapStruct(LC->load_command_data);
    ASSERT_EQ(LC->load_command_data.cmd, LC_SEGMENT);
    Current += LC->load_command_data.cmdsize;
  }
}

TEST(MachOTest, CPUType) {
#define CHECK_CPUTYPE(StrTriple, ExpectedCPUType)                              \
  ASSERT_EQ((MachO::CPUType)cantFail(MachO::getCPUType(Triple(StrTriple))),    \
            (ExpectedCPUType))
  CHECK_CPUTYPE("x86_64-apple-darwin", MachO::CPU_TYPE_X86_64);
  CHECK_CPUTYPE("x86_64h-apple-darwin", MachO::CPU_TYPE_X86_64);
  CHECK_CPUTYPE("i386-apple-darwin", MachO::CPU_TYPE_X86);
  CHECK_CPUTYPE("armv7-apple-darwin", MachO::CPU_TYPE_ARM);
  CHECK_CPUTYPE("thumbv7-apple-darwin", MachO::CPU_TYPE_ARM);
  CHECK_CPUTYPE("arm64-apple-darwin", MachO::CPU_TYPE_ARM64);
  CHECK_CPUTYPE("arm64e-apple-darwin", MachO::CPU_TYPE_ARM64);
  CHECK_CPUTYPE("arm64_32-apple-darwin", MachO::CPU_TYPE_ARM64_32);

  {
    // Not a mach-o.
    Expected<uint32_t> Type = MachO::getCPUType(Triple("x86_64-linux-unknown"));
    ASSERT_EQ(toString(Type.takeError()),
              "Unsupported triple for mach-o cpu type: x86_64-linux-unknown");
  }
  {
    // Not a valid mach-o architecture.
    Expected<uint32_t> Type = MachO::getCPUType(Triple("mips-apple-darwin"));
    ASSERT_EQ(toString(Type.takeError()),
              "Unsupported triple for mach-o cpu type: mips-apple-darwin");
  }
#undef CHECK_CPUTYPE
}

TEST(MachOTest, CPUSubType) {
#define CHECK_CPUSUBTYPE(StrTriple, ExpectedCPUSubType)                        \
  ASSERT_EQ(cantFail(MachO::getCPUSubType(Triple(StrTriple))),                 \
            ((uint32_t)ExpectedCPUSubType))
  CHECK_CPUSUBTYPE("x86_64-apple-darwin", MachO::CPU_SUBTYPE_X86_64_ALL);
  CHECK_CPUSUBTYPE("x86_64h-apple-darwin", MachO::CPU_SUBTYPE_X86_64_H);
  CHECK_CPUSUBTYPE("i386-apple-darwin", MachO::CPU_SUBTYPE_I386_ALL);
  CHECK_CPUSUBTYPE("arm-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7); // Default
  CHECK_CPUSUBTYPE("armv4t-apple-darwin", MachO::CPU_SUBTYPE_ARM_V4T);
  CHECK_CPUSUBTYPE("armv5t-apple-darwin", MachO::CPU_SUBTYPE_ARM_V5);
  CHECK_CPUSUBTYPE("armv5te-apple-darwin", MachO::CPU_SUBTYPE_ARM_V5);
  CHECK_CPUSUBTYPE("armv5tej-apple-darwin", MachO::CPU_SUBTYPE_ARM_V5);
  CHECK_CPUSUBTYPE("armv6-apple-darwin", MachO::CPU_SUBTYPE_ARM_V6);
  CHECK_CPUSUBTYPE("armv6k-apple-darwin", MachO::CPU_SUBTYPE_ARM_V6);
  CHECK_CPUSUBTYPE("armv7a-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7);
  CHECK_CPUSUBTYPE("armv7s-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7S);
  CHECK_CPUSUBTYPE("armv7k-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7K);
  CHECK_CPUSUBTYPE("armv6m-apple-darwin", MachO::CPU_SUBTYPE_ARM_V6M);
  CHECK_CPUSUBTYPE("armv7m-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7M);
  CHECK_CPUSUBTYPE("armv7em-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7EM);
  CHECK_CPUSUBTYPE("thumbv7-apple-darwin", MachO::CPU_SUBTYPE_ARM_V7);
  CHECK_CPUSUBTYPE("thumbv6-apple-darwin", MachO::CPU_SUBTYPE_ARM_V6);
  CHECK_CPUSUBTYPE("arm64-apple-darwin", MachO::CPU_SUBTYPE_ARM64_ALL);
  CHECK_CPUSUBTYPE("arm64e-apple-darwin", MachO::CPU_SUBTYPE_ARM64E);
  CHECK_CPUSUBTYPE("arm64_32-apple-darwin", MachO::CPU_SUBTYPE_ARM64_32_V8);

  {
    // Not a mach-o.
    Expected<uint32_t> Type =
        MachO::getCPUSubType(Triple("x86_64-linux-unknown"));
    ASSERT_EQ(
        toString(Type.takeError()),
        "Unsupported triple for mach-o cpu subtype: x86_64-linux-unknown");
  }
  {
    // Not a valid mach-o architecture.
    Expected<uint32_t> Type = MachO::getCPUSubType(Triple("mips-apple-darwin"));
    ASSERT_EQ(toString(Type.takeError()),
              "Unsupported triple for mach-o cpu subtype: mips-apple-darwin");
  }
#undef CHECK_CPUSUBTYPE
}

TEST(MachOTest, CPUSubTypePtrAuthABI) {
  {
    Expected<uint32_t> Type = MachO::getCPUSubType(
        Triple("x86_64-apple-darwin"), /*PtrAuthABIVersion=*/5,
        /*PtrAuthKernelABIVersion=*/false);
    ASSERT_EQ(toString(Type.takeError()),
              "ptrauth ABI version is only supported on arm64e.");
  }
  {
    Expected<uint32_t> Type = MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"), /*PtrAuthABIVersion=*/0x10,
        /*PtrAuthKernelABIVersion=*/false);
    ASSERT_EQ(toString(Type.takeError()),
              "The ptrauth ABI version needs to fit within 4 bits.");
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/5, /*PtrAuthKernelABIVersion=*/false));
    ASSERT_EQ(Type, 0x85000002U);
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/5, /*PtrAuthKernelABIVersion=*/true));
    ASSERT_EQ(Type, 0xC5000002U);
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/0xF, /*PtrAuthKernelABIVersion=*/false));
    ASSERT_EQ(Type, 0x8F000002U);
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/0xF, /*PtrAuthKernelABIVersion=*/true));
    ASSERT_EQ(Type, 0xCF000002U);
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/0, /*PtrAuthKernelABIVersion=*/false));
    ASSERT_EQ(Type, 0x80000002U);
  }
  {
    uint32_t Type = cantFail(MachO::getCPUSubType(
        Triple("arm64e-apple-darwin"),
        /*PtrAuthABIVersion=*/0, /*PtrAuthKernelABIVersion=*/true));
    ASSERT_EQ(Type, 0xC0000002U);
  }
}
