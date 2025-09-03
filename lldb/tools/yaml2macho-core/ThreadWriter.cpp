//===-- ThreadWriter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ThreadWriter.h"
#include "CoreSpec.h"
#include "Utility.h"
#include "llvm/BinaryFormat/MachO.h"
#include <algorithm>
#include <stdio.h>

#define ARM_THREAD_STATE 1
#define ARM_THREAD_STATE_COUNT 17
#define ARM_EXCEPTION_STATE 3
#define ARM_EXCEPTION_STATE_COUNT 3

std::vector<RegisterNameAndValue>::const_iterator
find_by_name(std::vector<RegisterNameAndValue>::const_iterator first,
             std::vector<RegisterNameAndValue>::const_iterator last,
             const char *name) {
  for (; first != last; ++first)
    if (first->name == name)
      return first;
  return last;
}

void add_reg_value(CoreSpec &spec, std::vector<uint8_t> &buf,
                   const std::vector<RegisterNameAndValue> &registers,
                   const char *regname, int regsize) {
  const auto it = find_by_name(registers.begin(), registers.end(), regname);
  if (it != registers.end()) {
    if (regsize == 8)
      add_uint64(buf, it->value);
    else
      add_uint32(buf, it->value);
  } else {
    if (regsize == 8)
      add_uint64(buf, 0);
    else
      add_uint32(buf, 0);
  }
}

void add_reg_value_32(CoreSpec &spec, std::vector<uint8_t> &buf,
                      const std::vector<RegisterNameAndValue> &registers,
                      const char *regname) {
  add_reg_value(spec, buf, registers, regname, 4);
}

void add_reg_value_64(CoreSpec &spec, std::vector<uint8_t> &buf,
                      const std::vector<RegisterNameAndValue> &registers,
                      const char *regname) {
  add_reg_value(spec, buf, registers, regname, 8);
}

void add_lc_threads_armv7(CoreSpec &spec,
                          std::vector<std::vector<uint8_t>> &load_commands) {
  for (const Thread &th : spec.threads) {
    std::vector<uint8_t> lc;
    int size_of_all_flavors = 0;
    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR)
        size_of_all_flavors += (ARM_THREAD_STATE_COUNT * 4);
      if (rs.flavor == RegisterFlavor::EXC)
        size_of_all_flavors += (ARM_EXCEPTION_STATE_COUNT * 4);
    }
    int cmdsize = 4 * 2;                  // cmd, cmdsize
    cmdsize += 4 * 2 * th.regsets.size(); // flavor, count (per register flavor)
    cmdsize += size_of_all_flavors;       // size of all the register set data

    add_uint32(lc, llvm::MachO::LC_THREAD); // thread_command.cmd
    add_uint32(lc, cmdsize);                // thread_command.cmdsize
    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR) {
        add_uint32(lc, ARM_THREAD_STATE);       // thread_command.flavor
        add_uint32(lc, ARM_THREAD_STATE_COUNT); // thread_command.count
        const char *names[] = {"r0",  "r1", "r2", "r3", "r4",   "r5",
                               "r6",  "r7", "r8", "r9", "r10",  "r11",
                               "r12", "sp", "lr", "pc", "cpsr", nullptr};
        for (int i = 0; names[i]; i++)
          add_reg_value_32(spec, lc, rs.registers, names[i]);
      }
      if (rs.flavor == RegisterFlavor::EXC) {
        add_uint32(lc, ARM_EXCEPTION_STATE);       // thread_command.flavor
        add_uint32(lc, ARM_EXCEPTION_STATE_COUNT); // thread_command.count
        const char *names[] = {"far", "esr", "exception", nullptr};
        for (int i = 0; names[i]; i++)
          add_reg_value_32(spec, lc, rs.registers, names[i]);
      }
    }
    load_commands.push_back(lc);
  }
}

#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT 68
#define ARM_EXCEPTION_STATE64 7
#define ARM_EXCEPTION_STATE64_COUNT 4

void add_lc_threads_arm64(CoreSpec &spec,
                          std::vector<std::vector<uint8_t>> &load_commands) {
  for (const Thread &th : spec.threads) {
    std::vector<uint8_t> lc;
    int size_of_all_flavors = 0;
    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR)
        size_of_all_flavors += (ARM_THREAD_STATE64_COUNT * 4);
      if (rs.flavor == RegisterFlavor::EXC)
        size_of_all_flavors += (ARM_EXCEPTION_STATE64_COUNT * 4);
    }
    int cmdsize = 4 * 2;                  // cmd, cmdsize
    cmdsize += 4 * 2 * th.regsets.size(); // flavor, count (per register flavor)
    cmdsize += size_of_all_flavors;       // size of all the register set data

    add_uint32(lc, llvm::MachO::LC_THREAD); // thread_command.cmd
    add_uint32(lc, cmdsize);                // thread_command.cmdsize

    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR) {
        add_uint32(lc, ARM_THREAD_STATE64);       // thread_command.flavor
        add_uint32(lc, ARM_THREAD_STATE64_COUNT); // thread_command.count
        const char *names[] = {"x0",  "x1",  "x2",  "x3",  "x4",  "x5",   "x6",
                               "x7",  "x8",  "x9",  "x10", "x11", "x12",  "x13",
                               "x14", "x15", "x16", "x17", "x18", "x19",  "x20",
                               "x21", "x22", "x23", "x24", "x25", "x26",  "x27",
                               "x28", "fp",  "lr",  "sp",  "pc",  nullptr};
        for (int i = 0; names[i]; i++)
          add_reg_value_64(spec, lc, rs.registers, names[i]);

        // cpsr is a 4-byte reg
        add_reg_value_32(spec, lc, rs.registers, "cpsr");
        // the 4 bytes of zeroes
        add_uint32(lc, 0);
      }
      if (rs.flavor == RegisterFlavor::EXC) {
        add_uint32(lc, ARM_EXCEPTION_STATE64); // thread_command.flavor
        add_uint32(lc,
                   ARM_EXCEPTION_STATE64_COUNT); // thread_command.count
        add_reg_value_64(spec, lc, rs.registers, "far");
        add_reg_value_32(spec, lc, rs.registers, "esr");
        add_reg_value_32(spec, lc, rs.registers, "exception");
      }
    }
    load_commands.push_back(lc);
  }
}

#define RV32_THREAD_STATE 2
#define RV32_THREAD_STATE_COUNT 33

void add_lc_threads_riscv(CoreSpec &spec,
                          std::vector<std::vector<uint8_t>> &load_commands) {
  for (const Thread &th : spec.threads) {
    std::vector<uint8_t> lc;
    int size_of_all_flavors = 0;
    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR)
        size_of_all_flavors += (RV32_THREAD_STATE_COUNT * 4);
    }
    int cmdsize = 4 * 2;                  // cmd, cmdsize
    cmdsize += 4 * 2 * th.regsets.size(); // flavor, count (per register flavor)
    cmdsize += size_of_all_flavors;       // size of all the register set data

    add_uint32(lc, llvm::MachO::LC_THREAD); // thread_command.cmd
    add_uint32(lc, cmdsize);                // thread_command.cmdsize
    for (const RegisterSet &rs : th.regsets) {
      if (rs.flavor == RegisterFlavor::GPR) {
        add_uint32(lc, RV32_THREAD_STATE);       // thread_command.flavor
        add_uint32(lc, RV32_THREAD_STATE_COUNT); // thread_command.count
        const char *names[] = {"zero", "ra", "sp", "gp", "tp", "t0",   "t1",
                               "t2",   "fp", "s1", "a0", "a1", "a2",   "a3",
                               "a4",   "a5", "a6", "a7", "s2", "s3",   "s4",
                               "s5",   "s6", "s7", "s8", "s9", "s10",  "s11",
                               "t3",   "t4", "t5", "t6", "pc", nullptr};
        for (int i = 0; names[i]; i++)
          add_reg_value_32(spec, lc, rs.registers, names[i]);
      }
    }
    load_commands.push_back(lc);
  }
}

void add_lc_threads(CoreSpec &spec,
                    std::vector<std::vector<uint8_t>> &load_commands) {
  if (spec.cputype == llvm::MachO::CPU_TYPE_ARM)
    add_lc_threads_armv7(spec, load_commands);
  else if (spec.cputype == llvm::MachO::CPU_TYPE_ARM64)
    add_lc_threads_arm64(spec, load_commands);
  else if (spec.cputype == llvm::MachO::CPU_TYPE_RISCV)
    add_lc_threads_riscv(spec, load_commands);
  else {
    fprintf(stderr,
            "Unrecognized cputype, could not write LC_THREAD.  Exiting.\n");
    exit(1);
  }
}
