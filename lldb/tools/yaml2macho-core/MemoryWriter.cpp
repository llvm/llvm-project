//===-- MemoryWriter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryWriter.h"
#include "CoreSpec.h"
#include "Utility.h"

#include "llvm/BinaryFormat/MachO.h"

void create_lc_segment_cmd(const CoreSpec &spec, std::vector<uint8_t> &cmds,
                           const MemoryRegion &memory, off_t data_offset) {
  if (spec.wordsize == 8) {
    // Add the bytes for a segment_command_64 from <mach-o/loader.h>
    add_uint32(spec, cmds, llvm::MachO::LC_SEGMENT_64);
    add_uint32(spec, cmds, sizeof(struct llvm::MachO::segment_command_64));
    for (int i = 0; i < 16; i++)
      cmds.push_back(0);
    add_uint64(spec, cmds, memory.addr); // segment_command_64.vmaddr
    add_uint64(spec, cmds, memory.size); // segment_command_64.vmsize
    add_uint64(spec, cmds, data_offset); // segment_command_64.fileoff
    add_uint64(spec, cmds, memory.size); // segment_command_64.filesize
  } else {
    // Add the bytes for a segment_command from <mach-o/loader.h>
    add_uint32(spec, cmds, llvm::MachO::LC_SEGMENT);
    add_uint32(spec, cmds, sizeof(struct llvm::MachO::segment_command));
    for (int i = 0; i < 16; i++)
      cmds.push_back(0);
    add_uint32(spec, cmds, memory.addr); // segment_command_64.vmaddr
    add_uint32(spec, cmds, memory.size); // segment_command_64.vmsize
    add_uint32(spec, cmds, data_offset); // segment_command_64.fileoff
    add_uint32(spec, cmds, memory.size); // segment_command_64.filesize
  }
  add_uint32(spec, cmds, 3); // segment_command_64.maxprot
  add_uint32(spec, cmds, 3); // segment_command_64.initprot
  add_uint32(spec, cmds, 0); // segment_command_64.nsects
  add_uint32(spec, cmds, 0); // segment_command_64.flags
}

void create_memory_bytes(const CoreSpec &spec, const MemoryRegion &memory,
                         std::vector<uint8_t> &buf) {
  if (memory.type == MemoryType::UInt8)
    for (uint8_t byte : memory.bytes)
      buf.push_back(byte);

  if (memory.type == MemoryType::UInt32)
    for (uint32_t word : memory.words)
      add_uint32(spec, buf, word);

  if (memory.type == MemoryType::UInt64)
    for (uint64_t word : memory.doublewords)
      add_uint64(spec, buf, word);
}
