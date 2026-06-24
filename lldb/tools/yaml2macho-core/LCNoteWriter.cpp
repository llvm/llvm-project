//===-- LCNoteWriter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LCNoteWriter.h"
#include "Utility.h"
#include "lldb/Utility/UUID.h"
#include "llvm/BinaryFormat/MachO.h"
#include <ctype.h>
#include <stdlib.h>

void create_lc_note_binary_load_cmd(const CoreSpec &spec,
                                    std::vector<uint8_t> &cmds,
                                    const Binary &binary,
                                    std::vector<uint8_t> &payload_bytes,
                                    off_t data_offset) {

  // Add the payload bytes to payload_bytes.
  size_t starting_payload_size = payload_bytes.size();
  add_uint32(payload_bytes, 1); // version
  lldb_private::UUID uuid;
  uuid.SetFromStringRef(binary.uuid);
  for (size_t i = 0; i < uuid.GetBytes().size(); i++)
    payload_bytes.push_back(uuid.GetBytes().data()[i]);
  if (binary.value_is_slide) {
    add_uint64(payload_bytes, UINT64_MAX);   // address
    add_uint64(payload_bytes, binary.value); // slide
  } else {
    add_uint64(payload_bytes, binary.value); // address
    add_uint64(payload_bytes, UINT64_MAX);   // slide
  }
  if (binary.name.empty()) {
    payload_bytes.push_back(0); // name_cstring
  } else {
    size_t len = binary.name.size();
    for (size_t i = 0; i < len; i++)
      payload_bytes.push_back(binary.name[i]);
    payload_bytes.push_back(0); // name_cstring
  }

  size_t payload_size = payload_bytes.size() - starting_payload_size;
  // Pad out the entry to a 4-byte aligned size.
  if (payload_bytes.size() % 4 != 0) {
    size_t pad_bytes =
        ((payload_bytes.size() + 4 - 1) & ~(4 - 1)) - payload_bytes.size();
    for (size_t i = 0; i < pad_bytes; i++)
      payload_bytes.push_back(0);
  }

  // Add the load command bytes to cmds.
  add_uint32(cmds, llvm::MachO::LC_NOTE);
  add_uint32(cmds, sizeof(struct llvm::MachO::note_command));
  char cmdname[16];
  memset(cmdname, '\0', sizeof(cmdname));
  strcpy(cmdname, "load binary");
  for (int i = 0; i < 16; i++)
    cmds.push_back(cmdname[i]);
  add_uint64(cmds, data_offset);
  add_uint64(cmds, payload_size);
}

void create_lc_note_addressable_bits(const CoreSpec &spec,
                                     std::vector<uint8_t> &cmds,
                                     const AddressableBits &addr_bits,
                                     std::vector<uint8_t> &payload_bytes,
                                     off_t data_offset) {
  // Add the payload bytes to payload_bytes.
  size_t starting_payload_size = payload_bytes.size();
  add_uint32(payload_bytes, 4); // version

  add_uint32(payload_bytes, *addr_bits.lowmem_bits);  // low memory
  add_uint32(payload_bytes, *addr_bits.highmem_bits); // high memory
  add_uint32(payload_bytes, 0);                       // reserved
  size_t payload_size = payload_bytes.size() - starting_payload_size;
  // Pad out the entry to a 4-byte aligned size.
  if (payload_bytes.size() % 4 != 0) {
    size_t pad_bytes =
        ((payload_bytes.size() + 4 - 1) & ~(4 - 1)) - payload_bytes.size();
    for (size_t i = 0; i < pad_bytes; i++)
      payload_bytes.push_back(0);
  }

  // Add the load command bytes to cmds.
  add_uint32(cmds, llvm::MachO::LC_NOTE);
  add_uint32(cmds, sizeof(struct llvm::MachO::note_command));
  char cmdname[16];
  memset(cmdname, '\0', sizeof(cmdname));
  strcpy(cmdname, "addrable bits");
  for (int i = 0; i < 16; i++)
    cmds.push_back(cmdname[i]);
  add_uint64(cmds, data_offset);
  add_uint64(cmds, payload_size);
}
