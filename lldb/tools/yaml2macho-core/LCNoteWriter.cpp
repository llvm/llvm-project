//===-- LCNoteWriter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LCNoteWriter.h"
#include "Utility.h"
#include <ctype.h>
#include <stdlib.h>

#include "llvm/BinaryFormat/MachO.h"

bool ishex(char p) {
  char upp = toupper(p);
  if (isdigit(upp) || (upp >= 'A' && upp <= 'F'))
    return true;
  return false;
}

void create_lc_note_binary_load_cmd(const CoreSpec &spec,
                                    std::vector<uint8_t> &cmds,
                                    std::string uuid, uint64_t slide,
                                    std::vector<uint8_t> &payload_bytes,
                                    off_t data_offset) {

  // Add the payload bytes to payload_bytes.
  size_t starting_payload_size = payload_bytes.size();
  add_uint32(spec, payload_bytes, 1); // version
  // uuid_t uuid
  const char *p = uuid.c_str();
  while (*p && *(p + 1)) {
    if (ishex(*p) && ishex(*(p + 1))) {
      char byte[3] = {'\0', '\0', '\0'};
      byte[0] = *p++;
      byte[1] = *p++;
      uint8_t val = strtoul(byte, nullptr, 16);
      payload_bytes.push_back(val);
    } else {
      p++;
    }
  }
  add_uint64(spec, payload_bytes, UINT64_MAX); // address
  add_uint64(spec, payload_bytes, slide);      // slide
  payload_bytes.push_back(0);                  // name_cstring

  size_t payload_size = payload_bytes.size() - starting_payload_size;
  // Pad out the entry to a 4-byte aligned size.
  if (payload_bytes.size() % 4 != 0) {
    size_t pad_bytes =
        ((payload_bytes.size() + 4 - 1) & (~4 - 1)) - payload_bytes.size();
    for (size_t i = 0; i < pad_bytes; i++)
      payload_bytes.push_back(0);
  }

  // Add the load command bytes to cmds.
  add_uint32(spec, cmds, llvm::MachO::LC_NOTE);
  add_uint32(spec, cmds, sizeof(struct llvm::MachO::note_command));
  char cmdname[16];
  memset(cmdname, '\0', sizeof(cmdname));
  strcpy(cmdname, "load binary");
  for (int i = 0; i < 16; i++)
    cmds.push_back(cmdname[i]);
  add_uint64(spec, cmds, data_offset);
  add_uint64(spec, cmds, payload_size);
}
