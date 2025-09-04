//===-- main.cppp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoreSpec.h"
#include "LCNoteWriter.h"
#include "MemoryWriter.h"
#include "ThreadWriter.h"
#include "Utility.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/CommandLine.h"
#include <stdio.h>
#include <string>
#include <sys/stat.h>

std::vector<std::string> get_fields_from_delimited_string(std::string str,
                                                          const char delim) {
  std::vector<std::string> result;
  std::string::size_type prev = std::string::npos;
  std::string::size_type next = str.find(delim);
  if (str.empty()) {
    return result;
  }
  if (next == std::string::npos) {
    result.push_back(str);
  } else {
    result.push_back(std::string(str, 0, next));
    prev = next;
    while ((next = str.find(delim, prev + 1)) != std::string::npos) {
      result.push_back(std::string(str, prev + 1, next - prev - 1));
      prev = next;
    }
    result.push_back(std::string(str, prev + 1));
  }
  return result;
}

llvm::cl::opt<std::string> InputFilename("i", llvm::cl::Required,
                                         llvm::cl::desc("input yaml filename"),
                                         llvm::cl::value_desc("input"));
llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::Required,
                   llvm::cl::desc("output core filenames"),
                   llvm::cl::value_desc("output"));
llvm::cl::list<std::string>
    UUIDs("u", llvm::cl::desc("uuid of binary loaded at slide 0"),
          llvm::cl::value_desc("uuid"));
llvm::cl::list<std::string>
    UUIDAndVAs("L", llvm::cl::desc("UUID,virtual-address-loaded-at"),
               llvm::cl::value_desc("--uuid-and-load-addr"));
llvm::cl::opt<int>
    AddressableBitsOverride("A",
                            llvm::cl::desc("number of bits used in addressing"),
                            llvm::cl::value_desc("--address-bits"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (InputFilename.empty() || OutputFilename.empty()) {
    fprintf(stderr, "Missing input or outpur file.\n");
    exit(1);
  }

  struct stat sb;

  if (stat(InputFilename.c_str(), &sb) == -1) {
    fprintf(stderr, "Unable to stat %s, exiting\n", InputFilename.c_str());
    exit(1);
  }

  FILE *input = fopen(InputFilename.c_str(), "r");
  if (!input) {
    fprintf(stderr, "Unable to open %s, exiting\n", InputFilename.c_str());
    exit(1);
  }
  auto file_corespec = std::make_unique<char[]>(sb.st_size);
  if (fread(file_corespec.get(), sb.st_size, 1, input) != 1) {
    fprintf(stderr, "Unable to read all of %s, exiting\n",
            InputFilename.c_str());
    exit(1);
  }
  CoreSpec spec = from_yaml(file_corespec.get(), sb.st_size);
  fclose(input);

  for (const std::string &uuid : UUIDs) {
    Binary binary;
    binary.uuid = uuid;
    binary.value = 0;
    binary.value_is_slide = true;
    spec.binaries.push_back(binary);
  }

  for (const std::string &uuid_and_va : UUIDAndVAs) {
    std::vector<std::string> parts =
        get_fields_from_delimited_string(uuid_and_va, ',');

    std::string uuid = parts[0];
    uint64_t va = std::strtoull(parts[1].c_str(), nullptr, 16);
    Binary binary;
    binary.uuid = uuid;
    binary.value = va;
    binary.value_is_slide = false;
    spec.binaries.push_back(binary);
  }

  if (AddressableBitsOverride) {
    AddressableBits bits;
    bits.lowmem_bits = bits.highmem_bits = AddressableBitsOverride;
    spec.addressable_bits = bits;
  }

  // An array of load commands
  std::vector<std::vector<uint8_t>> load_commands;

  // An array of corefile contents (memory regions)
  std::vector<uint8_t> payload;

  // First add all the load commands / payload so we can figure out how large
  // the load commands will be.

  add_lc_threads(spec, load_commands);
  for (size_t i = 0; i < spec.memory_regions.size(); i++) {
    std::vector<uint8_t> segment_command_bytes;
    create_lc_segment_cmd(spec, segment_command_bytes, spec.memory_regions[i],
                          0);
    load_commands.push_back(segment_command_bytes);
  }

  if (spec.binaries.size() > 0)
    for (const Binary &binary : spec.binaries) {
      std::vector<uint8_t> segment_command_bytes;
      std::vector<uint8_t> payload_bytes;
      create_lc_note_binary_load_cmd(spec, segment_command_bytes, binary,
                                     payload_bytes, 0);
      load_commands.push_back(segment_command_bytes);
    }
  if (spec.addressable_bits) {
    std::vector<uint8_t> segment_command_bytes;
    std::vector<uint8_t> payload_bytes;
    create_lc_note_addressable_bits(spec, segment_command_bytes,
                                    *spec.addressable_bits, payload_bytes, 0);
    load_commands.push_back(segment_command_bytes);
  }

  off_t size_of_load_commands = 0;
  for (const auto &lc : load_commands)
    size_of_load_commands += lc.size();

  off_t header_and_load_cmd_room =
      sizeof(llvm::MachO::mach_header_64) + size_of_load_commands;
  off_t initial_payload_fileoff = header_and_load_cmd_room;
  initial_payload_fileoff = (initial_payload_fileoff + 4096 - 1) & ~(4096 - 1);
  off_t payload_fileoff = initial_payload_fileoff;

  // Erase the load commands / payload now that we know how much space is
  // needed, redo it with real values.
  load_commands.clear();
  payload.clear();

  add_lc_threads(spec, load_commands);
  for (size_t i = 0; i < spec.memory_regions.size(); i++) {
    std::vector<uint8_t> segment_command_bytes;
    create_lc_segment_cmd(spec, segment_command_bytes, spec.memory_regions[i],
                          payload_fileoff);
    load_commands.push_back(segment_command_bytes);
    payload_fileoff += spec.memory_regions[i].size;
    payload_fileoff = (payload_fileoff + 4096 - 1) & ~(4096 - 1);
  }

  off_t payload_fileoff_before_lcnotes = payload_fileoff;
  std::vector<uint8_t> lc_note_payload_bytes;
  if (spec.binaries.size() > 0)
    for (const Binary &binary : spec.binaries) {
      std::vector<uint8_t> segment_command_bytes;
      std::vector<uint8_t> payload_bytes;
      create_lc_note_binary_load_cmd(spec, segment_command_bytes, binary,
                                     lc_note_payload_bytes, payload_fileoff);
      payload_fileoff =
          payload_fileoff_before_lcnotes + lc_note_payload_bytes.size();
      load_commands.push_back(segment_command_bytes);
    }
  if (spec.addressable_bits) {
    std::vector<uint8_t> segment_command_bytes;
    std::vector<uint8_t> payload_bytes;
    create_lc_note_addressable_bits(spec, segment_command_bytes,
                                    *spec.addressable_bits,
                                    lc_note_payload_bytes, payload_fileoff);
    payload_fileoff =
        payload_fileoff_before_lcnotes + lc_note_payload_bytes.size();
    load_commands.push_back(segment_command_bytes);
  }

  // Realign our payload offset if we added any LC_NOTEs.
  if (lc_note_payload_bytes.size() > 0)
    payload_fileoff = (payload_fileoff + 4096 - 1) & ~(4096 - 1);

  FILE *f = fopen(OutputFilename.c_str(), "w");
  if (f == nullptr) {
    fprintf(stderr, "Unable to open file %s for writing\n",
            OutputFilename.c_str());
    exit(1);
  }

  std::vector<uint8_t> mh;
  // Write the fields of a mach_header_64 struct
  if (spec.wordsize == 8)
    add_uint32(mh, llvm::MachO::MH_MAGIC_64); // magic
  else
    add_uint32(mh, llvm::MachO::MH_MAGIC); // magic
  add_uint32(mh, spec.cputype);            // cputype
  add_uint32(mh, spec.cpusubtype);         // cpusubtype
  add_uint32(mh, llvm::MachO::MH_CORE);    // filetype
  add_uint32(mh, load_commands.size());    // ncmds
  add_uint32(mh, size_of_load_commands);   // sizeofcmds
  add_uint32(mh, 0);                       // flags
  if (spec.wordsize == 8)
    add_uint32(mh, 0); // reserved

  fwrite(mh.data(), mh.size(), 1, f);

  for (const auto &lc : load_commands)
    fwrite(lc.data(), lc.size(), 1, f);

  // Reset the payload offset back to the first one.
  payload_fileoff = initial_payload_fileoff;
  if (spec.memory_regions.size() > 0) {
    for (size_t i = 0; i < spec.memory_regions.size(); i++) {
      std::vector<uint8_t> bytes;
      create_memory_bytes(spec, spec.memory_regions[i], bytes);
      fseek(f, payload_fileoff, SEEK_SET);
      fwrite(bytes.data(), bytes.size(), 1, f);

      payload_fileoff += bytes.size();
      payload_fileoff = (payload_fileoff + 4096 - 1) & ~(4096 - 1);
    }
  }

  if (lc_note_payload_bytes.size() > 0) {
    fseek(f, payload_fileoff, SEEK_SET);
    fwrite(lc_note_payload_bytes.data(), lc_note_payload_bytes.size(), 1, f);
    payload_fileoff += lc_note_payload_bytes.size();
    payload_fileoff = (payload_fileoff + 4096 - 1) & ~(4096 - 1);
  }

  fclose(f);
}
