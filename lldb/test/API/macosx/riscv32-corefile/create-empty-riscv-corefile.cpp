#include <inttypes.h>
#include <mach-o/loader.h>
#include <mach/thread_status.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/errno.h>
#include <uuid/uuid.h>
#include <vector>

#define CPU_TYPE_RISCV 24
#define CPU_SUBTYPE_RISCV_ALL 0
#define RV32_THREAD_STATE 2
// x0-x31 + pc, all 32-bit
#define RV32_THREAD_STATE_COUNT 33

union uint32_buf {
  uint8_t bytebuf[4];
  uint32_t val;
};

union uint64_buf {
  uint8_t bytebuf[8];
  uint64_t val;
};

void add_uint64(std::vector<uint8_t> &buf, uint64_t val) {
  uint64_buf conv;
  conv.val = val;
  for (int i = 0; i < 8; i++)
    buf.push_back(conv.bytebuf[i]);
}

void add_uint32(std::vector<uint8_t> &buf, uint32_t val) {
  uint32_buf conv;
  conv.val = val;
  for (int i = 0; i < 4; i++)
    buf.push_back(conv.bytebuf[i]);
}

std::vector<uint8_t> lc_thread_load_command() {
  std::vector<uint8_t> data;
  add_uint32(data, LC_THREAD); // thread_command.cmd
  add_uint32(data, 4 + 4 + 4 + 4 +
                       (RV32_THREAD_STATE_COUNT * 4)); // thread_command.cmdsize
  add_uint32(data, RV32_THREAD_STATE);                 // thread_command.flavor
  add_uint32(data, RV32_THREAD_STATE_COUNT);           // thread_command.count
  for (int i = 0; i < RV32_THREAD_STATE_COUNT; i++) {
    add_uint32(data, i | (i << 8) | (i << 16) | (i << 24));
  }
  return data;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr,
            "usage: create-empty-riscv-corefile output-corefile-name\n");
    exit(1);
  }

  cpu_type_t cputype = CPU_TYPE_RISCV;
  cpu_subtype_t cpusubtype = CPU_SUBTYPE_RISCV_ALL;

  // An array of load commands (in the form of byte arrays)
  std::vector<std::vector<uint8_t>> load_commands;

  // An array of corefile contents (page data, lc_note data, etc)
  std::vector<uint8_t> payload;

  // First add all the load commands / payload so we can figure out how large
  // the load commands will actually be.
  load_commands.push_back(lc_thread_load_command());

  int size_of_load_commands = 0;
  for (const auto &lc : load_commands)
    size_of_load_commands += lc.size();

  int header_and_load_cmd_room =
      sizeof(struct mach_header_64) + size_of_load_commands;

  // Erase the load commands / payload now that we know how much space is
  // needed, redo it.
  load_commands.clear();
  payload.clear();

  load_commands.push_back(lc_thread_load_command());

  struct mach_header mh;
  mh.magic = MH_MAGIC;
  mh.cputype = cputype;

  mh.cpusubtype = cpusubtype;
  mh.filetype = MH_CORE;
  mh.ncmds = load_commands.size();
  mh.sizeofcmds = size_of_load_commands;
  mh.flags = 0;

  FILE *f = fopen(argv[1], "w");

  if (f == nullptr) {
    fprintf(stderr, "Unable to open file %s for writing\n", argv[1]);
    exit(1);
  }

  fwrite(&mh, sizeof(struct mach_header), 1, f);

  for (const auto &lc : load_commands)
    fwrite(lc.data(), lc.size(), 1, f);

  fseek(f, header_and_load_cmd_room, SEEK_SET);

  fwrite(payload.data(), payload.size(), 1, f);

  fclose(f);
}
