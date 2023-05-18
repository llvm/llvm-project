#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>


// Normally these are picked up by including <mach/thread_status.h>
// but that does a compile time check for the build host arch and
// only defines the ARM register context constants when building on
// an arm system.  We're creating fake corefiles, and might be
// creating them on an intel system.
#define ARM_THREAD_STATE 1
#define ARM_THREAD_STATE_COUNT 17
#define ARM_EXCEPTION_STATE 3
#define ARM_EXCEPTION_STATE_COUNT 3
#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT 68
#define ARM_EXCEPTION_STATE64 7
#define ARM_EXCEPTION_STATE64_COUNT 4


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

std::vector<uint8_t> armv7_lc_thread_load_command() {
  std::vector<uint8_t> data;
  add_uint32(data, LC_THREAD);              // thread_command.cmd
  add_uint32(data, 104);                    // thread_command.cmdsize
  add_uint32(data, ARM_THREAD_STATE);       // thread_command.flavor
  add_uint32(data, ARM_THREAD_STATE_COUNT); // thread_command.count
  add_uint32(data, 0x00010000);             // r0
  add_uint32(data, 0x00020000);             // r1
  add_uint32(data, 0x00030000);             // r2
  add_uint32(data, 0x00040000);             // r3
  add_uint32(data, 0x00050000);             // r4
  add_uint32(data, 0x00060000);             // r5
  add_uint32(data, 0x00070000);             // r6
  add_uint32(data, 0x00080000);             // r7
  add_uint32(data, 0x00090000);             // r8
  add_uint32(data, 0x000a0000);             // r9
  add_uint32(data, 0x000b0000);             // r10
  add_uint32(data, 0x000c0000);             // r11
  add_uint32(data, 0x000d0000);             // r12
  add_uint32(data, 0x000e0000);             // sp
  add_uint32(data, 0x000f0000);             // lr
  add_uint32(data, 0x00100000);             // pc
  add_uint32(data, 0x00110000);             // cpsr

  add_uint32(data, ARM_EXCEPTION_STATE);       // thread_command.flavor
  add_uint32(data, ARM_EXCEPTION_STATE_COUNT); // thread_command.count
  add_uint32(data, 0x00003f5c);                // far
  add_uint32(data, 0xf2000000);                // esr
  add_uint32(data, 0x00000000);                // exception

  return data;
}

std::vector<uint8_t> arm64_lc_thread_load_command() {
  std::vector<uint8_t> data;
  add_uint32(data, LC_THREAD);                // thread_command.cmd
  add_uint32(data, 312);                      // thread_command.cmdsize
  add_uint32(data, ARM_THREAD_STATE64);       // thread_command.flavor
  add_uint32(data, ARM_THREAD_STATE64_COUNT); // thread_command.count
  add_uint64(data, 0x0000000000000001);       // x0
  add_uint64(data, 0x000000016fdff3c0);       // x1
  add_uint64(data, 0x000000016fdff3d0);       // x2
  add_uint64(data, 0x000000016fdff510);       // x3
  add_uint64(data, 0x0000000000000000);       // x4
  add_uint64(data, 0x0000000000000000);       // x5
  add_uint64(data, 0x0000000000000000);       // x6
  add_uint64(data, 0x0000000000000000);       // x7
  add_uint64(data, 0x000000010000d910);       // x8
  add_uint64(data, 0x0000000000000001);       // x9
  add_uint64(data, 0xe1e88de000000000);       // x10
  add_uint64(data, 0x0000000000000003);       // x11
  add_uint64(data, 0x0000000000000148);       // x12
  add_uint64(data, 0x0000000000004000);       // x13
  add_uint64(data, 0x0000000000000008);       // x14
  add_uint64(data, 0x0000000000000000);       // x15
  add_uint64(data, 0x0000000000000000);       // x16
  add_uint64(data, 0x0000000100003f5c);       // x17
  add_uint64(data, 0x0000000000000000);       // x18
  add_uint64(data, 0x0000000100003f5c);       // x19
  add_uint64(data, 0x000000010000c000);       // x20
  add_uint64(data, 0x000000010000d910);       // x21
  add_uint64(data, 0x000000016fdff250);       // x22
  add_uint64(data, 0x000000018ce12366);       // x23
  add_uint64(data, 0x000000016fdff1d0);       // x24
  add_uint64(data, 0x0000000000000001);       // x25
  add_uint64(data, 0x0000000000000000);       // x26
  add_uint64(data, 0x0000000000000000);       // x27
  add_uint64(data, 0x0000000000000000);       // x28
  add_uint64(data, 0x000000016fdff3a0);       // fp
  add_uint64(data, 0x000000018cd97f28);       // lr
  add_uint64(data, 0x000000016fdff140);       // sp
  add_uint64(data, 0x0000000100003f5c);       // pc
  add_uint32(data, 0x80001000);               // cpsr

  add_uint32(data, 0x00000000); // padding

  add_uint32(data, ARM_EXCEPTION_STATE64);       // thread_command.flavor
  add_uint32(data, ARM_EXCEPTION_STATE64_COUNT); // thread_command.count
  add_uint64(data, 0x0000000100003f5c);          // far
  add_uint32(data, 0xf2000000);                  // esr
  add_uint32(data, 0x00000000);                  // exception

  return data;
}

enum arch { unspecified, armv7, arm64 };

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr,
            "usage: create-arm-corefiles [armv7|arm64] <output-core-name>\n");
    exit(1);
  }

  arch arch = unspecified;

  if (strcmp(argv[1], "armv7") == 0)
    arch = armv7;
  else if (strcmp(argv[1], "arm64") == 0)
    arch = arm64;
  else {
    fprintf(stderr, "unrecognized architecture %s\n", argv[1]);
    exit(1);
  }

  // An array of load commands (in the form of byte arrays)
  std::vector<std::vector<uint8_t>> load_commands;

  // An array of corefile contents (page data, lc_note data, etc)
  std::vector<uint8_t> payload;

  // First add all the load commands / payload so we can figure out how large
  // the load commands will actually be.
  if (arch == armv7)
    load_commands.push_back(armv7_lc_thread_load_command());
  else if (arch == arm64)
    load_commands.push_back(arm64_lc_thread_load_command());

  int size_of_load_commands = 0;
  for (const auto &lc : load_commands)
    size_of_load_commands += lc.size();

  int header_and_load_cmd_room =
      sizeof(struct mach_header_64) + size_of_load_commands;

  // Erase the load commands / payload now that we know how much space is
  // needed, redo it.
  load_commands.clear();
  payload.clear();

  if (arch == armv7)
    load_commands.push_back(armv7_lc_thread_load_command());
  else if (arch == arm64)
    load_commands.push_back(arm64_lc_thread_load_command());

  struct mach_header_64 mh;
  mh.magic = MH_MAGIC_64;
  if (arch == armv7) {
    mh.cputype = CPU_TYPE_ARM;
    mh.cpusubtype = CPU_SUBTYPE_ARM_V7M;
  } else if (arch == arm64) {
    mh.cputype = CPU_TYPE_ARM64;
    mh.cpusubtype = CPU_SUBTYPE_ARM64_ALL;
  }
  mh.filetype = MH_CORE;
  mh.ncmds = load_commands.size();
  mh.sizeofcmds = size_of_load_commands;
  mh.flags = 0;
  mh.reserved = 0;

  FILE *f = fopen(argv[2], "w");

  if (f == nullptr) {
    fprintf(stderr, "Unable to open file %s for writing\n", argv[2]);
    exit(1);
  }

  fwrite(&mh, sizeof(struct mach_header_64), 1, f);

  for (const auto &lc : load_commands)
    fwrite(lc.data(), lc.size(), 1, f);

  fseek(f, header_and_load_cmd_room, SEEK_SET);

  fwrite(payload.data(), payload.size(), 1, f);

  fclose(f);
}
