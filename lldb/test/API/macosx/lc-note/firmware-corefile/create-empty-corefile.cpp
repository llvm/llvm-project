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

// Create an empty corefile with a "kern ver str" LC_NOTE
// or a "main bin spec" LC_NOTE..
// If an existing binary is given as a 3rd argument on the cmd line,
// the UUID from that binary will be encoded in the corefile.
// Otherwise a pre-set UUID will be put in the corefile that
// is created.

struct main_bin_spec_payload {
  uint32_t version;
  uint32_t type;
  uint64_t address;
  uuid_t uuid;
  uint32_t log2_pagesize;
  uint32_t unused;
};

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

std::vector<uint8_t> x86_lc_thread_load_command() {
  std::vector<uint8_t> data;
  add_uint32(data, LC_THREAD);                // thread_command.cmd
  add_uint32(data, 184);                      // thread_command.cmdsize
  add_uint32(data, x86_THREAD_STATE64);       // thread_command.flavor
  add_uint32(data, x86_THREAD_STATE64_COUNT); // thread_command.count
  add_uint64(data, 0x0000000000000000);       // rax
  add_uint64(data, 0x0000000000000400);       // rbx
  add_uint64(data, 0x0000000000000000);       // rcx
  add_uint64(data, 0x0000000000000000);       // rdx
  add_uint64(data, 0x0000000000000000);       // rdi
  add_uint64(data, 0x0000000000000000);       // rsi
  add_uint64(data, 0xffffff9246e2ba20);       // rbp
  add_uint64(data, 0xffffff9246e2ba10);       // rsp
  add_uint64(data, 0x0000000000000000);       // r8
  add_uint64(data, 0x0000000000000000);       // r9
  add_uint64(data, 0x0000000000000000);       // r10
  add_uint64(data, 0x0000000000000000);       // r11
  add_uint64(data, 0xffffff7f96ce5fe1);       // r12
  add_uint64(data, 0x0000000000000000);       // r13
  add_uint64(data, 0x0000000000000000);       // r14
  add_uint64(data, 0xffffff9246e2bac0);       // r15
  add_uint64(data, 0xffffff8015a8f6d0);       // rip
  add_uint64(data, 0x0000000000011111);       // rflags
  add_uint64(data, 0x0000000000022222);       // cs
  add_uint64(data, 0x0000000000033333);       // fs
  add_uint64(data, 0x0000000000044444);       // gs
  return data;
}

void add_lc_note_kern_ver_str_load_command(
    std::vector<std::vector<uint8_t>> &loadcmds, std::vector<uint8_t> &payload,
    int payload_file_offset, std::string uuid, uint64_t address) {
  std::string ident = "EFI UUID=";
  ident += uuid;

  if (address != 0xffffffffffffffff) {
    ident += "; stext=";
    char buf[24];
    sprintf(buf, "0x%" PRIx64, address);
    ident += buf;
  }

  std::vector<uint8_t> loadcmd_data;

  add_uint32(loadcmd_data, LC_NOTE); // note_command.cmd
  add_uint32(loadcmd_data, 40);      // note_command.cmdsize
  char lc_note_name[16];
  memset(lc_note_name, 0, 16);
  strcpy(lc_note_name, "kern ver str");

  // lc_note.data_owner
  for (int i = 0; i < 16; i++)
    loadcmd_data.push_back(lc_note_name[i]);

  // we start writing the payload at payload_file_offset to leave
  // room at the start for the header & the load commands.
  uint64_t current_payload_offset = payload.size() + payload_file_offset;

  add_uint64(loadcmd_data, current_payload_offset); // note_command.offset
  add_uint64(loadcmd_data, 4 + ident.size() + 1);   // note_command.size

  loadcmds.push_back(loadcmd_data);

  add_uint32(payload, 1); // kerneL_version_string.version
  for (int i = 0; i < ident.size() + 1; i++) {
    payload.push_back(ident[i]);
  }
}

void add_lc_note_main_bin_spec_load_command(
    std::vector<std::vector<uint8_t>> &loadcmds, std::vector<uint8_t> &payload,
    int payload_file_offset, std::string uuidstr, uint64_t address) {
  std::vector<uint8_t> loadcmd_data;

  add_uint32(loadcmd_data, LC_NOTE); // note_command.cmd
  add_uint32(loadcmd_data, 40);      // note_command.cmdsize
  char lc_note_name[16];
  memset(lc_note_name, 0, 16);
  strcpy(lc_note_name, "main bin spec");

  // lc_note.data_owner
  for (int i = 0; i < 16; i++)
    loadcmd_data.push_back(lc_note_name[i]);

  // we start writing the payload at payload_file_offset to leave
  // room at the start for the header & the load commands.
  uint64_t current_payload_offset = payload.size() + payload_file_offset;

  add_uint64(loadcmd_data, current_payload_offset); // note_command.offset
  add_uint64(loadcmd_data,
             sizeof(struct main_bin_spec_payload)); // note_command.size

  loadcmds.push_back(loadcmd_data);

  // Now write the "main bin spec" payload.
  add_uint32(payload, 1);          // version
  add_uint32(payload, 3);          // type == 3 [ firmware, standalone, etc ]
  add_uint64(payload, address);    // load address
  uuid_t uuid;
  uuid_parse(uuidstr.c_str(), uuid);
  for (int i = 0; i < sizeof(uuid_t); i++)
    payload.push_back(uuid[i]);
  add_uint32(payload, 0); // log2_pagesize unspecified
  add_uint32(payload, 0); // unused
}

void add_lc_segment(std::vector<std::vector<uint8_t>> &loadcmds,
                    std::vector<uint8_t> &payload, int payload_file_offset) {
  std::vector<uint8_t> loadcmd_data;
  struct segment_command_64 seg;
  seg.cmd = LC_SEGMENT_64;
  seg.cmdsize = sizeof(struct segment_command_64); // no sections
  memset(seg.segname, 0, 16);
  seg.vmaddr = 0xffffff7f96400000;
  seg.vmsize = 4096;
  seg.fileoff = payload.size() + payload_file_offset;
  seg.filesize = 0;
  seg.maxprot = 1;
  seg.initprot = 1;
  seg.nsects = 0;
  seg.flags = 0;

  uint8_t *p = (uint8_t *)&seg;
  for (int i = 0; i < sizeof(struct segment_command_64); i++) {
    loadcmd_data.push_back(*(p + i));
  }
  loadcmds.push_back(loadcmd_data);
}

std::string get_uuid_from_binary(const char *fn) {
  FILE *f = fopen(fn, "r");
  if (f == nullptr) {
    fprintf(stderr, "Unable to open binary '%s' to get uuid\n", fn);
    exit(1);
  }
  uint32_t num_of_load_cmds = 0;
  uint32_t size_of_load_cmds = 0;
  std::string uuid;
  off_t file_offset = 0;

  uint8_t magic[4];
  if (::fread(magic, 1, 4, f) != 4) {
    fprintf(stderr, "Failed to read magic number from input file %s\n", fn);
    exit(1);
  }
  uint8_t magic_32_be[] = {0xfe, 0xed, 0xfa, 0xce};
  uint8_t magic_32_le[] = {0xce, 0xfa, 0xed, 0xfe};
  uint8_t magic_64_be[] = {0xfe, 0xed, 0xfa, 0xcf};
  uint8_t magic_64_le[] = {0xcf, 0xfa, 0xed, 0xfe};

  if (memcmp(magic, magic_32_be, 4) == 0 ||
      memcmp(magic, magic_64_be, 4) == 0) {
    fprintf(stderr, "big endian corefiles not supported\n");
    exit(1);
  }

  ::fseeko(f, 0, SEEK_SET);
  if (memcmp(magic, magic_32_le, 4) == 0) {
    struct mach_header mh;
    if (::fread(&mh, 1, sizeof(mh), f) != sizeof(mh)) {
      fprintf(stderr, "error reading mach header from input file\n");
      exit(1);
    }
    if (mh.cputype != CPU_TYPE_X86_64) {
      fprintf(stderr,
              "This tool creates an x86_64 corefile but "
              "the supplied binary '%s' is cputype 0x%x\n",
              fn, (uint32_t)mh.cputype);
      exit(1);
    }
    num_of_load_cmds = mh.ncmds;
    size_of_load_cmds = mh.sizeofcmds;
    file_offset += sizeof(struct mach_header);
  } else {
    struct mach_header_64 mh;
    if (::fread(&mh, 1, sizeof(mh), f) != sizeof(mh)) {
      fprintf(stderr, "error reading mach header from input file\n");
      exit(1);
    }
    if (mh.cputype != CPU_TYPE_X86_64) {
      fprintf(stderr,
              "This tool creates an x86_64 corefile but "
              "the supplied binary '%s' is cputype 0x%x\n",
              fn, (uint32_t)mh.cputype);
      exit(1);
    }
    num_of_load_cmds = mh.ncmds;
    size_of_load_cmds = mh.sizeofcmds;
    file_offset += sizeof(struct mach_header_64);
  }

  off_t load_cmds_offset = file_offset;

  for (int i = 0; i < num_of_load_cmds &&
                  (file_offset - load_cmds_offset) < size_of_load_cmds;
       i++) {
    ::fseeko(f, file_offset, SEEK_SET);
    uint32_t cmd;
    uint32_t cmdsize;
    ::fread(&cmd, sizeof(uint32_t), 1, f);
    ::fread(&cmdsize, sizeof(uint32_t), 1, f);
    if (cmd == LC_UUID) {
      struct uuid_command uuidcmd;
      ::fseeko(f, file_offset, SEEK_SET);
      if (::fread(&uuidcmd, 1, sizeof(uuidcmd), f) != sizeof(uuidcmd)) {
        fprintf(stderr, "Unable to read LC_UUID load command.\n");
        exit(1);
      }
      uuid_string_t uuidstr;
      uuid_unparse(uuidcmd.uuid, uuidstr);
      uuid = uuidstr;
      break;
    }
    file_offset += cmdsize;
  }
  return uuid;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr,
            "usage: create-empty-corefile version-string|main-bin-spec "
            "<output-core-name> <binary-to-copy-uuid-from> <address>\n");
    fprintf(stderr,
            "     <address> is base 16, 0xffffffffffffffff means unknown\n");
    fprintf(
        stderr,
        "Create a Mach-O corefile with an either LC_NOTE 'kern ver str' or \n");
    fprintf(stderr, "an LC_NOTE 'main bin spec' load command without an "
                    "address specified, depending on\n");
    fprintf(stderr, "whether the 1st arg is version-string or main-bin-spec\n");
    exit(1);
  }
  if (strcmp(argv[1], "version-string") != 0 &&
      strcmp(argv[1], "main-bin-spec") != 0) {
    fprintf(stderr, "arg1 was not version-string or main-bin-spec\n");
    exit(1);
  }

  std::string uuid = get_uuid_from_binary(argv[3]);

  // An array of load commands (in the form of byte arrays)
  std::vector<std::vector<uint8_t>> load_commands;

  // An array of corefile contents (page data, lc_note data, etc)
  std::vector<uint8_t> payload;

  errno = 0;
  uint64_t address = strtoull(argv[4], NULL, 16);
  if (errno != 0) {
    fprintf(stderr, "Unable to parse address %s as base 16", argv[4]);
    exit(1);
  }

  // First add all the load commands / payload so we can figure out how large
  // the load commands will actually be.
  load_commands.push_back(x86_lc_thread_load_command());
  if (strcmp(argv[1], "version-string") == 0)
    add_lc_note_kern_ver_str_load_command(load_commands, payload, 0, uuid,
                                          address);
  else
    add_lc_note_main_bin_spec_load_command(load_commands, payload, 0, uuid,
                                           address);
  add_lc_segment(load_commands, payload, 0);

  int size_of_load_commands = 0;
  for (const auto &lc : load_commands)
    size_of_load_commands += lc.size();

  int header_and_load_cmd_room =
      sizeof(struct mach_header_64) + size_of_load_commands;

  // Erase the load commands / payload now that we know how much space is
  // needed, redo it.
  load_commands.clear();
  payload.clear();

  load_commands.push_back(x86_lc_thread_load_command());

  if (strcmp(argv[1], "version-string") == 0)
    add_lc_note_kern_ver_str_load_command(
        load_commands, payload, header_and_load_cmd_room, uuid, address);
  else
    add_lc_note_main_bin_spec_load_command(
        load_commands, payload, header_and_load_cmd_room, uuid, address);

  add_lc_segment(load_commands, payload, header_and_load_cmd_room);

  struct mach_header_64 mh;
  mh.magic = MH_MAGIC_64;
  mh.cputype = CPU_TYPE_X86_64;

  mh.cpusubtype = CPU_SUBTYPE_X86_64_ALL;
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
