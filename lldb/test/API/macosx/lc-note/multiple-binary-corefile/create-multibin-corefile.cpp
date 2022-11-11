#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <mach-o/loader.h>
#include <mach/thread_status.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <uuid/uuid.h>
#include <vector>

// Given a list of binaries, and optional slides to be applied,
// create a corefile whose memory is those binaries laid at at
// their slid addresses.
//
// Add a 'main bin spec' LC_NOTE for the first binary, and
// 'load binary' LC_NOTEs for any additional binaries, and
// these LC_NOTEs will ONLY have the vmaddr of the binary - no
// UUID, no slide, no filename.
//
// Test that lldb can use the load addresses, find the UUIDs,
// and load the binaries/dSYMs and put them at the correct load
// address.

struct main_bin_spec_payload {
  uint32_t version;
  uint32_t type;
  uint64_t address;
  uint64_t slide;
  uuid_t uuid;
  uint32_t log2_pagesize;
  uint32_t platform;
};

struct load_binary_payload {
  uint32_t version;
  uuid_t uuid;
  uint64_t address;
  uint64_t slide;
  const char name[4];
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

std::vector<uint8_t> lc_thread_load_command(cpu_type_t cputype) {
  std::vector<uint8_t> data;
  // Emit an LC_THREAD register context appropriate for the cputype
  // of the binary we're embedded.  The tests in this case do not
  // use the register values, so 0's are fine, lldb needs to see at
  // least one LC_THREAD in the corefile.
#if defined(__x86_64__)
  if (cputype == CPU_TYPE_X86_64) {
    add_uint32(data, LC_THREAD); // thread_command.cmd
    add_uint32(data,
               16 + (x86_THREAD_STATE64_COUNT * 4)); // thread_command.cmdsize
    add_uint32(data, x86_THREAD_STATE64);            // thread_command.flavor
    add_uint32(data, x86_THREAD_STATE64_COUNT);      // thread_command.count
    for (int i = 0; i < x86_THREAD_STATE64_COUNT; i++) {
      add_uint32(data, 0); // whatever, just some empty register values
    }
  }
#endif
#if defined(__arm64__) || defined(__aarch64__)
  if (cputype == CPU_TYPE_ARM64) {
    add_uint32(data, LC_THREAD); // thread_command.cmd
    add_uint32(data,
               16 + (ARM_THREAD_STATE64_COUNT * 4)); // thread_command.cmdsize
    add_uint32(data, ARM_THREAD_STATE64);            // thread_command.flavor
    add_uint32(data, ARM_THREAD_STATE64_COUNT);      // thread_command.count
    for (int i = 0; i < ARM_THREAD_STATE64_COUNT; i++) {
      add_uint32(data, 0); // whatever, just some empty register values
    }
  }
#endif
  return data;
}

void add_lc_note_main_bin_spec_load_command(
    std::vector<std::vector<uint8_t>> &loadcmds, std::vector<uint8_t> &payload,
    int payload_file_offset, std::string uuidstr, uint64_t address,
    uint64_t slide) {
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
  add_uint32(payload, 2);       // version
  add_uint32(payload, 3);       // type == 3 [ firmware, standalone, etc ]
  add_uint64(payload, address); // load address
  add_uint64(payload, slide);   // slide
  uuid_t uuid;
  uuid_parse(uuidstr.c_str(), uuid);
  for (int i = 0; i < sizeof(uuid_t); i++)
    payload.push_back(uuid[i]);
  add_uint32(payload, 0); // log2_pagesize unspecified
  add_uint32(payload, 0); // platform unspecified
}

void add_lc_note_load_binary_load_command(
    std::vector<std::vector<uint8_t>> &loadcmds, std::vector<uint8_t> &payload,
    int payload_file_offset, std::string uuidstr, uint64_t address,
    uint64_t slide) {
  std::vector<uint8_t> loadcmd_data;

  add_uint32(loadcmd_data, LC_NOTE); // note_command.cmd
  add_uint32(loadcmd_data, 40);      // note_command.cmdsize
  char lc_note_name[16];
  memset(lc_note_name, 0, 16);
  strcpy(lc_note_name, "load binary");

  // lc_note.data_owner
  for (int i = 0; i < 16; i++)
    loadcmd_data.push_back(lc_note_name[i]);

  // we start writing the payload at payload_file_offset to leave
  // room at the start for the header & the load commands.
  uint64_t current_payload_offset = payload.size() + payload_file_offset;

  add_uint64(loadcmd_data, current_payload_offset); // note_command.offset
  add_uint64(loadcmd_data,
             sizeof(struct load_binary_payload)); // note_command.size

  loadcmds.push_back(loadcmd_data);

  // Now write the "load binary" payload.
  add_uint32(payload, 1); // version
  uuid_t uuid;
  uuid_parse(uuidstr.c_str(), uuid);
  for (int i = 0; i < sizeof(uuid_t); i++)
    payload.push_back(uuid[i]);
  add_uint64(payload, address); // load address
  add_uint64(payload, slide);   // slide
  add_uint32(payload, 0);       // name
}

void add_lc_segment(std::vector<std::vector<uint8_t>> &loadcmds,
                    std::vector<uint8_t> &payload, int payload_file_offset,
                    uint64_t vmaddr, uint64_t size) {
  std::vector<uint8_t> loadcmd_data;
  struct segment_command_64 seg;
  seg.cmd = LC_SEGMENT_64;
  seg.cmdsize = sizeof(struct segment_command_64); // no sections
  memset(seg.segname, 0, 16);
  seg.vmaddr = vmaddr;
  seg.vmsize = size;
  seg.fileoff = payload.size() + payload_file_offset;
  seg.filesize = size;
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

std::string scan_binary(const char *fn, uint64_t &vmaddr, cpu_type_t &cputype,
                        cpu_subtype_t &cpusubtype) {
  FILE *f = fopen(fn, "r");
  if (f == nullptr) {
    fprintf(stderr, "Unable to open binary '%s' to get uuid\n", fn);
    exit(1);
  }
  uint32_t num_of_load_cmds = 0;
  uint32_t size_of_load_cmds = 0;
  std::string uuid;
  off_t file_offset = 0;
  vmaddr = UINT64_MAX;

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
    if (mh.cputype != CPU_TYPE_X86_64 && mh.cputype != CPU_TYPE_ARM64) {
      fprintf(stderr,
              "This tool creates an x86_64/arm64 corefile but "
              "the supplied binary '%s' is cputype 0x%x\n",
              fn, (uint32_t)mh.cputype);
      exit(1);
    }
    num_of_load_cmds = mh.ncmds;
    size_of_load_cmds = mh.sizeofcmds;
    file_offset += sizeof(struct mach_header);
    cputype = mh.cputype;
    cpusubtype = mh.cpusubtype;
  } else {
    struct mach_header_64 mh;
    if (::fread(&mh, 1, sizeof(mh), f) != sizeof(mh)) {
      fprintf(stderr, "error reading mach header from input file\n");
      exit(1);
    }
    if (mh.cputype != CPU_TYPE_X86_64 && mh.cputype != CPU_TYPE_ARM64) {
      fprintf(stderr,
              "This tool creates an x86_64/arm64 corefile but "
              "the supplied binary '%s' is cputype 0x%x\n",
              fn, (uint32_t)mh.cputype);
      exit(1);
    }
    num_of_load_cmds = mh.ncmds;
    size_of_load_cmds = mh.sizeofcmds;
    file_offset += sizeof(struct mach_header_64);
    cputype = mh.cputype;
    cpusubtype = mh.cpusubtype;
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
    if (vmaddr == UINT64_MAX && cmd == LC_SEGMENT_64) {
      struct segment_command_64 segcmd;
      ::fseeko(f, file_offset, SEEK_SET);
      if (::fread(&segcmd, 1, sizeof(segcmd), f) != sizeof(segcmd)) {
        fprintf(stderr, "Unable to read LC_SEGMENT_64 load command.\n");
        exit(1);
      }
      if (strcmp("__TEXT", segcmd.segname) == 0)
        vmaddr = segcmd.vmaddr;
    }
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
    }
    file_offset += cmdsize;
  }
  return uuid;
}

void slide_macho_binary(std::vector<uint8_t> &image, uint64_t slide) {
  uint8_t *p = image.data();
  struct mach_header_64 *mh = (struct mach_header_64 *)p;
  p += sizeof(struct mach_header_64);
  for (int lc_idx = 0; lc_idx < mh->ncmds; lc_idx++) {
    struct load_command *lc = (struct load_command *)p;
    if (lc->cmd == LC_SEGMENT_64) {
      struct segment_command_64 *seg = (struct segment_command_64 *)p;
      if (seg->maxprot != 0 && seg->nsects > 0) {
        seg->vmaddr += slide;
        uint8_t *j = p + sizeof(segment_command_64);
        for (int sect_idx = 0; sect_idx < seg->nsects; sect_idx++) {
          struct section_64 *sect = (struct section_64 *)j;
          sect->addr += slide;
          j += sizeof(struct section_64);
        }
      }
    }
    p += lc->cmdsize;
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: output-corefile binary1[@optional-slide] "
            "[binary2[@optional-slide] [binary3[@optional-slide] ...]]\n");
    exit(1);
  }

  // An array of load commands (in the form of byte arrays)
  std::vector<std::vector<uint8_t>> load_commands;

  // An array of corefile contents (page data, lc_note data, etc)
  std::vector<uint8_t> payload;

  std::vector<std::string> input_filenames;
  std::vector<uint64_t> input_slides;
  std::vector<uint64_t> input_filesizes;
  std::vector<uint64_t> input_filevmaddrs;
  uint64_t main_binary_cputype = CPU_TYPE_ARM64;
  uint64_t vmaddr = UINT64_MAX;
  cpu_type_t cputype;
  cpu_subtype_t cpusubtype;
  for (int i = 2; i < argc; i++) {
    std::string filename;
    std::string filename_and_opt_hex(argv[i]);
    uint64_t slide = 0;
    auto at_pos = filename_and_opt_hex.find_last_of('@');
    if (at_pos == std::string::npos) {
      filename = filename_and_opt_hex;
    } else {
      filename = filename_and_opt_hex.substr(0, at_pos);
      std::string hexstr = filename_and_opt_hex.substr(at_pos + 1);
      errno = 0;
      slide = (uint64_t)strtoull(hexstr.c_str(), nullptr, 16);
      if (errno != 0) {
        fprintf(stderr, "Unable to parse hex slide value in %s\n", argv[i]);
        exit(1);
      }
    }
    struct stat stbuf;
    if (stat(filename.c_str(), &stbuf) == -1) {
      fprintf(stderr, "Unable to stat '%s', exiting.\n", filename.c_str());
      exit(1);
    }
    input_filenames.push_back(filename);
    input_slides.push_back(slide);
    input_filesizes.push_back(stbuf.st_size);
    scan_binary(filename.c_str(), vmaddr, cputype, cpusubtype);
    input_filevmaddrs.push_back(vmaddr + slide);
    if (i == 2) {
      main_binary_cputype = cputype;
    }
  }

  const char *output_corefile_name = argv[1];
  std::string empty_uuidstr = "00000000-0000-0000-0000-000000000000";

  // First add all the load commands / payload so we can figure out how large
  // the load commands will actually be.
  load_commands.push_back(lc_thread_load_command(cputype));

  add_lc_note_main_bin_spec_load_command(load_commands, payload, 0,
                                         empty_uuidstr, 0, UINT64_MAX);
  for (int i = 1; i < input_filenames.size(); i++) {
    add_lc_note_load_binary_load_command(load_commands, payload, 0,
                                         empty_uuidstr, 0, UINT64_MAX);
  }

  for (int i = 0; i < input_filenames.size(); i++) {
    add_lc_segment(load_commands, payload, 0, 0, 0);
  }

  int size_of_load_commands = 0;
  for (const auto &lc : load_commands)
    size_of_load_commands += lc.size();

  int size_of_header_and_load_cmds =
      sizeof(struct mach_header_64) + size_of_load_commands;

  // Erase the load commands / payload now that we know how much space is
  // needed, redo it.
  load_commands.clear();
  payload.clear();

  // Push the LC_THREAD load command.
  load_commands.push_back(lc_thread_load_command(main_binary_cputype));

  const off_t payload_offset = size_of_header_and_load_cmds;

  add_lc_note_main_bin_spec_load_command(load_commands, payload, payload_offset,
                                         empty_uuidstr, input_filevmaddrs[0],
                                         UINT64_MAX);

  for (int i = 1; i < input_filenames.size(); i++) {
    add_lc_note_load_binary_load_command(load_commands, payload, payload_offset,
                                         empty_uuidstr, input_filevmaddrs[i],
                                         UINT64_MAX);
  }

  for (int i = 0; i < input_filenames.size(); i++) {
    add_lc_segment(load_commands, payload, payload_offset, input_filevmaddrs[i],
                   input_filesizes[i]);

    // Copy the contents of the binary into payload.
    int fd = open(input_filenames[i].c_str(), O_RDONLY);
    if (fd == -1) {
      fprintf(stderr, "Unable to open %s for reading\n",
              input_filenames[i].c_str());
      exit(1);
    }
    std::vector<uint8_t> binary_contents;
    for (int j = 0; j < input_filesizes[i]; j++) {
      uint8_t byte;
      read(fd, &byte, 1);
      binary_contents.push_back(byte);
    }
    close(fd);

    size_t cur_payload_size = payload.size();
    payload.resize(cur_payload_size + binary_contents.size());
    slide_macho_binary(binary_contents, input_slides[i]);
    memcpy(payload.data() + cur_payload_size, binary_contents.data(),
           binary_contents.size());
  }

  struct mach_header_64 mh;
  mh.magic = MH_MAGIC_64;
  mh.cputype = cputype;

  mh.cpusubtype = cpusubtype;
  mh.filetype = MH_CORE;
  mh.ncmds = load_commands.size();
  mh.sizeofcmds = size_of_load_commands;
  mh.flags = 0;
  mh.reserved = 0;

  FILE *f = fopen(output_corefile_name, "w");

  if (f == nullptr) {
    fprintf(stderr, "Unable to open file %s for writing\n",
            output_corefile_name);
    exit(1);
  }

  fwrite(&mh, sizeof(mh), 1, f);

  for (const auto &lc : load_commands)
    fwrite(lc.data(), lc.size(), 1, f);

  fwrite(payload.data(), payload.size(), 1, f);

  fclose(f);
}
