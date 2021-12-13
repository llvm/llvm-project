#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/machine.h>
#include <string.h>
#include <mach/machine/thread_state.h>
#include <inttypes.h>
#include <sys/syslimits.h>
#include <uuid/uuid.h>

// Given an executable binary with 
//   "fmain" (a function pointer to main)
//   "main"
// symbols, create a fake arm64e corefile that
// contains a memory segment for the fmain 
// function pointer, with the value of the 
// address of main() with ptrauth bits masked on.
//
// The corefile does not include the "addrable bits"
// LC_NOTE, so lldb will need to fall back on its 
// default value from the Darwin arm64 ABI.

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf (stderr, "usage: %s executable-binary output-file\n", argv[0]);
    exit(1);
  }
  FILE *exe = fopen(argv[1], "r");
  if (!exe) {
    fprintf (stderr, "Unable to open executable %s for reading\n", argv[1]);
    exit(1);
  }
  FILE *out = fopen(argv[2], "w");
  if (!out) {
    fprintf (stderr, "Unable to open %s for writing\n", argv[2]);
    exit(1);
  }

  char buf[PATH_MAX + 6];
  sprintf (buf, "nm '%s'", argv[1]);
  FILE *nm = popen(buf, "r");
  if (!nm) {
    fprintf (stderr, "Unable to run nm on '%s'", argv[1]);
    exit (1);
  }
  uint64_t main_addr = 0;
  uint64_t fmain_addr = 0;
  while (fgets (buf, sizeof(buf), nm)) {
    if (strstr (buf, "_fmain")) {
      fmain_addr = strtoul (buf, NULL, 16);
    }
    if (strstr (buf, "_main")) {
      main_addr = strtoul (buf, NULL, 16);
    }
  }
  pclose (nm);

  sprintf (buf, "dwarfdump -u '%s'", argv[1]);
  FILE *dwarfdump = popen(buf, "r");
  if (!dwarfdump) {
    fprintf (stderr, "Unable to run dwarfdump -u on '%s'\n", argv[1]);
    exit (1);
  }
  uuid_t uuid;
  uuid_clear (uuid);
  while (fgets (buf, sizeof(buf), dwarfdump)) {
    if (strncmp (buf, "UUID: ", 6) == 0) {
      buf[6 + 36] = '\0';
      if (uuid_parse (buf + 6, uuid) != 0) {
        fprintf (stderr, "Unable to parse UUID in '%s'\n", buf);
        exit (1);
      }
    }
  }
  if (uuid_is_null(uuid)) {
    fprintf (stderr, "Got a null uuid for the binary\n");
    exit (1);
  }

  if (main_addr == 0 || fmain_addr == 0) {
    fprintf(stderr, "Unable to find address of main or fmain in %s.\n",
        argv[1]);
    exit (1);
  }

  // Write out a corefile with contents in this order:
  //    1. mach header
  //    2. LC_THREAD load command
  //    3. LC_SEGMENT_64 load command
  //    4. LC_NOTE load command
  //    5. memory segment contents
  //    6. "load binary" note contents

  // struct thread_command {
  //       uint32_t        cmd;    
  //       uint32_t        cmdsize;
  //       uint32_t flavor      
  //       uint32_t count       
  //       struct XXX_thread_state state
  int size_of_thread_cmd = 4 + 4 + 4 + 4 + sizeof (arm_thread_state64_t);

  struct mach_header_64 mh;
  mh.magic = 0xfeedfacf;
  mh.cputype = CPU_TYPE_ARM64;
  mh.cpusubtype = CPU_SUBTYPE_ARM64E;
  mh.filetype = MH_CORE;
  mh.ncmds = 3; // LC_THREAD, LC_SEGMENT_64, LC_NOTE
  mh.sizeofcmds = size_of_thread_cmd + sizeof(struct segment_command_64) + sizeof(struct note_command);
  mh.flags = 0;
  mh.reserved = 0;

  fwrite(&mh, sizeof (mh), 1, out);

  struct note_command lcnote;
  struct segment_command_64 seg;
  seg.cmd = LC_SEGMENT_64;
  seg.cmdsize = sizeof(seg);
  memset (&seg.segname, 0, 16);
  seg.vmaddr = fmain_addr;
  seg.vmsize = 8;
  // Offset to segment contents
  seg.fileoff = sizeof (mh) + size_of_thread_cmd + sizeof(seg) + sizeof(lcnote);
  seg.filesize = 8;
  seg.maxprot = 3;
  seg.initprot = 3;
  seg.nsects = 0;
  seg.flags = 0;

  fwrite (&seg, sizeof (seg), 1, out);

  uint32_t cmd = LC_THREAD;
  fwrite (&cmd, sizeof (cmd), 1, out);
  uint32_t cmdsize = size_of_thread_cmd;
  fwrite (&cmdsize, sizeof (cmdsize), 1, out);
  uint32_t flavor = ARM_THREAD_STATE64;
  fwrite (&flavor, sizeof (flavor), 1, out);
  // count is number of uint32_t's of the register context
  uint32_t count = sizeof (arm_thread_state64_t) / 4;
  fwrite (&count, sizeof (count), 1, out);
  arm_thread_state64_t regstate;
  memset (&regstate, 0, sizeof (regstate));
  fwrite (&regstate, sizeof (regstate), 1, out);

  lcnote.cmd = LC_NOTE;
  lcnote.cmdsize = sizeof (lcnote);
  strcpy (lcnote.data_owner, "load binary");

  // 8 is the size of the LC_SEGMENT contents
  lcnote.offset = sizeof (mh) + size_of_thread_cmd + sizeof(seg) + sizeof(lcnote) + 8;

  // struct load_binary
  // {
  // uint32_t version;        // currently 1
  // uuid_t   uuid;           // all zeroes if uuid not specified
  // uint64_t load_address;   // virtual address where the macho is loaded, UINT64_MAX if unavail
  // uint64_t slide;          // slide to be applied to file address to get load address, 0 if unavail
  // char     name_cstring[]; // must be nul-byte terminated c-string, '\0' alone if name unavail
  // } __attribute__((packed));
  lcnote.size = 4 + 16 + 8 + 8 + sizeof("a.out");

  fwrite (&lcnote, sizeof(lcnote), 1, out);

  // Write the contents of the memory segment

  // Or together a random PAC value from a system using 39 bits 
  // of addressing with the address of main().  lldb will need
  // to correctly strip off the high bits to find the address of
  // main.
  uint64_t segment_contents = 0xe46bff0000000000 | main_addr;
  fwrite (&segment_contents, sizeof (segment_contents), 1, out);

  // Now write the contents of the "load binary" LC_NOTE.
  {
    uint32_t version = 1;
    fwrite (&version, sizeof (version), 1, out);
    fwrite (&uuid, sizeof (uuid), 1, out);
    uint64_t load_address = UINT64_MAX;
    fwrite (&load_address, sizeof (load_address), 1, out);
    uint64_t slide = 0;
    fwrite (&slide, sizeof (slide), 1, out);
    strcpy (buf, "a.out");
    fwrite (buf, 6, 1, out);
  }

  fclose (out);

  exit (0);
}
