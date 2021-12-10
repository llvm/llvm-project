#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/machine.h>
#include <string.h>
#include <mach/machine/thread_state.h>
#include <inttypes.h>
#include <sys/syslimits.h>

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

  if (main_addr == 0 || fmain_addr == 0) {
    fprintf(stderr, "Unable to find address of main or fmain in %s.\n",
        argv[1]);
    exit (1);
  }

  // Write out a corefile with contents in this order:
  //    1. mach header
  //    2. LC_THREAD load command
  //    3. LC_SEGMENT_64 load command
  //    4. memory segment contents

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
  mh.ncmds = 2; // LC_THREAD, LC_SEGMENT_64
  mh.sizeofcmds = size_of_thread_cmd + sizeof(struct segment_command_64);
  mh.flags = 0;
  mh.reserved = 0;

  fwrite(&mh, sizeof (mh), 1, out);

  struct segment_command_64 seg;
  seg.cmd = LC_SEGMENT_64;
  seg.cmdsize = sizeof(seg);
  memset (&seg.segname, 0, 16);
  seg.vmaddr = fmain_addr;
  seg.vmsize = 8;
  // Offset to segment contents
  seg.fileoff = sizeof (mh) + size_of_thread_cmd + sizeof(seg);
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


  // Or together a random PAC value from a system using 39 bits 
  // of addressing with the address of main().  lldb will need
  // to correctly strip off the high bits to find the address of
  // main.
  uint64_t segment_contents = 0xe46bff0000000000 | main_addr;

  fwrite (&segment_contents, sizeof (segment_contents), 1, out);

  fclose (out);

  exit (0);
}
