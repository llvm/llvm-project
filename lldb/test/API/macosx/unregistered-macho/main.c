#include <mach-o/loader.h>
#include <mach/machine.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

int main() {
  int size_of_load_cmds =
      sizeof(struct segment_command_64) + sizeof(struct uuid_command);
  uint8_t *macho_buf =
      (uint8_t *)malloc(sizeof(struct mach_header_64) + size_of_load_cmds);
  uint8_t *p = macho_buf;
  struct mach_header_64 mh;
  mh.magic = MH_MAGIC_64;
  mh.cputype = CPU_TYPE_ARM64;
  mh.cpusubtype = 0;
  mh.filetype = MH_EXECUTE;
  mh.ncmds = 2;
  mh.sizeofcmds = size_of_load_cmds;
  mh.flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE;

  memcpy(p, &mh, sizeof(mh));
  p += sizeof(mh);

  struct segment_command_64 seg;
  seg.cmd = LC_SEGMENT_64;
  seg.cmdsize = sizeof(seg);
  strcpy(seg.segname, "__TEXT");
  seg.vmaddr = 0x5000;
  seg.vmsize = 0x1000;
  seg.fileoff = 0;
  seg.filesize = 0;
  seg.maxprot = 0;
  seg.initprot = 0;
  seg.nsects = 0;
  seg.flags = 0;

  memcpy(p, &seg, sizeof(seg));
  p += sizeof(seg);

  struct uuid_command uuid;
  uuid.cmd = LC_UUID;
  uuid.cmdsize = sizeof(uuid);
  uuid_clear(uuid.uuid);
  uuid_parse("1b4e28ba-2fa1-11d2-883f-b9a761bde3fb", uuid.uuid);

  memcpy(p, &uuid, sizeof(uuid));
  p += sizeof(uuid);

  // If this needs to be debugged, the memory buffer can be written
  // to a file with
  // (lldb) mem rea -b -o /tmp/t -c `p - macho_buf` macho_buf
  // (lldb) platform shell otool -hlv /tmp/t
  // to verify that it is well formed.

  // And inside lldb, it should be inspectable via
  // (lldb) script print(lldb.frame.locals["macho_buf"][0].GetValueAsUnsigned())
  // 105553162403968
  // (lldb) process plugin packet send
  // 'jGetLoadedDynamicLibrariesInfos:{"solib_addresses":[105553162403968]}]'

  return 0; // break here
}
