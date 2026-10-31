#include <assert.h>
#include <mach-o/loader.h>
#include <mach/machine.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <uuid/uuid.h>

uint8_t *crossing_image;
uint8_t *large_image;
uint8_t *unreadable_unknown_image;
uint8_t *unreadable_uuid_image;
uint8_t *unreadable_version_image;
uint8_t *unreadable_segment_image;
uint8_t *invalid_size_image;
uint8_t *invalid_count_image;
uint8_t *invalid_prefix_image;
uint8_t *invalid_overrun_image;
uint8_t *huge_size_image;
uint8_t *breakpoint_address;

static const uint32_t unknown_command = 0x12345678;

static void write_header(uint8_t *buffer, uint32_t count, uint32_t size) {
  struct mach_header_64 header = {0};
  header.magic = MH_MAGIC_64;
#if defined(__x86_64__)
  header.cputype = CPU_TYPE_X86_64;
  header.cpusubtype = CPU_SUBTYPE_X86_64_ALL;
#else
  header.cputype = CPU_TYPE_ARM64;
  header.cpusubtype = CPU_SUBTYPE_ARM64_ALL;
#endif
  // Check that debugserver strips capability bits from the reported subtype.
  header.cpusubtype |= CPU_SUBTYPE_MASK;
  header.filetype = MH_DYLIB;
  header.ncmds = count;
  header.sizeofcmds = size;
  header.flags = MH_NOUNDEFS | MH_DYLDLINK;
  memcpy(buffer, &header, sizeof(header));
}

static size_t write_metadata_commands(uint8_t *buffer, int legacy_version) {
  uint8_t *cursor = buffer;
  struct uuid_command uuid = {0};
  uuid.cmd = LC_UUID;
  uuid.cmdsize = sizeof(uuid);
  uuid_parse("12345678-1234-5678-abcd-0123456789ab", uuid.uuid);
  memcpy(cursor, &uuid, sizeof(uuid));
  cursor += sizeof(uuid);

  struct segment_command_64 segment64 = {0};
  segment64.cmd = LC_SEGMENT_64;
  segment64.cmdsize = sizeof(segment64);
  memcpy(segment64.segname, "1234567890abcdef", 16);
  segment64.vmaddr = 0x123456789abcdef0;
  segment64.vmsize = 0x10000;
  segment64.fileoff = 0x1000;
  segment64.filesize = 0x2000;
  segment64.maxprot = 7;
  memcpy(cursor, &segment64, sizeof(segment64));
  cursor += sizeof(segment64);

  struct segment_command segment32 = {0};
  segment32.cmd = LC_SEGMENT;
  segment32.cmdsize = sizeof(segment32);
  memcpy(segment32.segname, "__TEXT32", 8);
  segment32.vmaddr = 0x5000;
  segment32.vmsize = 0x1000;
  segment32.fileoff = 0x20;
  segment32.filesize = 0x200;
  segment32.maxprot = 5;
  memcpy(cursor, &segment32, sizeof(segment32));
  cursor += sizeof(segment32);

  if (legacy_version) {
    struct version_min_command version = {0};
    version.cmd = LC_VERSION_MIN_MACOSX;
    version.cmdsize = sizeof(version);
    version.version = (10 << 16) | (12 << 8);
    memcpy(cursor, &version, sizeof(version));
    cursor += sizeof(version);
  } else {
    struct build_version_command version = {0};
    version.cmd = LC_BUILD_VERSION;
    version.cmdsize = sizeof(version);
    version.platform = PLATFORM_MACOS;
    version.minos = (13 << 16) | (2 << 8) | 1;
    memcpy(cursor, &version, sizeof(version));
    cursor += sizeof(version);
  }
  return cursor - buffer;
}

// Prepend an unknown command, including its header, to the metadata commands.
static uint8_t *make_padded_image(uint32_t unknown_size, int legacy) {
  uint8_t *buffer =
      calloc(1, sizeof(struct mach_header_64) + unknown_size + 256);
  assert(buffer);
  uint8_t *commands = buffer + sizeof(struct mach_header_64);
  struct load_command unknown = {unknown_command, unknown_size};
  memcpy(commands, &unknown, sizeof(unknown));
  size_t metadata_size =
      write_metadata_commands(commands + unknown_size, legacy);
  // One unknown command followed by four metadata commands.
  write_header(buffer, /*ncmds=*/5, unknown_size + metadata_size);
  return buffer;
}

static uint8_t *make_unreadable_payload(uint32_t command,
                                        uint32_t command_size) {
  size_t page_size = getpagesize();
  uint8_t *mapping = mmap(NULL, 2 * page_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
  assert(mapping != MAP_FAILED);
  uint8_t *buffer = mapping + page_size - sizeof(struct mach_header_64) -
                    sizeof(struct load_command);
  write_header(buffer, 1, command_size);
  struct load_command prefix = {command, command_size};
  memcpy(buffer + sizeof(struct mach_header_64), &prefix, sizeof(prefix));
  assert(mprotect(mapping + page_size, page_size, PROT_NONE) == 0);
  return buffer;
}

static uint8_t *make_bad_image(uint32_t count, uint32_t size, uint32_t command,
                               uint32_t command_size) {
  uint8_t *buffer = calloc(1, sizeof(struct mach_header_64) + 128);
  assert(buffer);
  write_header(buffer, count, size);
  struct load_command prefix = {command, command_size};
  memcpy(buffer + sizeof(struct mach_header_64), &prefix, sizeof(prefix));
  return buffer;
}

void prepare_load_command_images(void) {
  // The first 16 KB cache window starts after the Mach-O header. Put LC_UUID
  // eight bytes before that window's end: its command header fits, but its
  // UUID payload requires another read.
  crossing_image = make_padded_image(16 * 1024 - 8, 0);
  breakpoint_address = crossing_image + sizeof(struct mach_header_64) +
                       (16 * 1024 - 8) + sizeof(struct load_command);
  large_image = make_padded_image(2 * 1024 * 1024, 1);

  size_t page_size = getpagesize();
  uint8_t *mapping = mmap(NULL, 3 * page_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
  assert(mapping != MAP_FAILED);
  unreadable_unknown_image = mapping + page_size -
                             sizeof(struct mach_header_64) -
                             sizeof(struct load_command);
  uint8_t *commands = unreadable_unknown_image + sizeof(struct mach_header_64);
  uint32_t unknown_size = page_size + sizeof(struct load_command);
  struct load_command unknown = {unknown_command, unknown_size};
  memcpy(commands, &unknown, sizeof(unknown));
  size_t metadata_size = write_metadata_commands(commands + unknown_size, 0);
  // One unknown command followed by four metadata commands.
  write_header(unreadable_unknown_image, /*ncmds=*/5,
               unknown_size + metadata_size);
  assert(mprotect(mapping + page_size, page_size, PROT_NONE) == 0);

  unreadable_uuid_image =
      make_unreadable_payload(LC_UUID, sizeof(struct uuid_command));
  unreadable_version_image = make_unreadable_payload(
      LC_BUILD_VERSION, sizeof(struct build_version_command));
  unreadable_segment_image =
      make_unreadable_payload(LC_SEGMENT_64, sizeof(struct segment_command_64));
  invalid_size_image = make_bad_image(1, 8, unknown_command, 0);
  invalid_count_image = make_bad_image(2, 8, unknown_command, 8);
  invalid_prefix_image = make_bad_image(1, 8, LC_SEGMENT_64, 8);
  invalid_overrun_image = make_bad_image(1, 8, unknown_command, 16);
  // The declared region is intentionally huge and mostly inaccessible. Only
  // the unknown command's header is required, so do not allocate or read it
  // all.
  huge_size_image = make_unreadable_payload(unknown_command, UINT32_MAX - 7);
}
