#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// Two regions for the Memory64List size-accounting test:
//   readable_region        - fully readable.
//   unreadable_tail_region - a memfd one page long, mapped four pages long, so
//                            reads into the pages past the file's end fault.
// The unreadable tail makes save-core's per-range read fail part-way; a memfd
// keeps the test off the filesystem.
int main() {
  const size_t page = sysconf(_SC_PAGESIZE);

  const size_t readable_region_size = 2 * 1024 * 1024;
  uint8_t *readable_region = static_cast<uint8_t *>(
      mmap(nullptr, readable_region_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (readable_region == MAP_FAILED)
    return 1;
  memset(readable_region, 0xCD, readable_region_size);

  int fd = static_cast<int>(syscall(SYS_memfd_create, "lldb_hole", 0));
  if (fd < 0)
    return 1;
  if (ftruncate(fd, page) != 0) // the memfd is exactly one page
    return 1;
  const size_t tail_size =
      4 * page; // map four pages; pages 1..3 are past the end
  uint8_t *unreadable_tail_region = static_cast<uint8_t *>(
      mmap(nullptr, tail_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (unreadable_tail_region == MAP_FAILED)
    return 1;
  memset(unreadable_tail_region, 0xAB, page);

  printf("readable_region=%p unreadable_tail_region=%p page=%zu\n",
         (void *)readable_region, (void *)unreadable_tail_region, page);
  fflush(stdout);
  return 0; // Set a breakpoint here
}
