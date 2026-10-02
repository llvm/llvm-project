#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

// A mapping whose tail is unreadable: a memfd one page long, mapped four pages
// long, so reads into the pages past the file's end fault. save-core used to
// truncate the range there; a memfd keeps the test off the filesystem.
int main() {
  const size_t page = sysconf(_SC_PAGESIZE);
  int fd = static_cast<int>(syscall(SYS_memfd_create, "lldb_hole", 0));
  if (fd < 0)
    return 1;
  if (ftruncate(fd, page) != 0)
    return 1;
  uint8_t *region = static_cast<uint8_t *>(
      mmap(nullptr, 4 * page, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (region == MAP_FAILED)
    return 1;
  memset(region, 0xAB, page);
  printf("region = %p, page = %zu\n", (void *)region, page);
  fflush(stdout);
  return 0; // Set a breakpoint here
}
