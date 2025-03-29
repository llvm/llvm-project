#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

constexpr size_t num_pages = 7;
constexpr size_t accessible_pages[] = {0, 2, 4, 6};

bool is_accessible(size_t page) {
  return std::find(std::begin(accessible_pages), std::end(accessible_pages),
                   page) != std::end(accessible_pages);
}

// allocate_memory_with_holes returns a pointer to `num_pages` pages of memory,
// where some of the pages are inaccessible (even to debugging APIs). We use
// this to test lldb's ability to skip over inaccessible blocks.
#ifdef _WIN32
#include "Windows.h"

int getpagesize() {
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwPageSize;
}

char *allocate_memory_with_holes() {
  int pagesize = getpagesize();
  void *mem =
      VirtualAlloc(nullptr, num_pages * pagesize, MEM_RESERVE, PAGE_NOACCESS);
  if (!mem) {
    std::cerr << std::system_category().message(GetLastError()) << std::endl;
    exit(1);
  }
  char *bytes = static_cast<char *>(mem);
  for (size_t page = 0; page < num_pages; ++page) {
    if (!is_accessible(page))
      continue;
    if (!VirtualAlloc(bytes + page * pagesize, pagesize, MEM_COMMIT,
                      PAGE_READWRITE)) {
      std::cerr << std::system_category().message(GetLastError()) << std::endl;
      exit(1);
    }
  }
  return bytes;
}
#else
#include "sys/mman.h"
#include "unistd.h"

char *allocate_memory_with_holes() {
  int pagesize = getpagesize();
  void *mem = mmap(nullptr, num_pages * pagesize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  char *bytes = static_cast<char *>(mem);
  for (size_t page = 0; page < num_pages; ++page) {
    if (is_accessible(page))
      continue;
    if (munmap(bytes + page * pagesize, pagesize) != 0) {
      perror("munmap");
      exit(1);
    }
  }
  return bytes;
}
#endif

int main(int argc, char const *argv[]) {
  char *mem_with_holes = allocate_memory_with_holes();
  int pagesize = getpagesize();
  char *positions[] = {
      mem_with_holes,                // Beginning of memory
      mem_with_holes + 2 * pagesize, // After a hole
      mem_with_holes + 2 * pagesize +
          pagesize / 2, // Middle of a block, after an existing match.
      mem_with_holes + 5 * pagesize - 7, // End of a block
      mem_with_holes + 7 * pagesize - 7, // End of memory
  };
  for (char *p : positions)
    strcpy(p, "needle");

  return 0; // break here
}
