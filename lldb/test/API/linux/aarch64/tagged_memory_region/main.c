#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  void *the_page = mmap(0, sysconf(_SC_PAGESIZE), PROT_READ | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (the_page == MAP_FAILED)
    return 1;

  // Put something in the top byte (AArch64 Linux always enables top byte
  // ignore)
  the_page = (void *)((size_t)the_page | ((size_t)0x34 << 56));

  return 0; // Set break point at this line.
}
