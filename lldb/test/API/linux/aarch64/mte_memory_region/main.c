#include <asm/hwcap.h>
#include <asm/mman.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  if (!(getauxval(AT_HWCAP2) & HWCAP2_MTE))
    return 1;

  int got = prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE, 0, 0, 0);
  if (got)
    return 1;

  void *the_page = mmap(0, sysconf(_SC_PAGESIZE), PROT_MTE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (the_page == MAP_FAILED)
    return 1;

  return 0; // Set break point at this line.
}
