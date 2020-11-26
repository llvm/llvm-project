#include <asm/hwcap.h>
#include <asm/mman.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#define INCOMPATIBLE_TOOLCHAIN 47
#define INCOMPATIBLE_TARGET 48

// This is in a seperate non static function
// so that we can always breakpoint the return 0 here.
// Even if main never reaches it because HWCAP2_MTE
// is not defined.
// If it were in main then you would effectively have:
// return TEST_INCOMPATIBLE;
// return 0;
// So the two returns would have the same breakpoint location
// and we couldn't tell them apart.
int setup_mte_page(void) {
#ifdef HWCAP2_MTE
  if (!(getauxval(AT_HWCAP2) & HWCAP2_MTE))
    return INCOMPATIBLE_TARGET;

  int got = prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE, 0, 0, 0);
  if (got)
    return 1;

  void *the_page = mmap(0, sysconf(_SC_PAGESIZE), PROT_MTE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (the_page == MAP_FAILED)
    return 1;
#endif

  return 0; // Set break point at this line.
}

int main(int argc, char const *argv[]) {
#ifdef HWCAP2_MTE
  return setup_mte_page();
#else
  return INCOMPATIBLE_TOOLCHAIN;
#endif
}
