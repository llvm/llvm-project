#include <arm_acle.h>
#include <asm/hwcap.h>
#include <asm/mman.h>
#include <stdbool.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

// Set bits 59-56 to tag, removing any existing tag
static char *set_tag(char *ptr, size_t tag) {
  return (char *)(((size_t)ptr & ~((size_t)0xf << 56)) | (tag << 56));
}

int main(int argc, char const *argv[]) {
  // We assume that the test runner has checked we're on an MTE system

  // Only expect to get the fault type
  if (argc != 2)
    return 1;

  unsigned long prctl_arg2 = 0;
  if (!strcmp(argv[1], "sync"))
    prctl_arg2 = PR_MTE_TCF_SYNC;
  else if (!strcmp(argv[1], "async"))
    prctl_arg2 = PR_MTE_TCF_ASYNC;
  else
    return 1;

  // Set fault type
  if (prctl(PR_SET_TAGGED_ADDR_CTRL, prctl_arg2, 0, 0, 0))
    return 1;

  // Allocate some memory with tagging enabled that we
  // can read/write if we use correct tags.
  char *buf = mmap(0, sysconf(_SC_PAGESIZE), PROT_MTE | PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED)
    return 1;

  // Our pointer will have tag 9
  char *tagged_buf = set_tag(buf, 9);
  // Set allocation tags for the first 2 granules
  __arm_mte_set_tag(set_tag(tagged_buf, 9));
  __arm_mte_set_tag(set_tag(tagged_buf + 16, 10));

  // Confirm that we can write when tags match
  *tagged_buf = ' ';

  // Breakpoint here
  // Faults because tag 9 in the ptr != allocation tag of 10.
  // + 16 puts us in the second granule and +1 makes the fault address
  // misaligned relative to the granule size. This misalignment must
  // be accounted for by lldb-server.
  *(tagged_buf + 16 + 1) = '?';

  return 0;
}
