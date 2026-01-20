#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef SYS_pkey_alloc
#include <linux/unistd.h>
#endif

// Normally this functionality is provided by the libc as the pkey_* functions.
// However, POE and therefore protection keys are new to AArch64 so we need to
// be able to build with a libc without support for it.

static uint64_t por_read(void) {
  uint64_t por;
  __asm__ volatile("mrs %0, S3_3_C10_C2_4" /*POR_EL0*/ : "=r"(por));
  return por;
}

static void por_write(uint64_t por) {
  __asm__ volatile("msr S3_3_C10_C2_4, %0\n" /*POR_EL0*/
                   "isb" ::"r"(por)
                   : "memory");
}

// Syscall functions have _syscall suffix in case the C library does
// define them.

static int pkey_alloc_syscall(unsigned flags, unsigned access_rights) {
  long res = syscall(SYS_pkey_alloc, flags, access_rights);
  if (res < 0)
    exit(2);

  return (int)res;
}

static int pkey_free_syscall(int pkey) {
  long res = syscall(SYS_pkey_free, pkey);
  if (res < 0)
    exit(2);

  return (int)res;
}

static int pkey_mprotect_syscall(void *addr, size_t len, int prot, int pkey) {
  long res = syscall(SYS_pkey_mprotect, addr, len, prot, pkey);
  if (res < 0)
    exit(2);

  return (int)res;
}

static inline uint64_t set_perm(uint64_t por, int pkey, uint8_t perm) {
  // Each permissions key is 4 bits.
  const unsigned shift = (unsigned)pkey * 4u;
  const uint64_t mask = 0xFULL << shift;
  return (por & ~mask) | ((uint64_t)(perm & 0xF) << shift);
}

static void cause_write_fault(char *buffer) { buffer[0] = '?'; }

int expr_function() {
  por_write(set_perm(por_read(), 1, 0));
  return 1;
}

int main(void) {
  // pkeys have 2 parts. First bits in the page table tagging each page
  // with the key that page uses. Second the register holding the permissions
  // for that pkey.
  // On AArch64 we have 3 page table bits and space for 16 sets of permissions
  // in POR. Page table space is the limiting factor, so we can have a maximum
  // of 8 pkeys. To provide a default pkey, the kernel uses key 0.
  // Which leaves 7 keys available for programs to allocate.

  const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
  // pkeys can only subtract from the set of permissions in the page table,
  // so we set the page table to allow everything.
  const int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
  const int flags = MAP_PRIVATE | MAP_ANONYMOUS;

  // This page will have the default key 0.
  char *key_zero_page = mmap(NULL, page_size, prot, flags, -1, 0);
  if (key_zero_page == MAP_FAILED)
    exit(2);

  // Later we will use this to cause a protection key fault.
  char *read_only_page = NULL;

  // Allocate all possible pkeys. They can in theory be in a random order, so
  // we allocate them all up front instead of setting permissions as we go.
#define NUM_KEYS 7
  for (unsigned i = 0; i < NUM_KEYS; ++i) {
    int pkey = pkey_alloc_syscall(/*flags=*/0, /*access_rights=*/0);
    // Allocate a page to attach to that pkey.
    char *page = mmap(NULL, page_size, prot, flags, -1, 0);
    if (page == MAP_FAILED)
      exit(2);
    // Attach the pkey to the page.
    pkey_mprotect_syscall(page, page_size, prot, pkey);

    if (pkey == 6)
      read_only_page = page;
  }

  // Set permissions to result in a por value of 0x...01234567. The
  // final 7 is permission set 0 already set up by the kernel.
  // Allocated keys start at 1.
  for (unsigned i = 1; i < (NUM_KEYS + 1); ++i) {
    // pkey 0 is already set to read+write+execute, we will set all other
    // valid encodings. 0 is no access and 7 is read+write_execute.
    uint8_t perm = NUM_KEYS - i;
    por_write(set_perm(por_read(), i, perm));
  }

  // This page should allow reads.
  volatile char c = read_only_page[0]; // Set break point at this line.
  (void)c;

  // Will segfault if you try to write to it. This is done via a function so
  // that we can find the functions name in the backtrace later and make sure
  // it was not the read above that caused the fault.
  cause_write_fault(read_only_page);

  return 0;
}
