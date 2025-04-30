#include "kmp.h"
#include "kmp_debug.h"
#include "tinyalloc/kmp_ns_mark_page.h"

#include <sys/mman.h>

#if LIBOMP_NEXTSILICON_ATOMICS_BYPASS
/// Temporary aorkaround for SOF-6554
/// Enforce all OpenMP system allocations to be marked,
/// in the NextSilicon way, as non-migratable memory.
/// This is to avoid from OpenMP runtime code to use atomic
/// instructions over memory migrated behind the PCI.
static constexpr auto PROT_MARK_AS_UNMIGRATABLE = 0x100U;
static size_t __page_size = 0;
static uintptr_t __page_size_mask = 0;

inline static size_t __ceil_divide(size_t x, size_t y) {
  return (x + y - 1) / y;
}

static void __kmp_ns_mark_page_init() {
  static bool initialized = false;
  if (initialized)
    return;

  __page_size = getpagesize(); // page size is power of 2.
  __page_size_mask = ~(__page_size - 1);
  initialized = true;
  KA_TRACE(10, ("page size 0x%llx, page mask 0x%llx", __page_size,
                __page_size_mask));

  if (!__page_size) {
    // if cannot determine page size, something is terribly wrong.
    int err = errno;
    KA_TRACE(0, ("get page-size failed with error=%d\n", err));
    KMP_SYSFAIL(getpagesize, err);
  }
}

void __kmp_ns_mark_page(void *addr, size_t size, const char *caller) {
  if (!__kmp_ns_atomics_bypass_enable) {
    KA_TRACE(100, ("%s: (mprotect) mark addr %p for size 0x%llx, not "
                   "performed, FEATURE DISABLED!\n",
                   caller, addr, size));
    return;
  }
  __kmp_ns_mark_page_init();
  // Align address to nearest (below) page boundary
  // The alignment done + the size is then aligned to enclosing page boundary.
  uintptr_t addr_as_uintptr = reinterpret_cast<uintptr_t>(addr);
  uintptr_t page_aligned_addr = addr_as_uintptr & __page_size_mask;
  size_t alignment_size = addr_as_uintptr - page_aligned_addr;
  // note: __page_size cannot be 0!
  size_t page_aligned_size =
      __ceil_divide(size + alignment_size, __page_size) * __page_size;

  KA_TRACE(20, ("%s: (mprotect) mark addr %p (aligned to %p) size 0x%llx "
                "(aligned to 0x%llx)\n",
                caller, addr, page_aligned_addr, size, page_aligned_size));

  // denote the area as non-migratable.
  // the READ/WRITE is just in case the protection actually pushes through the
  // kernel. (this depends on the corresponding fix in the NextSilicon runtime).
  int rc =
      mprotect(reinterpret_cast<void *>(page_aligned_addr), page_aligned_size,
               PROT_MARK_AS_UNMIGRATABLE | PROT_READ | PROT_WRITE);
  if (rc) {
    // if we fail, we assume the feature is not-available, or we're not running
    // under NextSilicon runtime env.
    int err = errno;
    KA_TRACE(10, ("%s: mprotect for unmigratabale memory for %p (0x%llx) "
                  "failed with err %d\n",
                  caller, page_aligned_addr, page_aligned_size, err));
    __kmp_ns_atomics_bypass_enable = false;
  } else {
    KA_TRACE(20, ("%s: mprotect for unmigratabale memory for %p (0x%llx) ok\n",
                  caller, page_aligned_addr, page_aligned_size));
  }
}
#endif // LIBOMP_NEXTSILICON_ATOMICS_BYPASS