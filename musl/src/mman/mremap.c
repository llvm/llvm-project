#define _GNU_SOURCE
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>
#include <stdarg.h>
#include "syscall.h"

static void dummy(void) { }
weak_alias(dummy, __vm_wait);

__attribute__((noinline))
void *__mremap_internal(void *old_addr, size_t old_len, size_t new_len, int flags, void *new_addr)
{
	if (new_len >= PTRDIFF_MAX) {
		errno = ENOMEM;
		return MAP_FAILED;
	}

	return (void *)syscall(SYS_mremap, old_addr, old_len, new_len, flags, new_addr);
}


/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
void *__mremap_nextsilicon(void *old_addr, size_t old_len, size_t new_len, int flags, void *new_addr)
{
	return __mremap_internal(old_addr, old_len, new_len, flags, new_addr);
}

void *__mremap(void *old_addr, size_t old_len, size_t new_len, int flags, ...) {
	void *new_addr = NULL;
	va_list ap;

	if (flags & MREMAP_FIXED) {
		__vm_wait();
		va_start(ap, flags);
		new_addr = va_arg(ap, void *);
		va_end(ap);
	}

	return __mremap_nextsilicon(old_addr, old_len, new_len, flags, new_addr);
}

weak_alias(__mremap, mremap);
