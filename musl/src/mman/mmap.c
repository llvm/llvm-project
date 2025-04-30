#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>
#include "syscall.h"

static void dummy(void) { }
weak_alias(dummy, __vm_wait);

#define UNIT SYSCALL_MMAP2_UNIT
#define OFF_MASK ((-0x2000ULL << (8*sizeof(syscall_arg_t)-1)) | (UNIT-1))

__attribute__((noinline))
void *__mmap_internal(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
	long ret;
	if (off & OFF_MASK) {
		errno = EINVAL;
		return MAP_FAILED;
	}
	if (len >= PTRDIFF_MAX) {
		errno = ENOMEM;
		return MAP_FAILED;
	}
	if (flags & MAP_FIXED) {
		__vm_wait();
	}
#ifdef SYS_mmap2
	ret = __syscall(SYS_mmap2, start, len, prot, flags, fd, off/UNIT);
#else
	ret = __syscall(SYS_mmap, start, len, prot, flags, fd, off);
#endif
	/* Fixup incorrect EPERM from kernel. */
	if (ret == -EPERM && !start && (flags&MAP_ANON) && !(flags&MAP_FIXED))
		ret = -ENOMEM;
	return (void *)__syscall_ret(ret);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
void *__mmap_nextsilicon(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
    return __mmap_internal(start, len, prot, flags, fd, off);
}

void *__mmap(void *start, size_t len, int prot, int flags, int fd, off_t off)
{
    return __mmap_nextsilicon(start, len, prot, flags, fd, off);
}

weak_alias(__mmap, mmap);

weak_alias(mmap, mmap64);
