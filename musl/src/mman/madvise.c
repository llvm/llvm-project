#include <sys/mman.h>
#include "syscall.h"

__attribute__((noinline))
int __madvise_internal(void *addr, size_t len, int advice)
{
	return syscall(SYS_madvise, addr, len, advice);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __madvise_nextsilicon(void *addr, size_t len, int advice)
{
	return __madvise_internal(addr, len, advice);
}

int __madvise(void *addr, size_t len, int advice)
{
	return __madvise_nextsilicon(addr, len, advice);
}

weak_alias(__madvise, madvise);
