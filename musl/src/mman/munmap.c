#include <sys/mman.h>
#include "syscall.h"


static void dummy(void) { }
weak_alias(dummy, __vm_wait);

__attribute__((noinline))
int __munmap_internal(void *start, size_t len)
{
	__vm_wait();
	return syscall(SYS_munmap, start, len);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __munmap_nextsilicon(void *start, size_t len)
{
    return __munmap_internal(start, len);
}

int __munmap(void *start, size_t len)
{
    return __munmap_nextsilicon(start, len);
}

weak_alias(__munmap, munmap);
