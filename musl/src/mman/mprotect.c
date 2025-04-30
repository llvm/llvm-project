#include <sys/mman.h>
#include "libc.h"
#include "syscall.h"

__attribute__((noinline))
int __mprotect_internal(void *addr, size_t len, int prot)
{
	size_t start, end;
	start = (size_t)addr & -PAGE_SIZE;
	end = (size_t)((char *)addr + len + PAGE_SIZE-1) & -PAGE_SIZE;
	return syscall(SYS_mprotect, start, end-start, prot);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __mprotect_nextsilicon(void *addr, size_t len, int prot)
{
	return __mprotect_internal(addr, len, prot);
}

int __mprotect(void *addr, size_t len, int prot)
{
	return __mprotect_nextsilicon(addr, len, prot);
}

weak_alias(__mprotect, mprotect);
