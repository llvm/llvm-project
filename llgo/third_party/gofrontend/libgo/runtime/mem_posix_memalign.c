#include <errno.h>

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

void*
runtime_SysAlloc(uintptr n)
{
	void *p;

	mstats.sys += n;
	errno = posix_memalign(&p, PageSize, n);
	if (errno > 0) {
		perror("posix_memalign");
		exit(2);
	}
	return p;
}

void
runtime_SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call madvise MADV_DONTNEED
}

void
runtime_SysFree(void *v, uintptr n)
{
	mstats.sys -= n;
	free(v);
}

void*
runtime_SysReserve(void *v, uintptr n)
{
	USED(v);
	return runtime_SysAlloc(n);
}

void
runtime_SysMap(void *v, uintptr n)
{
	USED(v);
	USED(n);
}
