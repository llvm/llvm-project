/* Defining _XOPEN_SOURCE hides the declaration of madvise() on Solaris <
   11 and the MADV_DONTNEED definition on IRIX 6.5.  */
#undef _XOPEN_SOURCE

#include <errno.h>
#include <unistd.h>

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

#ifndef MAP_ANON
#ifdef MAP_ANONYMOUS
#define MAP_ANON MAP_ANONYMOUS
#else
#define USE_DEV_ZERO
#define MAP_ANON 0
#endif
#endif

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

#ifdef USE_DEV_ZERO
static int dev_zero = -1;
#endif

static int32
addrspace_free(void *v __attribute__ ((unused)), uintptr n __attribute__ ((unused)))
{
#ifdef HAVE_MINCORE
	size_t page_size = getpagesize();
	int32 errval;
	uintptr chunk;
	uintptr off;
	
	// NOTE: vec must be just 1 byte long here.
	// Mincore returns ENOMEM if any of the pages are unmapped,
	// but we want to know that all of the pages are unmapped.
	// To make these the same, we can only ask about one page
	// at a time. See golang.org/issue/7476.
	static byte vec[1];

	errno = 0;
	for(off = 0; off < n; off += chunk) {
		chunk = page_size * sizeof vec;
		if(chunk > (n - off))
			chunk = n - off;
		errval = mincore((char*)v + off, chunk, (void*)vec);
		// ENOMEM means unmapped, which is what we want.
		// Anything else we assume means the pages are mapped.
		if(errval == 0 || errno != ENOMEM)
			return 0;
	}
#endif
	return 1;
}

static void *
mmap_fixed(byte *v, uintptr n, int32 prot, int32 flags, int32 fd, uint32 offset)
{
	void *p;

	p = runtime_mmap((void *)v, n, prot, flags, fd, offset);
	if(p != v && addrspace_free(v, n)) {
		// On some systems, mmap ignores v without
		// MAP_FIXED, so retry if the address space is free.
		if(p != MAP_FAILED)
			runtime_munmap(p, n);
		p = runtime_mmap((void *)v, n, prot, flags|MAP_FIXED, fd, offset);
	}
	return p;
}

void*
runtime_SysAlloc(uintptr n, uint64 *stat)
{
	void *p;
	int fd = -1;

#ifdef USE_DEV_ZERO
	if (dev_zero == -1) {
		dev_zero = open("/dev/zero", O_RDONLY);
		if (dev_zero < 0) {
			runtime_printf("open /dev/zero: errno=%d\n", errno);
			exit(2);
		}
	}
	fd = dev_zero;
#endif

	p = runtime_mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, fd, 0);
	if (p == MAP_FAILED) {
		if(errno == EACCES) {
			runtime_printf("runtime: mmap: access denied\n");
			runtime_printf("if you're running SELinux, enable execmem for this process.\n");
			exit(2);
		}
		if(errno == EAGAIN) {
			runtime_printf("runtime: mmap: too much locked memory (check 'ulimit -l').\n");
			runtime_exit(2);
		}
		return nil;
	}
	runtime_xadd64(stat, n);
	return p;
}

void
runtime_SysUnused(void *v __attribute__ ((unused)), uintptr n __attribute__ ((unused)))
{
#ifdef MADV_DONTNEED
	runtime_madvise(v, n, MADV_DONTNEED);
#endif
}

void
runtime_SysUsed(void *v, uintptr n)
{
	USED(v);
	USED(n);
}

void
runtime_SysFree(void *v, uintptr n, uint64 *stat)
{
	runtime_xadd64(stat, -(uint64)n);
	runtime_munmap(v, n);
}

void
runtime_SysFault(void *v, uintptr n)
{
	int fd = -1;

#ifdef USE_DEV_ZERO
	if (dev_zero == -1) {
		dev_zero = open("/dev/zero", O_RDONLY);
		if (dev_zero < 0) {
			runtime_printf("open /dev/zero: errno=%d\n", errno);
			exit(2);
		}
	}
	fd = dev_zero;
#endif

	runtime_mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE|MAP_FIXED, fd, 0);
}

void*
runtime_SysReserve(void *v, uintptr n, bool *reserved)
{
	int fd = -1;
	void *p;

#ifdef USE_DEV_ZERO
	if (dev_zero == -1) {
		dev_zero = open("/dev/zero", O_RDONLY);
		if (dev_zero < 0) {
			runtime_printf("open /dev/zero: errno=%d\n", errno);
			exit(2);
		}
	}
	fd = dev_zero;
#endif

	// On 64-bit, people with ulimit -v set complain if we reserve too
	// much address space.  Instead, assume that the reservation is okay
	// if we can reserve at least 64K and check the assumption in SysMap.
	// Only user-mode Linux (UML) rejects these requests.
	if(sizeof(void*) == 8 && (n >> 16) > 1LLU<<16) {
		p = mmap_fixed(v, 64<<10, PROT_NONE, MAP_ANON|MAP_PRIVATE, fd, 0);
		if (p != v) {
			runtime_munmap(p, 64<<10);
			return nil;
		}
		runtime_munmap(p, 64<<10);
		*reserved = false;
		return v;
	}
	
	// Use the MAP_NORESERVE mmap() flag here because typically most of
	// this reservation will never be used. It does not make sense
	// reserve a huge amount of unneeded swap space. This is important on
	// systems which do not overcommit memory by default.
	p = runtime_mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE|MAP_NORESERVE, fd, 0);
	if(p == MAP_FAILED)
		return nil;
	*reserved = true;
	return p;
}

void
runtime_SysMap(void *v, uintptr n, bool reserved, uint64 *stat)
{
	void *p;
	int fd = -1;
	
	runtime_xadd64(stat, n);

#ifdef USE_DEV_ZERO
	if (dev_zero == -1) {
		dev_zero = open("/dev/zero", O_RDONLY);
		if (dev_zero < 0) {
			runtime_printf("open /dev/zero: errno=%d\n", errno);
			exit(2);
		}
	}
	fd = dev_zero;
#endif

	// On 64-bit, we don't actually have v reserved, so tread carefully.
	if(!reserved) {
		p = mmap_fixed(v, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, fd, 0);
		if(p == MAP_FAILED && errno == ENOMEM)
			runtime_throw("runtime: out of memory");
		if(p != v) {
			runtime_printf("runtime: address space conflict: map(%p) = %p\n", v, p);
			runtime_throw("runtime: address space conflict");
		}
		return;
	}

	p = runtime_mmap(v, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_FIXED|MAP_PRIVATE, fd, 0);
	if(p == MAP_FAILED && errno == ENOMEM)
		runtime_throw("runtime: out of memory");
	if(p != v)
		runtime_throw("runtime: cannot map pages in arena address space");
}
