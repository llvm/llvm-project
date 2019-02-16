// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-P malloc cache for small objects.
//
// See malloc.h for an overview.

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

extern volatile intgo runtime_MemProfileRate
  __asm__ (GOSYM_PREFIX "runtime.MemProfileRate");

// dummy MSpan that contains no free objects.
static MSpan emptymspan;

MCache*
runtime_allocmcache(void)
{
	intgo rate;
	MCache *c;
	int32 i;

	runtime_lock(&runtime_mheap.lock);
	c = runtime_FixAlloc_Alloc(&runtime_mheap.cachealloc);
	runtime_unlock(&runtime_mheap.lock);
	runtime_memclr((byte*)c, sizeof(*c));
	for(i = 0; i < NumSizeClasses; i++)
		c->alloc[i] = &emptymspan;

	// Set first allocation sample size.
	rate = runtime_MemProfileRate;
	if(rate > 0x3fffffff)	// make 2*rate not overflow
		rate = 0x3fffffff;
	if(rate != 0)
		c->next_sample = runtime_fastrand1() % (2*rate);

	return c;
}

void
runtime_freemcache(MCache *c)
{
	runtime_MCache_ReleaseAll(c);
	runtime_lock(&runtime_mheap.lock);
	runtime_purgecachedstats(c);
	runtime_FixAlloc_Free(&runtime_mheap.cachealloc, c);
	runtime_unlock(&runtime_mheap.lock);
}

// Gets a span that has a free object in it and assigns it
// to be the cached span for the given sizeclass.  Returns this span.
MSpan*
runtime_MCache_Refill(MCache *c, int32 sizeclass)
{
	MCacheList *l;
	MSpan *s;

	runtime_m()->locks++;
	// Return the current cached span to the central lists.
	s = c->alloc[sizeclass];
	if(s->freelist != nil)
		runtime_throw("refill on a nonempty span");
	if(s != &emptymspan)
		runtime_MCentral_UncacheSpan(&runtime_mheap.central[sizeclass].mcentral, s);

	// Push any explicitly freed objects to the central lists.
	// Not required, but it seems like a good time to do it.
	l = &c->free[sizeclass];
	if(l->nlist > 0) {
		runtime_MCentral_FreeList(&runtime_mheap.central[sizeclass].mcentral, l->list);
		l->list = nil;
		l->nlist = 0;
	}

	// Get a new cached span from the central lists.
	s = runtime_MCentral_CacheSpan(&runtime_mheap.central[sizeclass].mcentral);
	if(s == nil)
		runtime_throw("out of memory");
	if(s->freelist == nil) {
		runtime_printf("%d %d\n", s->ref, (int32)((s->npages << PageShift) / s->elemsize));
		runtime_throw("empty span");
	}
	c->alloc[sizeclass] = s;
	runtime_m()->locks--;
	return s;
}

void
runtime_MCache_Free(MCache *c, MLink *p, int32 sizeclass, uintptr size)
{
	MCacheList *l;

	// Put on free list.
	l = &c->free[sizeclass];
	p->next = l->list;
	l->list = p;
	l->nlist++;

	// We transfer a span at a time from MCentral to MCache,
	// so we'll do the same in the other direction.
	if(l->nlist >= (runtime_class_to_allocnpages[sizeclass]<<PageShift)/size) {
		runtime_MCentral_FreeList(&runtime_mheap.central[sizeclass].mcentral, l->list);
		l->list = nil;
		l->nlist = 0;
	}
}

void
runtime_MCache_ReleaseAll(MCache *c)
{
	int32 i;
	MSpan *s;
	MCacheList *l;

	for(i=0; i<NumSizeClasses; i++) {
		s = c->alloc[i];
		if(s != &emptymspan) {
			runtime_MCentral_UncacheSpan(&runtime_mheap.central[i].mcentral, s);
			c->alloc[i] = &emptymspan;
		}
		l = &c->free[i];
		if(l->nlist > 0) {
			runtime_MCentral_FreeList(&runtime_mheap.central[i].mcentral, l->list);
			l->list = nil;
			l->nlist = 0;
		}
	}
}
