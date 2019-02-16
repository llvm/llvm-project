// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page heap.
//
// See malloc.h for overview.
//
// When a MSpan is in the heap free list, state == MSpanFree
// and heapmap(s->start) == span, heapmap(s->start+s->npages-1) == span.
//
// When a MSpan is allocated, state == MSpanInUse
// and heapmap(i) == span for all s->start <= i < s->start+s->npages.

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

static MSpan *MHeap_AllocLocked(MHeap*, uintptr, int32);
static bool MHeap_Grow(MHeap*, uintptr);
static void MHeap_FreeLocked(MHeap*, MSpan*);
static MSpan *MHeap_AllocLarge(MHeap*, uintptr);
static MSpan *BestFit(MSpan*, uintptr, MSpan*);

static void
RecordSpan(void *vh, byte *p)
{
	MHeap *h;
	MSpan *s;
	MSpan **all;
	uint32 cap;

	h = vh;
	s = (MSpan*)p;
	if(h->nspan >= h->nspancap) {
		cap = 64*1024/sizeof(all[0]);
		if(cap < h->nspancap*3/2)
			cap = h->nspancap*3/2;
		all = (MSpan**)runtime_SysAlloc(cap*sizeof(all[0]), &mstats.other_sys);
		if(all == nil)
			runtime_throw("runtime: cannot allocate memory");
		if(h->allspans) {
			runtime_memmove(all, h->allspans, h->nspancap*sizeof(all[0]));
			// Don't free the old array if it's referenced by sweep.
			// See the comment in mgc0.c.
			if(h->allspans != runtime_mheap.sweepspans)
				runtime_SysFree(h->allspans, h->nspancap*sizeof(all[0]), &mstats.other_sys);
		}
		h->allspans = all;
		h->nspancap = cap;
	}
	h->allspans[h->nspan++] = s;
}

// Initialize the heap; fetch memory using alloc.
void
runtime_MHeap_Init(MHeap *h)
{
	uint32 i;

	runtime_FixAlloc_Init(&h->spanalloc, sizeof(MSpan), RecordSpan, h, &mstats.mspan_sys);
	runtime_FixAlloc_Init(&h->cachealloc, sizeof(MCache), nil, nil, &mstats.mcache_sys);
	runtime_FixAlloc_Init(&h->specialfinalizeralloc, sizeof(SpecialFinalizer), nil, nil, &mstats.other_sys);
	runtime_FixAlloc_Init(&h->specialprofilealloc, sizeof(SpecialProfile), nil, nil, &mstats.other_sys);
	// h->mapcache needs no init
	for(i=0; i<nelem(h->free); i++) {
		runtime_MSpanList_Init(&h->free[i]);
		runtime_MSpanList_Init(&h->busy[i]);
	}
	runtime_MSpanList_Init(&h->freelarge);
	runtime_MSpanList_Init(&h->busylarge);
	for(i=0; i<nelem(h->central); i++)
		runtime_MCentral_Init(&h->central[i].mcentral, i);
}

void
runtime_MHeap_MapSpans(MHeap *h)
{
	uintptr pagesize;
	uintptr n;

	// Map spans array, PageSize at a time.
	n = (uintptr)h->arena_used;
	n -= (uintptr)h->arena_start;
	n = n / PageSize * sizeof(h->spans[0]);
	n = ROUND(n, PageSize);
	pagesize = getpagesize();
	n = ROUND(n, pagesize);
	if(h->spans_mapped >= n)
		return;
	runtime_SysMap((byte*)h->spans + h->spans_mapped, n - h->spans_mapped, h->arena_reserved, &mstats.other_sys);
	h->spans_mapped = n;
}

// Sweeps spans in list until reclaims at least npages into heap.
// Returns the actual number of pages reclaimed.
static uintptr
MHeap_ReclaimList(MHeap *h, MSpan *list, uintptr npages)
{
	MSpan *s;
	uintptr n;
	uint32 sg;

	n = 0;
	sg = runtime_mheap.sweepgen;
retry:
	for(s = list->next; s != list; s = s->next) {
		if(s->sweepgen == sg-2 && runtime_cas(&s->sweepgen, sg-2, sg-1)) {
			runtime_MSpanList_Remove(s);
			// swept spans are at the end of the list
			runtime_MSpanList_InsertBack(list, s);
			runtime_unlock(&h->lock);
			n += runtime_MSpan_Sweep(s);
			runtime_lock(&h->lock);
			if(n >= npages)
				return n;
			// the span could have been moved elsewhere
			goto retry;
		}
		if(s->sweepgen == sg-1) {
			// the span is being sweept by background sweeper, skip
			continue;
		}
		// already swept empty span,
		// all subsequent ones must also be either swept or in process of sweeping
		break;
	}
	return n;
}

// Sweeps and reclaims at least npage pages into heap.
// Called before allocating npage pages.
static void
MHeap_Reclaim(MHeap *h, uintptr npage)
{
	uintptr reclaimed, n;

	// First try to sweep busy spans with large objects of size >= npage,
	// this has good chances of reclaiming the necessary space.
	for(n=npage; n < nelem(h->busy); n++) {
		if(MHeap_ReclaimList(h, &h->busy[n], npage))
			return;  // Bingo!
	}

	// Then -- even larger objects.
	if(MHeap_ReclaimList(h, &h->busylarge, npage))
		return;  // Bingo!

	// Now try smaller objects.
	// One such object is not enough, so we need to reclaim several of them.
	reclaimed = 0;
	for(n=0; n < npage && n < nelem(h->busy); n++) {
		reclaimed += MHeap_ReclaimList(h, &h->busy[n], npage-reclaimed);
		if(reclaimed >= npage)
			return;
	}

	// Now sweep everything that is not yet swept.
	runtime_unlock(&h->lock);
	for(;;) {
		n = runtime_sweepone();
		if(n == (uintptr)-1)  // all spans are swept
			break;
		reclaimed += n;
		if(reclaimed >= npage)
			break;
	}
	runtime_lock(&h->lock);
}

// Allocate a new span of npage pages from the heap
// and record its size class in the HeapMap and HeapMapCache.
MSpan*
runtime_MHeap_Alloc(MHeap *h, uintptr npage, int32 sizeclass, bool large, bool needzero)
{
	MSpan *s;

	runtime_lock(&h->lock);
	mstats.heap_alloc += runtime_m()->mcache->local_cachealloc;
	runtime_m()->mcache->local_cachealloc = 0;
	s = MHeap_AllocLocked(h, npage, sizeclass);
	if(s != nil) {
		mstats.heap_inuse += npage<<PageShift;
		if(large) {
			mstats.heap_objects++;
			mstats.heap_alloc += npage<<PageShift;
			// Swept spans are at the end of lists.
			if(s->npages < nelem(h->free))
				runtime_MSpanList_InsertBack(&h->busy[s->npages], s);
			else
				runtime_MSpanList_InsertBack(&h->busylarge, s);
		}
	}
	runtime_unlock(&h->lock);
	if(s != nil) {
		if(needzero && s->needzero)
			runtime_memclr((byte*)(s->start<<PageShift), s->npages<<PageShift);
		s->needzero = 0;
	}
	return s;
}

static MSpan*
MHeap_AllocLocked(MHeap *h, uintptr npage, int32 sizeclass)
{
	uintptr n;
	MSpan *s, *t;
	PageID p;

	// To prevent excessive heap growth, before allocating n pages
	// we need to sweep and reclaim at least n pages.
	if(!h->sweepdone)
		MHeap_Reclaim(h, npage);

	// Try in fixed-size lists up to max.
	for(n=npage; n < nelem(h->free); n++) {
		if(!runtime_MSpanList_IsEmpty(&h->free[n])) {
			s = h->free[n].next;
			goto HaveSpan;
		}
	}

	// Best fit in list of large spans.
	if((s = MHeap_AllocLarge(h, npage)) == nil) {
		if(!MHeap_Grow(h, npage))
			return nil;
		if((s = MHeap_AllocLarge(h, npage)) == nil)
			return nil;
	}

HaveSpan:
	// Mark span in use.
	if(s->state != MSpanFree)
		runtime_throw("MHeap_AllocLocked - MSpan not free");
	if(s->npages < npage)
		runtime_throw("MHeap_AllocLocked - bad npages");
	runtime_MSpanList_Remove(s);
	runtime_atomicstore(&s->sweepgen, h->sweepgen);
	s->state = MSpanInUse;
	mstats.heap_idle -= s->npages<<PageShift;
	mstats.heap_released -= s->npreleased<<PageShift;
	if(s->npreleased > 0)
		runtime_SysUsed((void*)(s->start<<PageShift), s->npages<<PageShift);
	s->npreleased = 0;

	if(s->npages > npage) {
		// Trim extra and put it back in the heap.
		t = runtime_FixAlloc_Alloc(&h->spanalloc);
		runtime_MSpan_Init(t, s->start + npage, s->npages - npage);
		s->npages = npage;
		p = t->start;
		p -= ((uintptr)h->arena_start>>PageShift);
		if(p > 0)
			h->spans[p-1] = s;
		h->spans[p] = t;
		h->spans[p+t->npages-1] = t;
		t->needzero = s->needzero;
		runtime_atomicstore(&t->sweepgen, h->sweepgen);
		t->state = MSpanInUse;
		MHeap_FreeLocked(h, t);
		t->unusedsince = s->unusedsince; // preserve age
	}
	s->unusedsince = 0;

	// Record span info, because gc needs to be
	// able to map interior pointer to containing span.
	s->sizeclass = sizeclass;
	s->elemsize = (sizeclass==0 ? s->npages<<PageShift : (uintptr)runtime_class_to_size[sizeclass]);
	s->types.compression = MTypes_Empty;
	p = s->start;
	p -= ((uintptr)h->arena_start>>PageShift);
	for(n=0; n<npage; n++)
		h->spans[p+n] = s;
	return s;
}

// Allocate a span of exactly npage pages from the list of large spans.
static MSpan*
MHeap_AllocLarge(MHeap *h, uintptr npage)
{
	return BestFit(&h->freelarge, npage, nil);
}

// Search list for smallest span with >= npage pages.
// If there are multiple smallest spans, take the one
// with the earliest starting address.
static MSpan*
BestFit(MSpan *list, uintptr npage, MSpan *best)
{
	MSpan *s;

	for(s=list->next; s != list; s=s->next) {
		if(s->npages < npage)
			continue;
		if(best == nil
		|| s->npages < best->npages
		|| (s->npages == best->npages && s->start < best->start))
			best = s;
	}
	return best;
}

// Try to add at least npage pages of memory to the heap,
// returning whether it worked.
static bool
MHeap_Grow(MHeap *h, uintptr npage)
{
	uintptr ask;
	void *v;
	MSpan *s;
	PageID p;

	// Ask for a big chunk, to reduce the number of mappings
	// the operating system needs to track; also amortizes
	// the overhead of an operating system mapping.
	// Allocate a multiple of 64kB (16 pages).
	npage = (npage+15)&~15;
	ask = npage<<PageShift;
	if(ask < HeapAllocChunk)
		ask = HeapAllocChunk;

	v = runtime_MHeap_SysAlloc(h, ask);
	if(v == nil) {
		if(ask > (npage<<PageShift)) {
			ask = npage<<PageShift;
			v = runtime_MHeap_SysAlloc(h, ask);
		}
		if(v == nil) {
			runtime_printf("runtime: out of memory: cannot allocate %D-byte block (%D in use)\n", (uint64)ask, mstats.heap_sys);
			return false;
		}
	}

	// Create a fake "in use" span and free it, so that the
	// right coalescing happens.
	s = runtime_FixAlloc_Alloc(&h->spanalloc);
	runtime_MSpan_Init(s, (uintptr)v>>PageShift, ask>>PageShift);
	p = s->start;
	p -= ((uintptr)h->arena_start>>PageShift);
	h->spans[p] = s;
	h->spans[p + s->npages - 1] = s;
	runtime_atomicstore(&s->sweepgen, h->sweepgen);
	s->state = MSpanInUse;
	MHeap_FreeLocked(h, s);
	return true;
}

// Look up the span at the given address.
// Address is guaranteed to be in map
// and is guaranteed to be start or end of span.
MSpan*
runtime_MHeap_Lookup(MHeap *h, void *v)
{
	uintptr p;
	
	p = (uintptr)v;
	p -= (uintptr)h->arena_start;
	return h->spans[p >> PageShift];
}

// Look up the span at the given address.
// Address is *not* guaranteed to be in map
// and may be anywhere in the span.
// Map entries for the middle of a span are only
// valid for allocated spans.  Free spans may have
// other garbage in their middles, so we have to
// check for that.
MSpan*
runtime_MHeap_LookupMaybe(MHeap *h, void *v)
{
	MSpan *s;
	PageID p, q;

	if((byte*)v < h->arena_start || (byte*)v >= h->arena_used)
		return nil;
	p = (uintptr)v>>PageShift;
	q = p;
	q -= (uintptr)h->arena_start >> PageShift;
	s = h->spans[q];
	if(s == nil || p < s->start || (byte*)v >= s->limit || s->state != MSpanInUse)
		return nil;
	return s;
}

// Free the span back into the heap.
void
runtime_MHeap_Free(MHeap *h, MSpan *s, int32 acct)
{
	runtime_lock(&h->lock);
	mstats.heap_alloc += runtime_m()->mcache->local_cachealloc;
	runtime_m()->mcache->local_cachealloc = 0;
	mstats.heap_inuse -= s->npages<<PageShift;
	if(acct) {
		mstats.heap_alloc -= s->npages<<PageShift;
		mstats.heap_objects--;
	}
	MHeap_FreeLocked(h, s);
	runtime_unlock(&h->lock);
}

static void
MHeap_FreeLocked(MHeap *h, MSpan *s)
{
	MSpan *t;
	PageID p;

	s->types.compression = MTypes_Empty;

	if(s->state != MSpanInUse || s->ref != 0 || s->sweepgen != h->sweepgen) {
		runtime_printf("MHeap_FreeLocked - span %p ptr %p state %d ref %d sweepgen %d/%d\n",
			s, s->start<<PageShift, s->state, s->ref, s->sweepgen, h->sweepgen);
		runtime_throw("MHeap_FreeLocked - invalid free");
	}
	mstats.heap_idle += s->npages<<PageShift;
	s->state = MSpanFree;
	runtime_MSpanList_Remove(s);
	// Stamp newly unused spans. The scavenger will use that
	// info to potentially give back some pages to the OS.
	s->unusedsince = runtime_nanotime();
	s->npreleased = 0;

	// Coalesce with earlier, later spans.
	p = s->start;
	p -= (uintptr)h->arena_start >> PageShift;
	if(p > 0 && (t = h->spans[p-1]) != nil && t->state != MSpanInUse) {
		s->start = t->start;
		s->npages += t->npages;
		s->npreleased = t->npreleased; // absorb released pages
		s->needzero |= t->needzero;
		p -= t->npages;
		h->spans[p] = s;
		runtime_MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime_FixAlloc_Free(&h->spanalloc, t);
	}
	if((p+s->npages)*sizeof(h->spans[0]) < h->spans_mapped && (t = h->spans[p+s->npages]) != nil && t->state != MSpanInUse) {
		s->npages += t->npages;
		s->npreleased += t->npreleased;
		s->needzero |= t->needzero;
		h->spans[p + s->npages - 1] = s;
		runtime_MSpanList_Remove(t);
		t->state = MSpanDead;
		runtime_FixAlloc_Free(&h->spanalloc, t);
	}

	// Insert s into appropriate list.
	if(s->npages < nelem(h->free))
		runtime_MSpanList_Insert(&h->free[s->npages], s);
	else
		runtime_MSpanList_Insert(&h->freelarge, s);
}

static void
forcegchelper(void *vnote)
{
	Note *note = (Note*)vnote;

	runtime_gc(1);
	runtime_notewakeup(note);
}

static uintptr
scavengelist(MSpan *list, uint64 now, uint64 limit)
{
	uintptr released, sumreleased, start, end, pagesize;
	MSpan *s;

	if(runtime_MSpanList_IsEmpty(list))
		return 0;

	sumreleased = 0;
	for(s=list->next; s != list; s=s->next) {
		if((now - s->unusedsince) > limit && s->npreleased != s->npages) {
			released = (s->npages - s->npreleased) << PageShift;
			mstats.heap_released += released;
			sumreleased += released;
			s->npreleased = s->npages;

			start = s->start << PageShift;
			end = start + (s->npages << PageShift);

			// Round start up and end down to ensure we
			// are acting on entire pages.
			pagesize = getpagesize();
			start = ROUND(start, pagesize);
			end &= ~(pagesize - 1);
			if(end > start)
				runtime_SysUnused((void*)start, end - start);
		}
	}
	return sumreleased;
}

static void
scavenge(int32 k, uint64 now, uint64 limit)
{
	uint32 i;
	uintptr sumreleased;
	MHeap *h;
	
	h = &runtime_mheap;
	sumreleased = 0;
	for(i=0; i < nelem(h->free); i++)
		sumreleased += scavengelist(&h->free[i], now, limit);
	sumreleased += scavengelist(&h->freelarge, now, limit);

	if(runtime_debug.gctrace > 0) {
		if(sumreleased > 0)
			runtime_printf("scvg%d: %D MB released\n", k, (uint64)sumreleased>>20);
		runtime_printf("scvg%d: inuse: %D, idle: %D, sys: %D, released: %D, consumed: %D (MB)\n",
			k, mstats.heap_inuse>>20, mstats.heap_idle>>20, mstats.heap_sys>>20,
			mstats.heap_released>>20, (mstats.heap_sys - mstats.heap_released)>>20);
	}
}

// Release (part of) unused memory to OS.
// Goroutine created at startup.
// Loop forever.
void
runtime_MHeap_Scavenger(void* dummy)
{
	G *g;
	MHeap *h;
	uint64 tick, now, forcegc, limit;
	int64 unixnow;
	uint32 k;
	Note note, *notep;

	USED(dummy);

	g = runtime_g();
	g->issystem = true;
	g->isbackground = true;

	// If we go two minutes without a garbage collection, force one to run.
	forcegc = 2*60*1e9;
	// If a span goes unused for 5 minutes after a garbage collection,
	// we hand it back to the operating system.
	limit = 5*60*1e9;
	// Make wake-up period small enough for the sampling to be correct.
	if(forcegc < limit)
		tick = forcegc/2;
	else
		tick = limit/2;

	h = &runtime_mheap;
	for(k=0;; k++) {
		runtime_noteclear(&note);
		runtime_notetsleepg(&note, tick);

		runtime_lock(&h->lock);
		unixnow = runtime_unixnanotime();
		if(unixnow - mstats.last_gc > forcegc) {
			runtime_unlock(&h->lock);
			// The scavenger can not block other goroutines,
			// otherwise deadlock detector can fire spuriously.
			// GC blocks other goroutines via the runtime_worldsema.
			runtime_noteclear(&note);
			notep = &note;
			__go_go(forcegchelper, (void*)notep);
			runtime_notetsleepg(&note, -1);
			if(runtime_debug.gctrace > 0)
				runtime_printf("scvg%d: GC forced\n", k);
			runtime_lock(&h->lock);
		}
		now = runtime_nanotime();
		scavenge(k, now, limit);
		runtime_unlock(&h->lock);
	}
}

void runtime_debug_freeOSMemory(void) __asm__("runtime_debug.freeOSMemory");

void
runtime_debug_freeOSMemory(void)
{
	runtime_gc(2);  // force GC and do eager sweep
	runtime_lock(&runtime_mheap.lock);
	scavenge(-1, ~(uintptr)0, 0);
	runtime_unlock(&runtime_mheap.lock);
}

// Initialize a new span with the given start and npages.
void
runtime_MSpan_Init(MSpan *span, PageID start, uintptr npages)
{
	span->next = nil;
	span->prev = nil;
	span->start = start;
	span->npages = npages;
	span->freelist = nil;
	span->ref = 0;
	span->sizeclass = 0;
	span->incache = false;
	span->elemsize = 0;
	span->state = MSpanDead;
	span->unusedsince = 0;
	span->npreleased = 0;
	span->types.compression = MTypes_Empty;
	span->specialLock.key = 0;
	span->specials = nil;
	span->needzero = 0;
	span->freebuf = nil;
}

// Initialize an empty doubly-linked list.
void
runtime_MSpanList_Init(MSpan *list)
{
	list->state = MSpanListHead;
	list->next = list;
	list->prev = list;
}

void
runtime_MSpanList_Remove(MSpan *span)
{
	if(span->prev == nil && span->next == nil)
		return;
	span->prev->next = span->next;
	span->next->prev = span->prev;
	span->prev = nil;
	span->next = nil;
}

bool
runtime_MSpanList_IsEmpty(MSpan *list)
{
	return list->next == list;
}

void
runtime_MSpanList_Insert(MSpan *list, MSpan *span)
{
	if(span->next != nil || span->prev != nil) {
		runtime_printf("failed MSpanList_Insert %p %p %p\n", span, span->next, span->prev);
		runtime_throw("MSpanList_Insert");
	}
	span->next = list->next;
	span->prev = list;
	span->next->prev = span;
	span->prev->next = span;
}

void
runtime_MSpanList_InsertBack(MSpan *list, MSpan *span)
{
	if(span->next != nil || span->prev != nil) {
		runtime_printf("failed MSpanList_Insert %p %p %p\n", span, span->next, span->prev);
		runtime_throw("MSpanList_Insert");
	}
	span->next = list;
	span->prev = list->prev;
	span->next->prev = span;
	span->prev->next = span;
}

// Adds the special record s to the list of special records for
// the object p.  All fields of s should be filled in except for
// offset & next, which this routine will fill in.
// Returns true if the special was successfully added, false otherwise.
// (The add will fail only if a record with the same p and s->kind
//  already exists.)
static bool
addspecial(void *p, Special *s)
{
	MSpan *span;
	Special **t, *x;
	uintptr offset;
	byte kind;

	span = runtime_MHeap_LookupMaybe(&runtime_mheap, p);
	if(span == nil)
		runtime_throw("addspecial on invalid pointer");

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	runtime_m()->locks++;
	runtime_MSpan_EnsureSwept(span);

	offset = (uintptr)p - (span->start << PageShift);
	kind = s->kind;

	runtime_lock(&span->specialLock);

	// Find splice point, check for existing record.
	t = &span->specials;
	while((x = *t) != nil) {
		if(offset == x->offset && kind == x->kind) {
			runtime_unlock(&span->specialLock);
			runtime_m()->locks--;
			return false; // already exists
		}
		if(offset < x->offset || (offset == x->offset && kind < x->kind))
			break;
		t = &x->next;
	}
	// Splice in record, fill in offset.
	s->offset = offset;
	s->next = x;
	*t = s;
	runtime_unlock(&span->specialLock);
	runtime_m()->locks--;
	return true;
}

// Removes the Special record of the given kind for the object p.
// Returns the record if the record existed, nil otherwise.
// The caller must FixAlloc_Free the result.
static Special*
removespecial(void *p, byte kind)
{
	MSpan *span;
	Special *s, **t;
	uintptr offset;

	span = runtime_MHeap_LookupMaybe(&runtime_mheap, p);
	if(span == nil)
		runtime_throw("removespecial on invalid pointer");

	// Ensure that the span is swept.
	// GC accesses specials list w/o locks. And it's just much safer.
	runtime_m()->locks++;
	runtime_MSpan_EnsureSwept(span);

	offset = (uintptr)p - (span->start << PageShift);

	runtime_lock(&span->specialLock);
	t = &span->specials;
	while((s = *t) != nil) {
		// This function is used for finalizers only, so we don't check for
		// "interior" specials (p must be exactly equal to s->offset).
		if(offset == s->offset && kind == s->kind) {
			*t = s->next;
			runtime_unlock(&span->specialLock);
			runtime_m()->locks--;
			return s;
		}
		t = &s->next;
	}
	runtime_unlock(&span->specialLock);
	runtime_m()->locks--;
	return nil;
}

// Adds a finalizer to the object p.  Returns true if it succeeded.
bool
runtime_addfinalizer(void *p, FuncVal *f, const FuncType *ft, const PtrType *ot)
{
	SpecialFinalizer *s;

	runtime_lock(&runtime_mheap.speciallock);
	s = runtime_FixAlloc_Alloc(&runtime_mheap.specialfinalizeralloc);
	runtime_unlock(&runtime_mheap.speciallock);
	s->special.kind = KindSpecialFinalizer;
	s->fn = f;
	s->ft = ft;
	s->ot = ot;
	if(addspecial(p, &s->special))
		return true;

	// There was an old finalizer
	runtime_lock(&runtime_mheap.speciallock);
	runtime_FixAlloc_Free(&runtime_mheap.specialfinalizeralloc, s);
	runtime_unlock(&runtime_mheap.speciallock);
	return false;
}

// Removes the finalizer (if any) from the object p.
void
runtime_removefinalizer(void *p)
{
	SpecialFinalizer *s;

	s = (SpecialFinalizer*)removespecial(p, KindSpecialFinalizer);
	if(s == nil)
		return; // there wasn't a finalizer to remove
	runtime_lock(&runtime_mheap.speciallock);
	runtime_FixAlloc_Free(&runtime_mheap.specialfinalizeralloc, s);
	runtime_unlock(&runtime_mheap.speciallock);
}

// Set the heap profile bucket associated with addr to b.
void
runtime_setprofilebucket(void *p, Bucket *b)
{
	SpecialProfile *s;

	runtime_lock(&runtime_mheap.speciallock);
	s = runtime_FixAlloc_Alloc(&runtime_mheap.specialprofilealloc);
	runtime_unlock(&runtime_mheap.speciallock);
	s->special.kind = KindSpecialProfile;
	s->b = b;
	if(!addspecial(p, &s->special))
		runtime_throw("setprofilebucket: profile already set");
}

// Do whatever cleanup needs to be done to deallocate s.  It has
// already been unlinked from the MSpan specials list.
// Returns true if we should keep working on deallocating p.
bool
runtime_freespecial(Special *s, void *p, uintptr size, bool freed)
{
	SpecialFinalizer *sf;
	SpecialProfile *sp;

	switch(s->kind) {
	case KindSpecialFinalizer:
		sf = (SpecialFinalizer*)s;
		runtime_queuefinalizer(p, sf->fn, sf->ft, sf->ot);
		runtime_lock(&runtime_mheap.speciallock);
		runtime_FixAlloc_Free(&runtime_mheap.specialfinalizeralloc, sf);
		runtime_unlock(&runtime_mheap.speciallock);
		return false; // don't free p until finalizer is done
	case KindSpecialProfile:
		sp = (SpecialProfile*)s;
		runtime_MProf_Free(sp->b, size, freed);
		runtime_lock(&runtime_mheap.speciallock);
		runtime_FixAlloc_Free(&runtime_mheap.specialprofilealloc, sp);
		runtime_unlock(&runtime_mheap.speciallock);
		return true;
	default:
		runtime_throw("bad special kind");
		return true;
	}
}

// Free all special records for p.
void
runtime_freeallspecials(MSpan *span, void *p, uintptr size)
{
	Special *s, **t, *list;
	uintptr offset;

	if(span->sweepgen != runtime_mheap.sweepgen)
		runtime_throw("runtime: freeallspecials: unswept span");
	// first, collect all specials into the list; then, free them
	// this is required to not cause deadlock between span->specialLock and proflock
	list = nil;
	offset = (uintptr)p - (span->start << PageShift);
	runtime_lock(&span->specialLock);
	t = &span->specials;
	while((s = *t) != nil) {
		if(offset + size <= s->offset)
			break;
		if(offset <= s->offset) {
			*t = s->next;
			s->next = list;
			list = s;
		} else
			t = &s->next;
	}
	runtime_unlock(&span->specialLock);

	while(list != nil) {
		s = list;
		list = s->next;
		if(!runtime_freespecial(s, p, size, true))
			runtime_throw("can't explicitly free an object with a finalizer");
	}
}

// Split an allocated span into two equal parts.
void
runtime_MHeap_SplitSpan(MHeap *h, MSpan *s)
{
	MSpan *t;
	MCentral *c;
	uintptr i;
	uintptr npages;
	PageID p;

	if(s->state != MSpanInUse)
		runtime_throw("MHeap_SplitSpan on a free span");
	if(s->sizeclass != 0 && s->ref != 1)
		runtime_throw("MHeap_SplitSpan doesn't have an allocated object");
	npages = s->npages;

	// remove the span from whatever list it is in now
	if(s->sizeclass > 0) {
		// must be in h->central[x].empty
		c = &h->central[s->sizeclass].mcentral;
		runtime_lock(&c->lock);
		runtime_MSpanList_Remove(s);
		runtime_unlock(&c->lock);
		runtime_lock(&h->lock);
	} else {
		// must be in h->busy/busylarge
		runtime_lock(&h->lock);
		runtime_MSpanList_Remove(s);
	}
	// heap is locked now

	if(npages == 1) {
		// convert span of 1 PageSize object to a span of 2 PageSize/2 objects.
		s->ref = 2;
		s->sizeclass = runtime_SizeToClass(PageSize/2);
		s->elemsize = PageSize/2;
	} else {
		// convert span of n>1 pages into two spans of n/2 pages each.
		if((s->npages & 1) != 0)
			runtime_throw("MHeap_SplitSpan on an odd size span");

		// compute position in h->spans
		p = s->start;
		p -= (uintptr)h->arena_start >> PageShift;

		// Allocate a new span for the first half.
		t = runtime_FixAlloc_Alloc(&h->spanalloc);
		runtime_MSpan_Init(t, s->start, npages/2);
		t->limit = (byte*)((t->start + npages/2) << PageShift);
		t->state = MSpanInUse;
		t->elemsize = npages << (PageShift - 1);
		t->sweepgen = s->sweepgen;
		if(t->elemsize <= MaxSmallSize) {
			t->sizeclass = runtime_SizeToClass(t->elemsize);
			t->ref = 1;
		}

		// the old span holds the second half.
		s->start += npages/2;
		s->npages = npages/2;
		s->elemsize = npages << (PageShift - 1);
		if(s->elemsize <= MaxSmallSize) {
			s->sizeclass = runtime_SizeToClass(s->elemsize);
			s->ref = 1;
		}

		// update span lookup table
		for(i = p; i < p + npages/2; i++)
			h->spans[i] = t;
	}

	// place the span into a new list
	if(s->sizeclass > 0) {
		runtime_unlock(&h->lock);
		c = &h->central[s->sizeclass].mcentral;
		runtime_lock(&c->lock);
		// swept spans are at the end of the list
		runtime_MSpanList_InsertBack(&c->empty, s);
		runtime_unlock(&c->lock);
	} else {
		// Swept spans are at the end of lists.
		if(s->npages < nelem(h->free))
			runtime_MSpanList_InsertBack(&h->busy[s->npages], s);
		else
			runtime_MSpanList_InsertBack(&h->busylarge, s);
		runtime_unlock(&h->lock);
	}
}
