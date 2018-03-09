// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Implementation of runtime/debug.WriteHeapDump.  Writes all
// objects in the heap plus additional info (roots, threads,
// finalizers, etc.) to a file.

// The format of the dumped file is described at
// http://code.google.com/p/go-wiki/wiki/heapdump13

#include "runtime.h"
#include "arch.h"
#include "malloc.h"
#include "mgc0.h"
#include "go-type.h"
#include "go-defer.h"
#include "go-panic.h"

#define hash __hash
#define KindNoPointers GO_NO_POINTERS

enum {
	FieldKindEol = 0,
	FieldKindPtr = 1,
	FieldKindString = 2,
	FieldKindSlice = 3,
	FieldKindIface = 4,
	FieldKindEface = 5,

	TagEOF = 0,
	TagObject = 1,
	TagOtherRoot = 2,
	TagType = 3,
	TagGoRoutine = 4,
	TagStackFrame = 5,
	TagParams = 6,
	TagFinalizer = 7,
	TagItab = 8,
	TagOSThread = 9,
	TagMemStats = 10,
	TagQueuedFinalizer = 11,
	TagData = 12,
	TagBss = 13,
	TagDefer = 14,
	TagPanic = 15,
	TagMemProf = 16,
	TagAllocSample = 17,

	TypeInfo_Conservative = 127,
};

// static uintptr* playgcprog(uintptr offset, uintptr *prog, void (*callback)(void*,uintptr,uintptr), void *arg);
// static void dumpfields(uintptr *prog);
static void dumpefacetypes(void *obj, uintptr size, const Type *type, uintptr kind);

// fd to write the dump to.
static uintptr dumpfd;

// buffer of pending write data
enum {
	BufSize = 4096,
};
static byte buf[BufSize];
static uintptr nbuf;

static void
hwrite(const byte *data, uintptr len)
{
	if(len + nbuf <= BufSize) {
		runtime_memmove(buf + nbuf, data, len);
		nbuf += len;
		return;
	}
	runtime_write(dumpfd, buf, nbuf);
	if(len >= BufSize) {
		runtime_write(dumpfd, data, len);
		nbuf = 0;
	} else {
		runtime_memmove(buf, data, len);
		nbuf = len;
	}
}

static void
flush(void)
{
	runtime_write(dumpfd, buf, nbuf);
	nbuf = 0;
}

// Cache of types that have been serialized already.
// We use a type's hash field to pick a bucket.
// Inside a bucket, we keep a list of types that
// have been serialized so far, most recently used first.
// Note: when a bucket overflows we may end up
// serializing a type more than once.  That's ok.
enum {
	TypeCacheBuckets = 256, // must be a power of 2
	TypeCacheAssoc = 4,
};
typedef struct TypeCacheBucket TypeCacheBucket;
struct TypeCacheBucket {
	const Type *t[TypeCacheAssoc];
};
static TypeCacheBucket typecache[TypeCacheBuckets];

// dump a uint64 in a varint format parseable by encoding/binary
static void
dumpint(uint64 v)
{
	byte buf[10];
	int32 n;
	n = 0;
	while(v >= 0x80) {
		buf[n++] = v | 0x80;
		v >>= 7;
	}
	buf[n++] = v;
	hwrite(buf, n);
}

static void
dumpbool(bool b)
{
	dumpint(b ? 1 : 0);
}

// dump varint uint64 length followed by memory contents
static void
dumpmemrange(const byte *data, uintptr len)
{
	dumpint(len);
	hwrite(data, len);
}

static void
dumpstr(String s)
{
	dumpmemrange(s.str, s.len);
}

static void
dumpcstr(const int8 *c)
{
	dumpmemrange((const byte*)c, runtime_findnull((const byte*)c));
}

// dump information for a type
static void
dumptype(const Type *t)
{
	TypeCacheBucket *b;
	int32 i, j;

	if(t == nil) {
		return;
	}

	// If we've definitely serialized the type before,
	// no need to do it again.
	b = &typecache[t->hash & (TypeCacheBuckets-1)];
	if(t == b->t[0]) return;
	for(i = 1; i < TypeCacheAssoc; i++) {
		if(t == b->t[i]) {
			// Move-to-front
			for(j = i; j > 0; j--) {
				b->t[j] = b->t[j-1];
			}
			b->t[0] = t;
			return;
		}
	}
	// Might not have been dumped yet.  Dump it and
	// remember we did so.
	for(j = TypeCacheAssoc-1; j > 0; j--) {
		b->t[j] = b->t[j-1];
	}
	b->t[0] = t;
	
	// dump the type
	dumpint(TagType);
	dumpint((uintptr)t);
	dumpint(t->__size);
	if(t->__uncommon == nil || t->__uncommon->__pkg_path == nil || t->__uncommon->__name == nil) {
		dumpstr(*t->__reflection);
	} else {
		dumpint(t->__uncommon->__pkg_path->len + 1 + t->__uncommon->__name->len);
		hwrite(t->__uncommon->__pkg_path->str, t->__uncommon->__pkg_path->len);
		hwrite((const byte*)".", 1);
		hwrite(t->__uncommon->__name->str, t->__uncommon->__name->len);
	}
	dumpbool(t->__size > PtrSize || (t->__code & KindNoPointers) == 0);
	// dumpfields((uintptr*)t->gc + 1);
}

// returns true if object is scannable
static bool
scannable(byte *obj)
{
	uintptr *b, off, shift;

	off = (uintptr*)obj - (uintptr*)runtime_mheap.arena_start;  // word offset
	b = (uintptr*)runtime_mheap.arena_start - off/wordsPerBitmapWord - 1;
	shift = off % wordsPerBitmapWord;
	return ((*b >> shift) & bitScan) != 0;
}

// dump an object
static void
dumpobj(byte *obj, uintptr size, const Type *type, uintptr kind)
{
	if(type != nil) {
		dumptype(type);
		dumpefacetypes(obj, size, type, kind);
	}

	dumpint(TagObject);
	dumpint((uintptr)obj);
	dumpint((uintptr)type);
	dumpint(kind);
	dumpmemrange(obj, size);
}

static void
dumpotherroot(const char *description, byte *to)
{
	dumpint(TagOtherRoot);
	dumpcstr((const int8 *)description);
	dumpint((uintptr)to);
}

static void
dumpfinalizer(byte *obj, FuncVal *fn, const FuncType* ft, const PtrType *ot)
{
	dumpint(TagFinalizer);
	dumpint((uintptr)obj);
	dumpint((uintptr)fn);
	dumpint((uintptr)fn->fn);
	dumpint((uintptr)ft);
	dumpint((uintptr)ot);
}

typedef struct ChildInfo ChildInfo;
struct ChildInfo {
	// Information passed up from the callee frame about
	// the layout of the outargs region.
	uintptr argoff;     // where the arguments start in the frame
	uintptr arglen;     // size of args region
	BitVector args;    // if args.n >= 0, pointer map of args region

	byte *sp;           // callee sp
	uintptr depth;      // depth in call stack (0 == most recent)
};

static void
dumpgoroutine(G *gp)
{
	// ChildInfo child;
	Defer *d;
	Panic *p;

	dumpint(TagGoRoutine);
	dumpint((uintptr)gp);
	dumpint((uintptr)0);
	dumpint(gp->goid);
	dumpint(gp->gopc);
	dumpint(gp->status);
	dumpbool(gp->issystem);
	dumpbool(gp->isbackground);
	dumpint(gp->waitsince);
	dumpcstr((const int8 *)gp->waitreason);
	dumpint((uintptr)0);
	dumpint((uintptr)gp->m);
	dumpint((uintptr)gp->defer);
	dumpint((uintptr)gp->panic);

	// dump stack
	// child.args.n = -1;
	// child.arglen = 0;
	// child.sp = nil;
	// child.depth = 0;
	// if(!ScanStackByFrames)
	// 	runtime_throw("need frame info to dump stacks");
	// runtime_gentraceback(pc, sp, lr, gp, 0, nil, 0x7fffffff, dumpframe, &child, false);

	// dump defer & panic records
	for(d = gp->defer; d != nil; d = d->__next) {
		dumpint(TagDefer);
		dumpint((uintptr)d);
		dumpint((uintptr)gp);
		dumpint((uintptr)d->__arg);
		dumpint((uintptr)d->__frame);
		dumpint((uintptr)d->__pfn);
		dumpint((uintptr)0);
		dumpint((uintptr)d->__next);
	}
	for (p = gp->panic; p != nil; p = p->__next) {
		dumpint(TagPanic);
		dumpint((uintptr)p);
		dumpint((uintptr)gp);
		dumpint((uintptr)p->__arg.__type_descriptor);
		dumpint((uintptr)p->__arg.__object);
		dumpint((uintptr)0);
		dumpint((uintptr)p->__next);
	}
}

static void
dumpgs(void)
{
	G *gp;
	uint32 i;

	// goroutines & stacks
	for(i = 0; i < runtime_allglen; i++) {
		gp = runtime_allg[i];
		switch(gp->status){
		default:
			runtime_printf("unexpected G.status %d\n", gp->status);
			runtime_throw("mark - bad status");
		case Gdead:
			break;
		case Grunnable:
		case Gsyscall:
		case Gwaiting:
			dumpgoroutine(gp);
			break;
		}
	}
}

static void
finq_callback(FuncVal *fn, void *obj, const FuncType *ft, const PtrType *ot)
{
	dumpint(TagQueuedFinalizer);
	dumpint((uintptr)obj);
	dumpint((uintptr)fn);
	dumpint((uintptr)fn->fn);
	dumpint((uintptr)ft);
	dumpint((uintptr)ot);
}


static void
dumproots(void)
{
	MSpan *s, **allspans;
	uint32 spanidx;
	Special *sp;
	SpecialFinalizer *spf;
	byte *p;

	// data segment
	// dumpint(TagData);
	// dumpint((uintptr)data);
	// dumpmemrange(data, edata - data);
	// dumpfields((uintptr*)gcdata + 1);

	// bss segment
	// dumpint(TagBss);
	// dumpint((uintptr)bss);
	// dumpmemrange(bss, ebss - bss);
	// dumpfields((uintptr*)gcbss + 1);
	
	// MSpan.types
	allspans = runtime_mheap.allspans;
	for(spanidx=0; spanidx<runtime_mheap.nspan; spanidx++) {
		s = allspans[spanidx];
		if(s->state == MSpanInUse) {
			// The garbage collector ignores type pointers stored in MSpan.types:
			//  - Compiler-generated types are stored outside of heap.
			//  - The reflect package has runtime-generated types cached in its data structures.
			//    The garbage collector relies on finding the references via that cache.
			switch(s->types.compression) {
			case MTypes_Empty:
			case MTypes_Single:
				break;
			case MTypes_Words:
			case MTypes_Bytes:
				dumpotherroot("runtime type info", (byte*)s->types.data);
				break;
			}

			// Finalizers
			for(sp = s->specials; sp != nil; sp = sp->next) {
				if(sp->kind != KindSpecialFinalizer)
					continue;
				spf = (SpecialFinalizer*)sp;
				p = (byte*)((s->start << PageShift) + spf->special.offset);
				dumpfinalizer(p, spf->fn, spf->ft, spf->ot);
			}
		}
	}

	// Finalizer queue
	runtime_iterate_finq(finq_callback);
}

// Bit vector of free marks.
// Needs to be as big as the largest number of objects per span.
static byte hfree[PageSize/8];

static void
dumpobjs(void)
{
	uintptr i, j, size, n, off, shift, *bitp, bits, ti, kind;
	MSpan *s;
	MLink *l;
	byte *p;
	const Type *t;

	for(i = 0; i < runtime_mheap.nspan; i++) {
		s = runtime_mheap.allspans[i];
		if(s->state != MSpanInUse)
			continue;
		p = (byte*)(s->start << PageShift);
		size = s->elemsize;
		n = (s->npages << PageShift) / size;
		if(n > PageSize/8)
			runtime_throw("free array doesn't have enough entries");
		for(l = s->freelist; l != nil; l = l->next) {
			hfree[((byte*)l - p) / size] = true;
		}
		for(j = 0; j < n; j++, p += size) {
			if(hfree[j]) {
				hfree[j] = false;
				continue;
			}
			off = (uintptr*)p - (uintptr*)runtime_mheap.arena_start;
			bitp = (uintptr*)runtime_mheap.arena_start - off/wordsPerBitmapWord - 1;
			shift = off % wordsPerBitmapWord;
			bits = *bitp >> shift;

			// Skip FlagNoGC allocations (stacks)
			if((bits & bitAllocated) == 0)
				continue;

			// extract type and kind
			ti = runtime_gettype(p);
			t = (Type*)(ti & ~(uintptr)(PtrSize-1));
			kind = ti & (PtrSize-1);
			
			// dump it
			if(kind == TypeInfo_Chan)
				t = ((const ChanType*)t)->__element_type; // use element type for chan encoding
			if(t == nil && scannable(p))
				kind = TypeInfo_Conservative; // special kind for conservatively scanned objects
			dumpobj(p, size, t, kind);
		}
	}
}

static void
dumpparams(void)
{
	byte *x;

	dumpint(TagParams);
	x = (byte*)1;
	if(*(byte*)&x == 1)
		dumpbool(false); // little-endian ptrs
	else
		dumpbool(true); // big-endian ptrs
	dumpint(PtrSize);
	dumpint(runtime_Hchansize);
	dumpint((uintptr)runtime_mheap.arena_start);
	dumpint((uintptr)runtime_mheap.arena_used);
	dumpint(0);
	dumpcstr((const int8 *)"");
	dumpint(runtime_ncpu);
}

static void
dumpms(void)
{
	M *mp;

	for(mp = runtime_allm; mp != nil; mp = mp->alllink) {
		dumpint(TagOSThread);
		dumpint((uintptr)mp);
		dumpint(mp->id);
		dumpint(0);
	}
}

static void
dumpmemstats(void)
{
	int32 i;

	dumpint(TagMemStats);
	dumpint(mstats.alloc);
	dumpint(mstats.total_alloc);
	dumpint(mstats.sys);
	dumpint(mstats.nlookup);
	dumpint(mstats.nmalloc);
	dumpint(mstats.nfree);
	dumpint(mstats.heap_alloc);
	dumpint(mstats.heap_sys);
	dumpint(mstats.heap_idle);
	dumpint(mstats.heap_inuse);
	dumpint(mstats.heap_released);
	dumpint(mstats.heap_objects);
	dumpint(mstats.stacks_inuse);
	dumpint(mstats.stacks_sys);
	dumpint(mstats.mspan_inuse);
	dumpint(mstats.mspan_sys);
	dumpint(mstats.mcache_inuse);
	dumpint(mstats.mcache_sys);
	dumpint(mstats.buckhash_sys);
	dumpint(mstats.gc_sys);
	dumpint(mstats.other_sys);
	dumpint(mstats.next_gc);
	dumpint(mstats.last_gc);
	dumpint(mstats.pause_total_ns);
	for(i = 0; i < 256; i++)
		dumpint(mstats.pause_ns[i]);
	dumpint(mstats.numgc);
}

static void
dumpmemprof_callback(Bucket *b, uintptr nstk, Location *stk, uintptr size, uintptr allocs, uintptr frees)
{
	uintptr i, pc;
	byte buf[20];

	dumpint(TagMemProf);
	dumpint((uintptr)b);
	dumpint(size);
	dumpint(nstk);
	for(i = 0; i < nstk; i++) {
		pc = stk[i].pc;
		if(stk[i].function.len == 0) {
			runtime_snprintf(buf, sizeof(buf), "%X", (uint64)pc);
			dumpcstr((int8*)buf);
			dumpcstr((const int8*)"?");
			dumpint(0);
		} else {
			dumpstr(stk[i].function);
			dumpstr(stk[i].filename);
			dumpint(stk[i].lineno);
		}
	}
	dumpint(allocs);
	dumpint(frees);
}

static void
dumpmemprof(void)
{
	MSpan *s, **allspans;
	uint32 spanidx;
	Special *sp;
	SpecialProfile *spp;
	byte *p;

	runtime_iterate_memprof(dumpmemprof_callback);

	allspans = runtime_mheap.allspans;
	for(spanidx=0; spanidx<runtime_mheap.nspan; spanidx++) {
		s = allspans[spanidx];
		if(s->state != MSpanInUse)
			continue;
		for(sp = s->specials; sp != nil; sp = sp->next) {
			if(sp->kind != KindSpecialProfile)
				continue;
			spp = (SpecialProfile*)sp;
			p = (byte*)((s->start << PageShift) + spp->special.offset);
			dumpint(TagAllocSample);
			dumpint((uintptr)p);
			dumpint((uintptr)spp->b);
		}
	}
}

static void
mdump(G *gp)
{
	const byte *hdr;
	uintptr i;
	MSpan *s;

	// make sure we're done sweeping
	for(i = 0; i < runtime_mheap.nspan; i++) {
		s = runtime_mheap.allspans[i];
		if(s->state == MSpanInUse)
			runtime_MSpan_EnsureSwept(s);
	}

	runtime_memclr((byte*)&typecache[0], sizeof(typecache));
	hdr = (const byte*)"go1.3 heap dump\n";
	hwrite(hdr, runtime_findnull(hdr));
	dumpparams();
	dumpobjs();
	dumpgs();
	dumpms();
	dumproots();
	dumpmemstats();
	dumpmemprof();
	dumpint(TagEOF);
	flush();

	gp->param = nil;
	gp->status = Grunning;
	runtime_gogo(gp);
}

void runtime_debug_WriteHeapDump(uintptr)
  __asm__(GOSYM_PREFIX "runtime_debug.WriteHeapDump");

void
runtime_debug_WriteHeapDump(uintptr fd)
{
	M *m;
	G *g;

	// Stop the world.
	runtime_semacquire(&runtime_worldsema, false);
	m = runtime_m();
	m->gcing = 1;
	m->locks++;
	runtime_stoptheworld();

	// Update stats so we can dump them.
	// As a side effect, flushes all the MCaches so the MSpan.freelist
	// lists contain all the free objects.
	runtime_updatememstats(nil);

	// Set dump file.
	dumpfd = fd;

	// Call dump routine on M stack.
	g = runtime_g();
	g->status = Gwaiting;
	g->waitreason = "dumping heap";
	runtime_mcall(mdump);

	// Reset dump file.
	dumpfd = 0;

	// Start up the world again.
	m->gcing = 0;
	runtime_semrelease(&runtime_worldsema);
	runtime_starttheworld();
	m->locks--;
}

// Runs the specified gc program.  Calls the callback for every
// pointer-like field specified by the program and passes to the
// callback the kind and offset of that field within the object.
// offset is the offset in the object of the start of the program.
// Returns a pointer to the opcode that ended the gc program (either
// GC_END or GC_ARRAY_NEXT).
/*
static uintptr*
playgcprog(uintptr offset, uintptr *prog, void (*callback)(void*,uintptr,uintptr), void *arg)
{
	uintptr len, elemsize, i, *end;

	for(;;) {
		switch(prog[0]) {
		case GC_END:
			return prog;
		case GC_PTR:
			callback(arg, FieldKindPtr, offset + prog[1]);
			prog += 3;
			break;
		case GC_APTR:
			callback(arg, FieldKindPtr, offset + prog[1]);
			prog += 2;
			break;
		case GC_ARRAY_START:
			len = prog[2];
			elemsize = prog[3];
			end = nil;
			for(i = 0; i < len; i++) {
				end = playgcprog(offset + prog[1] + i * elemsize, prog + 4, callback, arg);
				if(end[0] != GC_ARRAY_NEXT)
					runtime_throw("GC_ARRAY_START did not have matching GC_ARRAY_NEXT");
			}
			prog = end + 1;
			break;
		case GC_ARRAY_NEXT:
			return prog;
		case GC_CALL:
			playgcprog(offset + prog[1], (uintptr*)((byte*)prog + *(int32*)&prog[2]), callback, arg);
			prog += 3;
			break;
		case GC_CHAN_PTR:
			callback(arg, FieldKindPtr, offset + prog[1]);
			prog += 3;
			break;
		case GC_STRING:
			callback(arg, FieldKindString, offset + prog[1]);
			prog += 2;
			break;
		case GC_EFACE:
			callback(arg, FieldKindEface, offset + prog[1]);
			prog += 2;
			break;
		case GC_IFACE:
			callback(arg, FieldKindIface, offset + prog[1]);
			prog += 2;
			break;
		case GC_SLICE:
			callback(arg, FieldKindSlice, offset + prog[1]);
			prog += 3;
			break;
		case GC_REGION:
			playgcprog(offset + prog[1], (uintptr*)prog[3] + 1, callback, arg);
			prog += 4;
			break;
		default:
			runtime_printf("%D\n", (uint64)prog[0]);
			runtime_throw("bad gc op");
		}
	}
}

static void
dump_callback(void *p, uintptr kind, uintptr offset)
{
	USED(&p);
	dumpint(kind);
	dumpint(offset);
}

// dumpint() the kind & offset of each field in an object.
static void
dumpfields(uintptr *prog)
{
	playgcprog(0, prog, dump_callback, nil);
	dumpint(FieldKindEol);
}

static void
dumpeface_callback(void *p, uintptr kind, uintptr offset)
{
	Eface *e;

	if(kind != FieldKindEface)
		return;
	e = (Eface*)((byte*)p + offset);
	dumptype(e->__type_descriptor);
}
*/

// The heap dump reader needs to be able to disambiguate
// Eface entries.  So it needs to know every type that might
// appear in such an entry.  The following two routines accomplish
// that.

// Dump all the types that appear in the type field of
// any Eface contained in obj.
static void
dumpefacetypes(void *obj __attribute__ ((unused)), uintptr size, const Type *type, uintptr kind)
{
	uintptr i;

	switch(kind) {
	case TypeInfo_SingleObject:
		//playgcprog(0, (uintptr*)type->gc + 1, dumpeface_callback, obj);
		break;
	case TypeInfo_Array:
		for(i = 0; i <= size - type->__size; i += type->__size)
			//playgcprog(i, (uintptr*)type->gc + 1, dumpeface_callback, obj);
		break;
	case TypeInfo_Chan:
		if(type->__size == 0) // channels may have zero-sized objects in them
			break;
		for(i = runtime_Hchansize; i <= size - type->__size; i += type->__size)
			//playgcprog(i, (uintptr*)type->gc + 1, dumpeface_callback, obj);
		break;
	}
}
