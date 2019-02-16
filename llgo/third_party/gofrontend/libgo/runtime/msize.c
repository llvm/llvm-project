// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Malloc small size classes.
//
// See malloc.h for overview.
//
// The size classes are chosen so that rounding an allocation
// request up to the next size class wastes at most 12.5% (1.125x).
//
// Each size class has its own page count that gets allocated
// and chopped up when new objects of the size class are needed.
// That page count is chosen so that chopping up the run of
// pages into objects of the given size wastes at most 12.5% (1.125x)
// of the memory.  It is not necessary that the cutoff here be
// the same as above.
//
// The two sources of waste multiply, so the worst possible case
// for the above constraints would be that allocations of some
// size might have a 26.6% (1.266x) overhead.
// In practice, only one of the wastes comes into play for a
// given size (sizes < 512 waste mainly on the round-up,
// sizes > 512 waste mainly on the page chopping).
//
// TODO(rsc): Compute max waste for any given size.

#include "runtime.h"
#include "arch.h"
#include "malloc.h"

int32 runtime_class_to_size[NumSizeClasses];
int32 runtime_class_to_allocnpages[NumSizeClasses];

// The SizeToClass lookup is implemented using two arrays,
// one mapping sizes <= 1024 to their class and one mapping
// sizes >= 1024 and <= MaxSmallSize to their class.
// All objects are 8-aligned, so the first array is indexed by
// the size divided by 8 (rounded up).  Objects >= 1024 bytes
// are 128-aligned, so the second array is indexed by the
// size divided by 128 (rounded up).  The arrays are filled in
// by InitSizes.

int8 runtime_size_to_class8[1024/8 + 1];
int8 runtime_size_to_class128[(MaxSmallSize-1024)/128 + 1];

int32
runtime_SizeToClass(int32 size)
{
	if(size > MaxSmallSize)
		runtime_throw("SizeToClass - invalid size");
	if(size > 1024-8)
		return runtime_size_to_class128[(size-1024+127) >> 7];
	return runtime_size_to_class8[(size+7)>>3];
}

void
runtime_InitSizes(void)
{
	int32 align, sizeclass, size, nextsize, n;
	uint32 i;
	uintptr allocsize, npages;

	// Initialize the runtime_class_to_size table (and choose class sizes in the process).
	runtime_class_to_size[0] = 0;
	sizeclass = 1;	// 0 means no class
	align = 8;
	for(size = align; size <= MaxSmallSize; size += align) {
		if((size&(size-1)) == 0) {	// bump alignment once in a while
			if(size >= 2048)
				align = 256;
			else if(size >= 128)
				align = size / 8;
			else if(size >= 16)
				align = 16;	// required for x86 SSE instructions, if we want to use them
		}
		if((align&(align-1)) != 0)
			runtime_throw("InitSizes - bug");

		// Make the allocnpages big enough that
		// the leftover is less than 1/8 of the total,
		// so wasted space is at most 12.5%.
		allocsize = PageSize;
		while(allocsize%size > allocsize/8)
			allocsize += PageSize;
		npages = allocsize >> PageShift;

		// If the previous sizeclass chose the same
		// allocation size and fit the same number of
		// objects into the page, we might as well
		// use just this size instead of having two
		// different sizes.
		if(sizeclass > 1 &&
			(int32)npages == runtime_class_to_allocnpages[sizeclass-1] &&
			allocsize/size == allocsize/runtime_class_to_size[sizeclass-1]) {
			runtime_class_to_size[sizeclass-1] = size;
			continue;
		}

		runtime_class_to_allocnpages[sizeclass] = npages;
		runtime_class_to_size[sizeclass] = size;
		sizeclass++;
	}
	if(sizeclass != NumSizeClasses) {
		runtime_printf("sizeclass=%d NumSizeClasses=%d\n", sizeclass, NumSizeClasses);
		runtime_throw("InitSizes - bad NumSizeClasses");
	}

	// Initialize the size_to_class tables.
	nextsize = 0;
	for (sizeclass = 1; sizeclass < NumSizeClasses; sizeclass++) {
		for(; nextsize < 1024 && nextsize <= runtime_class_to_size[sizeclass]; nextsize+=8)
			runtime_size_to_class8[nextsize/8] = sizeclass;
		if(nextsize >= 1024)
			for(; nextsize <= runtime_class_to_size[sizeclass]; nextsize += 128)
				runtime_size_to_class128[(nextsize-1024)/128] = sizeclass;
	}

	// Double-check SizeToClass.
	if(0) {
		for(n=0; n < MaxSmallSize; n++) {
			sizeclass = runtime_SizeToClass(n);
			if(sizeclass < 1 || sizeclass >= NumSizeClasses || runtime_class_to_size[sizeclass] < n) {
				runtime_printf("size=%d sizeclass=%d runtime_class_to_size=%d\n", n, sizeclass, runtime_class_to_size[sizeclass]);
				runtime_printf("incorrect SizeToClass");
				goto dump;
			}
			if(sizeclass > 1 && runtime_class_to_size[sizeclass-1] >= n) {
				runtime_printf("size=%d sizeclass=%d runtime_class_to_size=%d\n", n, sizeclass, runtime_class_to_size[sizeclass]);
				runtime_printf("SizeToClass too big");
				goto dump;
			}
		}
	}

	// Copy out for statistics table.
	for(i=0; i<nelem(runtime_class_to_size); i++)
		mstats.by_size[i].size = runtime_class_to_size[i];
	return;

dump:
	if(1){
		runtime_printf("NumSizeClasses=%d\n", NumSizeClasses);
		runtime_printf("runtime_class_to_size:");
		for(sizeclass=0; sizeclass<NumSizeClasses; sizeclass++)
			runtime_printf(" %d", runtime_class_to_size[sizeclass]);
		runtime_printf("\n\n");
		runtime_printf("size_to_class8:");
		for(i=0; i<nelem(runtime_size_to_class8); i++)
			runtime_printf(" %d=>%d(%d)\n", i*8, runtime_size_to_class8[i],
				runtime_class_to_size[runtime_size_to_class8[i]]);
		runtime_printf("\n");
		runtime_printf("size_to_class128:");
		for(i=0; i<nelem(runtime_size_to_class128); i++)
			runtime_printf(" %d=>%d(%d)\n", i*128, runtime_size_to_class128[i],
				runtime_class_to_size[runtime_size_to_class128[i]]);
		runtime_printf("\n");
	}
	runtime_throw("InitSizes failed");
}

// Returns size of the memory block that mallocgc will allocate if you ask for the size.
uintptr
runtime_roundupsize(uintptr size)
{
	if(size < MaxSmallSize) {
		if(size <= 1024-8)
			return runtime_class_to_size[runtime_size_to_class8[(size+7)>>3]];
		else
			return runtime_class_to_size[runtime_size_to_class128[(size-1024+127) >> 7]];
	}
	if(size + PageSize < size)
		return size;
	return ROUND(size, PageSize);
}
