// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <features.h>
#include <sched.h>

// CPU_COUNT is only provided by glibc 2.6 or higher
#ifndef CPU_COUNT
#define CPU_COUNT(set) _CPU_COUNT((unsigned int *)(set), sizeof(*(set))/sizeof(unsigned int))
static int _CPU_COUNT(unsigned int *set, size_t len) {
	int cnt;

	cnt = 0;
	while (len--)
		cnt += __builtin_popcount(*set++);
	return cnt;
}
#endif

#include "runtime.h"
#include "defs.h"

int32
getproccount(void)
{
	cpu_set_t set;
	int32 r, cnt;

	cnt = 0;
	r = sched_getaffinity(0, sizeof(set), &set);
	if(r == 0)
		cnt += CPU_COUNT(&set);

	return cnt ? cnt : 1;
}
