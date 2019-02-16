// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
#include "signal_unix.h"

// Linux futex.
//
//	futexsleep(uint32 *addr, uint32 val)
//	futexwakeup(uint32 *addr)
//
// Futexsleep atomically checks if *addr == val and if so, sleeps on addr.
// Futexwakeup wakes up threads sleeping on addr.
// Futexsleep is allowed to wake up spuriously.

#include <errno.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <syscall.h>
#include <linux/futex.h>

typedef struct timespec Timespec;

// Atomically,
//	if(*addr == val) sleep
// Might be woken up spuriously; that's allowed.
// Don't sleep longer than ns; ns < 0 means forever.
void
runtime_futexsleep(uint32 *addr, uint32 val, int64 ns)
{
	Timespec ts;
	int32 nsec;

	// Some Linux kernels have a bug where futex of
	// FUTEX_WAIT returns an internal error code
	// as an errno.  Libpthread ignores the return value
	// here, and so can we: as it says a few lines up,
	// spurious wakeups are allowed.

	if(ns < 0) {
		syscall(__NR_futex, addr, FUTEX_WAIT, val, nil, nil, 0);
		return;
	}
	ts.tv_sec = runtime_timediv(ns, 1000000000LL, &nsec);
	ts.tv_nsec = nsec;
	syscall(__NR_futex, addr, FUTEX_WAIT, val, &ts, nil, 0);
}

// If any procs are sleeping on addr, wake up at most cnt.
void
runtime_futexwakeup(uint32 *addr, uint32 cnt)
{
	int64 ret;

	ret = syscall(__NR_futex, addr, FUTEX_WAKE, cnt, nil, nil, 0);

	if(ret >= 0)
		return;

	// I don't know that futex wakeup can return
	// EAGAIN or EINTR, but if it does, it would be
	// safe to loop and call futex again.
	runtime_printf("futexwakeup addr=%p returned %D\n", addr, ret);
	*(int32*)0x1006 = 0x1006;
}

void
runtime_osinit(void)
{
	runtime_ncpu = getproccount();
}

void
runtime_goenvs(void)
{
	runtime_goenvs_unix();
}
