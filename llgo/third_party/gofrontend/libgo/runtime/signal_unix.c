// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

#include <sys/time.h>

#include "runtime.h"
#include "defs.h"
#include "signal_unix.h"

extern SigTab runtime_sigtab[];

void
runtime_initsig(void)
{
	int32 i;
	SigTab *t;

	// First call: basic setup.
	for(i = 0; runtime_sigtab[i].sig != -1; i++) {
		t = &runtime_sigtab[i];
		if((t->flags == 0) || (t->flags & SigDefault))
			continue;

		// For some signals, we respect an inherited SIG_IGN handler
		// rather than insist on installing our own default handler.
		// Even these signals can be fetched using the os/signal package.
		switch(t->sig) {
		case SIGHUP:
		case SIGINT:
			if(runtime_getsig(i) == GO_SIG_IGN) {
				t->flags = SigNotify | SigIgnored;
				continue;
			}
		}

		t->flags |= SigHandling;
		runtime_setsig(i, runtime_sighandler, true);
	}
}

void
runtime_sigenable(uint32 sig)
{
	int32 i;
	SigTab *t;

	t = nil;
	for(i = 0; runtime_sigtab[i].sig != -1; i++) {
		if(runtime_sigtab[i].sig == (int32)sig) {
			t = &runtime_sigtab[i];
			break;
		}
	}

	if(t == nil)
		return;

	if((t->flags & SigNotify) && !(t->flags & SigHandling)) {
		t->flags |= SigHandling;
		if(runtime_getsig(i) == GO_SIG_IGN)
			t->flags |= SigIgnored;
		runtime_setsig(i, runtime_sighandler, true);
	}
}

void
runtime_sigdisable(uint32 sig)
{
	int32 i;
	SigTab *t;

	t = nil;
	for(i = 0; runtime_sigtab[i].sig != -1; i++) {
		if(runtime_sigtab[i].sig == (int32)sig) {
			t = &runtime_sigtab[i];
			break;
		}
	}

	if(t == nil)
		return;

	if((t->flags & SigNotify) && (t->flags & SigHandling)) {
		t->flags &= ~SigHandling;
		if(t->flags & SigIgnored)
			runtime_setsig(i, GO_SIG_IGN, true);
		else
			runtime_setsig(i, GO_SIG_DFL, true);
	}
}

void
runtime_sigignore(uint32 sig)
{
	int32 i;
	SigTab *t;

	t = nil;
	for(i = 0; runtime_sigtab[i].sig != -1; i++) {
		if(runtime_sigtab[i].sig == (int32)sig) {
			t = &runtime_sigtab[i];
			break;
		}
	}

	if(t == nil)
		return;

	if((t->flags & SigNotify) != 0) {
		t->flags &= ~SigHandling;
		runtime_setsig(i, GO_SIG_IGN, true);
	}
}

void
runtime_resetcpuprofiler(int32 hz)
{
	struct itimerval it;

	runtime_memclr((byte*)&it, sizeof it);
	if(hz == 0) {
		runtime_setitimer(ITIMER_PROF, &it, nil);
	} else {
		it.it_interval.tv_sec = 0;
		it.it_interval.tv_usec = 1000000 / hz;
		it.it_value = it.it_interval;
		runtime_setitimer(ITIMER_PROF, &it, nil);
	}
	runtime_m()->profilehz = hz;
}

void
os_sigpipe(void)
{
	int32 i;

	for(i = 0; runtime_sigtab[i].sig != -1; i++)
		if(runtime_sigtab[i].sig == SIGPIPE)
			break;
	runtime_setsig(i, GO_SIG_DFL, false);
	runtime_raise(SIGPIPE);
}

void
runtime_unblocksignals(void)
{
	sigset_t sigset_none;
	sigemptyset(&sigset_none);
	pthread_sigmask(SIG_SETMASK, &sigset_none, nil);
}

void
runtime_crash(void)
{
	int32 i;

#ifdef GOOS_darwin
	// OS X core dumps are linear dumps of the mapped memory,
	// from the first virtual byte to the last, with zeros in the gaps.
	// Because of the way we arrange the address space on 64-bit systems,
	// this means the OS X core file will be >128 GB and even on a zippy
	// workstation can take OS X well over an hour to write (uninterruptible).
	// Save users from making that mistake.
	if(sizeof(void*) == 8)
		return;
#endif

	runtime_unblocksignals();
	for(i = 0; runtime_sigtab[i].sig != -1; i++)
		if(runtime_sigtab[i].sig == SIGABRT)
			break;
	runtime_setsig(i, GO_SIG_DFL, false);
	runtime_raise(SIGABRT);
}

void
runtime_raise(int32 sig)
{
	raise(sig);
}
