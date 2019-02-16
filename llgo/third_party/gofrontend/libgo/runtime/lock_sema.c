// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin nacl netbsd openbsd plan9 solaris windows

#include "runtime.h"

// This implementation depends on OS-specific implementations of
//
//	uintptr runtime_semacreate(void)
//		Create a semaphore, which will be assigned to m->waitsema.
//		The zero value is treated as absence of any semaphore,
//		so be sure to return a non-zero value.
//
//	int32 runtime_semasleep(int64 ns)
//		If ns < 0, acquire m->waitsema and return 0.
//		If ns >= 0, try to acquire m->waitsema for at most ns nanoseconds.
//		Return 0 if the semaphore was acquired, -1 if interrupted or timed out.
//
//	int32 runtime_semawakeup(M *mp)
//		Wake up mp, which is or will soon be sleeping on mp->waitsema.
//

enum
{
	LOCKED = 1,

	ACTIVE_SPIN = 4,
	ACTIVE_SPIN_CNT = 30,
	PASSIVE_SPIN = 1,
};

void
runtime_lock(Lock *l)
{
	M *m;
	uintptr v;
	uint32 i, spin;

	m = runtime_m();
	if(m->locks++ < 0)
		runtime_throw("runtime_lock: lock count");

	// Speculative grab for lock.
	if(runtime_casp((void**)&l->key, nil, (void*)LOCKED))
		return;

	if(m->waitsema == 0)
		m->waitsema = runtime_semacreate();

	// On uniprocessor's, no point spinning.
	// On multiprocessors, spin for ACTIVE_SPIN attempts.
	spin = 0;
	if(runtime_ncpu > 1)
		spin = ACTIVE_SPIN;

	for(i=0;; i++) {
		v = (uintptr)runtime_atomicloadp((void**)&l->key);
		if((v&LOCKED) == 0) {
unlocked:
			if(runtime_casp((void**)&l->key, (void*)v, (void*)(v|LOCKED)))
				return;
			i = 0;
		}
		if(i<spin)
			runtime_procyield(ACTIVE_SPIN_CNT);
		else if(i<spin+PASSIVE_SPIN)
			runtime_osyield();
		else {
			// Someone else has it.
			// l->waitm points to a linked list of M's waiting
			// for this lock, chained through m->nextwaitm.
			// Queue this M.
			for(;;) {
				m->nextwaitm = (void*)(v&~LOCKED);
				if(runtime_casp((void**)&l->key, (void*)v, (void*)((uintptr)m|LOCKED)))
					break;
				v = (uintptr)runtime_atomicloadp((void**)&l->key);
				if((v&LOCKED) == 0)
					goto unlocked;
			}
			if(v&LOCKED) {
				// Queued.  Wait.
				runtime_semasleep(-1);
				i = 0;
			}
		}
	}
}

void
runtime_unlock(Lock *l)
{
	uintptr v;
	M *mp;

	for(;;) {
		v = (uintptr)runtime_atomicloadp((void**)&l->key);
		if(v == LOCKED) {
			if(runtime_casp((void**)&l->key, (void*)LOCKED, nil))
				break;
		} else {
			// Other M's are waiting for the lock.
			// Dequeue an M.
			mp = (void*)(v&~LOCKED);
			if(runtime_casp((void**)&l->key, (void*)v, mp->nextwaitm)) {
				// Dequeued an M.  Wake it.
				runtime_semawakeup(mp);
				break;
			}
		}
	}

	if(--runtime_m()->locks < 0)
		runtime_throw("runtime_unlock: lock count");
}

// One-time notifications.
void
runtime_noteclear(Note *n)
{
	n->key = 0;
}

void
runtime_notewakeup(Note *n)
{
	M *mp;

	do
		mp = runtime_atomicloadp((void**)&n->key);
	while(!runtime_casp((void**)&n->key, mp, (void*)LOCKED));

	// Successfully set waitm to LOCKED.
	// What was it before?
	if(mp == nil) {
		// Nothing was waiting.  Done.
	} else if(mp == (M*)LOCKED) {
		// Two notewakeups!  Not allowed.
		runtime_throw("notewakeup - double wakeup");
	} else {
		// Must be the waiting m.  Wake it up.
		runtime_semawakeup(mp);
	}
}

void
runtime_notesleep(Note *n)
{
	M *m;

	m = runtime_m();

  /* For gccgo it's OK to sleep in non-g0, and it happens in
     stoptheworld because we have not implemented preemption.

	if(runtime_g() != m->g0)
		runtime_throw("notesleep not on g0");
  */

	if(m->waitsema == 0)
		m->waitsema = runtime_semacreate();
	if(!runtime_casp((void**)&n->key, nil, m)) {  // must be LOCKED (got wakeup)
		if(n->key != LOCKED)
			runtime_throw("notesleep - waitm out of sync");
		return;
	}
	// Queued.  Sleep.
	m->blocked = true;
	runtime_semasleep(-1);
	m->blocked = false;
}

static bool
notetsleep(Note *n, int64 ns, int64 deadline, M *mp)
{
	M *m;

	m = runtime_m();

	// Conceptually, deadline and mp are local variables.
	// They are passed as arguments so that the space for them
	// does not count against our nosplit stack sequence.

	// Register for wakeup on n->waitm.
	if(!runtime_casp((void**)&n->key, nil, m)) {  // must be LOCKED (got wakeup already)
		if(n->key != LOCKED)
			runtime_throw("notetsleep - waitm out of sync");
		return true;
	}

	if(ns < 0) {
		// Queued.  Sleep.
		m->blocked = true;
		runtime_semasleep(-1);
		m->blocked = false;
		return true;
	}

	deadline = runtime_nanotime() + ns;
	for(;;) {
		// Registered.  Sleep.
		m->blocked = true;
		if(runtime_semasleep(ns) >= 0) {
			m->blocked = false;
			// Acquired semaphore, semawakeup unregistered us.
			// Done.
			return true;
		}
		m->blocked = false;

		// Interrupted or timed out.  Still registered.  Semaphore not acquired.
		ns = deadline - runtime_nanotime();
		if(ns <= 0)
			break;
		// Deadline hasn't arrived.  Keep sleeping.
	}

	// Deadline arrived.  Still registered.  Semaphore not acquired.
	// Want to give up and return, but have to unregister first,
	// so that any notewakeup racing with the return does not
	// try to grant us the semaphore when we don't expect it.
	for(;;) {
		mp = runtime_atomicloadp((void**)&n->key);
		if(mp == m) {
			// No wakeup yet; unregister if possible.
			if(runtime_casp((void**)&n->key, mp, nil))
				return false;
		} else if(mp == (M*)LOCKED) {
			// Wakeup happened so semaphore is available.
			// Grab it to avoid getting out of sync.
			m->blocked = true;
			if(runtime_semasleep(-1) < 0)
				runtime_throw("runtime: unable to acquire - semaphore out of sync");
			m->blocked = false;
			return true;
		} else
			runtime_throw("runtime: unexpected waitm - semaphore out of sync");
	}
}

bool
runtime_notetsleep(Note *n, int64 ns)
{
	M *m;
	bool res;

	m = runtime_m();

	if(runtime_g() != m->g0 && !m->gcing)
		runtime_throw("notetsleep not on g0");

	if(m->waitsema == 0)
		m->waitsema = runtime_semacreate();

	res = notetsleep(n, ns, 0, nil);
	return res;
}

// same as runtime_notetsleep, but called on user g (not g0)
// calls only nosplit functions between entersyscallblock/exitsyscall
bool
runtime_notetsleepg(Note *n, int64 ns)
{
	M *m;
	bool res;

	m = runtime_m();

	if(runtime_g() == m->g0)
		runtime_throw("notetsleepg on g0");

	if(m->waitsema == 0)
		m->waitsema = runtime_semacreate();

	runtime_entersyscallblock();
	res = notetsleep(n, ns, 0, nil);
	runtime_exitsyscall();
	return res;
}
