// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parallel for algorithm.

#include "runtime.h"
#include "arch.h"

struct ParForThread
{
	// the thread's iteration space [32lsb, 32msb)
	uint64 pos;
	// stats
	uint64 nsteal;
	uint64 nstealcnt;
	uint64 nprocyield;
	uint64 nosyield;
	uint64 nsleep;
	byte pad[CacheLineSize];
};

ParFor*
runtime_parforalloc(uint32 nthrmax)
{
	ParFor *desc;

	// The ParFor object is followed by CacheLineSize padding
	// and then nthrmax ParForThread.
	desc = (ParFor*)runtime_malloc(sizeof(ParFor) + CacheLineSize + nthrmax * sizeof(ParForThread));
	desc->thr = (ParForThread*)((byte*)(desc+1) + CacheLineSize);
	desc->nthrmax = nthrmax;
	return desc;
}

void
runtime_parforsetup(ParFor *desc, uint32 nthr, uint32 n, bool wait, const FuncVal *body)
{
	uint32 i, begin, end;
	uint64 *pos;

	if(desc == nil || nthr == 0 || nthr > desc->nthrmax || body == nil) {
		runtime_printf("desc=%p nthr=%d count=%d body=%p\n", desc, nthr, n, body);
		runtime_throw("parfor: invalid args");
	}

	desc->body = body;
	desc->done = 0;
	desc->nthr = nthr;
	desc->thrseq = 0;
	desc->cnt = n;
	desc->wait = wait;
	desc->nsteal = 0;
	desc->nstealcnt = 0;
	desc->nprocyield = 0;
	desc->nosyield = 0;
	desc->nsleep = 0;
	for(i=0; i<nthr; i++) {
		begin = (uint64)n*i / nthr;
		end = (uint64)n*(i+1) / nthr;
		pos = &desc->thr[i].pos;
		if(((uintptr)pos & 7) != 0)
			runtime_throw("parforsetup: pos is not aligned");
		*pos = (uint64)begin | (((uint64)end)<<32);
	}
}

void
runtime_parfordo(ParFor *desc)
{
	ParForThread *me;
	uint32 tid, begin, end, begin2, try, victim, i;
	uint64 *mypos, *victimpos, pos, newpos;
	const FuncVal *body;
	void (*bodyfn)(ParFor*, uint32);
	bool idle;

	// Obtain 0-based thread index.
	tid = runtime_xadd(&desc->thrseq, 1) - 1;
	if(tid >= desc->nthr) {
		runtime_printf("tid=%d nthr=%d\n", tid, desc->nthr);
		runtime_throw("parfor: invalid tid");
	}

	body = desc->body;
	bodyfn = (void (*)(ParFor*, uint32))(void*)body->fn;

	// If single-threaded, just execute the for serially.
	if(desc->nthr==1) {
		for(i=0; i<desc->cnt; i++)
		  __builtin_call_with_static_chain (bodyfn(desc, i), body);
		return;
	}

	me = &desc->thr[tid];
	mypos = &me->pos;
	for(;;) {
		for(;;) {
			// While there is local work,
			// bump low index and execute the iteration.
			pos = runtime_xadd64(mypos, 1);
			begin = (uint32)pos-1;
			end = (uint32)(pos>>32);
			if(begin < end) {
				__builtin_call_with_static_chain(bodyfn(desc, begin), body);
				continue;
			}
			break;
		}

		// Out of work, need to steal something.
		idle = false;
		for(try=0;; try++) {
			// If we don't see any work for long enough,
			// increment the done counter...
			if(try > desc->nthr*4 && !idle) {
				idle = true;
				runtime_xadd(&desc->done, 1);
			}
			// ...if all threads have incremented the counter,
			// we are done.
			if(desc->done + !idle == desc->nthr) {
				if(!idle)
					runtime_xadd(&desc->done, 1);
				goto exit;
			}
			// Choose a random victim for stealing.
			victim = runtime_fastrand1() % (desc->nthr-1);
			if(victim >= tid)
				victim++;
			victimpos = &desc->thr[victim].pos;
			for(;;) {
				// See if it has any work.
				pos = runtime_atomicload64(victimpos);
				begin = (uint32)pos;
				end = (uint32)(pos>>32);
				if(begin+1 >= end) {
					begin = end = 0;
					break;
				}
				if(idle) {
					runtime_xadd(&desc->done, -1);
					idle = false;
				}
				begin2 = begin + (end-begin)/2;
				newpos = (uint64)begin | (uint64)begin2<<32;
				if(runtime_cas64(victimpos, pos, newpos)) {
					begin = begin2;
					break;
				}
			}
			if(begin < end) {
				// Has successfully stolen some work.
				if(idle)
					runtime_throw("parfor: should not be idle");
				runtime_atomicstore64(mypos, (uint64)begin | (uint64)end<<32);
				me->nsteal++;
				me->nstealcnt += end-begin;
				break;
			}
			// Backoff.
			if(try < desc->nthr) {
				// nothing
			} else if (try < 4*desc->nthr) {
				me->nprocyield++;
				runtime_procyield(20);
			// If a caller asked not to wait for the others, exit now
			// (assume that most work is already done at this point).
			} else if (!desc->wait) {
				if(!idle)
					runtime_xadd(&desc->done, 1);
				goto exit;
			} else if (try < 6*desc->nthr) {
				me->nosyield++;
				runtime_osyield();
			} else {
				me->nsleep++;
				runtime_usleep(1);
			}
		}
	}
exit:
	runtime_xadd64(&desc->nsteal, me->nsteal);
	runtime_xadd64(&desc->nstealcnt, me->nstealcnt);
	runtime_xadd64(&desc->nprocyield, me->nprocyield);
	runtime_xadd64(&desc->nosyield, me->nosyield);
	runtime_xadd64(&desc->nsleep, me->nsleep);
	me->nsteal = 0;
	me->nstealcnt = 0;
	me->nprocyield = 0;
	me->nosyield = 0;
	me->nsleep = 0;
}

// For testing from Go.
void
runtime_parforiters(ParFor *desc, uintptr tid, uintptr *start, uintptr *end)
{
	*start = (uint32)desc->thr[tid].pos;
	*end = (uint32)(desc->thr[tid].pos>>32);
}
