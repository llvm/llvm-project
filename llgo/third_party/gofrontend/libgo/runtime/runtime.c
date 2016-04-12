// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <signal.h>
#include <unistd.h>

#include "config.h"

#include "runtime.h"
#include "arch.h"
#include "array.h"

enum {
	maxround = sizeof(uintptr),
};

// Keep a cached value to make gotraceback fast,
// since we call it on every call to gentraceback.
// The cached value is a uint32 in which the low bit
// is the "crash" setting and the top 31 bits are the
// gotraceback value.
static uint32 traceback_cache = ~(uint32)0;

extern volatile intgo runtime_MemProfileRate
  __asm__ (GOSYM_PREFIX "runtime.MemProfileRate");


// The GOTRACEBACK environment variable controls the
// behavior of a Go program that is crashing and exiting.
//	GOTRACEBACK=0   suppress all tracebacks
//	GOTRACEBACK=1   default behavior - show tracebacks but exclude runtime frames
//	GOTRACEBACK=2   show tracebacks including runtime frames
//	GOTRACEBACK=crash   show tracebacks including runtime frames, then crash (core dump etc)
int32
runtime_gotraceback(bool *crash)
{
	String s;
	const byte *p;
	uint32 x;

	if(crash != nil)
		*crash = false;
	if(runtime_m()->traceback != 0)
		return runtime_m()->traceback;
	x = runtime_atomicload(&traceback_cache);
	if(x == ~(uint32)0) {
		s = runtime_getenv("GOTRACEBACK");
		p = s.str;
		if(s.len == 0)
			x = 1<<1;
		else if(s.len == 5 && runtime_strcmp((const char *)p, "crash") == 0)
			x = (2<<1) | 1;
		else
			x = runtime_atoi(p, s.len)<<1;	
		runtime_atomicstore(&traceback_cache, x);
	}
	if(crash != nil)
		*crash = x&1;
	return x>>1;
}

static int32	argc;
static byte**	argv;

static Slice args;
Slice envs;

void (*runtime_sysargs)(int32, uint8**);

void
runtime_args(int32 c, byte **v)
{
	argc = c;
	argv = v;
	if(runtime_sysargs != nil)
		runtime_sysargs(c, v);
}

byte*
runtime_progname()
{
  return argc == 0 ? nil : argv[0];
}

void
runtime_goargs(void)
{
	String *s;
	int32 i;

	// for windows implementation see "os" package
	if(Windows)
		return;

	s = runtime_malloc(argc*sizeof s[0]);
	for(i=0; i<argc; i++)
		s[i] = runtime_gostringnocopy((const byte*)argv[i]);
	args.__values = (void*)s;
	args.__count = argc;
	args.__capacity = argc;
}

void
runtime_goenvs_unix(void)
{
	String *s;
	int32 i, n;

	for(n=0; argv[argc+1+n] != 0; n++)
		;

	s = runtime_malloc(n*sizeof s[0]);
	for(i=0; i<n; i++)
		s[i] = runtime_gostringnocopy(argv[argc+1+i]);
	envs.__values = (void*)s;
	envs.__count = n;
	envs.__capacity = n;
}

// Called from the syscall package.
Slice runtime_envs(void) __asm__ (GOSYM_PREFIX "syscall.runtime_envs");

Slice
runtime_envs()
{
	return envs;
}

Slice os_runtime_args(void) __asm__ (GOSYM_PREFIX "os.runtime_args");

Slice
os_runtime_args()
{
	return args;
}

int32
runtime_atoi(const byte *p, intgo len)
{
	int32 n;

	n = 0;
	while(len > 0 && '0' <= *p && *p <= '9') {
		n = n*10 + *p++ - '0';
		len--;
	}
	return n;
}

static struct root_list runtime_roots =
{ nil,
  { { &envs, sizeof envs },
    { &args, sizeof args },
    { nil, 0 } },
};

static void
TestAtomic64(void)
{
	uint64 z64, x64;

	z64 = 42;
	x64 = 0;
	PREFETCH(&z64);
	if(runtime_cas64(&z64, x64, 1))
		runtime_throw("cas64 failed");
	if(x64 != 0)
		runtime_throw("cas64 failed");
	x64 = 42;
	if(!runtime_cas64(&z64, x64, 1))
		runtime_throw("cas64 failed");
	if(x64 != 42 || z64 != 1)
		runtime_throw("cas64 failed");
	if(runtime_atomicload64(&z64) != 1)
		runtime_throw("load64 failed");
	runtime_atomicstore64(&z64, (1ull<<40)+1);
	if(runtime_atomicload64(&z64) != (1ull<<40)+1)
		runtime_throw("store64 failed");
	if(runtime_xadd64(&z64, (1ull<<40)+1) != (2ull<<40)+2)
		runtime_throw("xadd64 failed");
	if(runtime_atomicload64(&z64) != (2ull<<40)+2)
		runtime_throw("xadd64 failed");
	if(runtime_xchg64(&z64, (3ull<<40)+3) != (2ull<<40)+2)
		runtime_throw("xchg64 failed");
	if(runtime_atomicload64(&z64) != (3ull<<40)+3)
		runtime_throw("xchg64 failed");
}

void
runtime_check(void)
{
	__go_register_gc_roots(&runtime_roots);

	TestAtomic64();
}

uint32
runtime_fastrand1(void)
{
	M *m;
	uint32 x;

	m = runtime_m();
	x = m->fastrand;
	x += x;
	if(x & 0x80000000L)
		x ^= 0x88888eefUL;
	m->fastrand = x;
	return x;
}

int64
runtime_cputicks(void)
{
#if defined(__386__) || defined(__x86_64__)
  uint32 low, high;
  asm("rdtsc" : "=a" (low), "=d" (high));
  return (int64)(((uint64)high << 32) | (uint64)low);
#elif defined (__s390__) || defined (__s390x__)
  uint64 clock = 0;
  /* stckf may not write the return variable in case of a clock error, so make
     it read-write to prevent that the initialisation is optimised out.
     Note: Targets below z9-109 will crash when executing store clock fast, i.e.
     we don't support Go for machines older than that.  */
  asm volatile(".insn s,0xb27c0000,%0" /* stckf */ : "+Q" (clock) : : "cc" );
  return (int64)clock;
#else
  // FIXME: implement for other processors.
  return 0;
#endif
}

bool
runtime_showframe(String s, bool current)
{
	static int32 traceback = -1;

	if(current && runtime_m()->throwing > 0)
		return 1;
	if(traceback < 0)
		traceback = runtime_gotraceback(nil);
	return traceback > 1 || (__builtin_memchr(s.str, '.', s.len) != nil && __builtin_memcmp(s.str, "runtime.", 7) != 0);
}

static Lock ticksLock;
static int64 ticks;

int64
runtime_tickspersecond(void)
{
	int64 res, t0, t1, c0, c1;

	res = (int64)runtime_atomicload64((uint64*)&ticks);
	if(res != 0)
		return ticks;
	runtime_lock(&ticksLock);
	res = ticks;
	if(res == 0) {
		t0 = runtime_nanotime();
		c0 = runtime_cputicks();
		runtime_usleep(100*1000);
		t1 = runtime_nanotime();
		c1 = runtime_cputicks();
		if(t1 == t0)
			t1++;
		res = (c1-c0)*1000*1000*1000/(t1-t0);
		if(res == 0)
			res++;
		runtime_atomicstore64((uint64*)&ticks, res);
	}
	runtime_unlock(&ticksLock);
	return res;
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime_mpreinit(M *mp)
{
	mp->gsignal = runtime_malg(32*1024, &mp->gsignalstack, &mp->gsignalstacksize);	// OS X wants >=8K, Linux >=2K
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime_minit(void)
{
	M* m;
	sigset_t sigs;

	// Initialize signal handling.
	m = runtime_m();
	runtime_signalstack(m->gsignalstack, m->gsignalstacksize);
	if (sigemptyset(&sigs) != 0)
		runtime_throw("sigemptyset");
	pthread_sigmask(SIG_SETMASK, &sigs, nil);
}

// Called from dropm to undo the effect of an minit.
void
runtime_unminit(void)
{
	runtime_signalstack(nil, 0);
}


void
runtime_signalstack(byte *p, int32 n)
{
	stack_t st;

	st.ss_sp = p;
	st.ss_size = n;
	st.ss_flags = 0;
	if(p == nil)
		st.ss_flags = SS_DISABLE;
	if(sigaltstack(&st, nil) < 0)
		*(int *)0xf1 = 0xf1;
}

DebugVars	runtime_debug;

// Holds variables parsed from GODEBUG env var,
// except for "memprofilerate" since there is an
// existing var for that value which is int
// instead of in32 and might have an
// initial value.
static struct {
	const char* name;
	int32*	value;
} dbgvar[] = {
	{"allocfreetrace", &runtime_debug.allocfreetrace},
	{"efence", &runtime_debug.efence},
	{"gctrace", &runtime_debug.gctrace},
	{"gcdead", &runtime_debug.gcdead},
	{"scheddetail", &runtime_debug.scheddetail},
	{"schedtrace", &runtime_debug.schedtrace},
};

void
runtime_parsedebugvars(void)
{
	String s;
	const byte *p, *pn;
	intgo len;
	intgo i, n;
	bool tmp;
	
	// gotraceback caches the GOTRACEBACK setting in traceback_cache.
	// gotraceback can be called before the environment is available.
	// traceback_cache must be reset after the environment is made
	// available, in order for the environment variable to take effect.
	// The code is fixed differently in Go 1.4.
	// This is a limited fix for Go 1.3.3.
	traceback_cache = ~(uint32)0;
	runtime_gotraceback(&tmp);

	s = runtime_getenv("GODEBUG");
	if(s.len == 0)
		return;
	p = s.str;
	len = s.len;
	for(;;) {
		for(i=0; i<(intgo)nelem(dbgvar); i++) {
			n = runtime_findnull((const byte*)dbgvar[i].name);
			if(len > n && runtime_mcmp(p, "memprofilerate", n) == 0 && p[n] == '=')
				// Set the MemProfileRate directly since it
				// is an int, not int32, and should only lbe
				// set here if specified by GODEBUG
				runtime_MemProfileRate = runtime_atoi(p+n+1, len-(n+1));
			else if(len > n && runtime_mcmp(p, dbgvar[i].name, n) == 0 && p[n] == '=')
				*dbgvar[i].value = runtime_atoi(p+n+1, len-(n+1));
		}
		pn = (const byte *)runtime_strstr((const char *)p, ",");
		if(pn == nil || pn - p >= len)
			break;
		len -= (pn - p) - 1;
		p = pn + 1;
	}
}

// Poor mans 64-bit division.
// This is a very special function, do not use it if you are not sure what you are doing.
// int64 division is lowered into _divv() call on 386, which does not fit into nosplit functions.
// Handles overflow in a time-specific manner.
int32
runtime_timediv(int64 v, int32 div, int32 *rem)
{
	int32 res, bit;

	if(v >= (int64)div*0x7fffffffLL) {
		if(rem != nil)
			*rem = 0;
		return 0x7fffffff;
	}
	res = 0;
	for(bit = 30; bit >= 0; bit--) {
		if(v >= ((int64)div<<bit)) {
			v = v - ((int64)div<<bit);
			res += 1<<bit;
		}
	}
	if(rem != nil)
		*rem = v;
	return res;
}

// Setting the max stack size doesn't really do anything for gccgo.

uintptr runtime_maxstacksize = 1<<20; // enough until runtime.main sets it for real

void memclrBytes(Slice)
     __asm__ (GOSYM_PREFIX "runtime.memclrBytes");

void
memclrBytes(Slice s)
{
	runtime_memclr(s.__values, s.__count);
}
