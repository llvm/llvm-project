// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <complex.h>
#include <math.h>
#include <stdarg.h>
#include "runtime.h"
#include "array.h"
#include "go-type.h"

//static Lock debuglock;

// Clang requires this function to not be inlined (see below).
static void go_vprintf(const char*, va_list)
__attribute__((noinline));

// write to goroutine-local buffer if diverting output,
// or else standard error.
static void
gwrite(const void *v, intgo n)
{
	G* g = runtime_g();

	if(g == nil || g->writebuf == nil) {
		// Avoid -D_FORTIFY_SOURCE problems.
		int rv __attribute__((unused));

		rv = runtime_write(2, v, n);
		return;
	}

	if(g->writenbuf == 0)
		return;

	if(n > g->writenbuf)
		n = g->writenbuf;
	runtime_memmove(g->writebuf, v, n);
	g->writebuf += n;
	g->writenbuf -= n;
}

void
runtime_dump(byte *p, int32 n)
{
	int32 i;

	for(i=0; i<n; i++) {
		runtime_printpointer((byte*)(uintptr)(p[i]>>4));
		runtime_printpointer((byte*)(uintptr)(p[i]&0xf));
		if((i&15) == 15)
			runtime_prints("\n");
		else
			runtime_prints(" ");
	}
	if(n & 15)
		runtime_prints("\n");
}

void
runtime_prints(const char *s)
{
	gwrite(s, runtime_findnull((const byte*)s));
}

#if defined (__clang__) && (defined (__i386__) || defined (__x86_64__))
// LLVM's code generator does not currently support split stacks for vararg
// functions, so we disable the feature for this function under Clang. This
// appears to be OK as long as:
// - this function only calls non-inlined, internal-linkage (hence no dynamic
//   loader) functions compiled with split stacks (i.e. go_vprintf), which can
//   allocate more stack space as required;
// - this function itself does not occupy more than BACKOFF bytes of stack space
//   (see libgcc/config/i386/morestack.S).
// These conditions are currently known to be satisfied by Clang on x86-32 and
// x86-64. Note that signal handlers receive slightly less stack space than they
// would normally do if they happen to be called while this function is being
// run. If this turns out to be a problem we could consider increasing BACKOFF.

void
runtime_printf(const char *s, ...)
__attribute__((no_split_stack));

int32
runtime_snprintf(byte *buf, int32 n, const char *s, ...)
__attribute__((no_split_stack));

#endif

void
runtime_printf(const char *s, ...)
{
	va_list va;

	va_start(va, s);
	go_vprintf(s, va);
	va_end(va);
}

int32
runtime_snprintf(byte *buf, int32 n, const char *s, ...)
{
	G *g = runtime_g();
	va_list va;
	int32 m;

	g->writebuf = buf;
	g->writenbuf = n-1;
	va_start(va, s);
	go_vprintf(s, va);
	va_end(va);
	*g->writebuf = '\0';
	m = g->writebuf - buf;
	g->writenbuf = 0;
	g->writebuf = nil;
	return m;
}

// Very simple printf.  Only for debugging prints.
// Do not add to this without checking with Rob.
static void
go_vprintf(const char *s, va_list va)
{
	const char *p, *lp;

	//runtime_lock(&debuglock);

	lp = p = s;
	for(; *p; p++) {
		if(*p != '%')
			continue;
		if(p > lp)
			gwrite(lp, p-lp);
		p++;
		switch(*p) {
		case 'a':
			runtime_printslice(va_arg(va, Slice));
			break;
		case 'c':
			runtime_printbyte(va_arg(va, int32));
			break;
		case 'd':
			runtime_printint(va_arg(va, int32));
			break;
		case 'D':
			runtime_printint(va_arg(va, int64));
			break;
		case 'e':
			runtime_printeface(va_arg(va, Eface));
			break;
		case 'f':
			runtime_printfloat(va_arg(va, float64));
			break;
		case 'C':
			runtime_printcomplex(va_arg(va, complex double));
			break;
		case 'i':
			runtime_printiface(va_arg(va, Iface));
			break;
		case 'p':
			runtime_printpointer(va_arg(va, void*));
			break;
		case 's':
			runtime_prints(va_arg(va, char*));
			break;
		case 'S':
			runtime_printstring(va_arg(va, String));
			break;
		case 't':
			runtime_printbool(va_arg(va, int));
			break;
		case 'U':
			runtime_printuint(va_arg(va, uint64));
			break;
		case 'x':
			runtime_printhex(va_arg(va, uint32));
			break;
		case 'X':
			runtime_printhex(va_arg(va, uint64));
			break;
		}
		lp = p+1;
	}
	if(p > lp)
		gwrite(lp, p-lp);

	//runtime_unlock(&debuglock);
}

void
runtime_printpc(void *p __attribute__ ((unused)))
{
	runtime_prints("PC=");
	runtime_printhex((uint64)(uintptr)runtime_getcallerpc(p));
}

void
runtime_printbool(_Bool v)
{
	if(v) {
		gwrite("true", 4);
		return;
	}
	gwrite("false", 5);
}

void
runtime_printbyte(int8 c)
{
	gwrite(&c, 1);
}

void
runtime_printfloat(double v)
{
	byte buf[20];
	int32 e, s, i, n;
	float64 h;

	if(ISNAN(v)) {
		gwrite("NaN", 3);
		return;
	}
	if(isinf(v)) {
		if(signbit(v)) {
			gwrite("-Inf", 4);
		} else {
			gwrite("+Inf", 4);
		}
		return;
	}

	n = 7;	// digits printed
	e = 0;	// exp
	s = 0;	// sign
	if(v == 0) {
		if(isinf(1/v) && 1/v < 0)
			s = 1;
	} else {
		// sign
		if(v < 0) {
			v = -v;
			s = 1;
		}

		// normalize
		while(v >= 10) {
			e++;
			v /= 10;
		}
		while(v < 1) {
			e--;
			v *= 10;
		}

		// round
		h = 5;
		for(i=0; i<n; i++)
			h /= 10;

		v += h;
		if(v >= 10) {
			e++;
			v /= 10;
		}
	}

	// format +d.dddd+edd
	buf[0] = '+';
	if(s)
		buf[0] = '-';
	for(i=0; i<n; i++) {
		s = v;
		buf[i+2] = s+'0';
		v -= s;
		v *= 10.;
	}
	buf[1] = buf[2];
	buf[2] = '.';

	buf[n+2] = 'e';
	buf[n+3] = '+';
	if(e < 0) {
		e = -e;
		buf[n+3] = '-';
	}

	buf[n+4] = (e/100) + '0';
	buf[n+5] = (e/10)%10 + '0';
	buf[n+6] = (e%10) + '0';
	gwrite(buf, n+7);
}

void
runtime_printcomplex(complex double v)
{
	gwrite("(", 1);
	runtime_printfloat(creal(v));
	runtime_printfloat(cimag(v));
	gwrite("i)", 2);
}

void
runtime_printuint(uint64 v)
{
	byte buf[100];
	int32 i;

	for(i=nelem(buf)-1; i>0; i--) {
		buf[i] = v%10 + '0';
		if(v < 10)
			break;
		v = v/10;
	}
	gwrite(buf+i, nelem(buf)-i);
}

void
runtime_printint(int64 v)
{
	if(v < 0) {
		gwrite("-", 1);
		v = -v;
	}
	runtime_printuint(v);
}

void
runtime_printhex(uint64 v)
{
	static const char *dig = "0123456789abcdef";
	byte buf[100];
	int32 i;

	i=nelem(buf);
	for(; v>0; v/=16)
		buf[--i] = dig[v%16];
	if(i == nelem(buf))
		buf[--i] = '0';
	buf[--i] = 'x';
	buf[--i] = '0';
	gwrite(buf+i, nelem(buf)-i);
}

void
runtime_printpointer(void *p)
{
	runtime_printhex((uintptr)p);
}

void
runtime_printstring(String v)
{
	// if(v.len > runtime_maxstring) {
	//	gwrite("[string too long]", 17);
	//	return;
	// }
	if(v.len > 0)
		gwrite(v.str, v.len);
}

void
__go_print_space(void)
{
	gwrite(" ", 1);
}

void
__go_print_nl(void)
{
	gwrite("\n", 1);
}
