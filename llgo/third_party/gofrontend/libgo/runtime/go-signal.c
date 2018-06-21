/* go-signal.c -- signal handling for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

#include "runtime.h"
#include "go-assert.h"
#include "go-panic.h"
#include "signal_unix.h"

#ifndef SA_RESTART
  #define SA_RESTART 0
#endif

#ifdef USING_SPLIT_STACK

extern void __splitstack_getcontext(void *context[10]);

extern void __splitstack_setcontext(void *context[10]);

#endif

#define N SigNotify
#define K SigKill
#define T SigThrow
#define P SigPanic
#define D SigDefault

/* Signal actions.  This collects the sigtab tables for several
   different targets from the master library.  SIGKILL, SIGCONT, and
   SIGSTOP are not listed, as we don't want to set signal handlers for
   them.  */

SigTab runtime_sigtab[] = {
#ifdef SIGHUP
  { SIGHUP,	N + K },
#endif
#ifdef SIGINT
  { SIGINT, 	N + K },
#endif
#ifdef SIGQUIT
  { SIGQUIT, 	N + T },
#endif
#ifdef SIGILL
  { SIGILL, 	T },
#endif
#ifdef SIGTRAP
  { SIGTRAP, 	T },
#endif
#ifdef SIGABRT
  { SIGABRT, 	N + T },
#endif
#ifdef SIGBUS
  { SIGBUS, 	P },
#endif
#ifdef SIGFPE
  { SIGFPE, 	P },
#endif
#ifdef SIGUSR1
  { SIGUSR1, 	N },
#endif
#ifdef SIGSEGV
  { SIGSEGV, 	P },
#endif
#ifdef SIGUSR2
  { SIGUSR2, 	N },
#endif
#ifdef SIGPIPE
  { SIGPIPE, 	N },
#endif
#ifdef SIGALRM
  { SIGALRM, 	N },
#endif
#ifdef SIGTERM
  { SIGTERM, 	N + K },
#endif
#ifdef SIGSTKFLT
  { SIGSTKFLT, 	T },
#endif
#ifdef SIGCHLD
  { SIGCHLD, 	N },
#endif
#ifdef SIGTSTP
  { SIGTSTP, 	N + D },
#endif
#ifdef SIGTTIN
  { SIGTTIN, 	N + D },
#endif
#ifdef SIGTTOU
  { SIGTTOU, 	N + D },
#endif
#ifdef SIGURG
  { SIGURG, 	N },
#endif
#ifdef SIGXCPU
  { SIGXCPU, 	N },
#endif
#ifdef SIGXFSZ
  { SIGXFSZ, 	N },
#endif
#ifdef SIGVTALRM
  { SIGVTALRM, 	N },
#endif
#ifdef SIGPROF
  { SIGPROF, 	N },
#endif
#ifdef SIGWINCH
  { SIGWINCH, 	N },
#endif
#ifdef SIGIO
  { SIGIO, 	N },
#endif
#ifdef SIGPWR
  { SIGPWR, 	N },
#endif
#ifdef SIGSYS
  { SIGSYS, 	N },
#endif
#ifdef SIGEMT
  { SIGEMT,	T },
#endif
#ifdef SIGINFO
  { SIGINFO,	N },
#endif
#ifdef SIGTHR
  { SIGTHR,	N },
#endif
  { -1,		0 }
};
#undef N
#undef K
#undef T
#undef P
#undef D

/* Handle a signal, for cases where we don't panic.  We can split the
   stack here.  */

void
runtime_sighandler (int sig, Siginfo *info,
		    void *context __attribute__ ((unused)), G *gp)
{
  M *m;
  int i;

  m = runtime_m ();

#ifdef SIGPROF
  if (sig == SIGPROF)
    {
      if (m != NULL && gp != m->g0 && gp != m->gsignal)
	runtime_sigprof ();
      return;
    }
#endif

  if (m == NULL)
    {
      runtime_badsignal (sig);
      return;
    }

  for (i = 0; runtime_sigtab[i].sig != -1; ++i)
    {
      SigTab *t;
      bool notify, crash;

      t = &runtime_sigtab[i];

      if (t->sig != sig)
	continue;

      notify = false;
#ifdef SA_SIGINFO
      notify = info != NULL && info->si_code == SI_USER;
#endif
      if (notify || (t->flags & SigNotify) != 0)
	{
	  if (__go_sigsend (sig))
	    return;
	}
      if ((t->flags & SigKill) != 0)
	runtime_exit (2);
      if ((t->flags & SigThrow) == 0)
	return;

      runtime_startpanic ();

      {
	const char *name = NULL;

#ifdef HAVE_STRSIGNAL
	name = strsignal (sig);
#endif

	if (name == NULL)
	  runtime_printf ("Signal %d\n", sig);
	else
	  runtime_printf ("%s\n", name);
      }

      if (m->lockedg != NULL && m->ncgo > 0 && gp == m->g0)
	{
	  runtime_printf("signal arrived during cgo execution\n");
	  gp = m->lockedg;
	}

      runtime_printf ("\n");

      if (runtime_gotraceback (&crash))
	{
	  G *g;

	  g = runtime_g ();
	  runtime_traceback ();
	  runtime_tracebackothers (g);

	  /* The gc library calls runtime_dumpregs here, and provides
	     a function that prints the registers saved in context in
	     a readable form.  */
	}

      if (crash)
	runtime_crash ();

      runtime_exit (2);
    }

  __builtin_unreachable ();
}

/* The start of handling a signal which panics.  */

static void
sig_panic_leadin (G *gp)
{
  int i;
  sigset_t clear;

  if (!runtime_canpanic (gp))
    runtime_throw ("unexpected signal during runtime execution");

  /* The signal handler blocked signals; unblock them.  */
  i = sigfillset (&clear);
  __go_assert (i == 0);
  i = pthread_sigmask (SIG_UNBLOCK, &clear, NULL);
  __go_assert (i == 0);
}

#ifdef SA_SIGINFO

/* Signal dispatch for signals which panic, on systems which support
   SA_SIGINFO.  This is called on the thread stack, and as such it is
   permitted to split the stack.  */

static void
sig_panic_info_handler (int sig, Siginfo *info, void *context)
{
  G *g;

  g = runtime_g ();
  if (g == NULL || info->si_code == SI_USER)
    {
      runtime_sighandler (sig, info, context, g);
      return;
    }

  g->sig = sig;
  g->sigcode0 = info->si_code;
  g->sigcode1 = (uintptr_t) info->si_addr;

  /* It would be nice to set g->sigpc here as the gc library does, but
     I don't know how to get it portably.  */

  sig_panic_leadin (g);

  switch (sig)
    {
#ifdef SIGBUS
    case SIGBUS:
      if ((info->si_code == BUS_ADRERR && (uintptr_t) info->si_addr < 0x1000)
	  || g->paniconfault)
	runtime_panicstring ("invalid memory address or "
			     "nil pointer dereference");
      runtime_printf ("unexpected fault address %p\n", info->si_addr);
      runtime_throw ("fault");
#endif

#ifdef SIGSEGV
    case SIGSEGV:
      if (((info->si_code == 0
	    || info->si_code == SEGV_MAPERR
	    || info->si_code == SEGV_ACCERR)
	   && (uintptr_t) info->si_addr < 0x1000)
	  || g->paniconfault)
	runtime_panicstring ("invalid memory address or "
			     "nil pointer dereference");
      runtime_printf ("unexpected fault address %p\n", info->si_addr);
      runtime_throw ("fault");
#endif

#ifdef SIGFPE
    case SIGFPE:
      switch (info->si_code)
	{
	case FPE_INTDIV:
	  runtime_panicstring ("integer divide by zero");
	case FPE_INTOVF:
	  runtime_panicstring ("integer overflow");
	}
      runtime_panicstring ("floating point error");
#endif
    }

  /* All signals with SigPanic should be in cases above, and this
     handler should only be invoked for those signals.  */
  __builtin_unreachable ();
}

#else /* !defined (SA_SIGINFO) */

static void
sig_panic_handler (int sig)
{
  G *g;

  g = runtime_g ();
  if (g == NULL)
    {
      runtime_sighandler (sig, NULL, NULL, g);
      return;
    }

  g->sig = sig;
  g->sigcode0 = 0;
  g->sigcode1 = 0;

  sig_panic_leadin (g);

  switch (sig)
    {
#ifdef SIGBUS
    case SIGBUS:
      runtime_panicstring ("invalid memory address or "
			   "nil pointer dereference");
#endif

#ifdef SIGSEGV
    case SIGSEGV:
      runtime_panicstring ("invalid memory address or "
			   "nil pointer dereference");
#endif

#ifdef SIGFPE
    case SIGFPE:
      runtime_panicstring ("integer divide by zero or floating point error");
#endif
    }

  /* All signals with SigPanic should be in cases above, and this
     handler should only be invoked for those signals.  */
  __builtin_unreachable ();
}

#endif /* !defined (SA_SIGINFO) */

/* A signal handler used for signals which are not going to panic.
   This is called on the alternate signal stack so it may not split
   the stack.  */

static void
sig_tramp_info (int, Siginfo *, void *) __attribute__ ((no_split_stack));

static void
sig_tramp_info (int sig, Siginfo *info, void *context)
{
  G *gp;
  M *mp;
#ifdef USING_SPLIT_STACK
  void *stack_context[10];
#endif

  /* We are now running on the stack registered via sigaltstack.
     (Actually there is a small span of time between runtime_siginit
     and sigaltstack when the program starts.)  */
  gp = runtime_g ();
  mp = runtime_m ();

  if (gp != NULL)
    {
#ifdef USING_SPLIT_STACK
      __splitstack_getcontext (&stack_context[0]);
#endif
    }

  if (gp != NULL && mp->gsignal != NULL)
    {
      /* We are running on the signal stack.  Set the split stack
	 context so that the stack guards are checked correctly.  */
#ifdef USING_SPLIT_STACK
      __splitstack_setcontext (&mp->gsignal->stack_context[0]);
#endif
    }

  runtime_sighandler (sig, info, context, gp);

  /* We are going to return back to the signal trampoline and then to
     whatever we were doing before we got the signal.  Restore the
     split stack context so that stack guards are checked
     correctly.  */

  if (gp != NULL)
    {
#ifdef USING_SPLIT_STACK
      __splitstack_setcontext (&stack_context[0]);
#endif
    }
}

#ifndef SA_SIGINFO

static void sig_tramp (int sig) __attribute__ ((no_split_stack));

static void
sig_tramp (int sig)
{
  sig_tramp_info (sig, NULL, NULL);
}

#endif

void
runtime_setsig (int32 i, GoSighandler *fn, bool restart)
{
  struct sigaction sa;
  int r;
  SigTab *t;

  memset (&sa, 0, sizeof sa);

  r = sigfillset (&sa.sa_mask);
  __go_assert (r == 0);

  t = &runtime_sigtab[i];

  if ((t->flags & SigPanic) == 0)
    {
#ifdef SA_SIGINFO
      sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
      if (fn == runtime_sighandler)
	fn = (void *) sig_tramp_info;
      sa.sa_sigaction = (void *) fn;
#else
      sa.sa_flags = SA_ONSTACK;
      if (fn == runtime_sighandler)
	fn = (void *) sig_tramp;
      sa.sa_handler = (void *) fn;
#endif
    }
  else
    {
#ifdef SA_SIGINFO
      sa.sa_flags = SA_SIGINFO;
      if (fn == runtime_sighandler)
	fn = (void *) sig_panic_info_handler;
      sa.sa_sigaction = (void *) fn;
#else
      sa.sa_flags = 0;
      if (fn == runtime_sighandler)
	fn = (void *) sig_panic_handler;
      sa.sa_handler = (void *) fn;
#endif
    }

  if (restart)
    sa.sa_flags |= SA_RESTART;

  if (sigaction (t->sig, &sa, NULL) != 0)
    __go_assert (0);
}

GoSighandler*
runtime_getsig (int32 i)
{
  struct sigaction sa;
  int r;
  SigTab *t;

  memset (&sa, 0, sizeof sa);

  r = sigemptyset (&sa.sa_mask);
  __go_assert (r == 0);

  t = &runtime_sigtab[i];

  if (sigaction (t->sig, NULL, &sa) != 0)
    runtime_throw ("sigaction read failure");

  if ((void *) sa.sa_handler == sig_tramp_info)
    return runtime_sighandler;
#ifdef SA_SIGINFO
  if ((void *) sa.sa_handler == sig_panic_info_handler)
    return runtime_sighandler;
#else
  if ((void *) sa.sa_handler == sig_tramp
      || (void *) sa.sa_handler == sig_panic_handler)
    return runtime_sighandler;
#endif

  return (void *) sa.sa_handler;
}

/* Used by the os package to raise SIGPIPE.  */

void os_sigpipe (void) __asm__ (GOSYM_PREFIX "os.sigpipe");

void
os_sigpipe (void)
{
  struct sigaction sa;
  int i;

  memset (&sa, 0, sizeof sa);

  sa.sa_handler = SIG_DFL;

  i = sigemptyset (&sa.sa_mask);
  __go_assert (i == 0);

  if (sigaction (SIGPIPE, &sa, NULL) != 0)
    abort ();

  raise (SIGPIPE);
}

void
runtime_setprof(bool on)
{
	USED(on);
}
