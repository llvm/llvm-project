/* go-caller.c -- runtime.Caller and runtime.FuncForPC for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

/* Implement runtime.Caller.  */

#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "backtrace.h"

#include "runtime.h"

/* Get the function name, file name, and line number for a PC value.
   We use the backtrace library to get this.  */

/* Data structure to gather file/line information.  */

struct caller
{
  String fn;
  String file;
  intgo line;
};

/* Collect file/line information for a PC value.  If this is called
   more than once, due to inlined functions, we use the last call, as
   that is usually the most useful one.  */

static int
callback (void *data, uintptr_t pc __attribute__ ((unused)),
	  const char *filename, int lineno, const char *function)
{
  struct caller *c = (struct caller *) data;

  /* The libbacktrace library says that these strings might disappear,
     but with the current implementation they won't.  We can't easily
     allocate memory here, so for now assume that we can save a
     pointer to the strings.  */
  c->fn = runtime_gostringnocopy ((const byte *) function);
  c->file = runtime_gostringnocopy ((const byte *) filename);
  c->line = lineno;

  return 0;
}

/* The error callback for backtrace_pcinfo and backtrace_syminfo.  */

static void
error_callback (void *data __attribute__ ((unused)),
		const char *msg, int errnum)
{
  if (errnum == -1)
    return;
  if (errnum > 0)
    runtime_printf ("%s errno %d\n", msg, errnum);
  runtime_throw (msg);
}

/* The backtrace library state.  */

static void *back_state;

/* A lock to control creating back_state.  */

static Lock back_state_lock;

/* Fetch back_state, creating it if necessary.  */

struct backtrace_state *
__go_get_backtrace_state ()
{
  runtime_lock (&back_state_lock);
  if (back_state == NULL)
    {
      const char *filename;
      struct stat s;

      filename = (const char *) runtime_progname ();

      /* If there is no '/' in FILENAME, it was found on PATH, and
	 might not be the same as the file with the same name in the
	 current directory.  */
      if (__builtin_strchr (filename, '/') == NULL)
	filename = NULL;

      /* If the file is small, then it's not the real executable.
	 This is specifically to deal with Docker, which uses a bogus
	 argv[0] (http://gcc.gnu.org/PR61895).  It would be nice to
	 have a better check for whether this file is the real
	 executable.  */
      if (stat (filename, &s) < 0 || s.st_size < 1024)
	filename = NULL;

      back_state = backtrace_create_state (filename, 1, error_callback, NULL);
    }
  runtime_unlock (&back_state_lock);
  return back_state;
}

/* Return function/file/line information for PC.  */

_Bool
__go_file_line (uintptr pc, String *fn, String *file, intgo *line)
{
  struct caller c;

  runtime_memclr (&c, sizeof c);
  backtrace_pcinfo (__go_get_backtrace_state (), pc, callback,
		    error_callback, &c);
  *fn = c.fn;
  *file = c.file;
  *line = c.line;
  return c.file.len > 0;
}

/* Collect symbol information.  */

static void
syminfo_callback (void *data, uintptr_t pc __attribute__ ((unused)),
		  const char *symname __attribute__ ((unused)),
		  uintptr_t address, uintptr_t size __attribute__ ((unused)))
{
  uintptr_t *pval = (uintptr_t *) data;

  *pval = address;
}

/* Set *VAL to the value of the symbol for PC.  */

static _Bool
__go_symbol_value (uintptr_t pc, uintptr_t *val)
{
  *val = 0;
  backtrace_syminfo (__go_get_backtrace_state (), pc, syminfo_callback,
		     error_callback, val);
  return *val != 0;
}

/* The values returned by runtime.Caller.  */

struct caller_ret
{
  uintptr_t pc;
  String file;
  intgo line;
  _Bool ok;
};

struct caller_ret Caller (int n) __asm__ (GOSYM_PREFIX "runtime.Caller");

Func *FuncForPC (uintptr_t) __asm__ (GOSYM_PREFIX "runtime.FuncForPC");

/* Implement runtime.Caller.  */

struct caller_ret
Caller (int skip)
{
  struct caller_ret ret;
  Location loc;
  int32 n;

  runtime_memclr (&ret, sizeof ret);
  n = runtime_callers (skip + 1, &loc, 1, false);
  if (n < 1 || loc.pc == 0)
    return ret;
  ret.pc = loc.pc;
  ret.file = loc.filename;
  ret.line = loc.lineno;
  ret.ok = 1;
  return ret;
}

/* Implement runtime.FuncForPC.  */

Func *
FuncForPC (uintptr_t pc)
{
  Func *ret;
  String fn;
  String file;
  intgo line;
  uintptr_t val;

  if (!__go_file_line (pc, &fn, &file, &line))
    return NULL;

  ret = (Func *) runtime_malloc (sizeof (*ret));
  ret->name = fn;

  if (__go_symbol_value (pc, &val))
    ret->entry = val;
  else
    ret->entry = 0;

  return ret;
}

/* Look up the file and line information for a PC within a
   function.  */

struct funcline_go_return
{
  String retfile;
  intgo retline;
};

struct funcline_go_return
runtime_funcline_go (Func *f, uintptr targetpc)
  __asm__ (GOSYM_PREFIX "runtime.funcline_go");

struct funcline_go_return
runtime_funcline_go (Func *f __attribute__((unused)), uintptr targetpc)
{
  struct funcline_go_return ret;
  String fn;

  if (!__go_file_line (targetpc, &fn, &ret.retfile,  &ret.retline))
    runtime_memclr (&ret, sizeof ret);
  return ret;
}

/* Return the name of a function.  */
String runtime_funcname_go (Func *f)
  __asm__ (GOSYM_PREFIX "runtime.funcname_go");

String
runtime_funcname_go (Func *f)
{
  if (f == NULL)
    return runtime_gostringnocopy ((const byte *) "");
  return f->name;
}

/* Return the entry point of a function.  */
uintptr runtime_funcentry_go(Func *f)
  __asm__ (GOSYM_PREFIX "runtime.funcentry_go");

uintptr
runtime_funcentry_go (Func *f)
{
  return f->entry;
}
