/* go-libmain.c -- the startup function for a Go library.

   Copyright 2015 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "runtime.h"
#include "go-alloc.h"
#include "array.h"
#include "arch.h"
#include "malloc.h"

/* This is used when building a standalone Go library using the Go
   command's -buildmode=c-archive or -buildmode=c-shared option.  It
   starts up the Go code as a global constructor but does not take any
   other action.  The main program is written in some other language
   and calls exported Go functions as needed.  */

static void die (const char *, int);
static void initfn (int, char **, char **);
static void *gostart (void *);

/* Used to pass arguments to the thread that runs the Go startup.  */

struct args {
  int argc;
  char **argv;
};

/* We use .init_array so that we can get the command line arguments.
   This obviously assumes .init_array support; different systems may
   require other approaches.  */

typedef void (*initarrayfn) (int, char **, char **);

static initarrayfn initarray[1]
__attribute__ ((section (".init_array"), used)) =
  { initfn };

/* This function is called at program startup time.  It starts a new
   thread to do the actual Go startup, so that program startup is not
   paused waiting for the Go initialization functions.  Exported cgo
   functions will wait for initialization to complete if
   necessary.  */

static void
initfn (int argc, char **argv, char** env __attribute__ ((unused)))
{
  int err;
  pthread_attr_t attr;
  struct args *a;
  pthread_t tid;

  a = (struct args *) malloc (sizeof *a);
  if (a == NULL)
    die ("malloc", errno);
  a->argc = argc;
  a->argv = argv;

  err = pthread_attr_init (&attr);
  if (err != 0)
    die ("pthread_attr_init", err);
  err = pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
  if (err != 0)
    die ("pthread_attr_setdetachstate", err);

  err = pthread_create (&tid, &attr, gostart, (void *) a);
  if (err != 0)
    die ("pthread_create", err);

  err = pthread_attr_destroy (&attr);
  if (err != 0)
    die ("pthread_attr_destroy", err);
}

/* Start up the Go runtime.  */

static void *
gostart (void *arg)
{
  struct args *a = (struct args *) arg;

  runtime_isarchive = true;

  if (runtime_isstarted)
    return NULL;
  runtime_isstarted = true;

  runtime_check ();
  runtime_args (a->argc, (byte **) a->argv);
  runtime_osinit ();
  runtime_schedinit ();
  __go_go (runtime_main, NULL);
  runtime_mstart (runtime_m ());
  abort ();
}

/* If something goes wrong during program startup, crash.  There is no
   way to report failure and nobody to whom to report it.  */

static void
die (const char *fn, int err)
{
  fprintf (stderr, "%s: %d\n", fn, err);
  exit (EXIT_FAILURE);
}
