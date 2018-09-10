/* go-varargs.c -- functions for calling C varargs functions.

   Copyright 2013 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"

#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/ioctl.h>

/* The syscall package calls C functions.  The Go compiler can not
   represent a C varargs functions.  On some systems it's important
   that the declaration of a function match the call.  This function
   holds non-varargs C functions that the Go code can call.  */

int
__go_open (char *path, int mode, mode_t perm)
{
  return open (path, mode, perm);
}

int
__go_fcntl (int fd, int cmd, int arg)
{
  return fcntl (fd, cmd, arg);
}

int
__go_fcntl_flock (int fd, int cmd, struct flock *arg)
{
  return fcntl (fd, cmd, arg);
}

// This is for the net package.  We use uintptr_t to make sure that
// the types match, since the Go and C "int" types are not the same.
struct go_fcntl_ret {
  uintptr_t r;
  uintptr_t err;
};

struct go_fcntl_ret
__go_fcntl_uintptr (uintptr_t fd, uintptr_t cmd, uintptr_t arg)
{
  int r;
  struct go_fcntl_ret ret;

  r = fcntl ((int) fd, (int) cmd, (int) arg);
  ret.r = (uintptr_t) r;
  if (r < 0)
    ret.err = (uintptr_t) errno;
  else
    ret.err = 0;
  return ret;
}

int
__go_ioctl (int d, int request, int arg)
{
  return ioctl (d, request, arg);
}

int
__go_ioctl_ptr (int d, int request, void *arg)
{
  return ioctl (d, request, arg);
}

#ifdef HAVE_OPEN64

int
__go_open64 (char *path, int mode, mode_t perm)
{
  return open64 (path, mode, perm);
}

#endif

#ifdef HAVE_OPENAT

int
__go_openat (int fd, char *path, int flags, mode_t mode)
{
  return openat (fd, path, flags, mode);
}

#endif
