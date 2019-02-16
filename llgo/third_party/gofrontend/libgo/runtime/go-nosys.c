/* go-nosys.c -- functions missing from system.

   Copyright 2012 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

/* This file exists to provide definitions for functions that are
   missing from libc, according to the configure script.  This permits
   the Go syscall package to not worry about whether the functions
   exist or not.  */

#include "config.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifndef HAVE_OFF64_T
typedef signed int off64_t __attribute__ ((mode (DI)));
#endif

#ifndef HAVE_LOFF_T
typedef off64_t loff_t;
#endif

#ifndef HAVE_ACCEPT4
struct sockaddr;
int
accept4 (int sockfd __attribute__ ((unused)),
	 struct sockaddr *addr __attribute__ ((unused)),
	 socklen_t *addrlen __attribute__ ((unused)),
	 int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_DUP3
int
dup3 (int oldfd __attribute__ ((unused)),
      int newfd __attribute__ ((unused)),
      int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_EPOLL_CREATE1
int
epoll_create1 (int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_FACCESSAT
int
faccessat (int fd __attribute__ ((unused)),
	   const char *pathname __attribute__ ((unused)),
	   int mode __attribute__ ((unused)),
	   int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_FALLOCATE
int
fallocate (int fd __attribute__ ((unused)),
	   int mode __attribute__ ((unused)),
	   off_t offset __attribute__ ((unused)),
	   off_t len __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_FCHMODAT
int
fchmodat (int dirfd __attribute__ ((unused)),
	  const char *pathname __attribute__ ((unused)),
	  mode_t mode __attribute__ ((unused)),
	  int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_FCHOWNAT
int
fchownat (int dirfd __attribute__ ((unused)),
	  const char *pathname __attribute__ ((unused)),
	  uid_t owner __attribute__ ((unused)),
	  gid_t group __attribute__ ((unused)),
	  int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_FUTIMESAT
int
futimesat (int dirfd __attribute__ ((unused)),
	   const char *pathname __attribute__ ((unused)),
	   const struct timeval times[2] __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_GETXATTR
ssize_t
getxattr (const char *path __attribute__ ((unused)),
	  const char *name __attribute__ ((unused)),
	  void *value __attribute__ ((unused)),
	  size_t size __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_INOTIFY_ADD_WATCH
int
inotify_add_watch (int fd __attribute__ ((unused)),
		   const char* pathname __attribute__ ((unused)),
		   uint32_t mask __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_INOTIFY_INIT
int
inotify_init (void)
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_INOTIFY_INIT1
int
inotify_init1 (int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_INOTIFY_RM_WATCH
int
inotify_rm_watch (int fd __attribute__ ((unused)),
		  uint32_t wd __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_LISTXATTR
ssize_t
listxattr (const char *path __attribute__ ((unused)),
	   char *list __attribute__ ((unused)),
	   size_t size __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_MKDIRAT
int
mkdirat (int dirfd __attribute__ ((unused)),
	 const char *pathname __attribute__ ((unused)),
	 mode_t mode __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_MKNODAT
int
mknodat (int dirfd __attribute__ ((unused)),
	 const char *pathname __attribute__ ((unused)),
	 mode_t mode __attribute__ ((unused)),
	 dev_t dev __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_OPENAT
int
openat (int dirfd __attribute__ ((unused)),
	const char *pathname __attribute__ ((unused)),
	int oflag __attribute__ ((unused)),
	...)
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_PIPE2
int
pipe2 (int pipefd[2] __attribute__ ((unused)),
       int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_REMOVEXATTR
int
removexattr (const char *path __attribute__ ((unused)),
	     const char *name __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_RENAMEAT
int
renameat (int olddirfd __attribute__ ((unused)),
	  const char *oldpath __attribute__ ((unused)),
	  int newdirfd __attribute__ ((unused)),
	  const char *newpath __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_SETXATTR
int
setxattr (const char *path __attribute__ ((unused)),
	  const char *name __attribute__ ((unused)),
	  const void *value __attribute__ ((unused)),
	  size_t size __attribute__ ((unused)),
	  int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_SPLICE
int
splice (int fd __attribute__ ((unused)),
	loff_t *off_in __attribute__ ((unused)),
	int fd_out __attribute__ ((unused)),
	loff_t *off_out __attribute__ ((unused)),
	size_t len __attribute__ ((unused)),
	unsigned int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_SYNC_FILE_RANGE
int
sync_file_range (int fd __attribute__ ((unused)),
		 off64_t offset __attribute__ ((unused)),
		 off64_t nbytes __attribute__ ((unused)),
		 unsigned int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_TEE
int
tee (int fd_in __attribute__ ((unused)),
     int fd_out __attribute__ ((unused)),
     size_t len __attribute__ ((unused)),
     unsigned int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_UNLINKAT
int
unlinkat (int dirfd __attribute__ ((unused)),
	  const char *pathname __attribute__ ((unused)),
	  int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_UNSHARE
int
unshare (int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

#ifndef HAVE_UTIMENSAT
struct timespec;
int
utimensat(int dirfd __attribute__ ((unused)),
	  const char *pathname __attribute__ ((unused)),
	  const struct timespec times[2] __attribute__ ((unused)),
	  int flags __attribute__ ((unused)))
{
  errno = ENOSYS;
  return -1;
}
#endif

/* Long double math functions.  These are needed on old i386 systems
   that don't have them in libm.  The compiler translates calls to
   these functions on float64 to call an 80-bit floating point
   function instead, because when optimizing that function can be
   executed as an x87 instructure.  However, when not optimizing, this
   translates into a call to the math function.  So on systems that
   don't provide these functions, we provide a version that just calls
   the float64 version.  */

#ifndef HAVE_COSL
long double
cosl (long double a)
{
  return (long double) cos ((double) a);
}
#endif

#ifndef HAVE_EXPL
long double
expl (long double a)
{
  return (long double) exp ((double) a);
}
#endif

#ifndef HAVE_LOGL
long double
logl (long double a)
{
  return (long double) log ((double) a);
}
#endif

#ifndef HAVE_SINL
long double
sinl (long double a)
{
  return (long double) sin ((double) a);
}
#endif

#ifndef HAVE_TANL
long double
tanl (long double a)
{
  return (long double) tan ((double) a);
}
#endif

#ifndef HAVE_ACOSL
long double
acosl (long double a)
{
  return (long double) acos ((double) a);
}
#endif

#ifndef HAVE_ASINL
long double
asinl (long double a)
{
  return (long double) asin ((double) a);
}
#endif

#ifndef HAVE_ATANL
long double
atanl (long double a)
{
  return (long double) atan ((double) a);
}
#endif

#ifndef HAVE_ATAN2L
long double
atan2l (long double a, long double b)
{
  return (long double) atan2 ((double) a, (double) b);
}
#endif

#ifndef HAVE_EXPM1L
long double
expm1l (long double a)
{
  return (long double) expm1 ((double) a);
}
#endif

#ifndef HAVE_LDEXPL
long double
ldexpl (long double a, int exp)
{
  return (long double) ldexp ((double) a, exp);
}
#endif

#ifndef HAVE_LOG10L
long double
log10l (long double a)
{
  return (long double) log10 ((double) a);
}
#endif

#ifndef HAVE_LOG1PL
long double
log1pl (long double a)
{
  return (long double) log1p ((double) a);
}
#endif
