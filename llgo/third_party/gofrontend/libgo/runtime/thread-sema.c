// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "config.h"
#include "runtime.h"

#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <semaphore.h>

/* If we don't have sem_timedwait, use pthread_cond_timedwait instead.
   We don't always use condition variables because on some systems
   pthread_mutex_lock and pthread_mutex_unlock must be called by the
   same thread.  That is never true of semaphores.  */

struct go_sem
{
  sem_t sem;

#ifndef HAVE_SEM_TIMEDWAIT
  int timedwait;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
#endif
};

/* Create a semaphore.  */

uintptr
runtime_semacreate(void)
{
  struct go_sem *p;

  /* Call malloc rather than runtime_malloc.  This will allocate space
     on the C heap.  We can't call runtime_malloc here because it
     could cause a deadlock.  */
  p = malloc (sizeof (struct go_sem));
  if (sem_init (&p->sem, 0, 0) != 0)
    runtime_throw ("sem_init");

#ifndef HAVE_SEM_TIMEDWAIT
  if (pthread_mutex_init (&p->mutex, NULL) != 0)
    runtime_throw ("pthread_mutex_init");
  if (pthread_cond_init (&p->cond, NULL) != 0)
    runtime_throw ("pthread_cond_init");
#endif

  return (uintptr) p;
}

/* Acquire m->waitsema.  */

int32
runtime_semasleep (int64 ns)
{
  M *m;
  struct go_sem *sem;
  int r;

  m = runtime_m ();
  sem = (struct go_sem *) m->waitsema;
  if (ns >= 0)
    {
      int64 abs;
      struct timespec ts;
      int err;

      abs = ns + runtime_nanotime ();
      ts.tv_sec = abs / 1000000000LL;
      ts.tv_nsec = abs % 1000000000LL;

      err = 0;

#ifdef HAVE_SEM_TIMEDWAIT
      r = sem_timedwait (&sem->sem, &ts);
      if (r != 0)
	err = errno;
#else
      if (pthread_mutex_lock (&sem->mutex) != 0)
	runtime_throw ("pthread_mutex_lock");

      while ((r = sem_trywait (&sem->sem)) != 0)
	{
	  r = pthread_cond_timedwait (&sem->cond, &sem->mutex, &ts);
	  if (r != 0)
	    {
	      err = r;
	      break;
	    }
	}

      if (pthread_mutex_unlock (&sem->mutex) != 0)
	runtime_throw ("pthread_mutex_unlock");
#endif

      if (err != 0)
	{
	  if (err == ETIMEDOUT || err == EAGAIN || err == EINTR)
	    return -1;
	  runtime_throw ("sema_timedwait");
	}
      return 0;
    }

  while (sem_wait (&sem->sem) != 0)
    {
      if (errno == EINTR)
	continue;
      runtime_throw ("sem_wait");
    }

  return 0;
}

/* Wake up mp->waitsema.  */

void
runtime_semawakeup (M *mp)
{
  struct go_sem *sem;

  sem = (struct go_sem *) mp->waitsema;
  if (sem_post (&sem->sem) != 0)
    runtime_throw ("sem_post");

#ifndef HAVE_SEM_TIMEDWAIT
  if (pthread_mutex_lock (&sem->mutex) != 0)
    runtime_throw ("pthread_mutex_lock");
  if (pthread_cond_broadcast (&sem->cond) != 0)
    runtime_throw ("pthread_cond_broadcast");
  if (pthread_mutex_unlock (&sem->mutex) != 0)
    runtime_throw ("pthread_mutex_unlock");
#endif
}

void
runtime_osinit (void)
{
  runtime_ncpu = getproccount();
}

void
runtime_goenvs (void)
{
  runtime_goenvs_unix ();
}
