// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <errno.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "runtime.h"
#include "go-assert.h"

/* For targets which don't have the required sync support.  Really
   these should be provided by gcc itself.  FIXME.  */

#if !defined (HAVE_SYNC_BOOL_COMPARE_AND_SWAP_4) || !defined (HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8) || !defined (HAVE_SYNC_FETCH_AND_ADD_4) || !defined (HAVE_SYNC_ADD_AND_FETCH_8)

static pthread_mutex_t sync_lock = PTHREAD_MUTEX_INITIALIZER;

#endif

#ifndef HAVE_SYNC_BOOL_COMPARE_AND_SWAP_4

_Bool
__sync_bool_compare_and_swap_4 (uint32*, uint32, uint32)
  __attribute__ ((visibility ("hidden")));

_Bool
__sync_bool_compare_and_swap_4 (uint32* ptr, uint32 old, uint32 new)
{
  int i;
  _Bool ret;

  i = pthread_mutex_lock (&sync_lock);
  __go_assert (i == 0);

  if (*ptr != old)
    ret = 0;
  else
    {
      *ptr = new;
      ret = 1;
    }

  i = pthread_mutex_unlock (&sync_lock);
  __go_assert (i == 0);

  return ret;
}

#endif

#ifndef HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8

_Bool
__sync_bool_compare_and_swap_8 (uint64*, uint64, uint64)
  __attribute__ ((visibility ("hidden")));

_Bool
__sync_bool_compare_and_swap_8 (uint64* ptr, uint64 old, uint64 new)
{
  int i;
  _Bool ret;

  i = pthread_mutex_lock (&sync_lock);
  __go_assert (i == 0);

  if (*ptr != old)
    ret = 0;
  else
    {
      *ptr = new;
      ret = 1;
    }

  i = pthread_mutex_unlock (&sync_lock);
  __go_assert (i == 0);

  return ret;
}

#endif

#ifndef HAVE_SYNC_FETCH_AND_ADD_4

uint32
__sync_fetch_and_add_4 (uint32*, uint32)
  __attribute__ ((visibility ("hidden")));

uint32
__sync_fetch_and_add_4 (uint32* ptr, uint32 add)
{
  int i;
  uint32 ret;

  i = pthread_mutex_lock (&sync_lock);
  __go_assert (i == 0);

  ret = *ptr;
  *ptr += add;

  i = pthread_mutex_unlock (&sync_lock);
  __go_assert (i == 0);

  return ret;
}

#endif

#ifndef HAVE_SYNC_ADD_AND_FETCH_8

uint64
__sync_add_and_fetch_8 (uint64*, uint64)
  __attribute__ ((visibility ("hidden")));

uint64
__sync_add_and_fetch_8 (uint64* ptr, uint64 add)
{
  int i;
  uint64 ret;

  i = pthread_mutex_lock (&sync_lock);
  __go_assert (i == 0);

  *ptr += add;
  ret = *ptr;

  i = pthread_mutex_unlock (&sync_lock);
  __go_assert (i == 0);

  return ret;
}

#endif

uintptr
runtime_memlimit(void)
{
	struct rlimit rl;
	uintptr used;

	if(getrlimit(RLIMIT_AS, &rl) != 0)
		return 0;
	if(rl.rlim_cur >= 0x7fffffff)
		return 0;

	// Estimate our VM footprint excluding the heap.
	// Not an exact science: use size of binary plus
	// some room for thread stacks.
	used = (64<<20);
	if(used >= rl.rlim_cur)
		return 0;

	// If there's not at least 16 MB left, we're probably
	// not going to be able to do much.  Treat as no limit.
	rl.rlim_cur -= used;
	if(rl.rlim_cur < (16<<20))
		return 0;

	return rl.rlim_cur - used;
}
