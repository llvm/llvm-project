/* atomic.c -- implement atomic routines for Go.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stdint.h>

#include "runtime.h"

int32_t SwapInt32 (int32_t *, int32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapInt32")
  __attribute__ ((no_split_stack));

int32_t
SwapInt32 (int32_t *addr, int32_t new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

int64_t SwapInt64 (int64_t *, int64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapInt64")
  __attribute__ ((no_split_stack));

int64_t
SwapInt64 (int64_t *addr, int64_t new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

uint32_t SwapUint32 (uint32_t *, uint32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapUint32")
  __attribute__ ((no_split_stack));

uint32_t
SwapUint32 (uint32_t *addr, uint32_t new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

uint64_t SwapUint64 (uint64_t *, uint64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapUint64")
  __attribute__ ((no_split_stack));

uint64_t
SwapUint64 (uint64_t *addr, uint64_t new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

uintptr_t SwapUintptr (uintptr_t *, uintptr_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapUintptr")
  __attribute__ ((no_split_stack));

uintptr_t
SwapUintptr (uintptr_t *addr, uintptr_t new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

void *SwapPointer (void **, void *)
  __asm__ (GOSYM_PREFIX "sync_atomic.SwapPointer")
  __attribute__ ((no_split_stack));

void *
SwapPointer (void **addr, void *new)
{
  return __atomic_exchange_n (addr, new, __ATOMIC_SEQ_CST);
}

_Bool CompareAndSwapInt32 (int32_t *, int32_t, int32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapInt32")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapInt32 (int32_t *val, int32_t old, int32_t new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

_Bool CompareAndSwapInt64 (int64_t *, int64_t, int64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapInt64")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapInt64 (int64_t *val, int64_t old, int64_t new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

_Bool CompareAndSwapUint32 (uint32_t *, uint32_t, uint32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapUint32")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapUint32 (uint32_t *val, uint32_t old, uint32_t new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

_Bool CompareAndSwapUint64 (uint64_t *, uint64_t, uint64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapUint64")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapUint64 (uint64_t *val, uint64_t old, uint64_t new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

_Bool CompareAndSwapUintptr (uintptr_t *, uintptr_t, uintptr_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapUintptr")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapUintptr (uintptr_t *val, uintptr_t old, uintptr_t new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

_Bool CompareAndSwapPointer (void **, void *, void *)
  __asm__ (GOSYM_PREFIX "sync_atomic.CompareAndSwapPointer")
  __attribute__ ((no_split_stack));

_Bool
CompareAndSwapPointer (void **val, void *old, void *new)
{
  return __sync_bool_compare_and_swap (val, old, new);
}

int32_t AddInt32 (int32_t *, int32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.AddInt32")
  __attribute__ ((no_split_stack));

int32_t
AddInt32 (int32_t *val, int32_t delta)
{
  return __sync_add_and_fetch (val, delta);
}

uint32_t AddUint32 (uint32_t *, uint32_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.AddUint32")
  __attribute__ ((no_split_stack));

uint32_t
AddUint32 (uint32_t *val, uint32_t delta)
{
  return __sync_add_and_fetch (val, delta);
}

int64_t AddInt64 (int64_t *, int64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.AddInt64")
  __attribute__ ((no_split_stack));

int64_t
AddInt64 (int64_t *val, int64_t delta)
{
  return __sync_add_and_fetch (val, delta);
}

uint64_t AddUint64 (uint64_t *, uint64_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.AddUint64")
  __attribute__ ((no_split_stack));

uint64_t
AddUint64 (uint64_t *val, uint64_t delta)
{
  return __sync_add_and_fetch (val, delta);
}

uintptr_t AddUintptr (uintptr_t *, uintptr_t)
  __asm__ (GOSYM_PREFIX "sync_atomic.AddUintptr")
  __attribute__ ((no_split_stack));

uintptr_t
AddUintptr (uintptr_t *val, uintptr_t delta)
{
  return __sync_add_and_fetch (val, delta);
}

int32_t LoadInt32 (int32_t *addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadInt32")
  __attribute__ ((no_split_stack));

int32_t
LoadInt32 (int32_t *addr)
{
  int32_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

int64_t LoadInt64 (int64_t *addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadInt64")
  __attribute__ ((no_split_stack));

int64_t
LoadInt64 (int64_t *addr)
{
  int64_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

uint32_t LoadUint32 (uint32_t *addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadUint32")
  __attribute__ ((no_split_stack));

uint32_t
LoadUint32 (uint32_t *addr)
{
  uint32_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

uint64_t LoadUint64 (uint64_t *addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadUint64")
  __attribute__ ((no_split_stack));

uint64_t
LoadUint64 (uint64_t *addr)
{
  uint64_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

uintptr_t LoadUintptr (uintptr_t *addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadUintptr")
  __attribute__ ((no_split_stack));

uintptr_t
LoadUintptr (uintptr_t *addr)
{
  uintptr_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

void *LoadPointer (void **addr)
  __asm__ (GOSYM_PREFIX "sync_atomic.LoadPointer")
  __attribute__ ((no_split_stack));

void *
LoadPointer (void **addr)
{
  void *v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, v))
    v = *addr;
  return v;
}

void StoreInt32 (int32_t *addr, int32_t val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StoreInt32")
  __attribute__ ((no_split_stack));

void
StoreInt32 (int32_t *addr, int32_t val)
{
  int32_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}

void StoreInt64 (int64_t *addr, int64_t val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StoreInt64")
  __attribute__ ((no_split_stack));

void
StoreInt64 (int64_t *addr, int64_t val)
{
  int64_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}

void StoreUint32 (uint32_t *addr, uint32_t val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StoreUint32")
  __attribute__ ((no_split_stack));

void
StoreUint32 (uint32_t *addr, uint32_t val)
{
  uint32_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}

void StoreUint64 (uint64_t *addr, uint64_t val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StoreUint64")
  __attribute__ ((no_split_stack));

void
StoreUint64 (uint64_t *addr, uint64_t val)
{
  uint64_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}

void StoreUintptr (uintptr_t *addr, uintptr_t val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StoreUintptr")
  __attribute__ ((no_split_stack));

void
StoreUintptr (uintptr_t *addr, uintptr_t val)
{
  uintptr_t v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}

void StorePointer (void **addr, void *val)
  __asm__ (GOSYM_PREFIX "sync_atomic.StorePointer")
  __attribute__ ((no_split_stack));

void
StorePointer (void **addr, void *val)
{
  void *v;

  v = *addr;
  while (! __sync_bool_compare_and_swap (addr, v, val))
    v = *addr;
}
