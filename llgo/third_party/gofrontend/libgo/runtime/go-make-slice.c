/* go-make-slice.c -- make a slice.

   Copyright 2011 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stdint.h>

#include "runtime.h"
#include "go-alloc.h"
#include "go-assert.h"
#include "go-panic.h"
#include "go-type.h"
#include "array.h"
#include "arch.h"
#include "malloc.h"

/* Dummy word to use as base pointer for make([]T, 0).
   Since you cannot take the address of such a slice,
   you can't tell that they all have the same base pointer.  */
uintptr runtime_zerobase;

struct __go_open_array
__go_make_slice2 (const struct __go_type_descriptor *td, uintptr_t len,
		  uintptr_t cap)
{
  const struct __go_slice_type* std;
  intgo ilen;
  intgo icap;
  uintptr_t size;
  struct __go_open_array ret;

  __go_assert ((td->__code & GO_CODE_MASK) == GO_SLICE);
  std = (const struct __go_slice_type *) td;

  ilen = (intgo) len;
  if (ilen < 0
      || (uintptr_t) ilen != len
      || (std->__element_type->__size > 0
	  && len > MaxMem / std->__element_type->__size))
    runtime_panicstring ("makeslice: len out of range");

  icap = (intgo) cap;
  if (cap < len
      || (uintptr_t) icap != cap
      || (std->__element_type->__size > 0
	  && cap > MaxMem / std->__element_type->__size))
    runtime_panicstring ("makeslice: cap out of range");

  ret.__count = ilen;
  ret.__capacity = icap;

  size = cap * std->__element_type->__size;

  if (size == 0)
    ret.__values = &runtime_zerobase;
  else if ((std->__element_type->__code & GO_NO_POINTERS) != 0)
    ret.__values =
      runtime_mallocgc (size,
			(uintptr) std->__element_type | TypeInfo_Array,
			FlagNoScan);
  else
    ret.__values =
      runtime_mallocgc (size,
			(uintptr) std->__element_type | TypeInfo_Array,
			0);

  return ret;
}

struct __go_open_array
__go_make_slice1 (const struct __go_type_descriptor *td, uintptr_t len)
{
  return __go_make_slice2 (td, len, len);
}

struct __go_open_array
__go_make_slice2_big (const struct __go_type_descriptor *td, uint64_t len,
		      uint64_t cap)
{
  uintptr_t slen;
  uintptr_t scap;

  slen = (uintptr_t) len;
  if ((uint64_t) slen != len)
    runtime_panicstring ("makeslice: len out of range");

  scap = (uintptr_t) cap;
  if ((uint64_t) scap != cap)
    runtime_panicstring ("makeslice: cap out of range");

  return __go_make_slice2 (td, slen, scap);
}

struct __go_open_array
__go_make_slice1_big (const struct __go_type_descriptor *td, uint64_t len)
{
  return __go_make_slice2_big (td, len, len);
}
