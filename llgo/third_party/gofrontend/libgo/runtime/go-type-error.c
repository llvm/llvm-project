/* go-type-error.c -- invalid hash and equality functions.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "go-type.h"

/* A hash function used for a type which does not support hash
   functions.  */

uintptr_t
__go_type_hash_error (const void *val __attribute__ ((unused)),
		      uintptr_t key_size __attribute__ ((unused)))
{
  runtime_panicstring ("hash of unhashable type");
}

const FuncVal __go_type_hash_error_descriptor =
  { (void *) __go_type_hash_error };

/* An equality function for an interface.  */

_Bool
__go_type_equal_error (const void *v1 __attribute__ ((unused)),
		       const void *v2 __attribute__ ((unused)),
		       uintptr_t key_size __attribute__ ((unused)))
{
  runtime_panicstring ("comparing uncomparable types");
}

const FuncVal __go_type_equal_error_descriptor =
  { (void *) __go_type_equal_error };
