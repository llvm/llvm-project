/* go-unsafe-new.c -- unsafe.New function for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "arch.h"
#include "malloc.h"
#include "go-type.h"
#include "interface.h"

/* Implement unsafe_New, called from the reflect package.  */

void *unsafe_New (const struct __go_type_descriptor *)
  __asm__ (GOSYM_PREFIX "reflect.unsafe_New");

/* The dynamic type of the argument will be a pointer to a type
   descriptor.  */

void *
unsafe_New (const struct __go_type_descriptor *descriptor)
{
  return runtime_cnew (descriptor);
}
