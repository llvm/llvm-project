/* go-unsafe-newarray.c -- unsafe.NewArray function for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "runtime.h"
#include "arch.h"
#include "malloc.h"
#include "go-type.h"
#include "interface.h"

/* Implement unsafe_NewArray, called from the reflect package.  */

void *unsafe_NewArray (const struct __go_type_descriptor *, intgo)
  __asm__ (GOSYM_PREFIX "reflect.unsafe_NewArray");

/* The dynamic type of the argument will be a pointer to a type
   descriptor.  */

void *
unsafe_NewArray (const struct __go_type_descriptor *descriptor, intgo n)
{
  return runtime_cnewarray (descriptor, n);
}
