/* go-panic.h -- declare the go panic functions.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#ifndef LIBGO_GO_PANIC_H
#define LIBGO_GO_PANIC_H

#include "interface.h"

struct String;
struct __go_type_descriptor;
struct __go_defer_stack;

/* The stack of panic calls.  */

struct __go_panic_stack
{
  /* The next entry in the stack.  */
  struct __go_panic_stack *__next;

  /* The value associated with this panic.  */
  struct __go_empty_interface __arg;

  /* Whether this panic has been recovered.  */
  _Bool __was_recovered;

  /* Whether this panic was pushed on the stack because of an
     exception thrown in some other language.  */
  _Bool __is_foreign;
};

extern void __go_panic (struct __go_empty_interface)
  __attribute__ ((noreturn));

extern void __go_print_string (struct String);

extern struct __go_empty_interface __go_recover (void);

extern _Bool __go_can_recover (void *);

extern void __go_makefunc_can_recover (void *retaddr);

struct Location;
extern void __go_makefunc_ffi_can_recover (struct Location *, int);

extern void __go_makefunc_returning (void);

extern void __go_unwind_stack (void);

#endif /* !defined(LIBGO_GO_PANIC_H) */
