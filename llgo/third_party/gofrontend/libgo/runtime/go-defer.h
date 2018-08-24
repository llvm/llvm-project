/* go-defer.h -- the defer stack.

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

struct __go_panic_stack;

/* The defer stack is a list of these structures.  */

struct __go_defer_stack
{
  /* The next entry in the stack.  */
  struct __go_defer_stack *__next;

  /* The stack variable for the function which called this defer
     statement.  This is set to 1 if we are returning from that
     function, 0 if we are panicing through it.  */
  _Bool *__frame;

  /* The value of the panic stack when this function is deferred.
     This function can not recover this value from the panic stack.
     This can happen if a deferred function has a defer statement
     itself.  */
  struct __go_panic_stack *__panic;

  /* The function to call.  */
  void (*__pfn) (void *);

  /* The argument to pass to the function.  */
  void *__arg;

  /* The return address that a recover thunk matches against.  This is
     set by __go_set_defer_retaddr which is called by the thunks
     created by defer statements.  */
  const void *__retaddr;

  /* Set to true if a function created by reflect.MakeFunc is
     permitted to recover.  The return address of such a function
     function will be somewhere in libffi, so __retaddr is not
     useful.  */
  _Bool __makefunc_can_recover;

  /* Set to true if this defer stack entry is not part of the defer
     pool.  */
  _Bool __special;
};
