/* go-assert.h -- libgo specific assertions

   Copyright 2010 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#ifndef LIBGO_GO_ASSERT_H
#define LIBGO_GO_ASSERT_H

/* We use a Go specific assert function so that functions which call
   assert aren't required to always split the stack.  */

extern void __go_assert_fail (const char *file, unsigned int lineno)
  __attribute__ ((noreturn));

#define __go_assert(e) ((e) ? (void) 0 : __go_assert_fail (__FILE__, __LINE__))

#endif /* !defined(LIBGO_GO_ASSERT_H) */
