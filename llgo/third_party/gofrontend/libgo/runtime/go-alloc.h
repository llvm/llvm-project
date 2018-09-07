/* go-alloc.h -- allocate memory for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include <stddef.h>
#include <stdint.h>

extern void *__go_alloc (unsigned int __attribute__ ((mode (pointer))));
extern void __go_free (void *);
