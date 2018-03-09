/* go-ffi.c -- convert Go type description to libffi.

   Copyright 2014 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#include "config.h"
#include "go-type.h"

#ifdef USE_LIBFFI

#include "ffi.h"

void __go_func_to_cif (const struct __go_func_type *, _Bool, _Bool, ffi_cif *);

#endif
