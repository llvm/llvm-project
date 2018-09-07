// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <unistd.h>

#include "runtime.h"
#include "defs.h"

int32
getproccount(void)
{
	int32 n;
	n = (int32)sysconf(_SC_NPROCESSORS_ONLN);
	return n > 1 ? n : 1;
}
