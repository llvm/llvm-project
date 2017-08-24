// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include <sys/sysctl.h>

#include "runtime.h"
#include "defs.h"

int32
getproccount(void)
{
	int mib[2], out;
	size_t len;

	mib[0] = CTL_HW;
	mib[1] = HW_NCPU;
	len = sizeof(out);
	if(sysctl(mib, 2, &out, &len, NULL, 0) >= 0)
		return (int32)out;
	else
		return 0;
}
