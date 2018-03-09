// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

#include "runtime.h"
#include "array.h"
#include "arch.h"
#include "malloc.h"

extern Slice envs;

String
runtime_getenv(const char *s)
{
	int32 i, j;
	intgo len;
	const byte *v, *bs;
	String* envv;
	int32 envc;
	String ret;

	bs = (const byte*)s;
	len = runtime_findnull(bs);
	envv = (String*)envs.__values;
	envc = envs.__count;
	for(i=0; i<envc; i++){
		if(envv[i].len <= len)
			continue;
		v = (const byte*)envv[i].str;
		for(j=0; j<len; j++)
			if(bs[j] != v[j])
				goto nomatch;
		if(v[len] != '=')
			goto nomatch;
		ret.str = v+len+1;
		ret.len = envv[i].len-len-1;
		return ret;
	nomatch:;
	}
	ret.str = nil;
	ret.len = 0;
	return ret;
}
