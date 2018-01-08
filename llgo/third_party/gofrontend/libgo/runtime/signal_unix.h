// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <signal.h>

#define GO_SIG_DFL ((void*)SIG_DFL)
#define GO_SIG_IGN ((void*)SIG_IGN)

#ifdef SA_SIGINFO
typedef siginfo_t Siginfo;
#else
typedef void *Siginfo;
#endif

typedef void GoSighandler(int32, Siginfo*, void*, G*);
void	runtime_setsig(int32, GoSighandler*, bool);
GoSighandler* runtime_getsig(int32);

void	runtime_sighandler(int32 sig, Siginfo *info, void *context, G *gp);
void	runtime_raise(int32);

