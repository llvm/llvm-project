// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/epoll.h>

#include "runtime.h"
#include "defs.h"
#include "malloc.h"

#ifndef EPOLLRDHUP
#define EPOLLRDHUP 0x2000
#endif

#ifndef EPOLL_CLOEXEC
#define EPOLL_CLOEXEC 02000000
#endif

#ifndef HAVE_EPOLL_CREATE1
extern int epoll_create1(int __flags);
#endif

typedef struct epoll_event EpollEvent;

static int32
runtime_epollcreate(int32 size)
{
	int r;

	r = epoll_create(size);
	if(r >= 0)
		return r;
	return - errno;
}

static int32
runtime_epollcreate1(int32 flags)
{
	int r;

	r = epoll_create1(flags);
	if(r >= 0)
		return r;
	return - errno;
}

static int32
runtime_epollctl(int32 epfd, int32 op, int32 fd, EpollEvent *ev)
{
	int r;

	r = epoll_ctl(epfd, op, fd, ev);
	if(r >= 0)
		return r;
	return - errno;
}

static int32
runtime_epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout)
{
	int r;

	r = epoll_wait(epfd, ev, nev, timeout);
	if(r >= 0)
		return r;
	return - errno;
}

static void
runtime_closeonexec(int32 fd)
{
	fcntl(fd, F_SETFD, FD_CLOEXEC);
}

static int32 epfd = -1;  // epoll descriptor

void
runtime_netpollinit(void)
{
	epfd = runtime_epollcreate1(EPOLL_CLOEXEC);
	if(epfd >= 0)
		return;
	epfd = runtime_epollcreate(1024);
	if(epfd >= 0) {
		runtime_closeonexec(epfd);
		return;
	}
	runtime_printf("netpollinit: failed to create descriptor (%d)\n", -epfd);
	runtime_throw("netpollinit: failed to create descriptor");
}

int32
runtime_netpollopen(uintptr fd, PollDesc *pd)
{
	EpollEvent ev;
	int32 res;

	ev.events = EPOLLIN|EPOLLOUT|EPOLLRDHUP|EPOLLET;
	ev.data.ptr = (void*)pd;
	res = runtime_epollctl(epfd, EPOLL_CTL_ADD, (int32)fd, &ev);
	return -res;
}

int32
runtime_netpollclose(uintptr fd)
{
	EpollEvent ev;
	int32 res;

	res = runtime_epollctl(epfd, EPOLL_CTL_DEL, (int32)fd, &ev);
	return -res;
}

void
runtime_netpollarm(PollDesc* pd, int32 mode)
{
	USED(pd);
	USED(mode);
	runtime_throw("unused");
}

// polls for ready network connections
// returns list of goroutines that become runnable
G*
runtime_netpoll(bool block)
{
	static int32 lasterr;
	EpollEvent events[128], *ev;
	int32 n, i, waitms, mode;
	G *gp;

	if(epfd == -1)
		return nil;
	waitms = -1;
	if(!block)
		waitms = 0;
retry:
	n = runtime_epollwait(epfd, events, nelem(events), waitms);
	if(n < 0) {
		if(n != -EINTR && n != lasterr) {
			lasterr = n;
			runtime_printf("runtime: epollwait on fd %d failed with %d\n", epfd, -n);
		}
		goto retry;
	}
	gp = nil;
	for(i = 0; i < n; i++) {
		ev = &events[i];
		if(ev->events == 0)
			continue;
		mode = 0;
		if(ev->events & (EPOLLIN|EPOLLRDHUP|EPOLLHUP|EPOLLERR))
			mode += 'r';
		if(ev->events & (EPOLLOUT|EPOLLHUP|EPOLLERR))
			mode += 'w';
		if(mode)
			runtime_netpollready(&gp, (void*)ev->data.ptr, mode);
	}
	if(block && gp == nil)
		goto retry;
	return gp;
}

void
runtime_netpoll_scan(struct Workbuf** wbufp, void (*enqueue1)(struct Workbuf**, Obj))
{
	USED(wbufp);
	USED(enqueue1);
}
