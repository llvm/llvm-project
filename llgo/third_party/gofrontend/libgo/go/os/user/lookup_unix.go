// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd !android,linux netbsd openbsd solaris
// +build cgo

package user

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

/*
#cgo solaris CFLAGS: -D_POSIX_PTHREAD_SEMANTICS
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <stdlib.h>

static int mygetpwuid_r(int uid, struct passwd *pwd,
	char *buf, size_t buflen, struct passwd **result) {
	return getpwuid_r(uid, pwd, buf, buflen, result);
}

static int mygetpwnam_r(const char *name, struct passwd *pwd,
	char *buf, size_t buflen, struct passwd **result) {
	return getpwnam_r(name, pwd, buf, buflen, result);
}
*/

// bytePtrToString takes a NUL-terminated array of bytes and convert
// it to a Go string.
func bytePtrToString(p *byte) string {
	a := (*[10000]byte)(unsafe.Pointer(p))
	i := 0
	for a[i] != 0 {
		i++
	}
	return string(a[:i])
}

func current() (*User, error) {
	return lookupUnix(syscall.Getuid(), "", false)
}

func lookup(username string) (*User, error) {
	return lookupUnix(-1, username, true)
}

func lookupId(uid string) (*User, error) {
	i, e := strconv.Atoi(uid)
	if e != nil {
		return nil, e
	}
	return lookupUnix(i, "", false)
}

func lookupUnix(uid int, username string, lookupByName bool) (*User, error) {
	var pwd syscall.Passwd
	var result *syscall.Passwd

	// FIXME: Should let buf grow if necessary.
	const bufSize = 1024
	buf := make([]byte, bufSize)
	if lookupByName {
		nameC := syscall.StringBytePtr(username)
		syscall.Entersyscall()
		rv := libc_getpwnam_r(nameC,
			&pwd,
			&buf[0],
			bufSize,
			&result)
		syscall.Exitsyscall()
		if rv != 0 {
			return nil, fmt.Errorf("user: lookup username %s: %s", username, syscall.GetErrno())
		}
		if result == nil {
			return nil, UnknownUserError(username)
		}
	} else {
		syscall.Entersyscall()
		rv := libc_getpwuid_r(syscall.Uid_t(uid),
			&pwd,
			&buf[0],
			bufSize,
			&result)
		syscall.Exitsyscall()
		if rv != 0 {
			return nil, fmt.Errorf("user: lookup userid %d: %s", uid, syscall.GetErrno())
		}
		if result == nil {
			return nil, UnknownUserIdError(uid)
		}
	}
	u := &User{
		Uid:      strconv.Itoa(int(pwd.Pw_uid)),
		Gid:      strconv.Itoa(int(pwd.Pw_gid)),
		Username: bytePtrToString((*byte)(unsafe.Pointer(pwd.Pw_name))),
		Name:     bytePtrToString((*byte)(unsafe.Pointer(pwd.Pw_gecos))),
		HomeDir:  bytePtrToString((*byte)(unsafe.Pointer(pwd.Pw_dir))),
	}
	// The pw_gecos field isn't quite standardized.  Some docs
	// say: "It is expected to be a comma separated list of
	// personal data where the first item is the full name of the
	// user."
	if i := strings.Index(u.Name, ","); i >= 0 {
		u.Name = u.Name[:i]
	}
	return u, nil
}
