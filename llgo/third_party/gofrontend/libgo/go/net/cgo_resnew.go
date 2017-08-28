// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo
// +build darwin linux,!android netbsd solaris

package net

/*
#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
*/

import (
	"syscall"
)

//extern getnameinfo
func libc_getnameinfo(*syscall.RawSockaddr, syscall.Socklen_t, *byte, syscall.Size_t, *byte, syscall.Size_t, int) int

func cgoNameinfoPTR(b []byte, sa *syscall.RawSockaddr, salen syscall.Socklen_t) (int, error) {
	syscall.Entersyscall()
	gerrno := libc_getnameinfo(sa, salen, &b[0], syscall.Size_t(len(b)), nil, 0, syscall.NI_NAMEREQD)
	syscall.Exitsyscall()
	var err error
	if gerrno == syscall.EAI_SYSTEM {
		errno := syscall.GetErrno()
		if errno != 0 {
			err = errno
		}
	}
	return gerrno, err
}
