// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris
// +build cgo

package user

import "syscall"

// Declarations for the libc functions on Solaris.

//extern __posix_getpwnam_r
func libc_getpwnam_r(name *byte, pwd *syscall.Passwd, buf *byte, buflen syscall.Size_t, result **syscall.Passwd) int

//extern __posix_getpwuid_r
func libc_getpwuid_r(uid syscall.Uid_t, pwd *syscall.Passwd, buf *byte, buflen syscall.Size_t, result **syscall.Passwd) int
