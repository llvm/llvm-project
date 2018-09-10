// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Declarations for functions which are actually written in C.

package syscall

func Entersyscall()
func Exitsyscall()
func GetErrno() Errno
func SetErrno(Errno)

// These functions are used by CGO and SWIG.
func Cgocall()
func CgocallDone()
func CgocallBack()
func CgocallBackDone()
