// errstr.go -- Error strings when there is no strerror_r.

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"sync"
	"unsafe"
)

//sysnb	strerror(errnum int) (buf *byte)
//strerror(errnum _C_int) *byte

var errstr_lock sync.Mutex

func Errstr(errno int) string {
	errstr_lock.Lock()

	bp := strerror(errno)
	b := (*[1000]byte)(unsafe.Pointer(bp))
	i := 0
	for b[i] != 0 {
		i++
	}

	// Lowercase first letter: Bad -> bad, but STREAM -> STREAM.
	var s string
	if i > 1 && 'A' <= b[0] && b[0] <= 'Z' && 'a' <= b[1] && b[1] <= 'z' {
		c := b[0] + 'a' - 'A'
		s = string(c) + string(b[1:i])
	} else {
		s = string(b[:i])
	}

	errstr_lock.Unlock()

	return s
}
