// errstr_rtems.go -- RTEMS specific error strings.

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

//sysnb	strerror_r(errnum int, b []byte) (errstr *byte)
//strerror_r(errnum _C_int, b *byte, len Size_t) *byte

func Errstr(errnum int) string {
	a := make([]byte, 128)
	p := strerror_r(errnum, a)
	b := (*[1000]byte)(unsafe.Pointer(p))
	i := 0
	for b[i] != 0 {
		i++
	}
	// Lowercase first letter: Bad -> bad, but STREAM -> STREAM.
	if i > 1 && 'A' <= b[0] && b[0] <= 'Z' && 'a' <= b[1] && b[1] <= 'z' {
		c := b[0] + 'a' - 'A'
		return string(c) + string(b[1:i])
	}
	return string(b[:i])
}
