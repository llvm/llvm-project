// errstr.go -- Error strings.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

//sysnb	strerror_r(errnum int, buf []byte) (err Errno)
//strerror_r(errnum _C_int, buf *byte, buflen Size_t) _C_int

func Errstr(errnum int) string {
	for len := 128; ; len *= 2 {
		b := make([]byte, len)
		errno := strerror_r(errnum, b)
		if errno == 0 {
			i := 0
			for b[i] != 0 {
				i++
			}
			// Lowercase first letter: Bad -> bad, but
			// STREAM -> STREAM.
			if i > 1 && 'A' <= b[0] && b[0] <= 'Z' && 'a' <= b[1] && b[1] <= 'z' {
				b[0] += 'a' - 'A'
			}
			return string(b[:i])
		}
		if errno != ERANGE {
			return "errstr failure"
		}
	}
}
