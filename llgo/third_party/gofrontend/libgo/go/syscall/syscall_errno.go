// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// An Errno is an unsigned number describing an error condition.
// It implements the error interface.  The zero Errno is by convention
// a non-error, so code to convert from Errno to error should use:
//	err = nil
//	if errno != 0 {
//		err = errno
//	}
type Errno uintptr

func (e Errno) Error() string {
	return Errstr(int(e))
}

func (e Errno) Temporary() bool {
	return e == EINTR || e == EMFILE || e == ECONNRESET || e == ECONNABORTED || e.Timeout()
}

func (e Errno) Timeout() bool {
	return e == EAGAIN || e == EWOULDBLOCK || e == ETIMEDOUT
}
