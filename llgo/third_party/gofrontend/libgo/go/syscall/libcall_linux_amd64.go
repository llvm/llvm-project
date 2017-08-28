// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GNU/Linux library calls amd64 specific.

package syscall

//sys	Ioperm(from int, num int, on int) (err error)
//ioperm(from _C_long, num _C_long, on _C_int) _C_int

//sys	Iopl(level int) (err error)
//iopl(level _C_int) _C_int
