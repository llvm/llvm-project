// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// BSD library calls.

package syscall

import "unsafe"

//sys	sendfile(outfd int, infd int, offset *Offset_t, count int) (written int, err error)
//sendfile(outfd _C_int, infd _C_int, offset *Offset_t, count Size_t) Ssize_t
func Sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	var soff Offset_t
	var psoff *Offset_t
	if offset != nil {
		soff = Offset_t(*offset)
		psoff = &soff
	}
	written, err = sendfile(outfd, infd, psoff, count)
	if offset != nil {
		*offset = int64(soff)
	}
	return
}
