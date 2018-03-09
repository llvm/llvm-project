// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// General POSIX version of UtimesNano.

package syscall

import "unsafe"

func UtimesNano(path string, ts []Timespec) error {
	// TODO: The BSDs can do utimensat with SYS_UTIMENSAT but it
	// isn't supported by darwin so this uses utimes instead
	if len(ts) != 2 {
		return EINVAL
	}
	// Not as efficient as it could be because Timespec and
	// Timeval have different types in the different OSes
	tv := [2]Timeval{
		NsecToTimeval(TimespecToNsec(ts[0])),
		NsecToTimeval(TimespecToNsec(ts[1])),
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}
