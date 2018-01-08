// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GNU/Linux version of UtimesNano.

package syscall

import "unsafe"

//sys	utimensat(dirfd int, path string, times *[2]Timespec, flags int) (err error)
//utimensat(dirfd _C_int, path *byte, times *[2]Timespec, flags _C_int) _C_int
func UtimesNano(path string, ts []Timespec) (err error) {
	if len(ts) != 2 {
		return EINVAL
	}
	err = utimensat(_AT_FDCWD, path, (*[2]Timespec)(unsafe.Pointer(&ts[0])), 0)
	if err != ENOSYS {
		return err
	}
	// If the utimensat syscall isn't available (utimensat was added to Linux
	// in 2.6.22, Released, 8 July 2007) then fall back to utimes
	var tv [2]Timeval
	for i := 0; i < 2; i++ {
		tv[i].Sec = Timeval_sec_t(ts[i].Sec)
		tv[i].Usec = Timeval_usec_t(ts[i].Nsec / 1000)
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}
