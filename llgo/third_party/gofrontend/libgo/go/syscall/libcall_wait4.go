// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For systems with the wait4 library call.

package syscall

//sys	wait4(pid Pid_t, status *_C_int, options int, rusage *Rusage) (wpid Pid_t, err error)
//wait4(pid Pid_t, status *_C_int, options _C_int, rusage *Rusage) Pid_t

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	var status _C_int
	r, err := wait4(Pid_t(pid), &status, options, rusage)
	wpid = int(r)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}
