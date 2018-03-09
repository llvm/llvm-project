// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For systems with the waitpid library call.

package syscall

//sys	waitpid(pid Pid_t, status *_C_int, options int) (wpid Pid_t, err error)
//waitpid(pid Pid_t, status *_C_int, options _C_int) Pid_t

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	var status _C_int
	r, err := waitpid(Pid_t(pid), &status, options)
	wpid = int(r)
	if wstatus != nil {
		*wstatus = WaitStatus(status)
	}
	return
}
