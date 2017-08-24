// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stubs for fork, exec and wait.

package syscall

func ForkExec(argv0 string, argv []string, envv []string, dir string, fd []int) (pid int, err int) {
	return -1, ENOSYS
}

func Exec(argv0 string, argv []string, envv []string) (err int) {
	return ENOSYS
}

func Wait4(pid int, wstatus *WaitStatus, options int, rusage *Rusage) (wpid int, err error) {
	return -1, ENOSYS
}

func (w WaitStatus) Exited() bool    { return false }
func (w WaitStatus) Signaled() bool  { return false }
func (w WaitStatus) Stopped() bool   { return false }
func (w WaitStatus) Continued() bool { return false }
func (w WaitStatus) CoreDump() bool  { return false }
func (w WaitStatus) ExitStatus() int { return 0 }
func (w WaitStatus) Signal() int     { return 0 }
func (w WaitStatus) StopSignal() int { return 0 }
func (w WaitStatus) TrapCause() int  { return 0 }

func raw_ptrace(request int, pid int, addr *byte, data *byte) Errno {
	return ENOSYS
}
