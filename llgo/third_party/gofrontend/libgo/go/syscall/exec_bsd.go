// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package syscall

import (
	"unsafe"
)

type SysProcAttr struct {
	Chroot     string      // Chroot.
	Credential *Credential // Credential.
	Ptrace     bool        // Enable tracing.
	Setsid     bool        // Create session.
	Setpgid    bool        // Set process group ID to Pgid, or, if Pgid == 0, to new pid.
	Setctty    bool        // Set controlling terminal to fd Ctty
	Noctty     bool        // Detach fd 0 from controlling terminal
	Ctty       int         // Controlling TTY fd
	Foreground bool        // Place child's process group in foreground. (Implies Setpgid. Uses Ctty as fd of controlling TTY)
	Pgid       int         // Child's process group ID if Setpgid.
}

// Implemented in runtime package.
func runtime_BeforeFork()
func runtime_AfterFork()

// Fork, dup fd onto 0..len(fd), and exec(argv0, argvv, envv) in child.
// If a dup or exec fails, write the errno error to pipe.
// (Pipe is close-on-exec so if exec succeeds, it will be closed.)
// In the child, this function must not acquire any locks, because
// they might have been locked at the time of the fork.  This means
// no rescheduling, no malloc calls, and no new stack segments.
// For the same reason compiler does not race instrument it.
// The calls to RawSyscall are okay because they are assembly
// functions that do not grow the stack.
func forkAndExecInChild(argv0 *byte, argv, envv []*byte, chroot, dir *byte, attr *ProcAttr, sys *SysProcAttr, pipe int) (pid int, err Errno) {
	// Declare all variables at top in case any
	// declarations require heap allocation (e.g., err1).
	var (
		r1     Pid_t
		err1   Errno
		nextfd int
		i      int
	)

	// guard against side effects of shuffling fds below.
	// Make sure that nextfd is beyond any currently open files so
	// that we can't run the risk of overwriting any of them.
	fd := make([]int, len(attr.Files))
	nextfd = len(attr.Files)
	for i, ufd := range attr.Files {
		if nextfd < int(ufd) {
			nextfd = int(ufd)
		}
		fd[i] = int(ufd)
	}
	nextfd++

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	runtime_BeforeFork()
	r1, err1 = raw_fork()
	if err1 != 0 {
		runtime_AfterFork()
		return 0, err1
	}

	if r1 != 0 {
		// parent; return PID
		runtime_AfterFork()
		return int(r1), 0
	}

	// Fork succeeded, now in child.

	// Enable tracing if requested.
	if sys.Ptrace {
		err1 = raw_ptrace(_PTRACE_TRACEME, 0, nil, nil)
		if err1 != 0 {
			goto childerror
		}
	}

	// Session ID
	if sys.Setsid {
		err1 = raw_setsid()
		if err1 != 0 {
			goto childerror
		}
	}

	// Set process group
	if sys.Setpgid || sys.Foreground {
		// Place child in process group.
		err1 = raw_setpgid(0, sys.Pgid)
		if err1 != 0 {
			goto childerror
		}
	}

	if sys.Foreground {
		pgrp := Pid_t(sys.Pgid)
		if pgrp == 0 {
			pgrp = raw_getpid()
		}

		// Place process group in foreground.
		_, err1 = raw_ioctl_ptr(sys.Ctty, TIOCSPGRP, unsafe.Pointer(&pgrp))
		if err1 != 0 {
			goto childerror
		}
	}

	// Chroot
	if chroot != nil {
		err1 = raw_chroot(chroot)
		if err1 != 0 {
			goto childerror
		}
	}

	// User and groups
	if cred := sys.Credential; cred != nil {
		ngroups := len(cred.Groups)
		if ngroups == 0 {
			err2 := setgroups(0, nil)
			if err2 == nil {
				err1 = 0
			} else {
				err1 = err2.(Errno)
			}
		} else {
			groups := make([]Gid_t, ngroups)
			for i, v := range cred.Groups {
				groups[i] = Gid_t(v)
			}
			err2 := setgroups(ngroups, &groups[0])
			if err2 == nil {
				err1 = 0
			} else {
				err1 = err2.(Errno)
			}
		}
		if err1 != 0 {
			goto childerror
		}
		err2 := Setgid(int(cred.Gid))
		if err2 != nil {
			err1 = err2.(Errno)
			goto childerror
		}
		err2 = Setuid(int(cred.Uid))
		if err2 != nil {
			err1 = err2.(Errno)
			goto childerror
		}
	}

	// Chdir
	if dir != nil {
		err1 = raw_chdir(dir)
		if err1 != 0 {
			goto childerror
		}
	}

	// Pass 1: look for fd[i] < i and move those up above len(fd)
	// so that pass 2 won't stomp on an fd it needs later.
	if pipe < nextfd {
		err1 = raw_dup2(pipe, nextfd)
		if err1 != 0 {
			goto childerror
		}
		raw_fcntl(nextfd, F_SETFD, FD_CLOEXEC)
		pipe = nextfd
		nextfd++
	}
	for i = 0; i < len(fd); i++ {
		if fd[i] >= 0 && fd[i] < int(i) {
			err1 = raw_dup2(fd[i], nextfd)
			if err1 != 0 {
				goto childerror
			}
			raw_fcntl(nextfd, F_SETFD, FD_CLOEXEC)
			fd[i] = nextfd
			nextfd++
			if nextfd == pipe { // don't stomp on pipe
				nextfd++
			}
		}
	}

	// Pass 2: dup fd[i] down onto i.
	for i = 0; i < len(fd); i++ {
		if fd[i] == -1 {
			raw_close(i)
			continue
		}
		if fd[i] == int(i) {
			// dup2(i, i) won't clear close-on-exec flag on Linux,
			// probably not elsewhere either.
			_, err1 = raw_fcntl(fd[i], F_SETFD, 0)
			if err1 != 0 {
				goto childerror
			}
			continue
		}
		// The new fd is created NOT close-on-exec,
		// which is exactly what we want.
		err1 = raw_dup2(fd[i], i)
		if err1 != 0 {
			goto childerror
		}
	}

	// By convention, we don't close-on-exec the fds we are
	// started with, so if len(fd) < 3, close 0, 1, 2 as needed.
	// Programs that know they inherit fds >= 3 will need
	// to set them close-on-exec.
	for i = len(fd); i < 3; i++ {
		raw_close(i)
	}

	// Detach fd 0 from tty
	if sys.Noctty {
		_, err1 = raw_ioctl(0, TIOCNOTTY, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Set the controlling TTY to Ctty
	if sys.Setctty {
		_, err1 = raw_ioctl(sys.Ctty, TIOCSCTTY, 0)
		if err1 != 0 {
			goto childerror
		}
	}

	// Time to exec.
	err1 = raw_execve(argv0, &argv[0], &envv[0])

childerror:
	// send error code on pipe
	raw_write(pipe, (*byte)(unsafe.Pointer(&err1)), int(unsafe.Sizeof(err1)))
	for {
		raw_exit(253)
	}
}

// Try to open a pipe with O_CLOEXEC set on both file descriptors.
func forkExecPipe(p []int) error {
	err := Pipe(p)
	if err != nil {
		return err
	}
	_, err = fcntl(p[0], F_SETFD, FD_CLOEXEC)
	if err != nil {
		return err
	}
	_, err = fcntl(p[1], F_SETFD, FD_CLOEXEC)
	return err
}
