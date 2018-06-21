// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package syscall

import (
	"runtime"
	"unsafe"
)

//sysnb	raw_prctl(option int, arg2 int, arg3 int, arg4 int, arg5 int) (ret int, err Errno)
//prctl(option _C_int, arg2 _C_long, arg3 _C_long, arg4 _C_long, arg5 _C_long) _C_int

// SysProcIDMap holds Container ID to Host ID mappings used for User Namespaces in Linux.
// See user_namespaces(7).
type SysProcIDMap struct {
	ContainerID int // Container ID.
	HostID      int // Host ID.
	Size        int // Size.
}

type SysProcAttr struct {
	Chroot      string         // Chroot.
	Credential  *Credential    // Credential.
	Ptrace      bool           // Enable tracing.
	Setsid      bool           // Create session.
	Setpgid     bool           // Set process group ID to Pgid, or, if Pgid == 0, to new pid.
	Setctty     bool           // Set controlling terminal to fd Ctty (only meaningful if Setsid is set)
	Noctty      bool           // Detach fd 0 from controlling terminal
	Ctty        int            // Controlling TTY fd
	Foreground  bool           // Place child's process group in foreground. (Implies Setpgid. Uses Ctty as fd of controlling TTY)
	Pgid        int            // Child's process group ID if Setpgid.
	Pdeathsig   Signal         // Signal that the process will get when its parent dies (Linux only)
	Cloneflags  uintptr        // Flags for clone calls (Linux only)
	UidMappings []SysProcIDMap // User ID mappings for user namespaces.
	GidMappings []SysProcIDMap // Group ID mappings for user namespaces.
	// GidMappingsEnableSetgroups enabling setgroups syscall.
	// If false, then setgroups syscall will be disabled for the child process.
	// This parameter is no-op if GidMappings == nil. Otherwise for unprivileged
	// users this should be set to false for mappings work.
	GidMappingsEnableSetgroups bool
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
		r1     uintptr
		err1   Errno
		err2   Errno
		nextfd int
		i      int
		p      [2]int
	)

	// Record parent PID so child can test if it has died.
	ppid := raw_getpid()

	// Guard against side effects of shuffling fds below.
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

	// Allocate another pipe for parent to child communication for
	// synchronizing writing of User ID/Group ID mappings.
	if sys.UidMappings != nil || sys.GidMappings != nil {
		if err := forkExecPipe(p[:]); err != nil {
			return 0, err.(Errno)
		}
	}

	// About to call fork.
	// No more allocation or calls of non-assembly functions.
	runtime_BeforeFork()
	if runtime.GOARCH == "s390x" || runtime.GOARCH == "s390" {
		r1, _, err1 = RawSyscall6(SYS_CLONE, 0, uintptr(SIGCHLD)|sys.Cloneflags, 0, 0, 0, 0)
	} else {
		r1, _, err1 = RawSyscall6(SYS_CLONE, uintptr(SIGCHLD)|sys.Cloneflags, 0, 0, 0, 0, 0)
	}
	if err1 != 0 {
		runtime_AfterFork()
		return 0, err1
	}

	if r1 != 0 {
		// parent; return PID
		runtime_AfterFork()
		pid = int(r1)

		if sys.UidMappings != nil || sys.GidMappings != nil {
			Close(p[0])
			err := writeUidGidMappings(pid, sys)
			if err != nil {
				err2 = err.(Errno)
			}
			RawSyscall(SYS_WRITE, uintptr(p[1]), uintptr(unsafe.Pointer(&err2)), unsafe.Sizeof(err2))
			Close(p[1])
		}

		return pid, 0
	}

	// Fork succeeded, now in child.

	// Wait for User ID/Group ID mappings to be written.
	if sys.UidMappings != nil || sys.GidMappings != nil {
		if _, _, err1 = RawSyscall(SYS_CLOSE, uintptr(p[1]), 0, 0); err1 != 0 {
			goto childerror
		}
		r1, _, err1 = RawSyscall(SYS_READ, uintptr(p[0]), uintptr(unsafe.Pointer(&err2)), unsafe.Sizeof(err2))
		if err1 != 0 {
			goto childerror
		}
		if r1 != unsafe.Sizeof(err2) {
			err1 = EINVAL
			goto childerror
		}
		if err2 != 0 {
			err1 = err2
			goto childerror
		}
	}

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
		if ngroups > 0 {
			groups := unsafe.Pointer(&cred.Groups[0])
			err1 = raw_setgroups(ngroups, groups)
			if err1 != 0 {
				goto childerror
			}
		}
		err1 = raw_setgid(int(cred.Gid))
		if err1 != 0 {
			goto childerror
		}
		err1 = raw_setuid(int(cred.Uid))
		if err1 != 0 {
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

	// Parent death signal
	if sys.Pdeathsig != 0 {
		_, err1 = raw_prctl(PR_SET_PDEATHSIG, int(sys.Pdeathsig), 0, 0, 0)
		if err1 != 0 {
			goto childerror
		}

		// Signal self if parent is already dead. This might cause a
		// duplicate signal in rare cases, but it won't matter when
		// using SIGKILL.
		r1 := raw_getppid()
		if r1 != ppid {
			pid := raw_getpid()
			err1 = raw_kill(pid, sys.Pdeathsig)
			if err1 != 0 {
				goto childerror
			}
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
func forkExecPipe(p []int) (err error) {
	err = Pipe2(p, O_CLOEXEC)
	// pipe2 was added in 2.6.27 and our minimum requirement is 2.6.23, so it
	// might not be implemented.
	if err == ENOSYS {
		if err = Pipe(p); err != nil {
			return
		}
		if _, err = fcntl(p[0], F_SETFD, FD_CLOEXEC); err != nil {
			return
		}
		_, err = fcntl(p[1], F_SETFD, FD_CLOEXEC)
	}
	return
}

// writeIDMappings writes the user namespace User ID or Group ID mappings to the specified path.
func writeIDMappings(path string, idMap []SysProcIDMap) error {
	fd, err := Open(path, O_RDWR, 0)
	if err != nil {
		return err
	}

	data := ""
	for _, im := range idMap {
		data = data + itoa(im.ContainerID) + " " + itoa(im.HostID) + " " + itoa(im.Size) + "\n"
	}

	bytes, err := ByteSliceFromString(data)
	if err != nil {
		Close(fd)
		return err
	}

	if _, err := Write(fd, bytes); err != nil {
		Close(fd)
		return err
	}

	if err := Close(fd); err != nil {
		return err
	}

	return nil
}

// writeSetgroups writes to /proc/PID/setgroups "deny" if enable is false
// and "allow" if enable is true.
// This is needed since kernel 3.19, because you can't write gid_map without
// disabling setgroups() system call.
func writeSetgroups(pid int, enable bool) error {
	sgf := "/proc/" + itoa(pid) + "/setgroups"
	fd, err := Open(sgf, O_RDWR, 0)
	if err != nil {
		return err
	}

	var data []byte
	if enable {
		data = []byte("allow")
	} else {
		data = []byte("deny")
	}

	if _, err := Write(fd, data); err != nil {
		Close(fd)
		return err
	}

	return Close(fd)
}

// writeUidGidMappings writes User ID and Group ID mappings for user namespaces
// for a process and it is called from the parent process.
func writeUidGidMappings(pid int, sys *SysProcAttr) error {
	if sys.UidMappings != nil {
		uidf := "/proc/" + itoa(pid) + "/uid_map"
		if err := writeIDMappings(uidf, sys.UidMappings); err != nil {
			return err
		}
	}

	if sys.GidMappings != nil {
		// If the kernel is too old to support /proc/PID/setgroups, writeSetGroups will return ENOENT; this is OK.
		if err := writeSetgroups(pid, sys.GidMappingsEnableSetgroups); err != nil && err != ENOENT {
			return err
		}
		gidf := "/proc/" + itoa(pid) + "/gid_map"
		if err := writeIDMappings(gidf, sys.GidMappings); err != nil {
			return err
		}
	}

	return nil
}
