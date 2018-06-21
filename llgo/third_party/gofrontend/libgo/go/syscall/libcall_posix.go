// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// POSIX library calls.
// This file is compiled as ordinary Go code,
// but it is also input to mksyscall,
// which parses the //sys lines and generates library call stubs.
// Note that sometimes we use a lowercase //sys name and
// wrap it in our own nicer implementation.

package syscall

import "unsafe"

/*
 * Wrapped
 */

//sysnb	pipe(p *[2]_C_int) (err error)
//pipe(p *[2]_C_int) _C_int
func Pipe(p []int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe(&pp)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

//sys	utimes(path string, times *[2]Timeval) (err error)
//utimes(path *byte, times *[2]Timeval) _C_int
func Utimes(path string, tv []Timeval) (err error) {
	if len(tv) != 2 {
		return EINVAL
	}
	return utimes(path, (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

//sys	getcwd(buf *byte, size Size_t) (err error)
//getcwd(buf *byte, size Size_t) *byte

const ImplementsGetwd = true

func Getwd() (ret string, err error) {
	for len := Size_t(4096); ; len *= 2 {
		b := make([]byte, len)
		err := getcwd(&b[0], len)
		if err == nil {
			i := 0
			for b[i] != 0 {
				i++
			}
			return string(b[0:i]), nil
		}
		if err != ERANGE {
			return "", err
		}
	}
}

func Getcwd(buf []byte) (n int, err error) {
	err = getcwd(&buf[0], Size_t(len(buf)))
	if err == nil {
		i := 0
		for buf[i] != 0 {
			i++
		}
		n = i + 1
	}
	return
}

//sysnb	getgroups(size int, list *Gid_t) (nn int, err error)
//getgroups(size _C_int, list *Gid_t) _C_int

func Getgroups() (gids []int, err error) {
	n, err := getgroups(0, nil)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	// Sanity check group count.  Max is 1<<16 on GNU/Linux.
	if n < 0 || n > 1<<20 {
		return nil, EINVAL
	}

	a := make([]Gid_t, n)
	n, err = getgroups(n, &a[0])
	if err != nil {
		return nil, err
	}
	gids = make([]int, n)
	for i, v := range a[0:n] {
		gids[i] = int(v)
	}
	return
}

//sysnb	setgroups(n int, list *Gid_t) (err error)
//setgroups(n Size_t, list *Gid_t) _C_int

func Setgroups(gids []int) (err error) {
	if len(gids) == 0 {
		return setgroups(0, nil)
	}

	a := make([]Gid_t, len(gids))
	for i, v := range gids {
		a[i] = Gid_t(v)
	}
	return setgroups(len(a), &a[0])
}

type WaitStatus uint32

// The WaitStatus methods are implemented in C, to pick up the macros
// #defines in <sys/wait.h>.

func (w WaitStatus) Exited() bool
func (w WaitStatus) Signaled() bool
func (w WaitStatus) Stopped() bool
func (w WaitStatus) Continued() bool
func (w WaitStatus) CoreDump() bool
func (w WaitStatus) ExitStatus() int
func (w WaitStatus) Signal() Signal
func (w WaitStatus) StopSignal() Signal
func (w WaitStatus) TrapCause() int

//sys	Mkfifo(path string, mode uint32) (err error)
//mkfifo(path *byte, mode Mode_t) _C_int

//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, err error)
//select(nfd _C_int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) _C_int

const nfdbits = int(unsafe.Sizeof(fds_bits_type(0)) * 8)

type FdSet struct {
	Bits [(FD_SETSIZE + nfdbits - 1) / nfdbits]fds_bits_type
}

func FDSet(fd int, set *FdSet) {
	set.Bits[fd/nfdbits] |= (1 << (uint)(fd%nfdbits))
}

func FDClr(fd int, set *FdSet) {
	set.Bits[fd/nfdbits] &^= (1 << (uint)(fd%nfdbits))
}

func FDIsSet(fd int, set *FdSet) bool {
	if set.Bits[fd/nfdbits]&(1<<(uint)(fd%nfdbits)) != 0 {
		return true
	} else {
		return false
	}
}

func FDZero(set *FdSet) {
	for i := range set.Bits {
		set.Bits[i] = 0
	}
}

//sys	Access(path string, mode uint32) (err error)
//access(path *byte, mode _C_int) _C_int

//sys	Chdir(path string) (err error)
//chdir(path *byte) _C_int

//sys	Chmod(path string, mode uint32) (err error)
//chmod(path *byte, mode Mode_t) _C_int

//sys	Chown(path string, uid int, gid int) (err error)
//chown(path *byte, uid Uid_t, gid Gid_t) _C_int

//sys	Chroot(path string) (err error)
//chroot(path *byte) _C_int

//sys	Close(fd int) (err error)
//close(fd _C_int) _C_int

//sys	Creat(path string, mode uint32) (fd int, err error)
//creat(path *byte, mode Mode_t) _C_int

//sysnb	Dup(oldfd int) (fd int, err error)
//dup(oldfd _C_int) _C_int

//sysnb	Dup2(oldfd int, newfd int) (err error)
//dup2(oldfd _C_int, newfd _C_int) _C_int

//sys	Exit(code int)
//exit(code _C_int)

//sys	Fchdir(fd int) (err error)
//fchdir(fd _C_int) _C_int

//sys	Fchmod(fd int, mode uint32) (err error)
//fchmod(fd _C_int, mode Mode_t) _C_int

//sys	Fchown(fd int, uid int, gid int) (err error)
//fchown(fd _C_int, uid Uid_t, gid Gid_t) _C_int

//sys	fcntl(fd int, cmd int, arg int) (val int, err error)
//__go_fcntl(fd _C_int, cmd _C_int, arg _C_int) _C_int

//sys	FcntlFlock(fd uintptr, cmd int, lk *Flock_t) (err error)
//__go_fcntl_flock(fd _C_int, cmd _C_int, arg *Flock_t) _C_int

//sys	Fdatasync(fd int) (err error)
//fdatasync(fd _C_int) _C_int

//sys	Fsync(fd int) (err error)
//fsync(fd _C_int) _C_int

//sysnb Getegid() (egid int)
//getegid() Gid_t

//sysnb Geteuid() (euid int)
//geteuid() Uid_t

//sysnb Getgid() (gid int)
//getgid() Gid_t

//sysnb	Getpagesize() (pagesize int)
//getpagesize() _C_int

//sysnb	Getpgid(pid int) (pgid int, err error)
//getpgid(pid Pid_t) Pid_t

//sysnb	Getpgrp() (pid int)
//getpgrp() Pid_t

//sysnb	Getpid() (pid int)
//getpid() Pid_t

//sysnb	Getppid() (ppid int)
//getppid() Pid_t

//sys Getpriority(which int, who int) (prio int, err error)
//getpriority(which _C_int, who _C_int) _C_int

//sysnb	Getrusage(who int, rusage *Rusage) (err error)
//getrusage(who _C_int, rusage *Rusage) _C_int

//sysnb	gettimeofday(tv *Timeval, tz *byte) (err error)
//gettimeofday(tv *Timeval, tz *byte) _C_int
func Gettimeofday(tv *Timeval) (err error) {
	return gettimeofday(tv, nil)
}

//sysnb Getuid() (uid int)
//getuid() Uid_t

//sysnb	Kill(pid int, sig Signal) (err error)
//kill(pid Pid_t, sig _C_int) _C_int

//sys	Lchown(path string, uid int, gid int) (err error)
//lchown(path *byte, uid Uid_t, gid Gid_t) _C_int

//sys	Link(oldpath string, newpath string) (err error)
//link(oldpath *byte, newpath *byte) _C_int

//sys	Mkdir(path string, mode uint32) (err error)
//mkdir(path *byte, mode Mode_t) _C_int

//sys	Mknod(path string, mode uint32, dev int) (err error)
//mknod(path *byte, mode Mode_t, dev _dev_t) _C_int

//sys	Mount(source string, target string, fstype string, flags uintptr, data string) (err error)
//mount(source *byte, target *byte, fstype *byte, flags _C_long, data *byte) _C_int

//sys	Nanosleep(time *Timespec, leftover *Timespec) (err error)
//nanosleep(time *Timespec, leftover *Timespec) _C_int

//sys	Pause() (err error)
//pause() _C_int

//sys	read(fd int, p []byte) (n int, err error)
//read(fd _C_int, buf *byte, count Size_t) Ssize_t

//sys	readlen(fd int, p *byte, np int) (n int, err error)
//read(fd _C_int, buf *byte, count Size_t) Ssize_t

//sys	Readlink(path string, buf []byte) (n int, err error)
//readlink(path *byte, buf *byte, bufsiz Size_t) Ssize_t

//sys	Rename(oldpath string, newpath string) (err error)
//rename(oldpath *byte, newpath *byte) _C_int

//sys	Rmdir(path string) (err error)
//rmdir(path *byte) _C_int

//sys	Setdomainname(p []byte) (err error)
//setdomainname(name *byte, len Size_t) _C_int

//sys	Sethostname(p []byte) (err error)
//sethostname(name *byte, len Size_t) _C_int

//sysnb	Setgid(gid int) (err error)
//setgid(gid Gid_t) _C_int

//sysnb Setregid(rgid int, egid int) (err error)
//setregid(rgid Gid_t, egid Gid_t) _C_int

//sysnb	Setpgid(pid int, pgid int) (err error)
//setpgid(pid Pid_t, pgid Pid_t) _C_int

//sys Setpriority(which int, who int, prio int) (err error)
//setpriority(which _C_int, who _C_int, prio _C_int) _C_int

//sysnb	Setreuid(ruid int, euid int) (err error)
//setreuid(ruid Uid_t, euid Uid_t) _C_int

//sysnb	Setsid() (pid int, err error)
//setsid() Pid_t

//sysnb	settimeofday(tv *Timeval, tz *byte) (err error)
//settimeofday(tv *Timeval, tz *byte) _C_int

func Settimeofday(tv *Timeval) (err error) {
	return settimeofday(tv, nil)
}

//sysnb	Setuid(uid int) (err error)
//setuid(uid Uid_t) _C_int

//sys	Symlink(oldpath string, newpath string) (err error)
//symlink(oldpath *byte, newpath *byte) _C_int

//sys	Sync()
//sync()

//sysnb	Time(t *Time_t) (tt Time_t, err error)
//time(t *Time_t) Time_t

//sysnb	Times(tms *Tms) (ticks uintptr, err error)
//times(tms *Tms) _clock_t

//sysnb	Umask(mask int) (oldmask int)
//umask(mask Mode_t) Mode_t

//sys	Unlink(path string) (err error)
//unlink(path *byte) _C_int

//sys	Utime(path string, buf *Utimbuf) (err error)
//utime(path *byte, buf *Utimbuf) _C_int

//sys	write(fd int, p []byte) (n int, err error)
//write(fd _C_int, buf *byte, count Size_t) Ssize_t

//sys	writelen(fd int, p *byte, np int) (n int, err error)
//write(fd _C_int, buf *byte, count Size_t) Ssize_t

//sys	munmap(addr uintptr, length uintptr) (err error)
//munmap(addr *byte, length Size_t) _C_int

//sys Madvise(b []byte, advice int) (err error)
//madvise(addr *byte, len Size_t, advice _C_int) _C_int

//sys	Mprotect(b []byte, prot int) (err error)
//mprotect(addr *byte, len Size_t, prot _C_int) _C_int

//sys	Mlock(b []byte) (err error)
//mlock(addr *byte, len Size_t) _C_int

//sys	Munlock(b []byte) (err error)
//munlock(addr *byte, len Size_t) _C_int

//sys	Mlockall(flags int) (err error)
//mlockall(flags _C_int) _C_int

//sys	Munlockall() (err error)
//munlockall() _C_int

func TimespecToNsec(ts Timespec) int64 { return int64(ts.Sec)*1e9 + int64(ts.Nsec) }

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = Timespec_sec_t(nsec / 1e9)
	ts.Nsec = Timespec_nsec_t(nsec % 1e9)
	return
}

func TimevalToNsec(tv Timeval) int64 { return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3 }

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999 // round up to microsecond
	tv.Sec = Timeval_sec_t(nsec / 1e9)
	tv.Usec = Timeval_usec_t(nsec % 1e9 / 1e3)
	return
}

//sysnb	Tcgetattr(fd int, p *Termios) (err error)
//tcgetattr(fd _C_int, p *Termios) _C_int

//sys	Tcsetattr(fd int, actions int, p *Termios) (err error)
//tcsetattr(fd _C_int, actions _C_int, p *Termios) _C_int
