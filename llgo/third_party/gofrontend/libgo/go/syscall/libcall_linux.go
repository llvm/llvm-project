// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GNU/Linux library calls.

package syscall

import "unsafe"

//sys	Openat(dirfd int, path string, flags int, mode uint32) (fd int, err error)
//__go_openat(dirfd _C_int, path *byte, flags _C_int, mode Mode_t) _C_int

//sys	futimesat(dirfd int, path *byte, times *[2]Timeval) (err error)
//futimesat(dirfd _C_int, path *byte, times *[2]Timeval) _C_int
func Futimesat(dirfd int, path string, tv []Timeval) (err error) {
	if len(tv) != 2 {
		return EINVAL
	}
	return futimesat(dirfd, StringBytePtr(path), (*[2]Timeval)(unsafe.Pointer(&tv[0])))
}

func Futimes(fd int, tv []Timeval) (err error) {
	// Believe it or not, this is the best we can do on GNU/Linux
	// (and is what glibc does).
	return Utimes("/proc/self/fd/"+itoa(fd), tv)
}

//sys	ptrace(request int, pid int, addr uintptr, data uintptr) (err error)
//ptrace(request _C_int, pid Pid_t, addr *byte, data *byte) _C_long

//sysnb raw_ptrace(request int, pid int, addr *byte, data *byte) (err Errno)
//ptrace(request _C_int, pid Pid_t, addr *byte, data *byte) _C_long

func ptracePeek(req int, pid int, addr uintptr, out []byte) (count int, err error) {
	// The peek requests are machine-size oriented, so we wrap it
	// to retrieve arbitrary-length data.

	// The ptrace syscall differs from glibc's ptrace.
	// Peeks returns the word in *data, not as the return value.

	var buf [sizeofPtr]byte

	// Leading edge.  PEEKTEXT/PEEKDATA don't require aligned
	// access (PEEKUSER warns that it might), but if we don't
	// align our reads, we might straddle an unmapped page
	// boundary and not get the bytes leading up to the page
	// boundary.
	n := 0
	if addr%sizeofPtr != 0 {
		err = ptrace(req, pid, addr-addr%sizeofPtr, uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return 0, err
		}
		n += copy(out, buf[addr%sizeofPtr:])
		out = out[n:]
	}

	// Remainder.
	for len(out) > 0 {
		// We use an internal buffer to gaurantee alignment.
		// It's not documented if this is necessary, but we're paranoid.
		err = ptrace(req, pid, addr+uintptr(n), uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return n, err
		}
		copied := copy(out, buf[0:])
		n += copied
		out = out[copied:]
	}

	return n, nil
}

func PtracePeekText(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PTRACE_PEEKTEXT, pid, addr, out)
}

func PtracePeekData(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PTRACE_PEEKDATA, pid, addr, out)
}

func ptracePoke(pokeReq int, peekReq int, pid int, addr uintptr, data []byte) (count int, err error) {
	// As for ptracePeek, we need to align our accesses to deal
	// with the possibility of straddling an invalid page.

	// Leading edge.
	n := 0
	if addr%sizeofPtr != 0 {
		var buf [sizeofPtr]byte
		err = ptrace(peekReq, pid, addr-addr%sizeofPtr, uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return 0, err
		}
		n += copy(buf[addr%sizeofPtr:], data)
		word := *((*uintptr)(unsafe.Pointer(&buf[0])))
		err = ptrace(pokeReq, pid, addr-addr%sizeofPtr, word)
		if err != nil {
			return 0, err
		}
		data = data[n:]
	}

	// Interior.
	for len(data) > int(sizeofPtr) {
		word := *((*uintptr)(unsafe.Pointer(&data[0])))
		err = ptrace(pokeReq, pid, addr+uintptr(n), word)
		if err != nil {
			return n, err
		}
		n += int(sizeofPtr)
		data = data[sizeofPtr:]
	}

	// Trailing edge.
	if len(data) > 0 {
		var buf [sizeofPtr]byte
		err = ptrace(peekReq, pid, addr+uintptr(n), uintptr(unsafe.Pointer(&buf[0])))
		if err != nil {
			return n, err
		}
		copy(buf[0:], data)
		word := *((*uintptr)(unsafe.Pointer(&buf[0])))
		err = ptrace(pokeReq, pid, addr+uintptr(n), word)
		if err != nil {
			return n, err
		}
		n += len(data)
	}

	return n, nil
}

func PtracePokeText(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PTRACE_POKETEXT, PTRACE_PEEKTEXT, pid, addr, data)
}

func PtracePokeData(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PTRACE_POKEDATA, PTRACE_PEEKDATA, pid, addr, data)
}

func PtraceSetOptions(pid int, options int) (err error) {
	return ptrace(PTRACE_SETOPTIONS, pid, 0, uintptr(options))
}

func PtraceGetEventMsg(pid int) (msg uint, err error) {
	var data _C_long
	err = ptrace(PTRACE_GETEVENTMSG, pid, 0, uintptr(unsafe.Pointer(&data)))
	msg = uint(data)
	return
}

func PtraceCont(pid int, signal int) (err error) {
	return ptrace(PTRACE_CONT, pid, 0, uintptr(signal))
}

func PtraceSingleStep(pid int) (err error) { return ptrace(PTRACE_SINGLESTEP, pid, 0, 0) }

func PtraceAttach(pid int) (err error) { return ptrace(PTRACE_ATTACH, pid, 0, 0) }

func PtraceDetach(pid int) (err error) { return ptrace(PTRACE_DETACH, pid, 0, 0) }

//sys	reboot(magic1 uint, magic2 uint, cmd int, arg string) (err error)
//reboot(magic1 _C_uint, magic2 _C_uint, cmd _C_int, arg *byte) _C_int
func Reboot(cmd int) (err error) {
	return reboot(LINUX_REBOOT_MAGIC1, LINUX_REBOOT_MAGIC2, cmd, "")
}

//sys	accept4(fd int, sa *RawSockaddrAny, len *Socklen_t, flags int) (nfd int, err error)
//accept4(fd _C_int, sa *RawSockaddrAny, len *Socklen_t, flags _C_int) _C_int

func Accept4(fd int, flags int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len Socklen_t = SizeofSockaddrAny
	nfd, err = accept4(fd, &rsa, &len, flags)
	if err != nil {
		return -1, nil, err
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		return -1, nil, err
	}
	return nfd, sa, nil
}

//sys	Acct(path string) (err error)
//acct(path *byte) _C_int

//sys	Adjtimex(buf *Timex) (state int, err error)
//adjtimex(buf *Timex) _C_int

//sysnb	Dup3(oldfd int, newfd int, flags int) (err error)
//dup3(oldfd _C_int, newfd _C_int, flags _C_int) _C_int

//sys	Faccessat(dirfd int, path string, mode uint32, flags int) (err error)
//faccessat(dirfd _C_int, pathname *byte, mode _C_int, flags _C_int) _C_int

//sys	Fallocate(fd int, mode uint32, off int64, len int64) (err error)
//fallocate(fd _C_int, mode _C_int, offset Offset_t, len Offset_t) _C_int

//sys	Fchmodat(dirfd int, path string, mode uint32, flags int) (err error)
//fchmodat(dirfd _C_int, pathname *byte, mode Mode_t, flags _C_int) _C_int

//sys	Fchownat(dirfd int, path string, uid int, gid int, flags int) (err error)
//fchownat(dirfd _C_int, path *byte, owner Uid_t, group Gid_t, flags _C_int) _C_int

//sys	Flock(fd int, how int) (err error)
//flock(fd _C_int, how _C_int) _C_int

//sys	Fstatfs(fd int, buf *Statfs_t) (err error)
//fstatfs(fd _C_int, buf *Statfs_t) _C_int

func Gettid() (tid int) {
	r1, _, _ := Syscall(SYS_GETTID, 0, 0, 0)
	return int(r1)
}

func Getdents(fd int, buf []byte) (n int, err error) {
	var p *byte
	if len(buf) > 0 {
		p = &buf[0]
	} else {
		p = (*byte)(unsafe.Pointer(&_zero))
	}
	s := SYS_GETDENTS64
	if s == 0 {
		s = SYS_GETDENTS
	}
	r1, _, errno := Syscall(uintptr(s), uintptr(fd), uintptr(unsafe.Pointer(p)), uintptr(len(buf)))
	n = int(r1)
	if n < 0 {
		err = errno
	}
	return
}

func clen(n []byte) int {
	for i := 0; i < len(n); i++ {
		if n[i] == 0 {
			return i
		}
	}
	return len(n)
}

func ReadDirent(fd int, buf []byte) (n int, err error) {
	return Getdents(fd, buf)
}

func ParseDirent(buf []byte, max int, names []string) (consumed int, count int, newnames []string) {
	origlen := len(buf)
	count = 0
	for max != 0 && len(buf) > 0 {
		dirent := (*Dirent)(unsafe.Pointer(&buf[0]))
		buf = buf[dirent.Reclen:]
		if dirent.Ino == 0 { // File absent in directory.
			continue
		}
		bytes := (*[10000]byte)(unsafe.Pointer(&dirent.Name[0]))
		var name = string(bytes[0:clen(bytes[:])])
		if name == "." || name == ".." { // Useless names
			continue
		}
		max--
		count++
		names = append(names, name)
	}
	return origlen - len(buf), count, names
}

//sys	Getxattr(path string, attr string, dest []byte) (sz int, err error)
//getxattr(path *byte, attr *byte, buf *byte, count Size_t) Ssize_t

//sys	InotifyAddWatch(fd int, pathname string, mask uint32) (watchdesc int, err error)
//inotify_add_watch(fd _C_int, pathname *byte, mask uint32) _C_int

//sysnb	InotifyInit() (fd int, err error)
//inotify_init() _C_int

//sysnb	InotifyInit1(flags int) (fd int, err error)
//inotify_init1(flags _C_int) _C_int

//sysnb	InotifyRmWatch(fd int, watchdesc uint32) (success int, err error)
//inotify_rm_watch(fd _C_int, wd uint32) _C_int

//sys	Klogctl(typ int, buf []byte) (n int, err error)
//klogctl(typ _C_int, bufp *byte, len _C_int) _C_int

//sys	Listxattr(path string, dest []byte) (sz int, err error)
//listxattr(path *byte, list *byte, size Size_t) Ssize_t

//sys	Mkdirat(dirfd int, path string, mode uint32) (err error)
//mkdirat(dirfd _C_int, path *byte, mode Mode_t) _C_int

//sys	Mknodat(dirfd int, path string, mode uint32, dev int) (err error)
//mknodat(dirfd _C_int, path *byte, mode Mode_t, dev _dev_t) _C_int

//sysnb	pipe2(p *[2]_C_int, flags int) (err error)
//pipe2(p *[2]_C_int, flags _C_int) _C_int
func Pipe2(p []int, flags int) (err error) {
	if len(p) != 2 {
		return EINVAL
	}
	var pp [2]_C_int
	err = pipe2(&pp, flags)
	p[0] = int(pp[0])
	p[1] = int(pp[1])
	return
}

//sys	PivotRoot(newroot string, putold string) (err error)
//pivot_root(newroot *byte, putold *byte) _C_int

//sys	Removexattr(path string, attr string) (err error)
//removexattr(path *byte, name *byte) _C_int

//sys	Renameat(olddirfd int, oldpath string, newdirfd int, newpath string) (err error)
//renameat(olddirfd _C_int, oldpath *byte, newdirfd _C_int, newpath *byte) _C_int

//sys	sendfile(outfd int, infd int, offset *Offset_t, count int) (written int, err error)
//sendfile64(outfd _C_int, infd _C_int, offset *Offset_t, count Size_t) Ssize_t
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

//sys	Setfsgid(gid int) (err error)
//setfsgid(gid Gid_t) _C_int

//sys	Setfsuid(uid int) (err error)
//setfsuid(uid Uid_t) _C_int

//sysnb	Setresgid(rgid int, egid int, sgid int) (err error)
//setresgid(rgid Gid_t, egid Gid_t, sgid Gid_t) _C_int

//sysnb	Setresuid(ruid int, eguid int, suid int) (err error)
//setresuid(ruid Uid_t, euid Uid_t, suid Uid_t) _C_int

//sys	Setxattr(path string, attr string, data []byte, flags int) (err error)
//setxattr(path *byte, name *byte, value *byte, size Size_t, flags _C_int) _C_int

//sys	splice(rfd int, roff *_loff_t, wfd int, woff *_loff_t, len int, flags int) (n int64, err error)
//splice(rfd _C_int, roff *_loff_t, wfd _C_int, woff *_loff_t, len Size_t, flags _C_uint) Ssize_t
func Splice(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int64, err error) {
	var lroff _loff_t
	var plroff *_loff_t
	if roff != nil {
		lroff = _loff_t(*roff)
		plroff = &lroff
	}
	var lwoff _loff_t
	var plwoff *_loff_t
	if woff != nil {
		lwoff = _loff_t(*woff)
		plwoff = &lwoff
	}
	n, err = splice(rfd, plroff, wfd, plwoff, len, flags)
	if roff != nil {
		*roff = int64(lroff)
	}
	if woff != nil {
		*woff = int64(lwoff)
	}
	return
}

//sys	Statfs(path string, buf *Statfs_t) (err error)
//statfs(path *byte, buf *Statfs_t) _C_int

//sys	SyncFileRange(fd int, off int64, n int64, flags int) (err error)
//sync_file_range(fd _C_int, off Offset_t, n Offset_t, flags _C_uint) _C_int

//sysnb	Sysinfo(info *Sysinfo_t) (err error)
//sysinfo(info *Sysinfo_t) _C_int

//sys	Tee(rfd int, wfd int, len int, flags int) (n int64, err error)
//tee(rfd _C_int, wfd _C_int, len Size_t, flags _C_uint) Ssize_t

func Tgkill(tgid int, tid int, sig Signal) error {
	r1, _, errno := Syscall(SYS_TGKILL, uintptr(tgid), uintptr(tid), uintptr(sig))
	if r1 < 0 {
		return errno
	}
	return nil
}

//sys	unlinkat(dirfd int, path string, flags int) (err error)
//unlinkat(dirfd _C_int, path *byte, flags _C_int) _C_int

func Unlinkat(dirfd int, path string) (err error) {
	return unlinkat(dirfd, path, 0)
}

//sys	Unmount(target string, flags int) (err error) = SYS_UMOUNT2
//umount2(target *byte, flags _C_int) _C_int

//sys	Unshare(flags int) (err error)
//unshare(flags _C_int) _C_int
