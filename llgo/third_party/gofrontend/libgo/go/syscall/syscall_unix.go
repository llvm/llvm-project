// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package syscall

import (
	"runtime"
	"sync"
	"unsafe"
)

var (
	Stdin  = 0
	Stdout = 1
	Stderr = 2
)

//extern syscall
func c_syscall32(trap int32, a1, a2, a3, a4, a5, a6 int32) int32

//extern syscall
func c_syscall64(trap int64, a1, a2, a3, a4, a5, a6 int64) int64

const (
	darwin64Bit    = runtime.GOOS == "darwin" && sizeofPtr == 8
	dragonfly64Bit = runtime.GOOS == "dragonfly" && sizeofPtr == 8
	netbsd32Bit    = runtime.GOOS == "netbsd" && sizeofPtr == 4
)

// Do a system call.  We look at the size of uintptr to see how to pass
// the arguments, so that we don't pass a 64-bit value when the function
// expects a 32-bit one.
func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	Entersyscall()
	SetErrno(0)
	var r uintptr
	if unsafe.Sizeof(r) == 4 {
		r1 := c_syscall32(int32(trap), int32(a1), int32(a2), int32(a3), 0, 0, 0)
		r = uintptr(r1)
	} else {
		r1 := c_syscall64(int64(trap), int64(a1), int64(a2), int64(a3), 0, 0, 0)
		r = uintptr(r1)
	}
	err = GetErrno()
	Exitsyscall()
	return r, 0, err
}

func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	Entersyscall()
	SetErrno(0)
	var r uintptr
	if unsafe.Sizeof(r) == 4 {
		r1 := c_syscall32(int32(trap), int32(a1), int32(a2), int32(a3),
			int32(a4), int32(a5), int32(a6))
		r = uintptr(r1)
	} else {
		r1 := c_syscall64(int64(trap), int64(a1), int64(a2), int64(a3),
			int64(a4), int64(a5), int64(a6))
		r = uintptr(r1)
	}
	err = GetErrno()
	Exitsyscall()
	return r, 0, err
}

func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno) {
	var r uintptr
	SetErrno(0)
	if unsafe.Sizeof(r) == 4 {
		r1 := c_syscall32(int32(trap), int32(a1), int32(a2), int32(a3), 0, 0, 0)
		r = uintptr(r1)
	} else {
		r1 := c_syscall64(int64(trap), int64(a1), int64(a2), int64(a3), 0, 0, 0)
		r = uintptr(r1)
	}
	err = GetErrno()
	return r, 0, err
}

func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno) {
	var r uintptr
	SetErrno(0)
	if unsafe.Sizeof(r) == 4 {
		r1 := c_syscall32(int32(trap), int32(a1), int32(a2), int32(a3),
			int32(a4), int32(a5), int32(a6))
		r = uintptr(r1)
	} else {
		r1 := c_syscall64(int64(trap), int64(a1), int64(a2), int64(a3),
			int64(a4), int64(a5), int64(a6))
		r = uintptr(r1)
	}
	err = GetErrno()
	return r, 0, err
}

// Mmap manager, for use by operating system-specific implementations.
// Gccgo only has one implementation but we do this to correspond to gc.

type mmapper struct {
	sync.Mutex
	active map[*byte][]byte // active mappings; key is last byte in mapping
	mmap   func(addr, length uintptr, prot, flags, fd int, offset int64) (uintptr, error)
	munmap func(addr uintptr, length uintptr) error
}

func (m *mmapper) Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	if length <= 0 {
		return nil, EINVAL
	}

	// Map the requested memory.
	addr, errno := m.mmap(0, uintptr(length), prot, flags, fd, offset)
	if errno != nil {
		return nil, errno
	}

	// Slice memory layout
	var sl = struct {
		addr uintptr
		len  int
		cap  int
	}{addr, length, length}

	// Use unsafe to turn sl into a []byte.
	b := *(*[]byte)(unsafe.Pointer(&sl))

	// Register mapping in m and return it.
	p := &b[cap(b)-1]
	m.Lock()
	defer m.Unlock()
	m.active[p] = b
	return b, nil
}

func (m *mmapper) Munmap(data []byte) (err error) {
	if len(data) == 0 || len(data) != cap(data) {
		return EINVAL
	}

	// Find the base of the mapping.
	p := &data[cap(data)-1]
	m.Lock()
	defer m.Unlock()
	b := m.active[p]
	if b == nil || &b[0] != &data[0] {
		return EINVAL
	}

	// Unmap the memory and update m.
	if errno := m.munmap(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b))); errno != nil {
		return errno
	}
	delete(m.active, p)
	return nil
}

var mapper = &mmapper{
	active: make(map[*byte][]byte),
	mmap:   mmap,
	munmap: munmap,
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return mapper.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return mapper.Munmap(b)
}

// Do the interface allocations only once for common
// Errno values.
var (
	errEAGAIN error = EAGAIN
	errEINVAL error = EINVAL
	errENOENT error = ENOENT
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e Errno) error {
	switch e {
	case 0:
		return nil
	case EAGAIN:
		return errEAGAIN
	case EINVAL:
		return errEINVAL
	case ENOENT:
		return errENOENT
	}
	return e
}

// A Signal is a number describing a process signal.
// It implements the os.Signal interface.
type Signal int

func (s Signal) Signal() {}

func Signame(s Signal) string

func (s Signal) String() string {
	return Signame(s)
}

func Read(fd int, p []byte) (n int, err error) {
	n, err = read(fd, p)
	if raceenabled {
		if n > 0 {
			raceWriteRange(unsafe.Pointer(&p[0]), n)
		}
		if err == nil {
			raceAcquire(unsafe.Pointer(&ioSync))
		}
	}
	return
}

func Write(fd int, p []byte) (n int, err error) {
	if raceenabled {
		raceReleaseMerge(unsafe.Pointer(&ioSync))
	}
	n, err = write(fd, p)
	if raceenabled && n > 0 {
		raceReadRange(unsafe.Pointer(&p[0]), n)
	}
	return
}

var ioSync int64
