// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Waiting for FDs via select(2).

package net

import (
	"errors"
	"os"
	"syscall"
)

type pollster struct {
	readFds, writeFds, repeatFds *syscall.FdSet
	maxFd                        int
	readyReadFds, readyWriteFds  *syscall.FdSet
	nReady                       int
	lastFd                       int
	closed                       bool
}

func newpollster() (p *pollster, err error) {
	p = new(pollster)
	p.readFds = new(syscall.FdSet)
	p.writeFds = new(syscall.FdSet)
	p.repeatFds = new(syscall.FdSet)
	p.readyReadFds = new(syscall.FdSet)
	p.readyWriteFds = new(syscall.FdSet)
	p.maxFd = -1
	p.nReady = 0
	p.lastFd = 0
	return p, nil
}

func (p *pollster) AddFD(fd int, mode int, repeat bool) (bool, error) {
	// pollServer is locked.

	if p.closed {
		return false, errors.New("pollster closed")
	}

	if mode == 'r' {
		syscall.FDSet(fd, p.readFds)
	} else {
		syscall.FDSet(fd, p.writeFds)
	}

	if repeat {
		syscall.FDSet(fd, p.repeatFds)
	}

	if fd > p.maxFd {
		p.maxFd = fd
	}

	return true, nil
}

func (p *pollster) DelFD(fd int, mode int) bool {
	// pollServer is locked.

	if p.closed {
		return false
	}

	if mode == 'r' {
		if !syscall.FDIsSet(fd, p.readFds) {
			print("Select unexpected fd=", fd, " for read\n")
			return false
		}
		syscall.FDClr(fd, p.readFds)
	} else {
		if !syscall.FDIsSet(fd, p.writeFds) {
			print("Select unexpected fd=", fd, " for write\n")
			return false
		}
		syscall.FDClr(fd, p.writeFds)
	}

	// Doesn't matter if not already present.
	syscall.FDClr(fd, p.repeatFds)

	// We don't worry about maxFd here.

	return true
}

func (p *pollster) WaitFD(s *pollServer, nsec int64) (fd int, mode int, err error) {
	if p.nReady == 0 {
		var timeout *syscall.Timeval
		var tv syscall.Timeval
		timeout = nil
		if nsec > 0 {
			tv = syscall.NsecToTimeval(nsec)
			timeout = &tv
		}

		var n int
		var e error
		var tmpReadFds, tmpWriteFds syscall.FdSet
		for {
			if p.closed {
				return -1, 0, errors.New("pollster closed")
			}

			// Temporary syscall.FdSet's into which the values are copied
			// because select mutates the values.
			tmpReadFds = *p.readFds
			tmpWriteFds = *p.writeFds

			s.Unlock()
			n, e = syscall.Select(p.maxFd+1, &tmpReadFds, &tmpWriteFds, nil, timeout)
			s.Lock()

			if e != syscall.EINTR {
				break
			}
		}
		if e == syscall.EBADF {
			// Some file descriptor has been closed.
			tmpReadFds = syscall.FdSet{}
			tmpWriteFds = syscall.FdSet{}
			n = 0
			for i := 0; i < p.maxFd+1; i++ {
				if syscall.FDIsSet(i, p.readFds) {
					var s syscall.Stat_t
					if syscall.Fstat(i, &s) == syscall.EBADF {
						syscall.FDSet(i, &tmpReadFds)
						n++
					}
				} else if syscall.FDIsSet(i, p.writeFds) {
					var s syscall.Stat_t
					if syscall.Fstat(i, &s) == syscall.EBADF {
						syscall.FDSet(i, &tmpWriteFds)
						n++
					}
				}
			}
		} else if e != nil {
			return -1, 0, os.NewSyscallError("select", e)
		}
		if n == 0 {
			return -1, 0, nil
		}

		p.nReady = n
		*p.readyReadFds = tmpReadFds
		*p.readyWriteFds = tmpWriteFds
		p.lastFd = 0
	}

	flag := false
	for i := p.lastFd; i < p.maxFd+1; i++ {
		if syscall.FDIsSet(i, p.readyReadFds) {
			flag = true
			mode = 'r'
			syscall.FDClr(i, p.readyReadFds)
		} else if syscall.FDIsSet(i, p.readyWriteFds) {
			flag = true
			mode = 'w'
			syscall.FDClr(i, p.readyWriteFds)
		}
		if flag {
			if !syscall.FDIsSet(i, p.repeatFds) {
				p.DelFD(i, mode)
			}
			p.nReady--
			p.lastFd = i
			return i, mode, nil
		}
	}

	// Will not reach here.  Just to shut up the compiler.
	return -1, 0, nil
}

func (p *pollster) Close() error {
	p.closed = true
	return nil
}
