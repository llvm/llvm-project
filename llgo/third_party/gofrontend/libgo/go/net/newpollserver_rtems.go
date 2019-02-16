// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
)

func selfConnectedTCPSocket() (pr, pw *os.File, err error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	sockfd, e := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, 0)
	if e != 0 {
		syscall.ForkLock.RUnlock()
		return nil, nil, os.Errno(e)
	}
	syscall.CloseOnExec(sockfd)
	syscall.ForkLock.RUnlock()

	// Allow reuse of recently-used addresses.
	syscall.SetsockoptInt(sockfd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)

	var laTCP *TCPAddr
	var la syscall.Sockaddr
	if laTCP, err = ResolveTCPAddr("127.0.0.1:0"); err != nil {
	Error:
		return nil, nil, err
	}
	if la, err = laTCP.sockaddr(syscall.AF_INET); err != nil {
		goto Error
	}
	e = syscall.Bind(sockfd, la)
	if e != 0 {
	Errno:
		syscall.Close(sockfd)
		return nil, nil, os.Errno(e)
	}

	laddr, _ := syscall.Getsockname(sockfd)
	e = syscall.Connect(sockfd, laddr)
	if e != 0 {
		goto Errno
	}

	fd := os.NewFile(sockfd, "wakeupSocket")
	return fd, fd, nil
}

func newPollServer() (s *pollServer, err error) {
	s = new(pollServer)
	s.cr = make(chan *netFD, 1)
	s.cw = make(chan *netFD, 1)
	// s.pr and s.pw are indistinguishable.
	if s.pr, s.pw, err = selfConnectedTCPSocket(); err != nil {
		return nil, err
	}
	var e int
	if e = syscall.SetNonblock(s.pr.Fd(), true); e != 0 {
	Errno:
		err = &os.PathError{"setnonblock", s.pr.Name(), os.Errno(e)}
	Error:
		s.pr.Close()
		return nil, err
	}
	if s.poll, err = newpollster(); err != nil {
		goto Error
	}
	if _, err = s.poll.AddFD(s.pr.Fd(), 'r', true); err != nil {
		s.poll.Close()
		goto Error
	}
	s.pending = make(map[int]*netFD)
	go s.Run()
	return s, nil
}
