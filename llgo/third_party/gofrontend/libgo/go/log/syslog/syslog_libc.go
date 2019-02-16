// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo specific implementation of syslog for Solaris.  Solaris uses
// STREAMS to communicate with syslogd.  That is enough of a pain that
// we just call the libc function.

package syslog

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

func unixSyslog() (conn serverConn, err error) {
	return libcConn(0), nil
}

type libcConn int

func syslog_c(int, *byte)

func (libcConn) writeString(p Priority, hostname, tag, msg, nl string) error {
	timestamp := time.Now().Format(time.RFC3339)
	log := fmt.Sprintf("%s %s %s[%d]: %s%s", timestamp, hostname, tag, os.Getpid(), msg, nl)
	buf, err := syscall.BytePtrFromString(log)
	if err != nil {
		return err
	}
	syscall.Entersyscall()
	syslog_c(int(p), buf)
	syscall.Exitsyscall()
	return nil
}

func (libcConn) close() error {
	return nil
}
