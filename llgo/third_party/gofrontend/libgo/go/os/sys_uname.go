// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For systems which only store the hostname in uname (Solaris).

package os

import "syscall"

func hostname() (name string, err error) {
	var u syscall.Utsname
	if errno := syscall.Uname(&u); errno != nil {
		return "", NewSyscallError("uname", errno)
	}
	b := make([]byte, len(u.Nodename))
	i := 0
	for ; i < len(u.Nodename); i++ {
		if u.Nodename[i] == 0 {
			break
		}
		b[i] = byte(u.Nodename[i])
	}
	return string(b[:i]), nil
}
