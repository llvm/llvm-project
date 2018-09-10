// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// 32-bit Solaris 2/x86 needs to use _nuname internally, cf. <sys/utsname.h>.
//sysnb	Uname(buf *Utsname) (err error)
//_nuname(buf *Utsname) _C_int

//sysnb raw_ptrace(request int, pid int, addr *byte, data *byte) (err Errno)
//ptrace(request _C_int, pid Pid_t, addr *byte, data *byte) _C_long
