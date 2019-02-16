// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// 64-bit ptrace(3C) doesn't exist
func raw_ptrace(request int, pid int, addr *byte, data *byte) Errno {
	return ENOSYS
}
