// socket_linux_type.go -- Socket handling specific to GNU/Linux.

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// Type needed if not on ppc64le or ppc64

type RawSockaddr struct {
	Family uint16
	Data   [14]int8
}
