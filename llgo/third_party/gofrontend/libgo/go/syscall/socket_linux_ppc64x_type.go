// socket_linux_ppc64x_type.go -- Socket handling specific to ppc64 GNU/Linux.

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// Type needed on ppc64le & ppc64

type RawSockaddr struct {
	Family uint16
	Data   [14]uint8
}
