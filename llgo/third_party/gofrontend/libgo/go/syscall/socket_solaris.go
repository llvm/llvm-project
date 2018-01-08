// socket_solaris.go -- Socket handling specific to Solaris.

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const SizeofSockaddrInet4 = 16
const SizeofSockaddrInet6 = 32
const SizeofSockaddrUnix = 110

type RawSockaddrInet4 struct {
	Family uint16
	Port   uint16
	Addr   [4]byte /* in_addr */
	Zero   [8]uint8
}

func (sa *RawSockaddrInet4) setLen() Socklen_t {
	return SizeofSockaddrInet4
}

type RawSockaddrInet6 struct {
	Family   uint16
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte /* in6_addr */
	Scope_id uint32
	Src_id   uint32
}

func (sa *RawSockaddrInet6) setLen() Socklen_t {
	return SizeofSockaddrInet6
}

type RawSockaddrUnix struct {
	Family uint16
	Path   [108]int8
}

func (sa *RawSockaddrUnix) setLen(int) {
}

func (sa *RawSockaddrUnix) getLen() (int, error) {
	n := 0
	for n < len(sa.Path) && sa.Path[n] != 0 {
		n++
	}
	return n, nil
}

func (sa *RawSockaddrUnix) adjustAbstract(sl Socklen_t) Socklen_t {
	return sl
}

type RawSockaddr struct {
	Family uint16
	Data   [14]int8
}

// BindToDevice binds the socket associated with fd to device.
func BindToDevice(fd int, device string) (err error) {
	return ENOSYS
}

func anyToSockaddrOS(rsa *RawSockaddrAny) (Sockaddr, error) {
	return nil, EAFNOSUPPORT
}
