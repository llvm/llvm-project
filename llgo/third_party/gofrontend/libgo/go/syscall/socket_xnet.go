// socket_xnet.go -- Socket handling specific to Solaris.
// Enforce use of XPG 4.2 versions of socket functions.

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

//sys	bind(fd int, sa *RawSockaddrAny, len Socklen_t) (err error)
//__xnet_bind(fd _C_int, sa *RawSockaddrAny, len Socklen_t) _C_int

//sys	connect(s int, addr *RawSockaddrAny, addrlen Socklen_t) (err error)
//__xnet_connect(s _C_int, addr *RawSockaddrAny, addrlen Socklen_t) _C_int

//sysnb	socket(domain int, typ int, proto int) (fd int, err error)
//__xnet_socket(domain _C_int, typ _C_int, protocol _C_int) _C_int

//sysnb	socketpair(domain int, typ int, proto int, fd *[2]_C_int) (err error)
//__xnet_socketpair(domain _C_int, typ _C_int, protocol _C_int, fd *[2]_C_int) _C_int

//sys	getsockopt(s int, level int, name int, val unsafe.Pointer, vallen *Socklen_t) (err error)
//__xnet_getsockopt(s _C_int, level _C_int, name _C_int, val *byte, vallen *Socklen_t) _C_int

//sys	sendto(s int, buf []byte, flags int, to *RawSockaddrAny, tolen Socklen_t) (err error)
//__xnet_sendto(s _C_int, buf *byte, len Size_t, flags _C_int, to *RawSockaddrAny, tolen Socklen_t) Ssize_t

//sys	recvmsg(s int, msg *Msghdr, flags int) (n int, err error)
//__xnet_recvmsg(s _C_int, msg *Msghdr, flags _C_int) Ssize_t

//sys	sendmsg(s int, msg *Msghdr, flags int) (n int, err error)
//__xnet_sendmsg(s _C_int, msg *Msghdr, flags _C_int) Ssize_t
