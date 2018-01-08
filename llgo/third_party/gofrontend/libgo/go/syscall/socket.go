// socket.go -- Socket handling.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Low-level socket interface.
// Only for implementing net package.
// DO NOT USE DIRECTLY.

package syscall

import "unsafe"

// For testing: clients can set this flag to force
// creation of IPv6 sockets to return EAFNOSUPPORT.
var SocketDisableIPv6 bool

type Sockaddr interface {
	sockaddr() (ptr *RawSockaddrAny, len Socklen_t, err error) // lowercase; only we can define Sockaddrs
}

type RawSockaddrAny struct {
	Addr RawSockaddr
	Pad  [96]int8
}

const SizeofSockaddrAny = 0x6c

type SockaddrInet4 struct {
	Port int
	Addr [4]byte
	raw  RawSockaddrInet4
}

func (sa *SockaddrInet4) sockaddr() (*RawSockaddrAny, Socklen_t, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET
	n := sa.raw.setLen()
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return (*RawSockaddrAny)(unsafe.Pointer(&sa.raw)), n, nil
}

type SockaddrInet6 struct {
	Port   int
	ZoneId uint32
	Addr   [16]byte
	raw    RawSockaddrInet6
}

func (sa *SockaddrInet6) sockaddr() (*RawSockaddrAny, Socklen_t, error) {
	if sa.Port < 0 || sa.Port > 0xFFFF {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_INET6
	n := sa.raw.setLen()
	p := (*[2]byte)(unsafe.Pointer(&sa.raw.Port))
	p[0] = byte(sa.Port >> 8)
	p[1] = byte(sa.Port)
	sa.raw.Scope_id = sa.ZoneId
	for i := 0; i < len(sa.Addr); i++ {
		sa.raw.Addr[i] = sa.Addr[i]
	}
	return (*RawSockaddrAny)(unsafe.Pointer(&sa.raw)), n, nil
}

type SockaddrUnix struct {
	Name string
	raw  RawSockaddrUnix
}

func (sa *SockaddrUnix) sockaddr() (*RawSockaddrAny, Socklen_t, error) {
	name := sa.Name
	n := len(name)
	if n >= len(sa.raw.Path) {
		return nil, 0, EINVAL
	}
	sa.raw.Family = AF_UNIX
	sa.raw.setLen(n)
	for i := 0; i < n; i++ {
		sa.raw.Path[i] = int8(name[i])
	}
	// length is family (uint16), name, NUL.
	sl := Socklen_t(2)
	if n > 0 {
		sl += Socklen_t(n) + 1
	}
	sl = sa.raw.adjustAbstract(sl)

	// length is family (uint16), name, NUL.
	return (*RawSockaddrAny)(unsafe.Pointer(&sa.raw)), sl, nil
}

func anyToSockaddr(rsa *RawSockaddrAny) (Sockaddr, error) {
	switch rsa.Addr.Family {
	case AF_UNIX:
		pp := (*RawSockaddrUnix)(unsafe.Pointer(rsa))
		sa := new(SockaddrUnix)
		n, err := pp.getLen()
		if err != nil {
			return nil, err
		}
		bytes := (*[len(pp.Path)]byte)(unsafe.Pointer(&pp.Path[0]))
		sa.Name = string(bytes[0:n])
		return sa, nil

	case AF_INET:
		pp := (*RawSockaddrInet4)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet4)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, nil

	case AF_INET6:
		pp := (*RawSockaddrInet6)(unsafe.Pointer(rsa))
		sa := new(SockaddrInet6)
		p := (*[2]byte)(unsafe.Pointer(&pp.Port))
		sa.Port = int(p[0])<<8 + int(p[1])
		for i := 0; i < len(sa.Addr); i++ {
			sa.Addr[i] = pp.Addr[i]
		}
		return sa, nil
	}
	return anyToSockaddrOS(rsa)
}

//sys	accept(fd int, sa *RawSockaddrAny, len *Socklen_t) (nfd int, err error)
//accept(fd _C_int, sa *RawSockaddrAny, len *Socklen_t) _C_int

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len Socklen_t = SizeofSockaddrAny
	nfd, err = accept(fd, &rsa, &len)
	if err != nil {
		return
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

//sysnb	getsockname(fd int, sa *RawSockaddrAny, len *Socklen_t) (err error)
//getsockname(fd _C_int, sa *RawSockaddrAny, len *Socklen_t) _C_int

func Getsockname(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len Socklen_t = SizeofSockaddrAny
	if err = getsockname(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(&rsa)
}

//sysnb getpeername(fd int, sa *RawSockaddrAny, len *Socklen_t) (err error)
//getpeername(fd _C_int, sa *RawSockaddrAny, len *Socklen_t) _C_int

func Getpeername(fd int) (sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len Socklen_t = SizeofSockaddrAny
	if err = getpeername(fd, &rsa, &len); err != nil {
		return
	}
	return anyToSockaddr(&rsa)
}

func Bind(fd int, sa Sockaddr) (err error) {
	ptr, n, err := sa.sockaddr()
	if err != nil {
		return err
	}
	return bind(fd, ptr, n)
}

func Connect(fd int, sa Sockaddr) (err error) {
	ptr, n, err := sa.sockaddr()
	if err != nil {
		return err
	}
	return connect(fd, ptr, n)
}

func Socket(domain, typ, proto int) (fd int, err error) {
	if domain == AF_INET6 && SocketDisableIPv6 {
		return -1, EAFNOSUPPORT
	}
	fd, err = socket(domain, typ, proto)
	return
}

func Socketpair(domain, typ, proto int) (fd [2]int, err error) {
	var fdx [2]_C_int
	err = socketpair(domain, typ, proto, &fdx)
	if err == nil {
		fd[0] = int(fdx[0])
		fd[1] = int(fdx[1])
	}
	return
}

func GetsockoptByte(fd, level, opt int) (value byte, err error) {
	var n byte
	vallen := Socklen_t(1)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return n, err
}

func GetsockoptInt(fd, level, opt int) (value int, err error) {
	var n int32
	vallen := Socklen_t(4)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&n), &vallen)
	return int(n), err
}

func GetsockoptInet4Addr(fd, level, opt int) (value [4]byte, err error) {
	vallen := Socklen_t(4)
	err = getsockopt(fd, level, opt, unsafe.Pointer(&value[0]), &vallen)
	return value, err
}

func GetsockoptIPMreq(fd, level, opt int) (*IPMreq, error) {
	var value IPMreq
	vallen := Socklen_t(SizeofIPMreq)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptIPMreqn(fd, level, opt int) (*IPMreqn, error) {
	var value IPMreqn
	vallen := Socklen_t(SizeofIPMreqn)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

func GetsockoptIPv6Mreq(fd, level, opt int) (*IPv6Mreq, error) {
	var value IPv6Mreq
	vallen := Socklen_t(SizeofIPv6Mreq)
	err := getsockopt(fd, level, opt, unsafe.Pointer(&value), &vallen)
	return &value, err
}

//sys	setsockopt(s int, level int, name int, val unsafe.Pointer, vallen Socklen_t) (err error)
//setsockopt(s _C_int, level _C_int, optname _C_int, val *byte, vallen Socklen_t) _C_int

func SetsockoptByte(fd, level, opt int, value byte) (err error) {
	var n = byte(value)
	return setsockopt(fd, level, opt, unsafe.Pointer(&n), 1)
}

func SetsockoptInt(fd, level, opt int, value int) (err error) {
	var n = int32(value)
	return setsockopt(fd, level, opt, unsafe.Pointer(&n), 4)
}

func SetsockoptInet4Addr(fd, level, opt int, value [4]byte) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(&value[0]), 4)
}

func SetsockoptTimeval(fd, level, opt int, tv *Timeval) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(tv), Socklen_t(unsafe.Sizeof(*tv)))
}

func SetsockoptICMPv6Filter(fd, level, opt int, filter *ICMPv6Filter) error {
	return setsockopt(fd, level, opt, unsafe.Pointer(filter), SizeofICMPv6Filter)
}

type Linger struct {
	Onoff  int32
	Linger int32
}

func SetsockoptLinger(fd, level, opt int, l *Linger) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(l), Socklen_t(unsafe.Sizeof(*l)))
}

func SetsockoptIPMreq(fd, level, opt int, mreq *IPMreq) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), Socklen_t(unsafe.Sizeof(*mreq)))
}

func SetsockoptIPMreqn(fd, level, opt int, mreq *IPMreqn) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), Socklen_t(unsafe.Sizeof(*mreq)))
}

func SetsockoptIPv6Mreq(fd, level, opt int, mreq *IPv6Mreq) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(mreq), Socklen_t(unsafe.Sizeof(*mreq)))
}

func SetsockoptString(fd, level, opt int, s string) (err error) {
	return setsockopt(fd, level, opt, unsafe.Pointer(&[]byte(s)[0]), Socklen_t(len(s)))
}

//sys	recvfrom(fd int, p []byte, flags int, from *RawSockaddrAny, fromlen *Socklen_t) (n int, err error)
//recvfrom(fd _C_int, buf *byte, len Size_t, flags _C_int, from *RawSockaddrAny, fromlen *Socklen_t) Ssize_t

func Recvfrom(fd int, p []byte, flags int) (n int, from Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len Socklen_t = SizeofSockaddrAny
	if n, err = recvfrom(fd, p, flags, &rsa, &len); err != nil {
		return
	}
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(&rsa)
	}
	return
}

func Sendto(fd int, p []byte, flags int, to Sockaddr) (err error) {
	ptr, n, err := to.sockaddr()
	if err != nil {
		return err
	}
	return sendto(fd, p, flags, ptr, n)
}

func Recvmsg(fd int, p, oob []byte, flags int) (n, oobn int, recvflags int, from Sockaddr, err error) {
	var msg Msghdr
	var rsa RawSockaddrAny
	msg.Name = (*byte)(unsafe.Pointer(&rsa))
	msg.Namelen = uint32(SizeofSockaddrAny)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = (*byte)(unsafe.Pointer(&p[0]))
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// receive at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = (*byte)(unsafe.Pointer(&oob[0]))
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = recvmsg(fd, &msg, flags); err != nil {
		return
	}
	oobn = int(msg.Controllen)
	recvflags = int(msg.Flags)
	// source address is only specified if the socket is unconnected
	if rsa.Addr.Family != AF_UNSPEC {
		from, err = anyToSockaddr(&rsa)
	}
	return
}

func Sendmsg(fd int, p, oob []byte, to Sockaddr, flags int) (err error) {
	_, err = SendmsgN(fd, p, oob, to, flags)
	return
}

func SendmsgN(fd int, p, oob []byte, to Sockaddr, flags int) (n int, err error) {
	var ptr *RawSockaddrAny
	var salen Socklen_t
	if to != nil {
		var err error
		ptr, salen, err = to.sockaddr()
		if err != nil {
			return 0, err
		}
	}
	var msg Msghdr
	msg.Name = (*byte)(unsafe.Pointer(ptr))
	msg.Namelen = uint32(salen)
	var iov Iovec
	if len(p) > 0 {
		iov.Base = (*byte)(unsafe.Pointer(&p[0]))
		iov.SetLen(len(p))
	}
	var dummy byte
	if len(oob) > 0 {
		// send at least one normal byte
		if len(p) == 0 {
			iov.Base = &dummy
			iov.SetLen(1)
		}
		msg.Control = (*byte)(unsafe.Pointer(&oob[0]))
		msg.SetControllen(len(oob))
	}
	msg.Iov = &iov
	msg.Iovlen = 1
	if n, err = sendmsg(fd, &msg, flags); err != nil {
		return 0, err
	}
	if len(oob) > 0 && len(p) == 0 {
		n = 0
	}
	return n, nil
}

//sys	Listen(fd int, n int) (err error)
//listen(fd _C_int, n _C_int) _C_int

//sys	Shutdown(fd int, how int) (err error)
//shutdown(fd _C_int, how _C_int) _C_int

func (iov *Iovec) SetLen(length int) {
	iov.Len = Iovec_len_t(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = Msghdr_controllen_t(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = Cmsghdr_len_t(length)
}
