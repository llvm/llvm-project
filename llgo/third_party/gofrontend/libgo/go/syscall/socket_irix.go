// socket_irix.go -- Socket handling specific to IRIX 6.

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

const SizeofSockaddrInet4 = 16
const SizeofSockaddrInet6 = 28
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
	if sa.Path[0] == 0 {
		// "Abstract" Unix domain socket.
		// Rewrite leading NUL as @ for textual display.
		// (This is the standard convention.)
		// Not friendly to overwrite in place,
		// but the callers below don't care.
		sa.Path[0] = '@'
	}

	// Assume path ends at NUL.
	// This is not technically the GNU/Linux semantics for
	// abstract Unix domain sockets--they are supposed
	// to be uninterpreted fixed-size binary blobs--but
	// everyone uses this convention.
	n := 0
	for n < len(sa.Path)-3 && sa.Path[n] != 0 {
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

// <netdb.h> only provides struct addrinfo, AI_* and EAI_* if  _NO_XOPEN4
// && _NO_XOPEN5, but -D_XOPEN_SOURCE=500 is required for msg_control etc.
// in struct msghgr, so simply provide them here.
type Addrinfo struct {
	Ai_flags     int32
	Ai_family    int32
	Ai_socktype  int32
	Ai_protocol  int32
	Ai_addrlen   int32
	Ai_canonname *uint8
	Ai_addr      *_sockaddr
	Ai_next      *Addrinfo
}

const (
	AI_PASSIVE     = 0x00000001
	AI_CANONNAME   = 0x00000002
	AI_NUMERICHOST = 0x00000004
	AI_NUMERICSERV = 0x00000008
	AI_ALL         = 0x00000100
	AI_ADDRCONFIG  = 0x00000400
	AI_V4MAPPED    = 0x00000800
	AI_DEFAULT     = (AI_V4MAPPED | AI_ADDRCONFIG)
)

const (
	EAI_ADDRFAMILY = 1
	EAI_AGAIN      = 2
	EAI_BADFLAGS   = 3
	EAI_FAIL       = 4
	EAI_FAMILY     = 5
	EAI_MEMORY     = 6
	EAI_NODATA     = 7
	EAI_NONAME     = 8
	EAI_SERVICE    = 9
	EAI_SOCKTYPE   = 10
	EAI_SYSTEM     = 11
	EAI_BADHINTS   = 12
	EAI_OVERFLOW   = 13
	EAI_MAX        = 14
)

func anyToSockaddrOS(rsa *RawSockaddrAny) (Sockaddr, error) {
	return nil, EAFNOSUPPORT
}

// <netinet/in.h.h> only provides IPV6_* etc. if  _NO_XOPEN4 && _NO_XOPEN5,
// so as above simply provide them here.
const (
	IPV6_UNICAST_HOPS   = 48
	IPV6_MULTICAST_IF   = IP_MULTICAST_IF
	IPV6_MULTICAST_HOPS = IP_MULTICAST_TTL
	IPV6_MULTICAST_LOOP = IP_MULTICAST_LOOP
)
