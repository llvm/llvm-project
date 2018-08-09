// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

/*
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
*/

import (
	"syscall"
	"unsafe"
)

//extern getaddrinfo
func libc_getaddrinfo(node *byte, service *byte, hints *syscall.Addrinfo, res **syscall.Addrinfo) int

//extern freeaddrinfo
func libc_freeaddrinfo(res *syscall.Addrinfo)

//extern gai_strerror
func libc_gai_strerror(errcode int) *byte

// bytePtrToString takes a NUL-terminated array of bytes and convert
// it to a Go string.
func bytePtrToString(p *byte) string {
	a := (*[10000]byte)(unsafe.Pointer(p))
	i := 0
	for a[i] != 0 {
		i++
	}
	return string(a[:i])
}

// An addrinfoErrno represents a getaddrinfo, getnameinfo-specific
// error number. It's a signed number and a zero value is a non-error
// by convention.
type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return bytePtrToString(libc_gai_strerror(int(eai))) }
func (eai addrinfoErrno) Temporary() bool { return eai == syscall.EAI_AGAIN }
func (eai addrinfoErrno) Timeout() bool   { return false }

func cgoLookupHost(name string) (hosts []string, err error, completed bool) {
	addrs, err, completed := cgoLookupIP(name)
	for _, addr := range addrs {
		hosts = append(hosts, addr.String())
	}
	return
}

func cgoLookupPort(network, service string) (port int, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var hints syscall.Addrinfo
	switch network {
	case "": // no hints
	case "tcp", "tcp4", "tcp6":
		hints.Ai_socktype = syscall.SOCK_STREAM
		hints.Ai_protocol = syscall.IPPROTO_TCP
	case "udp", "udp4", "udp6":
		hints.Ai_socktype = syscall.SOCK_DGRAM
		hints.Ai_protocol = syscall.IPPROTO_UDP
	default:
		return 0, &DNSError{Err: "unknown network", Name: network + "/" + service}, true
	}
	if len(network) >= 4 {
		switch network[3] {
		case '4':
			hints.Ai_family = syscall.AF_INET
		case '6':
			hints.Ai_family = syscall.AF_INET6
		}
	}

	s := syscall.StringBytePtr(service)
	var res *syscall.Addrinfo
	syscall.Entersyscall()
	gerrno := libc_getaddrinfo(nil, s, &hints, &res)
	syscall.Exitsyscall()
	if gerrno != 0 {
		switch gerrno {
		case syscall.EAI_SYSTEM:
			errno := syscall.GetErrno()
			if errno == 0 { // see golang.org/issue/6232
				errno = syscall.EMFILE
			}
			err = errno
		default:
			err = addrinfoErrno(gerrno)
		}
		return 0, &DNSError{Err: err.Error(), Name: network + "/" + service}, true
	}
	defer libc_freeaddrinfo(res)

	for r := res; r != nil; r = r.Ai_next {
		switch r.Ai_family {
		case syscall.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.Ai_addr))
			p := (*[2]byte)(unsafe.Pointer(&sa.Port))
			return int(p[0])<<8 | int(p[1]), nil, true
		case syscall.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.Ai_addr))
			p := (*[2]byte)(unsafe.Pointer(&sa.Port))
			return int(p[0])<<8 | int(p[1]), nil, true
		}
	}
	return 0, &DNSError{Err: "unknown port", Name: network + "/" + service}, true
}

func cgoLookupIPCNAME(name string) (addrs []IPAddr, cname string, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var hints syscall.Addrinfo
	hints.Ai_flags = int32(cgoAddrInfoFlags)
	hints.Ai_socktype = syscall.SOCK_STREAM

	h := syscall.StringBytePtr(name)
	var res *syscall.Addrinfo
	syscall.Entersyscall()
	gerrno := libc_getaddrinfo(h, nil, &hints, &res)
	syscall.Exitsyscall()
	if gerrno != 0 {
		switch gerrno {
		case syscall.EAI_SYSTEM:
			errno := syscall.GetErrno()
			if errno == 0 {
				// err should not be nil, but sometimes getaddrinfo returns
				// gerrno == C.EAI_SYSTEM with err == nil on Linux.
				// The report claims that it happens when we have too many
				// open files, so use syscall.EMFILE (too many open files in system).
				// Most system calls would return ENFILE (too many open files),
				// so at the least EMFILE should be easy to recognize if this
				// comes up again. golang.org/issue/6232.
				errno = syscall.EMFILE
			}
			err = errno
		case syscall.EAI_NONAME:
			err = errNoSuchHost
		default:
			err = addrinfoErrno(gerrno)
		}
		return nil, "", &DNSError{Err: err.Error(), Name: name}, true
	}
	defer libc_freeaddrinfo(res)

	if res != nil {
		cname = bytePtrToString((*byte)(unsafe.Pointer(res.Ai_canonname)))
		if cname == "" {
			cname = name
		}
		if len(cname) > 0 && cname[len(cname)-1] != '.' {
			cname += "."
		}
	}
	for r := res; r != nil; r = r.Ai_next {
		// We only asked for SOCK_STREAM, but check anyhow.
		if r.Ai_socktype != syscall.SOCK_STREAM {
			continue
		}
		switch r.Ai_family {
		case syscall.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.Ai_addr))
			addr := IPAddr{IP: copyIP(sa.Addr[:])}
			addrs = append(addrs, addr)
		case syscall.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.Ai_addr))
			addr := IPAddr{IP: copyIP(sa.Addr[:]), Zone: zoneToString(int(sa.Scope_id))}
			addrs = append(addrs, addr)
		}
	}
	return addrs, cname, nil, true
}

func cgoLookupIP(name string) (addrs []IPAddr, err error, completed bool) {
	addrs, _, err, completed = cgoLookupIPCNAME(name)
	return
}

func cgoLookupCNAME(name string) (cname string, err error, completed bool) {
	_, cname, err, completed = cgoLookupIPCNAME(name)
	return
}

// These are roughly enough for the following:
//
// Source		Encoding			Maximum length of single name entry
// Unicast DNS		ASCII or			<=253 + a NUL terminator
//			Unicode in RFC 5892		252 * total number of labels + delimiters + a NUL terminator
// Multicast DNS	UTF-8 in RFC 5198 or		<=253 + a NUL terminator
//			the same as unicast DNS ASCII	<=253 + a NUL terminator
// Local database	various				depends on implementation
const (
	nameinfoLen    = 64
	maxNameinfoLen = 4096
)

func cgoLookupPTR(addr string) ([]string, error, bool) {
	acquireThread()
	defer releaseThread()

	ip := ParseIP(addr)
	if ip == nil {
		return nil, &DNSError{Err: "invalid address", Name: addr}, true
	}
	sa, salen := cgoSockaddr(ip)
	if sa == nil {
		return nil, &DNSError{Err: "invalid address " + ip.String(), Name: addr}, true
	}
	var err error
	var b []byte
	var gerrno int
	for l := nameinfoLen; l <= maxNameinfoLen; l *= 2 {
		b = make([]byte, l)
		gerrno, err = cgoNameinfoPTR(b, sa, salen)
		if gerrno == 0 || gerrno != syscall.EAI_OVERFLOW {
			break
		}
	}
	if gerrno != 0 {
		switch gerrno {
		case syscall.EAI_SYSTEM:
			if err == nil { // see golang.org/issue/6232
				err = syscall.EMFILE
			}
		default:
			err = addrinfoErrno(gerrno)
		}
		return nil, &DNSError{Err: err.Error(), Name: addr}, true
	}

	for i := 0; i < len(b); i++ {
		if b[i] == 0 {
			b = b[:i]
			break
		}
	}
	// Add trailing dot to match pure Go reverse resolver
	// and all other lookup routines. See golang.org/issue/12189.
	if len(b) > 0 && b[len(b)-1] != '.' {
		b = append(b, '.')
	}
	return []string{string(b)}, nil, true
}

func cgoSockaddr(ip IP) (*syscall.RawSockaddr, syscall.Socklen_t) {
	if ip4 := ip.To4(); ip4 != nil {
		return cgoSockaddrInet4(ip4), syscall.Socklen_t(syscall.SizeofSockaddrInet4)
	}
	if ip6 := ip.To16(); ip6 != nil {
		return cgoSockaddrInet6(ip6), syscall.Socklen_t(syscall.SizeofSockaddrInet6)
	}
	return nil, 0
}

func copyIP(x IP) IP {
	if len(x) < 16 {
		return x.To16()
	}
	y := make(IP, len(x))
	copy(y, x)
	return y
}
