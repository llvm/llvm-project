//===-- SocketAddress.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/SocketAddress.h"
#include <stddef.h>
#include <stdio.h>

// C Includes
#if !defined(_WIN32)
#include <arpa/inet.h>
#endif

#include <assert.h>
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Host/PosixApi.h"

// WindowsXP needs an inet_ntop implementation
#ifdef _WIN32

#ifndef INET6_ADDRSTRLEN // might not be defined in older Windows SDKs
#define INET6_ADDRSTRLEN 46
#endif

// TODO: implement shortened form "::" for runs of zeros
const char* inet_ntop(int af, const void * src,
                      char * dst, socklen_t size)
{
    if (size==0)
    {
        return nullptr;
    }
    
    switch (af)
    {
        case AF_INET:
        {
            {
                const char* formatted = inet_ntoa(*static_cast<const in_addr*>(src));
                if (formatted && strlen(formatted) < size)
                {
                    return ::strcpy(dst, formatted);
                }
            }
            return nullptr;
        case AF_INET6:
            {
                char tmp[INET6_ADDRSTRLEN] = {0};
                const uint16_t* src16 = static_cast<const uint16_t*>(src);
                int full_size = ::snprintf(tmp, sizeof(tmp),
                                          "%x:%x:%x:%x:%x:%x:%x:%x",
                                          ntohs(src16[0]), ntohs(src16[1]), ntohs(src16[2]), ntohs(src16[3]),
                                          ntohs(src16[4]), ntohs(src16[5]), ntohs(src16[6]), ntohs(src16[7])
                                          );
                if (full_size < static_cast<int>(size))
                {
                    return ::strcpy(dst, tmp);
                }
                return nullptr;
            }
        }
    }
    return nullptr;
} 
#endif
    

using namespace lldb_private;

//----------------------------------------------------------------------
// SocketAddress constructor
//----------------------------------------------------------------------
SocketAddress::SocketAddress ()
{
    Clear ();
}

SocketAddress::SocketAddress (const struct sockaddr &s)
{
    m_socket_addr.sa = s;
}


SocketAddress::SocketAddress (const struct sockaddr_in &s)
{
    m_socket_addr.sa_ipv4 = s;
}


SocketAddress::SocketAddress (const struct sockaddr_in6 &s)
{
    m_socket_addr.sa_ipv6 = s;
}


SocketAddress::SocketAddress (const struct sockaddr_storage &s)
{
    m_socket_addr.sa_storage = s;
}

//----------------------------------------------------------------------
// SocketAddress copy constructor
//----------------------------------------------------------------------
SocketAddress::SocketAddress (const SocketAddress& rhs) :
    m_socket_addr (rhs.m_socket_addr)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SocketAddress::~SocketAddress()
{
}

void
SocketAddress::Clear ()
{
    memset (&m_socket_addr, 0, sizeof(m_socket_addr));
}

bool
SocketAddress::IsValid () const
{
    return GetLength () != 0;
}

static socklen_t 
GetFamilyLength (sa_family_t family)
{
    switch (family)
    {
        case AF_INET:  return sizeof(struct sockaddr_in);
        case AF_INET6: return sizeof(struct sockaddr_in6);
    }
    assert(0 && "Unsupported address family");
    return 0;
}

socklen_t
SocketAddress::GetLength () const
{
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    return m_socket_addr.sa.sa_len;
#else
    return GetFamilyLength (GetFamily());
#endif
}

socklen_t
SocketAddress::GetMaxLength ()
{
    return sizeof (sockaddr_t);
}

sa_family_t
SocketAddress::GetFamily () const
{
    return m_socket_addr.sa.sa_family;
}

void
SocketAddress::SetFamily (sa_family_t family)
{
    m_socket_addr.sa.sa_family = family;
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    m_socket_addr.sa.sa_len = GetFamilyLength (family);
#endif
}

std::string
SocketAddress::GetIPAddress () const
{
    char str[INET6_ADDRSTRLEN] = {0};
    switch (GetFamily())
    {
        case AF_INET:
            if (inet_ntop(GetFamily(), &m_socket_addr.sa_ipv4.sin_addr, str, sizeof(str)))
                return str;
            break;
        case AF_INET6:
            if (inet_ntop(GetFamily(), &m_socket_addr.sa_ipv6.sin6_addr, str, sizeof(str)))
                return str;
            break;
    }
    return "";
}

uint16_t
SocketAddress::GetPort () const
{
    switch (GetFamily())
    {
        case AF_INET:   return ntohs(m_socket_addr.sa_ipv4.sin_port);
        case AF_INET6:  return ntohs(m_socket_addr.sa_ipv6.sin6_port);
    }
    return 0;
}

bool
SocketAddress::SetPort (uint16_t port)
{
    switch (GetFamily())
    {
        case AF_INET:   
            m_socket_addr.sa_ipv4.sin_port = htons(port);
            return true;

        case AF_INET6:  
            m_socket_addr.sa_ipv6.sin6_port = htons(port);
            return true;
    }
    return false;
}

//----------------------------------------------------------------------
// SocketAddress assignment operator
//----------------------------------------------------------------------
const SocketAddress&
SocketAddress::operator=(const SocketAddress& rhs)
{
    if (this != &rhs)
        m_socket_addr = rhs.m_socket_addr;
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct addrinfo *addr_info)
{
    Clear();
    if (addr_info && 
        addr_info->ai_addr &&
        addr_info->ai_addrlen > 0&& 
        addr_info->ai_addrlen <= sizeof m_socket_addr)
    {
        ::memcpy (&m_socket_addr, 
                  addr_info->ai_addr, 
                  addr_info->ai_addrlen);
    }
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct sockaddr &s)
{
    m_socket_addr.sa = s;
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct sockaddr_in &s)
{
    m_socket_addr.sa_ipv4 = s;
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct sockaddr_in6 &s)
{
    m_socket_addr.sa_ipv6 = s;
    return *this;
}

const SocketAddress&
SocketAddress::operator=(const struct sockaddr_storage &s)
{
    m_socket_addr.sa_storage = s;
    return *this;
}

bool
SocketAddress::getaddrinfo (const char *host,
                            const char *service,
                            int ai_family,
                            int ai_socktype,
                            int ai_protocol,
                            int ai_flags)
{
    Clear ();

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = ai_family;
    hints.ai_socktype = ai_socktype;
    hints.ai_protocol = ai_protocol;
    hints.ai_flags = ai_flags;

    bool result = false;
    struct addrinfo *service_info_list = NULL;
    int err = ::getaddrinfo (host, service, &hints, &service_info_list);
    if (err == 0 && service_info_list)
    {
        *this = service_info_list;
        result = IsValid ();
    }

    if (service_info_list)
        ::freeaddrinfo(service_info_list);

    return result;
}


bool
SocketAddress::SetToLocalhost (sa_family_t family, uint16_t port)
{
    switch (family)
    {
        case AF_INET:   
            SetFamily (AF_INET);
            if (SetPort (port))
            {
                m_socket_addr.sa_ipv4.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
                return true;
            }
            break;

        case AF_INET6:  
            SetFamily (AF_INET6);
            if (SetPort (port))
            {
                m_socket_addr.sa_ipv6.sin6_addr = in6addr_loopback;
                return true;
            }            
            break;

    }
    Clear();
    return false;
}

bool
SocketAddress::SetToAnyAddress (sa_family_t family, uint16_t port)
{
    switch (family)
    {
        case AF_INET:
            SetFamily (AF_INET);
            if (SetPort (port))
            {
                m_socket_addr.sa_ipv4.sin_addr.s_addr = htonl (INADDR_ANY);
                return true;
            }
            break;
            
        case AF_INET6:
            SetFamily (AF_INET6);
            if (SetPort (port))
            {
                m_socket_addr.sa_ipv6.sin6_addr = in6addr_any;
                return true;
            }
            break;
            
    }
    Clear();
    return false;
}
