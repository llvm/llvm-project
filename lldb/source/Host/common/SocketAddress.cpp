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

// C Includes
#include <assert.h>
#include <string.h>

// C++ Includes
// Other libraries and framework includes
// Project includes

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
}

socklen_t
SocketAddress::GetLength () const
{
#if defined(__APPLE__)
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
#if defined(__APPLE__)
    m_socket_addr.sa.sa_len = GetFamilyLength (family);
#endif
}

in_port_t
SocketAddress::GetPort () const
{
    switch (GetFamily())
    {
        case AF_INET:   return m_socket_addr.sa_ipv4.sin_port;
        case AF_INET6:  return m_socket_addr.sa_ipv6.sin6_port;
    }
    return 0;
}

bool
SocketAddress::SetPort (in_port_t port)
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
SocketAddress::SetAddress (const struct addrinfo *hints_ptr,
                           const char *host, 
                           const char *service,
                           struct addrinfo *addr_info_ptr)
{
    struct addrinfo *service_info_list = NULL;
    int err = ::getaddrinfo (host, service, hints_ptr, &service_info_list);
    if (err == 0 && service_info_list)
    {
        if (addr_info_ptr)
            *addr_info_ptr = *service_info_list;
        *this = service_info_list;
    }
    else
        Clear();
    
    :: freeaddrinfo (service_info_list);
    
    const bool is_valid = IsValid();
    if (!is_valid)
    {
        if (addr_info_ptr)
            ::memset (addr_info_ptr, 0, sizeof(struct addrinfo));
    }
    return is_valid;
}


bool
SocketAddress::SetToLocalhost (sa_family_t family, in_port_t port)
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
