//===-- UdpSocket.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/UDPSocket.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/Config.h"

#ifndef LLDB_DISABLE_POSIX
#include <arpa/inet.h>
#include <sys/socket.h>
#endif

#include <memory>

using namespace lldb;
using namespace lldb_private;

namespace {

const int kDomain = AF_INET;
const int kType   = SOCK_DGRAM;

const Error kNotSupported("Not supported");

}

UDPSocket::UDPSocket(NativeSocket socket)
    : Socket(socket, ProtocolUdp, true)
{
}

UDPSocket::UDPSocket(bool child_processes_inherit, Error &error)
    : UDPSocket(CreateSocket(kDomain, kType, 0, child_processes_inherit, error))
{
}

size_t
UDPSocket::Send(const void *buf, const size_t num_bytes)
{
    return ::sendto (m_socket,
                     static_cast<const char*>(buf),
                     num_bytes,
                     0,
                     m_send_sockaddr,
                     m_send_sockaddr.GetLength());
}

Error
UDPSocket::Connect(llvm::StringRef name)
{
    return kNotSupported;
}

Error
UDPSocket::Listen(llvm::StringRef name, int backlog)
{
    return kNotSupported;
}

Error
UDPSocket::Accept(llvm::StringRef name, bool child_processes_inherit, Socket *&socket)
{
    return kNotSupported;
}

Error
UDPSocket::Connect(llvm::StringRef name, bool child_processes_inherit, Socket *&send_socket, Socket *&recv_socket)
{
    std::unique_ptr<UDPSocket> final_send_socket;
    std::unique_ptr<UDPSocket> final_recv_socket;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("UDPSocket::%s (host/port = %s)", __FUNCTION__, name.data());

    Error error;
    std::string host_str;
    std::string port_str;
    int32_t port = INT32_MIN;
    if (!DecodeHostAndPort (name, host_str, port_str, port, &error))
        return error;

    // Setup the receiving end of the UDP connection on this localhost
    // on port zero. After we bind to port zero we can read the port.
    final_recv_socket.reset(new UDPSocket(child_processes_inherit, error));
    if (error.Success())
    {
        // Socket was created, now lets bind to the requested port
        SocketAddress addr;
        addr.SetToAnyAddress (AF_INET, 0);

        if (::bind (final_recv_socket->GetNativeSocket(), addr, addr.GetLength()) == -1)
        {
            // Bind failed...
            SetLastError (error);
        }
    }

    assert(error.Fail() == !(final_recv_socket && final_recv_socket->IsValid()));
    if (error.Fail())
        return error;

    // At this point we have setup the receive port, now we need to
    // setup the UDP send socket

    struct addrinfo hints;
    struct addrinfo *service_info_list = nullptr;

    ::memset (&hints, 0, sizeof(hints));
    hints.ai_family = kDomain;
    hints.ai_socktype = kType;
    int err = ::getaddrinfo (host_str.c_str(), port_str.c_str(), &hints, &service_info_list);
    if (err != 0)
    {
        error.SetErrorStringWithFormat("getaddrinfo(%s, %s, &hints, &info) returned error %i (%s)",
                                       host_str.c_str(),
                                       port_str.c_str(),
                                       err,
                                       gai_strerror(err));
        return error;
    }

    for (struct addrinfo *service_info_ptr = service_info_list;
         service_info_ptr != nullptr;
         service_info_ptr = service_info_ptr->ai_next)
    {
        auto send_fd = CreateSocket (service_info_ptr->ai_family,
                                     service_info_ptr->ai_socktype,
                                     service_info_ptr->ai_protocol,
                                     child_processes_inherit,
                                     error);
        if (error.Success())
        {
            final_send_socket.reset(new UDPSocket(send_fd));
            final_send_socket->m_send_sockaddr = service_info_ptr;
            break;
        }
        else
            continue;
    }

    :: freeaddrinfo (service_info_list);

    if (!final_send_socket)
        return error;

    send_socket = final_send_socket.release();
    recv_socket = final_recv_socket.release();
    error.Clear();
    return error;
}
