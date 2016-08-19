//===-- GDBRemoteTestUtils.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread> when
// exceptions are disabled.
#include <eh.h>
#endif

#include "GDBRemoteTestUtils.h"

#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"

#include <future>

namespace lldb_private
{
namespace process_gdb_remote
{

void
GDBRemoteTest::SetUpTestCase()
{
#if defined(_MSC_VER)
    WSADATA data;
    ::WSAStartup(MAKEWORD(2, 2), &data);
#endif
}

void
GDBRemoteTest::TearDownTestCase()
{
#if defined(_MSC_VER)
    ::WSACleanup();
#endif
}

void
Connect(GDBRemoteCommunication &client, GDBRemoteCommunication &server)
{
    bool child_processes_inherit = false;
    Error error;
    TCPSocket listen_socket(child_processes_inherit, error);
    ASSERT_FALSE(error.Fail());
    error = listen_socket.Listen("127.0.0.1:0", 5);
    ASSERT_FALSE(error.Fail());

    Socket *accept_socket;
    std::future<Error> accept_error = std::async(std::launch::async, [&] {
        return listen_socket.Accept("127.0.0.1:0", child_processes_inherit, accept_socket);
    });

    char connect_remote_address[64];
    snprintf(connect_remote_address, sizeof(connect_remote_address), "connect://localhost:%u",
             listen_socket.GetLocalPortNumber());

    std::unique_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
    ASSERT_EQ(conn_ap->Connect(connect_remote_address, nullptr), lldb::eConnectionStatusSuccess);

    client.SetConnection(conn_ap.release());
    ASSERT_TRUE(accept_error.get().Success());
    server.SetConnection(new ConnectionFileDescriptor(accept_socket));
}

} // namespace process_gdb_remote
} // namespace lldb_private
