//===-- GDBRemoteTestUtils.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteTestUtils.h"

#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"

#include <future>

namespace lldb_private {
namespace process_gdb_remote {

void GDBRemoteTest::SetUpTestCase() {
#if defined(_MSC_VER)
  WSADATA data;
  ::WSAStartup(MAKEWORD(2, 2), &data);
#endif
}

void GDBRemoteTest::TearDownTestCase() {
#if defined(_MSC_VER)
  ::WSACleanup();
#endif
}

llvm::Error GDBRemoteTest::Connect(GDBRemoteCommunication &client,
                                   GDBRemoteCommunication &server) {
  bool child_processes_inherit = false;
  TCPSocket listen_socket(true, child_processes_inherit);
  if (llvm::Error error = listen_socket.Listen("127.0.0.1:0", 5).ToError())
    return error;

  Socket *accept_socket;
  std::future<Status> accept_status = std::async(
      std::launch::async, [&] { return listen_socket.Accept(accept_socket); });

  llvm::SmallString<32> remote_addr;
  llvm::raw_svector_ostream(remote_addr)
      << "connect://localhost:" << listen_socket.GetLocalPortNumber();

  std::unique_ptr<ConnectionFileDescriptor> conn_up(
      new ConnectionFileDescriptor());
  if (conn_up->Connect(remote_addr, nullptr) != lldb::eConnectionStatusSuccess)
    return llvm::make_error<llvm::StringError>("Unable to connect",
                                               llvm::inconvertibleErrorCode());

  client.SetConnection(conn_up.release());
  if (llvm::Error error = accept_status.get().ToError())
    return error;

  server.SetConnection(new ConnectionFileDescriptor(accept_socket));
  return llvm::Error::success();
}

} // namespace process_gdb_remote
} // namespace lldb_private
