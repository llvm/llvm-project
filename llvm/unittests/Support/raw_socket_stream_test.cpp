#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <future>
#include <iostream>
#include <stdlib.h>

using namespace llvm;

namespace {

TEST(raw_socket_streamTest, CLIENT_TO_SERVER_AND_SERVER_TO_CLIENT) {

  SmallString<100> SocketPath("/tmp/test_raw_socket_stream.sock");
  std::error_code ECServer, ECClient;

  int ServerFD = raw_socket_stream::MakeServerSocket(SocketPath, 3, ECServer);

  raw_socket_stream Client(SocketPath, ECClient);
  EXPECT_TRUE(!ECClient);

  raw_socket_stream Client2(SocketPath, ECClient);

  raw_socket_stream Server(ServerFD, SocketPath, ECServer);
  EXPECT_TRUE(!ECServer);

  Client << "01234567";
  Client.flush();

  Client2 << "abcdefgh";
  Client2.flush();

  Expected<std::string> from_client = Server.read_impl();

  if (auto E = from_client.takeError()) {
    return; // FIXME: Do something.
  }
  EXPECT_EQ("01234567", (*from_client));

  Server << "76543210";
  Server.flush();

  Expected<std::string> from_server = Client.read_impl();
  if (auto E = from_server.takeError()) {
    return;
    // YIKES! 😩
  }
  EXPECT_EQ("76543210", (*from_server));
}
} // namespace