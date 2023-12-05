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

  char Bytes[8];

  Expected<ListeningSocket> MaybeServerListener = ListeningSocket::createUnix(SocketPath);

  ListeningSocket ServerListener = std::move(*MaybeServerListener);

  Expected<raw_socket_stream> MaybeClient = raw_socket_stream::createConnectedUnix(SocketPath);

  raw_socket_stream Client = std::move(*MaybeClient);

  Expected<raw_socket_stream> MaybeServer = ServerListener.accept();

  raw_socket_stream Server = std::move(*MaybeServer);

  Client << "01234567";
  Client.flush();

  ssize_t BytesRead = Server.read(Bytes, 8);

  std::string string(reinterpret_cast<char *>(Bytes), 8);

  ASSERT_EQ(8, BytesRead);
  ASSERT_EQ("01234567", string);
}
} // namespace