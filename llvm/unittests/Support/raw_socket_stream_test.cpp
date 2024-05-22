#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <future>
#include <iostream>
#include <stdlib.h>

#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

using namespace llvm;

namespace {

bool hasUnixSocketSupport() {
#ifdef _WIN32
  VersionTuple Ver = GetWindowsOSVersion();
  if (Ver < VersionTuple(10, 0, 0, 17063))
    return false;
#endif
  return true;
}

TEST(raw_socket_streamTest, CLIENT_TO_SERVER_AND_SERVER_TO_CLIENT) {
  if (!hasUnixSocketSupport())
    GTEST_SKIP();

  SmallString<100> SocketPath;
  llvm::sys::fs::createUniquePath("test_raw_socket_stream.sock", SocketPath,
                                  true);

  char Bytes[8];

  Expected<ListeningSocket> MaybeServerListener =
      ListeningSocket::createUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeServerListener, llvm::Succeeded());

  ListeningSocket ServerListener = std::move(*MaybeServerListener);

  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeClient, llvm::Succeeded());

  raw_socket_stream &Client = **MaybeClient;

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener.accept();
  ASSERT_THAT_EXPECTED(MaybeServer, llvm::Succeeded());

  raw_socket_stream &Server = **MaybeServer;

  Client << "01234567";
  Client.flush();

  ssize_t BytesRead = Server.read(Bytes, 8);

  std::string string(Bytes, 8);

  ASSERT_EQ(8, BytesRead);
  ASSERT_EQ("01234567", string);
}
} // namespace