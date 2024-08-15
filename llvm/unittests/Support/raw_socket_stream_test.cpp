#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_socket_stream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <future>
#include <stdlib.h>
#include <thread>

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
  llvm::sys::fs::createUniquePath("client_server_comms.sock", SocketPath, true);

  // Make sure socket file does not exist. May still be there from the last test
  std::remove(SocketPath.c_str());

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

  char Bytes[8];
  ssize_t BytesRead = Server.read(Bytes, 8);

  std::string string(Bytes, 8);
  ASSERT_EQ(Server.has_error(), false);

  ASSERT_EQ(8, BytesRead);
  ASSERT_EQ("01234567", string);
}

TEST(raw_socket_streamTest, READ_WITH_TIMEOUT) {
  if (!hasUnixSocketSupport())
    GTEST_SKIP();

  SmallString<100> SocketPath;
  llvm::sys::fs::createUniquePath("read_with_timeout.sock", SocketPath, true);

  // Make sure socket file does not exist. May still be there from the last test
  std::remove(SocketPath.c_str());

  Expected<ListeningSocket> MaybeServerListener =
      ListeningSocket::createUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeServerListener, llvm::Succeeded());
  ListeningSocket ServerListener = std::move(*MaybeServerListener);

  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeClient, llvm::Succeeded());

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener.accept();
  ASSERT_THAT_EXPECTED(MaybeServer, llvm::Succeeded());
  raw_socket_stream &Server = **MaybeServer;

  char Bytes[8];
  ssize_t BytesRead = Server.read(Bytes, 8, std::chrono::milliseconds(100));
  ASSERT_EQ(BytesRead, -1);
  ASSERT_EQ(Server.has_error(), true);
  ASSERT_EQ(Server.error(), std::errc::timed_out);
  Server.clear_error();
}

TEST(raw_socket_streamTest, ACCEPT_WITH_TIMEOUT) {
  if (!hasUnixSocketSupport())
    GTEST_SKIP();

  SmallString<100> SocketPath;
  llvm::sys::fs::createUniquePath("accept_with_timeout.sock", SocketPath, true);

  // Make sure socket file does not exist. May still be there from the last test
  std::remove(SocketPath.c_str());

  Expected<ListeningSocket> MaybeServerListener =
      ListeningSocket::createUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeServerListener, llvm::Succeeded());
  ListeningSocket ServerListener = std::move(*MaybeServerListener);

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener.accept(std::chrono::milliseconds(100));
  ASSERT_EQ(llvm::errorToErrorCode(MaybeServer.takeError()),
            std::errc::timed_out);
}

TEST(raw_socket_streamTest, ACCEPT_WITH_SHUTDOWN) {
  if (!hasUnixSocketSupport())
    GTEST_SKIP();

  SmallString<100> SocketPath;
  llvm::sys::fs::createUniquePath("accept_with_shutdown.sock", SocketPath,
                                  true);

  // Make sure socket file does not exist. May still be there from the last test
  std::remove(SocketPath.c_str());

  Expected<ListeningSocket> MaybeServerListener =
      ListeningSocket::createUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeServerListener, llvm::Succeeded());
  ListeningSocket ServerListener = std::move(*MaybeServerListener);

  // Create a separate thread to close the socket after a delay. Simulates a
  // signal handler calling ServerListener::shutdown
  std::thread CloseThread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ServerListener.shutdown();
  });

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener.accept();

  // Wait for the CloseThread to finish
  CloseThread.join();
  ASSERT_EQ(llvm::errorToErrorCode(MaybeServer.takeError()),
            std::errc::operation_canceled);
}
} // namespace
