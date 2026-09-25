#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_socket_stream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
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

struct raw_socket_streamTest : ::testing::Test {
  SmallString<100> SocketPath;
  std::optional<ListeningSocket> ServerListener;

  void SetUp() override {
    if (!hasUnixSocketSupport())
      GTEST_SKIP();

    llvm::sys::fs::createUniquePath("llvm-%%%%%%%%.sock", SocketPath, true);
    Expected<ListeningSocket> MaybeServerListener =
        ListeningSocket::createUnix(SocketPath);
    if (!MaybeServerListener) {
      std::error_code EC = errorToErrorCode(MaybeServerListener.takeError());
      if (EC == std::errc::filename_too_long)
        GTEST_SKIP() << EC.message() << ": " << SocketPath;
      FAIL() << EC.message();
      return;
    }

    ServerListener.emplace(std::move(*MaybeServerListener));
  }

  void TearDown() override { std::remove(SocketPath.c_str()); }
};

TEST_F(raw_socket_streamTest, CLIENT_TO_SERVER_AND_SERVER_TO_CLIENT) {
  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeClient, llvm::Succeeded());

  raw_socket_stream &Client = **MaybeClient;

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener->accept();
  ASSERT_THAT_EXPECTED(MaybeServer, llvm::Succeeded());

  raw_socket_stream &Server = **MaybeServer;

  Client << "01234567";
  Client.flush();

  char Bytes[8];
  ssize_t BytesRead = Server.read(Bytes, 8);

  std::string Str(Bytes, 8);
  ASSERT_EQ(Server.has_error(), false);

  ASSERT_EQ(8, BytesRead);
  ASSERT_EQ("01234567", Str);
}

TEST_F(raw_socket_streamTest, READ_WITH_TIMEOUT) {
  Expected<std::unique_ptr<raw_socket_stream>> MaybeClient =
      raw_socket_stream::createConnectedUnix(SocketPath);
  ASSERT_THAT_EXPECTED(MaybeClient, llvm::Succeeded());

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener->accept();
  ASSERT_THAT_EXPECTED(MaybeServer, llvm::Succeeded());
  raw_socket_stream &Server = **MaybeServer;

  char Bytes[8];
  ssize_t BytesRead = Server.read(Bytes, 8, std::chrono::milliseconds(100));
  ASSERT_EQ(BytesRead, -1);
  ASSERT_EQ(Server.has_error(), true);
  ASSERT_EQ(Server.error(), std::errc::timed_out);
  Server.clear_error();
}

TEST_F(raw_socket_streamTest, ACCEPT_WITH_TIMEOUT) {
  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener->accept(std::chrono::milliseconds(100));
  ASSERT_EQ(llvm::errorToErrorCode(MaybeServer.takeError()),
            std::errc::timed_out);
}

TEST_F(raw_socket_streamTest, ACCEPT_WITH_SHUTDOWN) {
  // Create a separate thread to close the socket after a delay. Simulates a
  // signal handler calling ServerListener::shutdown
  std::thread CloseThread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ServerListener->shutdown();
  });

  Expected<std::unique_ptr<raw_socket_stream>> MaybeServer =
      ServerListener->accept();

  // Wait for the CloseThread to finish
  CloseThread.join();
  ASSERT_EQ(llvm::errorToErrorCode(MaybeServer.takeError()),
            std::errc::operation_canceled);
}
} // namespace
