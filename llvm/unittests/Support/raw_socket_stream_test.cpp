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

  llvm::Expected<std::string> MaybeText = Server.readFromSocket();
  ASSERT_THAT_EXPECTED(MaybeText, llvm::Succeeded());
  ASSERT_EQ("01234567", *MaybeText);
}

TEST(raw_socket_streamTest, LARGE_READ) {
  if (!hasUnixSocketSupport())
    GTEST_SKIP();

  SmallString<100> SocketPath;
  llvm::sys::fs::createUniquePath("large_read.sock", SocketPath, true);

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

  // raw_socket_stream::readFromSocket pre-allocates a buffer 1024 bytes large.
  // Test to make sure readFromSocket can handle messages larger then size of
  // pre-allocated block
  constexpr int TextLength = 1342;
  constexpr char Text[TextLength] =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
      "eiusmod tempor incididunt ut labore et dolore magna aliqua. Vel orci "
      "porta non pulvinar neque laoreet suspendisse interdum consectetur. "
      "Nulla facilisi etiam dignissim diam quis. Porttitor massa id neque "
      "aliquam vestibulum morbi blandit cursus. Purus viverra accumsan in "
      "nisl. Nunc non blandit massa enim nec dui nunc mattis enim. Rhoncus "
      "dolor purus non enim praesent elementum facilisis leo. Parturient "
      "montes nascetur ridiculus mus mauris. Urna condimentum mattis "
      "pellentesque id nibh tortor id aliquet lectus. Orci eu lobortis "
      "elementum nibh. Sagittis eu volutpat odio facilisis. Molestie a "
      "iaculis at erat pellentesque adipiscing. Tincidunt augue interdum "
      "velit euismod in pellentesque massa placerat. Cras ornare arcu dui "
      "vivamus arcu felis bibendum ut tristique. Tellus elementum sagittis "
      "vitae et leo duis. Scelerisque fermentum dui faucibus in ornare "
      "quam. Ipsum a arcu cursus vitae congue. Sit amet nisl suscipit "
      "adipiscing. Sociis natoque penatibus et magnis. Cras semper auctor "
      "neque vitae tempus quam pellentesque. Neque gravida in fermentum et "
      "sollicitudin ac orci phasellus egestas. Vitae suscipit tellus mauris "
      "a diam maecenas sed. Lectus arcu bibendum at varius vel pharetra. "
      "Dignissim sodales ut eu sem integer vitae justo. Id cursus metus "
      "aliquam eleifend mi.";
  Client << Text;
  Client.flush();

  llvm::Expected<std::string> MaybeText = Server.readFromSocket();
  ASSERT_THAT_EXPECTED(MaybeText, llvm::Succeeded());
  ASSERT_EQ(Text, *MaybeText);
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

  llvm::Expected<std::string> MaybeBytesRead =
      Server.readFromSocket(std::chrono::milliseconds(100));
  ASSERT_EQ(llvm::errorToErrorCode(MaybeBytesRead.takeError()),
            std::errc::timed_out);
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
