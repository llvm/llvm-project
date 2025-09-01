//===-- Socket.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Socket.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Host/SocketAddress.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/common/UDPSocket.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/WindowsError.h"

#if LLDB_ENABLE_POSIX
#include "lldb/Host/posix/DomainSocket.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include "lldb/Host/linux/AbstractSocket.h"
#endif

using namespace lldb;
using namespace lldb_private;

#if defined(_WIN32)
typedef const char *set_socket_option_arg_type;
typedef char *get_socket_option_arg_type;
const NativeSocket Socket::kInvalidSocketValue = INVALID_SOCKET;
const shared_fd_t SharedSocket::kInvalidFD = LLDB_INVALID_PIPE;
#else  // #if defined(_WIN32)
typedef const void *set_socket_option_arg_type;
typedef void *get_socket_option_arg_type;
const NativeSocket Socket::kInvalidSocketValue = -1;
const shared_fd_t SharedSocket::kInvalidFD = Socket::kInvalidSocketValue;
#endif // #if defined(_WIN32)

static bool IsInterrupted() {
#if defined(_WIN32)
  return ::WSAGetLastError() == WSAEINTR;
#else
  return errno == EINTR;
#endif
}

SharedSocket::SharedSocket(const Socket *socket, Status &error) {
#ifdef _WIN32
  m_socket = socket->GetNativeSocket();
  m_fd = kInvalidFD;

  // Create a pipe to transfer WSAPROTOCOL_INFO to the child process.
  error = m_socket_pipe.CreateNew();
  if (error.Fail())
    return;

  m_fd = m_socket_pipe.GetReadPipe();
#else
  m_fd = socket->GetNativeSocket();
  error = Status();
#endif
}

Status SharedSocket::CompleteSending(lldb::pid_t child_pid) {
#ifdef _WIN32
  // Transfer WSAPROTOCOL_INFO to the child process.
  m_socket_pipe.CloseReadFileDescriptor();

  WSAPROTOCOL_INFO protocol_info;
  if (::WSADuplicateSocket(m_socket, child_pid, &protocol_info) ==
      SOCKET_ERROR) {
    int last_error = ::WSAGetLastError();
    return Status::FromErrorStringWithFormat(
        "WSADuplicateSocket() failed, error: %d", last_error);
  }

  llvm::Expected<size_t> num_bytes = m_socket_pipe.Write(
      &protocol_info, sizeof(protocol_info), std::chrono::seconds(10));
  if (!num_bytes)
    return Status::FromError(num_bytes.takeError());
  if (*num_bytes != sizeof(protocol_info))
    return Status::FromErrorStringWithFormatv(
        "Write(WSAPROTOCOL_INFO) failed: wrote {0}/{1} bytes", *num_bytes,
        sizeof(protocol_info));
#endif
  return Status();
}

Status SharedSocket::GetNativeSocket(shared_fd_t fd, NativeSocket &socket) {
#ifdef _WIN32
  socket = Socket::kInvalidSocketValue;
  // Read WSAPROTOCOL_INFO from the parent process and create NativeSocket.
  WSAPROTOCOL_INFO protocol_info;
  {
    Pipe socket_pipe(fd, LLDB_INVALID_PIPE);
    llvm::Expected<size_t> num_bytes = socket_pipe.Read(
        &protocol_info, sizeof(protocol_info), std::chrono::seconds(10));
    if (!num_bytes)
      return Status::FromError(num_bytes.takeError());
    if (*num_bytes != sizeof(protocol_info)) {
      return Status::FromErrorStringWithFormatv(
          "Read(WSAPROTOCOL_INFO) failed: read {0}/{1} bytes", *num_bytes,
          sizeof(protocol_info));
    }
  }
  socket = ::WSASocket(FROM_PROTOCOL_INFO, FROM_PROTOCOL_INFO,
                       FROM_PROTOCOL_INFO, &protocol_info, 0, 0);
  if (socket == INVALID_SOCKET) {
    return Status::FromErrorStringWithFormatv(
        "WSASocket(FROM_PROTOCOL_INFO) failed: error {0}", ::WSAGetLastError());
  }
  return Status();
#else
  socket = fd;
  return Status();
#endif
}

struct SocketScheme {
  const char *m_scheme;
  const Socket::SocketProtocol m_protocol;
};

static SocketScheme socket_schemes[] = {
    {"tcp", Socket::ProtocolTcp},
    {"udp", Socket::ProtocolUdp},
    {"unix", Socket::ProtocolUnixDomain},
    {"unix-abstract", Socket::ProtocolUnixAbstract},
};

const char *
Socket::FindSchemeByProtocol(const Socket::SocketProtocol protocol) {
  for (auto s : socket_schemes) {
    if (s.m_protocol == protocol)
      return s.m_scheme;
  }
  return nullptr;
}

bool Socket::FindProtocolByScheme(const char *scheme,
                                  Socket::SocketProtocol &protocol) {
  for (auto s : socket_schemes) {
    if (!strcmp(s.m_scheme, scheme)) {
      protocol = s.m_protocol;
      return true;
    }
  }
  return false;
}

Socket::Socket(SocketProtocol protocol, bool should_close)
    : IOObject(eFDTypeSocket), m_protocol(protocol),
      m_socket(kInvalidSocketValue), m_should_close_fd(should_close) {}

Socket::~Socket() { Close(); }

llvm::Error Socket::Initialize() {
#if defined(_WIN32)
  auto wVersion = WINSOCK_VERSION;
  WSADATA wsaData;
  int err = ::WSAStartup(wVersion, &wsaData);
  if (err == 0) {
    if (wsaData.wVersion < wVersion) {
      WSACleanup();
      return llvm::createStringError("WSASock version is not expected.");
    }
  } else {
    return llvm::errorCodeToError(llvm::mapWindowsError(::WSAGetLastError()));
  }
#endif

  return llvm::Error::success();
}

void Socket::Terminate() {
#if defined(_WIN32)
  ::WSACleanup();
#endif
}

std::unique_ptr<Socket> Socket::Create(const SocketProtocol protocol,
                                       Status &error) {
  error.Clear();

  const bool should_close = true;
  std::unique_ptr<Socket> socket_up;
  switch (protocol) {
  case ProtocolTcp:
    socket_up = std::make_unique<TCPSocket>(should_close);
    break;
  case ProtocolUdp:
    socket_up = std::make_unique<UDPSocket>(should_close);
    break;
  case ProtocolUnixDomain:
#if LLDB_ENABLE_POSIX
    socket_up = std::make_unique<DomainSocket>(should_close);
#else
    error = Status::FromErrorString(
        "Unix domain sockets are not supported on this platform.");
#endif
    break;
  case ProtocolUnixAbstract:
#ifdef __linux__
    socket_up = std::make_unique<AbstractSocket>();
#else
    error = Status::FromErrorString(
        "Abstract domain sockets are not supported on this platform.");
#endif
    break;
  }

  if (error.Fail())
    socket_up.reset();

  return socket_up;
}

llvm::Expected<Socket::Pair>
Socket::CreatePair(std::optional<SocketProtocol> protocol) {
  constexpr SocketProtocol kBestProtocol =
      LLDB_ENABLE_POSIX ? ProtocolUnixDomain : ProtocolTcp;
  switch (protocol.value_or(kBestProtocol)) {
  case ProtocolTcp:
    return TCPSocket::CreatePair();
#if LLDB_ENABLE_POSIX
  case ProtocolUnixDomain:
  case ProtocolUnixAbstract:
    return DomainSocket::CreatePair();
#endif
  default:
    return llvm::createStringError("Unsupported protocol");
  }
}

llvm::Expected<std::unique_ptr<Socket>>
Socket::TcpConnect(llvm::StringRef host_and_port) {
  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOG(log, "host_and_port = {0}", host_and_port);

  Status error;
  std::unique_ptr<Socket> connect_socket = Create(ProtocolTcp, error);
  if (error.Fail())
    return error.ToError();

  error = connect_socket->Connect(host_and_port);
  if (error.Success())
    return std::move(connect_socket);

  return error.ToError();
}

llvm::Expected<std::unique_ptr<TCPSocket>>
Socket::TcpListen(llvm::StringRef host_and_port, int backlog) {
  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOG(log, "host_and_port = {0}", host_and_port);

  std::unique_ptr<TCPSocket> listen_socket(
      new TCPSocket(/*should_close=*/true));

  Status error = listen_socket->Listen(host_and_port, backlog);
  if (error.Fail())
    return error.ToError();

  return std::move(listen_socket);
}

llvm::Expected<std::unique_ptr<UDPSocket>>
Socket::UdpConnect(llvm::StringRef host_and_port) {
  return UDPSocket::CreateConnected(host_and_port);
}

llvm::Expected<Socket::HostAndPort>
Socket::DecodeHostAndPort(llvm::StringRef host_and_port) {
  static llvm::Regex g_regex("([^:]+|\\[[0-9a-fA-F:]+.*\\]):([0-9]+)");
  HostAndPort ret;
  llvm::SmallVector<llvm::StringRef, 3> matches;
  if (g_regex.match(host_and_port, &matches)) {
    ret.hostname = matches[1].str();
    // IPv6 addresses are wrapped in [] when specified with ports
    if (ret.hostname.front() == '[' && ret.hostname.back() == ']')
      ret.hostname = ret.hostname.substr(1, ret.hostname.size() - 2);
    if (to_integer(matches[2], ret.port, 10))
      return ret;
  } else {
    // If this was unsuccessful, then check if it's simply an unsigned 16-bit
    // integer, representing a port with an empty host.
    if (to_integer(host_and_port, ret.port, 10))
      return ret;
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid host:port specification: '%s'",
                                 host_and_port.str().c_str());
}

IOObject::WaitableHandle Socket::GetWaitableHandle() {
  return (IOObject::WaitableHandle)m_socket;
}

Status Socket::Read(void *buf, size_t &num_bytes) {
  Status error;
  int bytes_received = 0;
  do {
    bytes_received = ::recv(m_socket, static_cast<char *>(buf), num_bytes, 0);
  } while (bytes_received < 0 && IsInterrupted());

  if (bytes_received < 0) {
    SetLastError(error);
    num_bytes = 0;
  } else
    num_bytes = bytes_received;

  Log *log = GetLog(LLDBLog::Communication);
  if (log) {
    LLDB_LOGF(log,
              "%p Socket::Read() (socket = %" PRIu64
              ", src = %p, src_len = %" PRIu64 ", flags = 0) => %" PRIi64
              " (error = %s)",
              static_cast<void *>(this), static_cast<uint64_t>(m_socket), buf,
              static_cast<uint64_t>(num_bytes),
              static_cast<int64_t>(bytes_received), error.AsCString());
  }

  return error;
}

Status Socket::Write(const void *buf, size_t &num_bytes) {
  const size_t src_len = num_bytes;
  Status error;
  int bytes_sent = 0;
  do {
    bytes_sent = Send(buf, num_bytes);
  } while (bytes_sent < 0 && IsInterrupted());

  if (bytes_sent < 0) {
    SetLastError(error);
    num_bytes = 0;
  } else
    num_bytes = bytes_sent;

  Log *log = GetLog(LLDBLog::Communication);
  if (log) {
    LLDB_LOGF(log,
              "%p Socket::Write() (socket = %" PRIu64
              ", src = %p, src_len = %" PRIu64 ", flags = 0) => %" PRIi64
              " (error = %s)",
              static_cast<void *>(this), static_cast<uint64_t>(m_socket), buf,
              static_cast<uint64_t>(src_len), static_cast<int64_t>(bytes_sent),
              error.AsCString());
  }

  return error;
}

Status Socket::Close() {
  Status error;
  if (!IsValid() || !m_should_close_fd)
    return error;

  Log *log = GetLog(LLDBLog::Connection);
  LLDB_LOGF(log, "%p Socket::Close (fd = %" PRIu64 ")",
            static_cast<void *>(this), static_cast<uint64_t>(m_socket));

  bool success = CloseSocket(m_socket) == 0;
  // A reference to a FD was passed in, set it to an invalid value
  m_socket = kInvalidSocketValue;
  if (!success) {
    SetLastError(error);
  }

  return error;
}

int Socket::GetOption(NativeSocket sockfd, int level, int option_name,
                      int &option_value) {
  get_socket_option_arg_type option_value_p =
      reinterpret_cast<get_socket_option_arg_type>(&option_value);
  socklen_t option_value_size = sizeof(int);
  return ::getsockopt(sockfd, level, option_name, option_value_p,
                      &option_value_size);
}

int Socket::SetOption(NativeSocket sockfd, int level, int option_name,
                      int option_value) {
  set_socket_option_arg_type option_value_p =
      reinterpret_cast<set_socket_option_arg_type>(&option_value);
  return ::setsockopt(sockfd, level, option_name, option_value_p,
                      sizeof(option_value));
}

size_t Socket::Send(const void *buf, const size_t num_bytes) {
  return ::send(m_socket, static_cast<const char *>(buf), num_bytes, 0);
}

void Socket::SetLastError(Status &error) {
#if defined(_WIN32)
  error = Status(::WSAGetLastError(), lldb::eErrorTypeWin32);
#else
  error = Status::FromErrno();
#endif
}

Status Socket::GetLastError() {
  std::error_code EC;
#ifdef _WIN32
  EC = llvm::mapWindowsError(WSAGetLastError());
#else
  EC = std::error_code(errno, std::generic_category());
#endif
  return EC;
}

int Socket::CloseSocket(NativeSocket sockfd) {
#ifdef _WIN32
  return ::closesocket(sockfd);
#else
  return ::close(sockfd);
#endif
}

NativeSocket Socket::CreateSocket(const int domain, const int type,
                                  const int protocol, Status &error) {
  error.Clear();
  auto socket_type = type;
#ifdef SOCK_CLOEXEC
  socket_type |= SOCK_CLOEXEC;
#endif
  auto sock = ::socket(domain, socket_type, protocol);
  if (sock == kInvalidSocketValue)
    SetLastError(error);

  return sock;
}

Status Socket::Accept(const Timeout<std::micro> &timeout, Socket *&socket) {
  socket = nullptr;
  MainLoop accept_loop;
  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>> expected_handles =
      Accept(accept_loop,
             [&accept_loop, &socket](std::unique_ptr<Socket> sock) {
               socket = sock.release();
               accept_loop.RequestTermination();
             });
  if (!expected_handles)
    return Status::FromError(expected_handles.takeError());
  if (timeout) {
    accept_loop.AddCallback(
        [](MainLoopBase &loop) { loop.RequestTermination(); }, *timeout);
  }
  if (Status status = accept_loop.Run(); status.Fail())
    return status;
  if (socket)
    return Status();
  return Status(std::make_error_code(std::errc::timed_out));
}

NativeSocket Socket::AcceptSocket(NativeSocket sockfd, struct sockaddr *addr,
                                  socklen_t *addrlen, Status &error) {
  error.Clear();
#if defined(SOCK_CLOEXEC) && defined(HAVE_ACCEPT4)
  int flags = SOCK_CLOEXEC;
  NativeSocket fd = llvm::sys::RetryAfterSignal(
      static_cast<NativeSocket>(-1), ::accept4, sockfd, addr, addrlen, flags);
#else
  NativeSocket fd = llvm::sys::RetryAfterSignal(
      static_cast<NativeSocket>(-1), ::accept, sockfd, addr, addrlen);
#endif
  if (fd == kInvalidSocketValue)
    SetLastError(error);
  return fd;
}

llvm::raw_ostream &lldb_private::operator<<(llvm::raw_ostream &OS,
                                            const Socket::HostAndPort &HP) {
  return OS << '[' << HP.hostname << ']' << ':' << HP.port;
}

std::optional<Socket::ProtocolModePair>
Socket::GetProtocolAndMode(llvm::StringRef scheme) {
  // Keep in sync with ConnectionFileDescriptor::Connect.
  return llvm::StringSwitch<std::optional<ProtocolModePair>>(scheme)
      .Case("listen", ProtocolModePair{SocketProtocol::ProtocolTcp,
                                       SocketMode::ModeAccept})
      .Cases("accept", "unix-accept",
             ProtocolModePair{SocketProtocol::ProtocolUnixDomain,
                              SocketMode::ModeAccept})
      .Case("unix-abstract-accept",
            ProtocolModePair{SocketProtocol::ProtocolUnixAbstract,
                             SocketMode::ModeAccept})
      .Cases("connect", "tcp-connect",
             ProtocolModePair{SocketProtocol::ProtocolTcp,
                              SocketMode::ModeConnect})
      .Case("udp", ProtocolModePair{SocketProtocol::ProtocolTcp,
                                    SocketMode::ModeConnect})
      .Case("unix-connect", ProtocolModePair{SocketProtocol::ProtocolUnixDomain,
                                             SocketMode::ModeConnect})
      .Case("unix-abstract-connect",
            ProtocolModePair{SocketProtocol::ProtocolUnixAbstract,
                             SocketMode::ModeConnect})
      .Default(std::nullopt);
}
