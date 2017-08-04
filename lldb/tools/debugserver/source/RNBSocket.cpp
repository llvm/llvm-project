//===-- RNBSocket.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 12/12/07.
//
//===----------------------------------------------------------------------===//

#include "RNBSocket.h"
#include "DNBError.h"
#include "DNBLog.h"
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <termios.h>

#ifdef WITH_LOCKDOWN
#include "lockdown.h"
#endif

/* Once we have a RNBSocket object with a port # specified,
   this function is called to wait for an incoming connection.
   This function blocks while waiting for that connection.  */

bool ResolveIPV4HostName(const char *hostname, in_addr_t &addr) {
  if (hostname == NULL || hostname[0] == '\0' ||
      strcmp(hostname, "localhost") == 0 ||
      strcmp(hostname, "127.0.0.1") == 0) {
    addr = htonl(INADDR_LOOPBACK);
    return true;
  } else if (strcmp(hostname, "*") == 0) {
    addr = htonl(INADDR_ANY);
    return true;
  } else {
    // See if an IP address was specified as numbers
    int inet_pton_result = ::inet_pton(AF_INET, hostname, &addr);

    if (inet_pton_result == 1)
      return true;

    struct hostent *host_entry = gethostbyname(hostname);
    if (host_entry) {
      std::string ip_str(
          ::inet_ntoa(*(struct in_addr *)*host_entry->h_addr_list));
      inet_pton_result = ::inet_pton(AF_INET, ip_str.c_str(), &addr);
      if (inet_pton_result == 1)
        return true;
    }
  }
  return false;
}

rnb_err_t RNBSocket::Listen(const char *listen_host, uint16_t port,
                            PortBoundCallback callback,
                            const void *callback_baton) {
  // DNBLogThreadedIf(LOG_RNB_COMM, "%8u RNBSocket::%s called",
  // (uint32_t)m_timer.ElapsedMicroSeconds(true), __FUNCTION__);
  // Disconnect without saving errno
  Disconnect(false);

  // Now figure out the hostname that will be attaching and palce it into
  struct sockaddr_in listen_addr;
  ::memset(&listen_addr, 0, sizeof listen_addr);
  listen_addr.sin_len = sizeof listen_addr;
  listen_addr.sin_family = AF_INET;
  listen_addr.sin_port = htons(port);
  listen_addr.sin_addr.s_addr = INADDR_ANY;

  if (!ResolveIPV4HostName(listen_host, listen_addr.sin_addr.s_addr)) {
    DNBLogThreaded("error: failed to resolve connecting host '%s'",
                   listen_host);
    return rnb_err;
  }

  DNBError err;
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listen_fd == -1)
    err.SetError(errno, DNBError::POSIX);

  if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
    err.LogThreaded("::socket ( domain = AF_INET, type = SOCK_STREAM, protocol "
                    "= IPPROTO_TCP ) => socket = %i",
                    listen_fd);

  if (err.Fail())
    return rnb_err;

  // enable local address reuse
  SetSocketOption(listen_fd, SOL_SOCKET, SO_REUSEADDR, 1);

  struct sockaddr_in sa;
  ::memset(&sa, 0, sizeof sa);
  sa.sin_len = sizeof sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  sa.sin_addr.s_addr = INADDR_ANY; // Let incoming connections bind to any host
                                   // network interface (this is NOT who can
                                   // connect to us)
  int error = ::bind(listen_fd, (struct sockaddr *)&sa, sizeof(sa));
  if (error == -1)
    err.SetError(errno, DNBError::POSIX);

  if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
    err.LogThreaded(
        "::bind ( socket = %i, (struct sockaddr *) &sa, sizeof(sa)) )",
        listen_fd);

  if (err.Fail()) {
    ClosePort(listen_fd, false);
    return rnb_err;
  }

  error = ::listen(listen_fd, 5);
  if (error == -1)
    err.SetError(errno, DNBError::POSIX);

  if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
    err.LogThreaded("::listen ( socket = %i, backlog = 1 )", listen_fd);

  if (err.Fail()) {
    ClosePort(listen_fd, false);
    return rnb_err;
  }

  if (callback) {
    // We were asked to listen on port zero which means we
    // must now read the actual port that was given to us
    // as port zero is a special code for "find an open port
    // for me".
    if (port == 0) {
      socklen_t sa_len = sizeof(sa);
      if (getsockname(listen_fd, (struct sockaddr *)&sa, &sa_len) == 0) {
        port = ntohs(sa.sin_port);
        callback(callback_baton, port);
      }
    } else {
      callback(callback_baton, port);
    }
  }

  struct sockaddr_in accept_addr;
  ::memset(&accept_addr, 0, sizeof accept_addr);
  accept_addr.sin_len = sizeof accept_addr;

  bool accept_connection = false;

  // Loop until we are happy with our connection
  while (!accept_connection) {
    socklen_t accept_addr_len = sizeof accept_addr;
    m_fd =
        ::accept(listen_fd, (struct sockaddr *)&accept_addr, &accept_addr_len);

    if (m_fd == -1)
      err.SetError(errno, DNBError::POSIX);

    if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
      err.LogThreaded(
          "::accept ( socket = %i, address = %p, address_len = %u )", listen_fd,
          &accept_addr, accept_addr_len);

    if (err.Fail())
      break;

    if (listen_addr.sin_addr.s_addr == INADDR_ANY)
      accept_connection = true;
    else {
      if (accept_addr_len == listen_addr.sin_len &&
          accept_addr.sin_addr.s_addr == listen_addr.sin_addr.s_addr) {
        accept_connection = true;
      } else {
        ::close(m_fd);
        m_fd = -1;
        const uint8_t *accept_ip =
            (const uint8_t *)&accept_addr.sin_addr.s_addr;
        const uint8_t *listen_ip =
            (const uint8_t *)&listen_addr.sin_addr.s_addr;
        ::fprintf(stderr, "error: rejecting incoming connection from "
                          "%u.%u.%u.%u (expecting %u.%u.%u.%u)\n",
                  accept_ip[0], accept_ip[1], accept_ip[2], accept_ip[3],
                  listen_ip[0], listen_ip[1], listen_ip[2], listen_ip[3]);
        DNBLogThreaded("error: rejecting connection from %u.%u.%u.%u "
                       "(expecting %u.%u.%u.%u)",
                       accept_ip[0], accept_ip[1], accept_ip[2], accept_ip[3],
                       listen_ip[0], listen_ip[1], listen_ip[2], listen_ip[3]);
      }
    }
  }

  ClosePort(listen_fd, false);

  if (err.Fail()) {
    return rnb_err;
  } else {
    // Keep our TCP packets coming without any delays.
    SetSocketOption(m_fd, IPPROTO_TCP, TCP_NODELAY, 1);
  }

  return rnb_success;
}

rnb_err_t RNBSocket::Connect(const char *host, uint16_t port) {
  Disconnect(false);

  // Create the socket
  m_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (m_fd == -1)
    return rnb_err;

  // Enable local address reuse
  SetSocketOption(m_fd, SOL_SOCKET, SO_REUSEADDR, 1);

  struct sockaddr_in sa;
  ::memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);

  if (!ResolveIPV4HostName(host, sa.sin_addr.s_addr)) {
    DNBLogThreaded("error: failed to resolve host '%s'", host);
    Disconnect(false);
    return rnb_err;
  }

  if (-1 == ::connect(m_fd, (const struct sockaddr *)&sa, sizeof(sa))) {
    Disconnect(false);
    return rnb_err;
  }

  // Keep our TCP packets coming without any delays.
  SetSocketOption(m_fd, IPPROTO_TCP, TCP_NODELAY, 1);
  return rnb_success;
}

rnb_err_t RNBSocket::useFD(int fd) {
  if (fd < 0) {
    DNBLogThreadedIf(LOG_RNB_COMM, "Bad file descriptor passed in.");
    return rnb_err;
  }

  m_fd = fd;
  return rnb_success;
}

#ifdef WITH_LOCKDOWN
rnb_err_t RNBSocket::ConnectToService() {
  DNBLog("Connecting to com.apple.%s service...", DEBUGSERVER_PROGRAM_NAME);
  // Disconnect from any previous connections
  Disconnect(false);
  if (::secure_lockdown_checkin(&m_ld_conn, NULL, NULL) != kLDESuccess) {
    DNBLogThreadedIf(LOG_RNB_COMM,
                     "::secure_lockdown_checkin(&m_fd, NULL, NULL) failed");
    m_fd = -1;
    return rnb_not_connected;
  }
  m_fd = ::lockdown_get_socket(m_ld_conn);
  if (m_fd == -1) {
    DNBLogThreadedIf(LOG_RNB_COMM, "::lockdown_get_socket() failed");
    return rnb_not_connected;
  }
  m_fd_from_lockdown = true;
  return rnb_success;
}
#endif

rnb_err_t RNBSocket::OpenFile(const char *path) {
  DNBError err;
  m_fd = open(path, O_RDWR);
  if (m_fd == -1) {
    err.SetError(errno, DNBError::POSIX);
    err.LogThreaded("can't open file '%s'", path);
    return rnb_not_connected;
  } else {
    struct termios stdin_termios;

    if (::tcgetattr(m_fd, &stdin_termios) == 0) {
      stdin_termios.c_lflag &= ~ECHO;   // Turn off echoing
      stdin_termios.c_lflag &= ~ICANON; // Get one char at a time
      ::tcsetattr(m_fd, TCSANOW, &stdin_termios);
    }
  }
  return rnb_success;
}

int RNBSocket::SetSocketOption(int fd, int level, int option_name,
                               int option_value) {
  return ::setsockopt(fd, level, option_name, &option_value,
                      sizeof(option_value));
}

rnb_err_t RNBSocket::Disconnect(bool save_errno) {
#ifdef WITH_LOCKDOWN
  if (m_fd_from_lockdown) {
    m_fd_from_lockdown = false;
    m_fd = -1;
    lockdown_disconnect(m_ld_conn);
    return rnb_success;
  }
#endif
  return ClosePort(m_fd, save_errno);
}

rnb_err_t RNBSocket::Read(std::string &p) {
  char buf[1024];
  p.clear();

  // Note that BUF is on the stack so we must be careful to keep any
  // writes to BUF from overflowing or we'll have security issues.

  if (m_fd == -1)
    return rnb_err;

  // DNBLogThreadedIf(LOG_RNB_COMM, "%8u RNBSocket::%s calling read()",
  // (uint32_t)m_timer.ElapsedMicroSeconds(true), __FUNCTION__);
  DNBError err;
  ssize_t bytesread = read(m_fd, buf, sizeof(buf));
  if (bytesread <= 0)
    err.SetError(errno, DNBError::POSIX);
  else
    p.append(buf, bytesread);

  if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
    err.LogThreaded("::read ( %i, %p, %llu ) => %i", m_fd, buf, sizeof(buf),
                    (uint64_t)bytesread);

  // Our port went away - we have to mark this so IsConnected will return the
  // truth.
  if (bytesread == 0) {
    m_fd = -1;
    return rnb_not_connected;
  } else if (bytesread == -1) {
    m_fd = -1;
    return rnb_err;
  }
  // Strip spaces from the end of the buffer
  while (!p.empty() && isspace(p[p.size() - 1]))
    p.erase(p.size() - 1);

  // Most data in the debugserver packets valid printable characters...
  DNBLogThreadedIf(LOG_RNB_COMM, "read: %s", p.c_str());
  return rnb_success;
}

rnb_err_t RNBSocket::Write(const void *buffer, size_t length) {
  if (m_fd == -1)
    return rnb_err;

  DNBError err;
  ssize_t bytessent = write(m_fd, buffer, length);
  if (bytessent < 0)
    err.SetError(errno, DNBError::POSIX);

  if (err.Fail() || DNBLogCheckLogBit(LOG_RNB_COMM))
    err.LogThreaded("::write ( socket = %i, buffer = %p, length = %llu) => %i",
                    m_fd, buffer, length, (uint64_t)bytessent);

  if (bytessent < 0)
    return rnb_err;

  if ((size_t)bytessent != length)
    return rnb_err;

  DNBLogThreadedIf(
      LOG_RNB_PACKETS, "putpkt: %*s", (int)length,
      (char *)
          buffer); // All data is string based in debugserver, so this is safe
  DNBLogThreadedIf(LOG_RNB_COMM, "sent: %*s", (int)length, (char *)buffer);

  return rnb_success;
}

rnb_err_t RNBSocket::ClosePort(int &fd, bool save_errno) {
  int close_err = 0;
  if (fd > 0) {
    errno = 0;
    close_err = close(fd);
    fd = -1;
  }
  return close_err != 0 ? rnb_err : rnb_success;
}
