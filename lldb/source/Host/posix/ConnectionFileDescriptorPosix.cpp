//===-- ConnectionFileDescriptorPosix.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
// Enable this special support for Apple builds where we can have unlimited
// select bounds. We tried switching to poll() and kqueue and we were panicing
// the kernel, so we have to stick with select for now.
#define _DARWIN_UNLIMITED_SELECT
#endif

#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/IOObject.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/SocketAddress.h"
#include "lldb/Host/StringConvert.h"

// C Includes
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifndef LLDB_DISABLE_POSIX
#include <termios.h>
#endif

// C++ Includes
#include <sstream>

// Other libraries and framework includes
#include "llvm/Support/ErrorHandling.h"
#if defined(__APPLE__)
#include "llvm/ADT/SmallVector.h"
#endif
// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Interpreter/Args.h"

using namespace lldb;
using namespace lldb_private;

const char *ConnectionFileDescriptor::LISTEN_SCHEME = "listen";
const char *ConnectionFileDescriptor::ACCEPT_SCHEME = "accept";
const char *ConnectionFileDescriptor::UNIX_ACCEPT_SCHEME = "unix-accept";
const char *ConnectionFileDescriptor::CONNECT_SCHEME = "connect";
const char *ConnectionFileDescriptor::TCP_CONNECT_SCHEME = "tcp-connect";
const char *ConnectionFileDescriptor::UDP_SCHEME = "udp";
const char *ConnectionFileDescriptor::UNIX_CONNECT_SCHEME = "unix-connect";
const char *ConnectionFileDescriptor::UNIX_ABSTRACT_CONNECT_SCHEME =
    "unix-abstract-connect";
const char *ConnectionFileDescriptor::FD_SCHEME = "fd";
const char *ConnectionFileDescriptor::FILE_SCHEME = "file";

namespace {

const char *GetURLAddress(const char *url, const char *scheme) {
  const auto prefix = std::string(scheme) + "://";
  if (strstr(url, prefix.c_str()) != url)
    return nullptr;

  return url + prefix.size();
}
}

ConnectionFileDescriptor::ConnectionFileDescriptor(bool child_processes_inherit)
    : Connection(), m_pipe(), m_mutex(Mutex::eMutexTypeRecursive),
      m_shutting_down(false), m_waiting_for_accept(false),
      m_child_processes_inherit(child_processes_inherit) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION |
                                                  LIBLLDB_LOG_OBJECT));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::ConnectionFileDescriptor ()",
                static_cast<void *>(this));
}

ConnectionFileDescriptor::ConnectionFileDescriptor(int fd, bool owns_fd)
    : Connection(), m_pipe(), m_mutex(Mutex::eMutexTypeRecursive),
      m_shutting_down(false), m_waiting_for_accept(false),
      m_child_processes_inherit(false) {
  m_write_sp.reset(new File(fd, owns_fd));
  m_read_sp.reset(new File(fd, false));

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION |
                                                  LIBLLDB_LOG_OBJECT));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::ConnectionFileDescriptor (fd = "
                "%i, owns_fd = %i)",
                static_cast<void *>(this), fd, owns_fd);
  OpenCommandPipe();
}

ConnectionFileDescriptor::ConnectionFileDescriptor(Socket *socket)
    : Connection(), m_pipe(), m_mutex(Mutex::eMutexTypeRecursive),
      m_shutting_down(false), m_waiting_for_accept(false),
      m_child_processes_inherit(false) {
  InitializeSocket(socket);
}

ConnectionFileDescriptor::~ConnectionFileDescriptor() {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION |
                                                  LIBLLDB_LOG_OBJECT));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::~ConnectionFileDescriptor ()",
                static_cast<void *>(this));
  Disconnect(NULL);
  CloseCommandPipe();
}

void ConnectionFileDescriptor::OpenCommandPipe() {
  CloseCommandPipe();

  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  // Make the command file descriptor here:
  Error result = m_pipe.CreateNew(m_child_processes_inherit);
  if (!result.Success()) {
    if (log)
      log->Printf("%p ConnectionFileDescriptor::OpenCommandPipe () - could not "
                  "make pipe: %s",
                  static_cast<void *>(this), result.AsCString());
  } else {
    if (log)
      log->Printf("%p ConnectionFileDescriptor::OpenCommandPipe() - success "
                  "readfd=%d writefd=%d",
                  static_cast<void *>(this), m_pipe.GetReadFileDescriptor(),
                  m_pipe.GetWriteFileDescriptor());
  }
}

void ConnectionFileDescriptor::CloseCommandPipe() {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::CloseCommandPipe()",
                static_cast<void *>(this));

  m_pipe.Close();
}

bool ConnectionFileDescriptor::IsConnected() const {
  return (m_read_sp && m_read_sp->IsValid()) ||
         (m_write_sp && m_write_sp->IsValid());
}

ConnectionStatus ConnectionFileDescriptor::Connect(const char *s,
                                                   Error *error_ptr) {
  Mutex::Locker locker(m_mutex);
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::Connect (url = '%s')",
                static_cast<void *>(this), s);

  OpenCommandPipe();

  if (s && s[0]) {
    const char *addr = nullptr;
    if ((addr = GetURLAddress(s, LISTEN_SCHEME))) {
      // listen://HOST:PORT
      return SocketListenAndAccept(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, ACCEPT_SCHEME))) {
      // unix://SOCKNAME
      return NamedSocketAccept(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, UNIX_ACCEPT_SCHEME))) {
      // unix://SOCKNAME
      return NamedSocketAccept(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, CONNECT_SCHEME))) {
      return ConnectTCP(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, TCP_CONNECT_SCHEME))) {
      return ConnectTCP(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, UDP_SCHEME))) {
      return ConnectUDP(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, UNIX_CONNECT_SCHEME))) {
      // unix-connect://SOCKNAME
      return NamedSocketConnect(addr, error_ptr);
    } else if ((addr = GetURLAddress(s, UNIX_ABSTRACT_CONNECT_SCHEME))) {
      // unix-abstract-connect://SOCKNAME
      return UnixAbstractSocketConnect(addr, error_ptr);
    }
#ifndef LLDB_DISABLE_POSIX
    else if ((addr = GetURLAddress(s, FD_SCHEME))) {
      // Just passing a native file descriptor within this current process
      // that is already opened (possibly from a service or other source).
      bool success = false;
      int fd = StringConvert::ToSInt32(addr, -1, 0, &success);

      if (success) {
        // We have what looks to be a valid file descriptor, but we
        // should make sure it is. We currently are doing this by trying to
        // get the flags from the file descriptor and making sure it
        // isn't a bad fd.
        errno = 0;
        int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags == -1 || errno == EBADF) {
          if (error_ptr)
            error_ptr->SetErrorStringWithFormat("stale file descriptor: %s", s);
          m_read_sp.reset();
          m_write_sp.reset();
          return eConnectionStatusError;
        } else {
          // Don't take ownership of a file descriptor that gets passed
          // to us since someone else opened the file descriptor and
          // handed it to us.
          // TODO: Since are using a URL to open connection we should
          // eventually parse options using the web standard where we
          // have "fd://123?opt1=value;opt2=value" and we can have an
          // option be "owns=1" or "owns=0" or something like this to
          // allow us to specify this. For now, we assume we must
          // assume we don't own it.

          std::unique_ptr<TCPSocket> tcp_socket;
          tcp_socket.reset(new TCPSocket(fd, false));
          // Try and get a socket option from this file descriptor to
          // see if this is a socket and set m_is_socket accordingly.
          int resuse;
          bool is_socket =
              !!tcp_socket->GetOption(SOL_SOCKET, SO_REUSEADDR, resuse);
          if (is_socket) {
            m_read_sp = std::move(tcp_socket);
            m_write_sp = m_read_sp;
          } else {
            m_read_sp.reset(new File(fd, false));
            m_write_sp.reset(new File(fd, false));
          }
          m_uri.assign(addr);
          return eConnectionStatusSuccess;
        }
      }

      if (error_ptr)
        error_ptr->SetErrorStringWithFormat("invalid file descriptor: \"%s\"",
                                            s);
      m_read_sp.reset();
      m_write_sp.reset();
      return eConnectionStatusError;
    } else if ((addr = GetURLAddress(s, FILE_SCHEME))) {
      // file:///PATH
      const char *path = addr;
      int fd = -1;
      do {
        fd = ::open(path, O_RDWR);
      } while (fd == -1 && errno == EINTR);

      if (fd == -1) {
        if (error_ptr)
          error_ptr->SetErrorToErrno();
        return eConnectionStatusError;
      }

      if (::isatty(fd)) {
        // Set up serial terminal emulation
        struct termios options;
        ::tcgetattr(fd, &options);

        // Set port speed to maximum
        ::cfsetospeed(&options, B115200);
        ::cfsetispeed(&options, B115200);

        // Raw input, disable echo and signals
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

        // Make sure only one character is needed to return from a read
        options.c_cc[VMIN] = 1;
        options.c_cc[VTIME] = 0;

        ::tcsetattr(fd, TCSANOW, &options);
      }

      int flags = ::fcntl(fd, F_GETFL, 0);
      if (flags >= 0) {
        if ((flags & O_NONBLOCK) == 0) {
          flags |= O_NONBLOCK;
          ::fcntl(fd, F_SETFL, flags);
        }
      }
      m_read_sp.reset(new File(fd, true));
      m_write_sp.reset(new File(fd, false));
      return eConnectionStatusSuccess;
    }
#endif
    if (error_ptr)
      error_ptr->SetErrorStringWithFormat("unsupported connection URL: '%s'",
                                          s);
    return eConnectionStatusError;
  }
  if (error_ptr)
    error_ptr->SetErrorString("invalid connect arguments");
  return eConnectionStatusError;
}

bool ConnectionFileDescriptor::InterruptRead() {
  size_t bytes_written = 0;
  Error result = m_pipe.Write("i", 1, bytes_written);
  return result.Success();
}

ConnectionStatus ConnectionFileDescriptor::Disconnect(Error *error_ptr) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf("%p ConnectionFileDescriptor::Disconnect ()",
                static_cast<void *>(this));

  ConnectionStatus status = eConnectionStatusSuccess;

  if (!IsConnected()) {
    if (log)
      log->Printf(
          "%p ConnectionFileDescriptor::Disconnect(): Nothing to disconnect",
          static_cast<void *>(this));
    return eConnectionStatusSuccess;
  }

  if (m_read_sp && m_read_sp->IsValid() &&
      m_read_sp->GetFdType() == IOObject::eFDTypeSocket)
    static_cast<Socket &>(*m_read_sp).PreDisconnect();

  // Try to get the ConnectionFileDescriptor's mutex.  If we fail, that is quite
  // likely
  // because somebody is doing a blocking read on our file descriptor.  If
  // that's the case,
  // then send the "q" char to the command file channel so the read will wake up
  // and the connection
  // will then know to shut down.

  m_shutting_down = true;

  Mutex::Locker locker;
  bool got_lock = locker.TryLock(m_mutex);

  if (!got_lock) {
    if (m_pipe.CanWrite()) {
      size_t bytes_written = 0;
      Error result = m_pipe.Write("q", 1, bytes_written);
      if (log)
        log->Printf("%p ConnectionFileDescriptor::Disconnect(): Couldn't get "
                    "the lock, sent 'q' to %d, error = '%s'.",
                    static_cast<void *>(this), m_pipe.GetWriteFileDescriptor(),
                    result.AsCString());
    } else if (log) {
      log->Printf("%p ConnectionFileDescriptor::Disconnect(): Couldn't get the "
                  "lock, but no command pipe is available.",
                  static_cast<void *>(this));
    }
    locker.Lock(m_mutex);
  }

  Error error = m_read_sp->Close();
  Error error2 = m_write_sp->Close();
  if (error.Fail() || error2.Fail())
    status = eConnectionStatusError;
  if (error_ptr)
    *error_ptr = error.Fail() ? error : error2;

  // Close any pipes we were using for async interrupts
  m_pipe.Close();

  m_uri.clear();
  m_shutting_down = false;
  return status;
}

size_t ConnectionFileDescriptor::Read(void *dst, size_t dst_len,
                                      uint32_t timeout_usec,
                                      ConnectionStatus &status,
                                      Error *error_ptr) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));

  Mutex::Locker locker;
  bool got_lock = locker.TryLock(m_mutex);
  if (!got_lock) {
    if (log)
      log->Printf("%p ConnectionFileDescriptor::Read () failed to get the "
                  "connection lock.",
                  static_cast<void *>(this));
    if (error_ptr)
      error_ptr->SetErrorString("failed to get the connection lock for read.");

    status = eConnectionStatusTimedOut;
    return 0;
  }

  if (m_shutting_down) {
    status = eConnectionStatusError;
    return 0;
  }

  status = BytesAvailable(timeout_usec, error_ptr);
  if (status != eConnectionStatusSuccess)
    return 0;

  Error error;
  size_t bytes_read = dst_len;
  error = m_read_sp->Read(dst, bytes_read);

  if (log) {
    log->Printf("%p ConnectionFileDescriptor::Read()  fd = %" PRIu64
                ", dst = %p, dst_len = %" PRIu64 ") => %" PRIu64 ", error = %s",
                static_cast<void *>(this),
                static_cast<uint64_t>(m_read_sp->GetWaitableHandle()),
                static_cast<void *>(dst), static_cast<uint64_t>(dst_len),
                static_cast<uint64_t>(bytes_read), error.AsCString());
  }

  if (bytes_read == 0) {
    error.Clear(); // End-of-file.  Do not automatically close; pass along for
                   // the end-of-file handlers.
    status = eConnectionStatusEndOfFile;
  }

  if (error_ptr)
    *error_ptr = error;

  if (error.Fail()) {
    uint32_t error_value = error.GetError();
    switch (error_value) {
    case EAGAIN: // The file was marked for non-blocking I/O, and no data were
                 // ready to be read.
      if (m_read_sp->GetFdType() == IOObject::eFDTypeSocket)
        status = eConnectionStatusTimedOut;
      else
        status = eConnectionStatusSuccess;
      return 0;

    case EFAULT:  // Buf points outside the allocated address space.
    case EINTR:   // A read from a slow device was interrupted before any data
                  // arrived by the delivery of a signal.
    case EINVAL:  // The pointer associated with fildes was negative.
    case EIO:     // An I/O error occurred while reading from the file system.
                  // The process group is orphaned.
                  // The file is a regular file, nbyte is greater than 0,
                  // the starting position is before the end-of-file, and
                  // the starting position is greater than or equal to the
                  // offset maximum established for the open file
                  // descriptor associated with fildes.
    case EISDIR:  // An attempt is made to read a directory.
    case ENOBUFS: // An attempt to allocate a memory buffer fails.
    case ENOMEM:  // Insufficient memory is available.
      status = eConnectionStatusError;
      break; // Break to close....

    case ENOENT:     // no such file or directory
    case EBADF:      // fildes is not a valid file or socket descriptor open for
                     // reading.
    case ENXIO:      // An action is requested of a device that does not exist..
                     // A requested action cannot be performed by the device.
    case ECONNRESET: // The connection is closed by the peer during a read
                     // attempt on a socket.
    case ENOTCONN:   // A read is attempted on an unconnected socket.
      status = eConnectionStatusLostConnection;
      break; // Break to close....

    case ETIMEDOUT: // A transmission timeout occurs during a read attempt on a
                    // socket.
      status = eConnectionStatusTimedOut;
      return 0;

    default:
      if (log)
        log->Printf(
            "%p ConnectionFileDescriptor::Read (), unexpected error: %s",
            static_cast<void *>(this), strerror(error_value));
      status = eConnectionStatusError;
      break; // Break to close....
    }

    return 0;
  }
  return bytes_read;
}

size_t ConnectionFileDescriptor::Write(const void *src, size_t src_len,
                                       ConnectionStatus &status,
                                       Error *error_ptr) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf(
        "%p ConnectionFileDescriptor::Write (src = %p, src_len = %" PRIu64 ")",
        static_cast<void *>(this), static_cast<const void *>(src),
        static_cast<uint64_t>(src_len));

  if (!IsConnected()) {
    if (error_ptr)
      error_ptr->SetErrorString("not connected");
    status = eConnectionStatusNoConnection;
    return 0;
  }

  Error error;

  size_t bytes_sent = src_len;
  error = m_write_sp->Write(src, bytes_sent);

  if (log) {
    log->Printf("%p ConnectionFileDescriptor::Write(fd = %" PRIu64
                ", src = %p, src_len = %" PRIu64 ") => %" PRIu64
                " (error = %s)",
                static_cast<void *>(this),
                static_cast<uint64_t>(m_write_sp->GetWaitableHandle()),
                static_cast<const void *>(src), static_cast<uint64_t>(src_len),
                static_cast<uint64_t>(bytes_sent), error.AsCString());
  }

  if (error_ptr)
    *error_ptr = error;

  if (error.Fail()) {
    switch (error.GetError()) {
    case EAGAIN:
    case EINTR:
      status = eConnectionStatusSuccess;
      return 0;

    case ECONNRESET: // The connection is closed by the peer during a read
                     // attempt on a socket.
    case ENOTCONN:   // A read is attempted on an unconnected socket.
      status = eConnectionStatusLostConnection;
      break; // Break to close....

    default:
      status = eConnectionStatusError;
      break; // Break to close....
    }

    return 0;
  }

  status = eConnectionStatusSuccess;
  return bytes_sent;
}

std::string ConnectionFileDescriptor::GetURI() { return m_uri; }

// This ConnectionFileDescriptor::BytesAvailable() uses select().
//
// PROS:
//  - select is consistent across most unix platforms
//  - The Apple specific version allows for unlimited fds in the fd_sets by
//    setting the _DARWIN_UNLIMITED_SELECT define prior to including the
//    required header files.
// CONS:
//  - on non-Apple platforms, only supports file descriptors up to FD_SETSIZE.
//     This implementation  will assert if it runs into that hard limit to let
//     users know that another ConnectionFileDescriptor::BytesAvailable() should
//     be used or a new version of ConnectionFileDescriptor::BytesAvailable()
//     should be written for the system that is running into the limitations.

#if defined(__APPLE__)
#define FD_SET_DATA(fds) fds.data()
#else
#define FD_SET_DATA(fds) &fds
#endif

ConnectionStatus ConnectionFileDescriptor::BytesAvailable(uint32_t timeout_usec,
                                                          Error *error_ptr) {
  // Don't need to take the mutex here separately since we are only called from
  // Read.  If we
  // ever get used more generally we will need to lock here as well.

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_CONNECTION));
  if (log)
    log->Printf(
        "%p ConnectionFileDescriptor::BytesAvailable (timeout_usec = %u)",
        static_cast<void *>(this), timeout_usec);

  struct timeval *tv_ptr;
  struct timeval tv;
  if (timeout_usec == UINT32_MAX) {
    // Inifinite wait...
    tv_ptr = nullptr;
  } else {
    TimeValue time_value;
    time_value.OffsetWithMicroSeconds(timeout_usec);
    tv.tv_sec = time_value.seconds();
    tv.tv_usec = time_value.microseconds();
    tv_ptr = &tv;
  }

  // Make a copy of the file descriptors to make sure we don't
  // have another thread change these values out from under us
  // and cause problems in the loop below where like in FS_SET()
  const IOObject::WaitableHandle handle = m_read_sp->GetWaitableHandle();
  const int pipe_fd = m_pipe.GetReadFileDescriptor();

  if (handle != IOObject::kInvalidHandleValue) {
#if defined(_MSC_VER)
    // select() won't accept pipes on Windows.  The entire Windows codepath
    // needs to be
    // converted over to using WaitForMultipleObjects and event HANDLEs, but for
    // now at least
    // this will allow ::select() to not return an error.
    const bool have_pipe_fd = false;
#else
    const bool have_pipe_fd = pipe_fd >= 0;
#if !defined(__APPLE__)
    assert(handle < FD_SETSIZE);
    if (have_pipe_fd)
      assert(pipe_fd < FD_SETSIZE);
#endif
#endif
    while (handle == m_read_sp->GetWaitableHandle()) {
      const int nfds = std::max<int>(handle, pipe_fd) + 1;
#if defined(__APPLE__)
      llvm::SmallVector<fd_set, 1> read_fds;
      read_fds.resize((nfds / FD_SETSIZE) + 1);
      for (size_t i = 0; i < read_fds.size(); ++i)
        FD_ZERO(&read_fds[i]);
// FD_SET doesn't bounds check, it just happily walks off the end
// but we have taken care of making the extra storage with our
// SmallVector of fd_set objects
#else
      fd_set read_fds;
      FD_ZERO(&read_fds);
#endif
      FD_SET(handle, FD_SET_DATA(read_fds));
      if (have_pipe_fd)
        FD_SET(pipe_fd, FD_SET_DATA(read_fds));

      Error error;

      if (log) {
        if (have_pipe_fd)
          log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select "
                      "(nfds=%i, fds={%i, %i}, NULL, NULL, timeout=%p)...",
                      static_cast<void *>(this), nfds, handle, pipe_fd,
                      static_cast<void *>(tv_ptr));
        else
          log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select "
                      "(nfds=%i, fds={%i}, NULL, NULL, timeout=%p)...",
                      static_cast<void *>(this), nfds, handle,
                      static_cast<void *>(tv_ptr));
      }

      const int num_set_fds =
          ::select(nfds, FD_SET_DATA(read_fds), NULL, NULL, tv_ptr);
      if (num_set_fds < 0)
        error.SetErrorToErrno();
      else
        error.Clear();

      if (log) {
        if (have_pipe_fd)
          log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select "
                      "(nfds=%i, fds={%i, %i}, NULL, NULL, timeout=%p) "
                      "=> %d, error = %s",
                      static_cast<void *>(this), nfds, handle, pipe_fd,
                      static_cast<void *>(tv_ptr), num_set_fds,
                      error.AsCString());
        else
          log->Printf("%p ConnectionFileDescriptor::BytesAvailable()  ::select "
                      "(nfds=%i, fds={%i}, NULL, NULL, timeout=%p) => "
                      "%d, error = %s",
                      static_cast<void *>(this), nfds, handle,
                      static_cast<void *>(tv_ptr), num_set_fds,
                      error.AsCString());
      }

      if (error_ptr)
        *error_ptr = error;

      if (error.Fail()) {
        switch (error.GetError()) {
        case EBADF: // One of the descriptor sets specified an invalid
                    // descriptor.
          return eConnectionStatusLostConnection;

        case EINVAL: // The specified time limit is invalid. One of its
                     // components is negative or too large.
        default:     // Other unknown error
          return eConnectionStatusError;

        case EAGAIN: // The kernel was (perhaps temporarily) unable to
                     // allocate the requested number of file descriptors,
                     // or we have non-blocking IO
        case EINTR:  // A signal was delivered before the time limit
          // expired and before any of the selected events
          // occurred.
          break; // Lets keep reading to until we timeout
        }
      } else if (num_set_fds == 0) {
        return eConnectionStatusTimedOut;
      } else if (num_set_fds > 0) {
        if (FD_ISSET(handle, FD_SET_DATA(read_fds)))
          return eConnectionStatusSuccess;
        if (have_pipe_fd && FD_ISSET(pipe_fd, FD_SET_DATA(read_fds))) {
          // There is an interrupt or exit command in the command pipe
          // Read the data from that pipe:
          char buffer[1];

          ssize_t bytes_read;

          do {
            bytes_read = ::read(pipe_fd, buffer, sizeof(buffer));
          } while (bytes_read < 0 && errno == EINTR);

          switch (buffer[0]) {
          case 'q':
            if (log)
              log->Printf("%p ConnectionFileDescriptor::BytesAvailable() "
                          "got data: %c from the command channel.",
                          static_cast<void *>(this), buffer[0]);
            return eConnectionStatusEndOfFile;
          case 'i':
            // Interrupt the current read
            return eConnectionStatusInterrupted;
          }
        }
      }
    }
  }

  if (error_ptr)
    error_ptr->SetErrorString("not connected");
  return eConnectionStatusLostConnection;
}

ConnectionStatus
ConnectionFileDescriptor::NamedSocketAccept(const char *socket_name,
                                            Error *error_ptr) {
  Socket *socket = nullptr;
  Error error =
      Socket::UnixDomainAccept(socket_name, m_child_processes_inherit, socket);
  if (error_ptr)
    *error_ptr = error;
  m_write_sp.reset(socket);
  m_read_sp = m_write_sp;
  if (error.Fail()) {
    return eConnectionStatusError;
  }
  m_uri.assign(socket_name);
  return eConnectionStatusSuccess;
}

ConnectionStatus
ConnectionFileDescriptor::NamedSocketConnect(const char *socket_name,
                                             Error *error_ptr) {
  Socket *socket = nullptr;
  Error error =
      Socket::UnixDomainConnect(socket_name, m_child_processes_inherit, socket);
  if (error_ptr)
    *error_ptr = error;
  m_write_sp.reset(socket);
  m_read_sp = m_write_sp;
  if (error.Fail()) {
    return eConnectionStatusError;
  }
  m_uri.assign(socket_name);
  return eConnectionStatusSuccess;
}

lldb::ConnectionStatus
ConnectionFileDescriptor::UnixAbstractSocketConnect(const char *socket_name,
                                                    Error *error_ptr) {
  Socket *socket = nullptr;
  Error error = Socket::UnixAbstractConnect(socket_name,
                                            m_child_processes_inherit, socket);
  if (error_ptr)
    *error_ptr = error;
  m_write_sp.reset(socket);
  m_read_sp = m_write_sp;
  if (error.Fail()) {
    return eConnectionStatusError;
  }
  m_uri.assign(socket_name);
  return eConnectionStatusSuccess;
}

ConnectionStatus
ConnectionFileDescriptor::SocketListenAndAccept(const char *s,
                                                Error *error_ptr) {
  m_port_predicate.SetValue(0, eBroadcastNever);

  Socket *socket = nullptr;
  m_waiting_for_accept = true;
  Error error = Socket::TcpListen(s, m_child_processes_inherit, socket,
                                  &m_port_predicate);
  if (error_ptr)
    *error_ptr = error;
  if (error.Fail())
    return eConnectionStatusError;

  std::unique_ptr<Socket> listening_socket_up;

  listening_socket_up.reset(socket);
  socket = nullptr;
  error = listening_socket_up->Accept(s, m_child_processes_inherit, socket);
  listening_socket_up.reset();
  if (error_ptr)
    *error_ptr = error;
  if (error.Fail())
    return eConnectionStatusError;

  InitializeSocket(socket);
  return eConnectionStatusSuccess;
}

ConnectionStatus ConnectionFileDescriptor::ConnectTCP(const char *s,
                                                      Error *error_ptr) {
  Socket *socket = nullptr;
  Error error = Socket::TcpConnect(s, m_child_processes_inherit, socket);
  if (error_ptr)
    *error_ptr = error;
  m_write_sp.reset(socket);
  m_read_sp = m_write_sp;
  if (error.Fail()) {
    return eConnectionStatusError;
  }
  m_uri.assign(s);
  return eConnectionStatusSuccess;
}

ConnectionStatus ConnectionFileDescriptor::ConnectUDP(const char *s,
                                                      Error *error_ptr) {
  Socket *send_socket = nullptr;
  Socket *recv_socket = nullptr;
  Error error = Socket::UdpConnect(s, m_child_processes_inherit, send_socket,
                                   recv_socket);
  if (error_ptr)
    *error_ptr = error;
  m_write_sp.reset(send_socket);
  m_read_sp.reset(recv_socket);
  if (error.Fail()) {
    return eConnectionStatusError;
  }
  m_uri.assign(s);
  return eConnectionStatusSuccess;
}

uint16_t ConnectionFileDescriptor::GetListeningPort(uint32_t timeout_sec) {
  uint16_t bound_port = 0;
  if (timeout_sec == UINT32_MAX)
    m_port_predicate.WaitForValueNotEqualTo(0, bound_port);
  else {
    TimeValue timeout = TimeValue::Now();
    timeout.OffsetWithSeconds(timeout_sec);
    m_port_predicate.WaitForValueNotEqualTo(0, bound_port, &timeout);
  }
  return bound_port;
}

bool ConnectionFileDescriptor::GetChildProcessesInherit() const {
  return m_child_processes_inherit;
}

void ConnectionFileDescriptor::SetChildProcessesInherit(
    bool child_processes_inherit) {
  m_child_processes_inherit = child_processes_inherit;
}

void ConnectionFileDescriptor::InitializeSocket(Socket *socket) {
  assert(socket->GetSocketProtocol() == Socket::ProtocolTcp);
  TCPSocket *tcp_socket = static_cast<TCPSocket *>(socket);

  m_write_sp.reset(socket);
  m_read_sp = m_write_sp;
  StreamString strm;
  strm.Printf("connect://%s:%u", tcp_socket->GetRemoteIPAddress().c_str(),
              tcp_socket->GetRemotePortNumber());
  m_uri.swap(strm.GetString());
}
