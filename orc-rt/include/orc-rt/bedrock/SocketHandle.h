//===- SocketHandle.h - An owning socket handle -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The system's socket type, and an owner for one.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SOCKETHANDLE_H
#define ORC_RT_BEDROCK_SOCKETHANDLE_H

#include <cstdint>
#include <utility>

namespace orc_rt {

#if defined(_WIN32)

/// Winsock's SOCKET, spelled as the integer it is a typedef for so that this
/// header does not pull in <winsock2.h>. The two are the same type.
///
/// TODO: Untested. There is no Windows reset() yet, and Winsock also needs
/// WSAStartup called somewhere.
using NativeSocketHandle = uintptr_t;

/// Winsock's INVALID_SOCKET.
inline constexpr NativeSocketHandle InvalidNativeSocketHandle =
    ~static_cast<NativeSocketHandle>(0);

#else

using NativeSocketHandle = int;

inline constexpr NativeSocketHandle InvalidNativeSocketHandle = -1;

#endif

/// Owns a socket, closing it on destruction.
///
/// Ownership only: says nothing about the socket's family or protocol, or
/// whether it is connected.
class SocketHandle {
public:
  SocketHandle() = default;

  /// Adopts H. Adopting InvalidNativeSocketHandle gives an empty handle, so the
  /// result of a socket call can be adopted before it is checked.
  explicit SocketHandle(NativeSocketHandle H) noexcept : H(H) {}

  SocketHandle(const SocketHandle &) = delete;
  SocketHandle &operator=(const SocketHandle &) = delete;

  SocketHandle(SocketHandle &&Other) noexcept : H(Other.release()) {}
  SocketHandle &operator=(SocketHandle &&Other) noexcept {
    // Take before closing: the other order closes the socket on a self-move.
    NativeSocketHandle Incoming = Other.release();
    reset();
    H = Incoming;
    return *this;
  }

  ~SocketHandle() { reset(); }

  explicit operator bool() const noexcept {
    return H != InvalidNativeSocketHandle;
  }

  /// The socket, which remains this handle's to close.
  NativeSocketHandle get() const noexcept { return H; }

  /// Surrenders the socket to the caller, who must close it.
  [[nodiscard]] NativeSocketHandle release() noexcept {
    return std::exchange(H, InvalidNativeSocketHandle);
  }

  /// Closes the socket, if any. A failed close is logged rather than reported:
  /// the socket is gone either way.
  void reset() noexcept;

private:
  NativeSocketHandle H = InvalidNativeSocketHandle;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SOCKETHANDLE_H
