//===- DlfcnLoaderAdapter.h - dlfcn.h APIs via an adapter -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DlfcnLoaderAdapter: an abstract base class that implements the dlfcn.h APIs,
// including dlerror's per-thread error state, in terms of virtual *Impl
// methods supplied by subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H
#define ORC_RT_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H

#include "orc-rt-c/bedrock/sys/posix/DlfcnLoaderAdapter.h"
#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt/support/Error.h"

namespace orc_rt {

class DlfcnLoaderAdapter;

inline orc_rt_DlfcnLoaderAdapterRef wrap(DlfcnLoaderAdapter *DLA) noexcept {
  return reinterpret_cast<orc_rt_DlfcnLoaderAdapterRef>(DLA);
}

inline DlfcnLoaderAdapter *
unwrap(orc_rt_DlfcnLoaderAdapterRef DLARef) noexcept {
  return reinterpret_cast<DlfcnLoaderAdapter *>(DLARef);
}

/// Implements the dlfcn.h APIs in terms of virtual *Impl methods supplied by
/// subclasses, adding dlfcn.h's error-reporting conventions: when an *Impl
/// method fails, the corresponding operation returns its POSIX failure value,
/// logs the failure, and records the error's message for dlerror.
///
/// The C entry points in orc-rt-c/bedrock/sys/posix/DlfcnLoaderAdapter.h
/// forward to these operations, so that JIT'd code can call them.
///
/// A subclass passes its constructor a SimpleSymbolTable, to which the adapter
/// adds itself as the ORC_RT_DLFCN_LOADER_ADAPTER_INSTANCE instance symbol.
/// JIT'd code can reach the adapter once that table is published to the
/// session (see Session::addInstanceSymbols), which the creator should do only
/// once the whole system the adapter belongs to (e.g. with its loader engine)
/// has been constructed.
///
/// Published instance symbols can't be removed, so the adapter must stay alive
/// while any JIT'd code may call it. Session shutdown waits for calls into
/// JIT'd code to return before shutting down Services and running shutdown
/// callbacks, so it is enough for the adapter (or its owner) to be a Service,
/// or to be destroyed by a shutdown callback. (This relies on those calls
/// being made with keepalives: see Session::callWithKeepalive.)
class DlfcnLoaderAdapter {
public:
  DlfcnLoaderAdapter(const DlfcnLoaderAdapter &) = delete;
  DlfcnLoaderAdapter &operator=(const DlfcnLoaderAdapter &) = delete;
  DlfcnLoaderAdapter(DlfcnLoaderAdapter &&) = delete;
  DlfcnLoaderAdapter &operator=(DlfcnLoaderAdapter &&) = delete;
  virtual ~DlfcnLoaderAdapter();

  // dlopen, dlclose, dlsym, and dladdr forward to the corresponding *Impl
  // methods. dlopen, dlclose, and dlsym clear this thread's unread dlerror
  // message, as native dlfcn operations do on Darwin and glibc.

  /// Returns a handle on success, or null on failure.
  void *dlopen(const char *Path, int Mode) noexcept;

  /// Returns 0 on success, or -1 on failure.
  int dlclose(void *Handle) noexcept;

  /// Returns the symbol's address on success, or null on failure.
  void *dlsym(void *Handle, const char *Symbol) noexcept;

  /// Returns non-zero and fills in Info if Addr is in a known image, or 0
  /// otherwise. As with native dladdr, not finding Addr isn't an error, and
  /// dladdr leaves this thread's dlerror state alone. It only records a dlerror
  /// message if the lookup itself fails (dladdrImpl returns an Error), e.g.
  /// because a remote symbol table couldn't be reached.
  int dladdr(const void *Addr, orc_rt_Dl_info *Info) noexcept;

  /// Returns a message describing the most recent failure of an adapter
  /// operation on this thread, or null if there has been none since the last
  /// call to dlerror or the last successful dlopen, dlclose, or dlsym.
  ///
  /// - Error state is per-thread. It is currently shared by all adapters, but
  ///   clients shouldn't rely on that.
  /// - Each message is returned once: the next call returns null.
  /// - The string remains valid until the next adapter operation on this
  ///   thread, and must not be modified or freed.
  /// - If the message for a failure could not be kept because the thread is
  ///   exiting, a fixed message is returned instead. The original message is
  ///   logged.
  char *dlerror() noexcept;

protected:
  /// Adds this adapter to ST as the ORC_RT_DLFCN_LOADER_ADAPTER_INSTANCE
  /// instance symbol. Sets Err if ST already defines that symbol, in which case
  /// the creator should discard ST and this adapter. ST should be a fresh table
  /// for the adapter's system (see the class comment).
  DlfcnLoaderAdapter(SimpleSymbolTable &ST, Error &Err) noexcept;

private:
  // Subclasses implement the operations here. Each returns its result on
  // success, or an Error describing the failure; the public operations turn
  // failures into POSIX failure values and dlerror messages. dladdrImpl
  // returns whether Addr was found, and should return an Error only if the
  // lookup itself failed.
  virtual Expected<void *> dlopenImpl(const char *Path, int Mode) noexcept = 0;
  virtual Error dlcloseImpl(void *Handle) noexcept = 0;
  virtual Expected<void *> dlsymImpl(void *Handle,
                                     const char *Symbol) noexcept = 0;
  virtual Expected<bool> dladdrImpl(const void *Addr,
                                    orc_rt_Dl_info *Info) noexcept = 0;

  /// Resets this thread's dlerror state, then returns true if Err is success.
  /// Otherwise logs the failure, records Err's message for dlerror, and
  /// returns false. DescribeOp() returns a description of the failed operation
  /// and its arguments for the log, e.g. dlopen("libfoo.so", 0x2). It is only
  /// called on failure.
  template <typename DescribeOpFn>
  bool updateErrorState(Error Err, DescribeOpFn &&DescribeOp) noexcept;

  /// As above, for operations that return a value: returns Result's value on
  /// success, or ResultOnError on failure.
  template <typename T, typename DescribeOpFn>
  T updateErrorState(Expected<T> Result, T ResultOnError,
                     DescribeOpFn &&DescribeOp) noexcept;
};

} // namespace orc_rt

#endif // ORC_RT_BEDROCK_SYS_POSIX_DLFCNLOADERADAPTER_H
