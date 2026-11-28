//===- DlfcnLoaderAdapter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DlfcnLoaderAdapter and its C API.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sys/posix/DlfcnLoaderAdapter.h"

#include "orc-rt-c/support/Compiler.h"
#include "orc-rt-c/support/Logging.h"
#include "orc-rt-internal/support/StringExtras.h"

#include <string>

namespace orc_rt {

extern "C" {

void *orc_rt_DlfcnLoaderAdapter_dlopen(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                       const char *Path, int Mode) noexcept {
  return unwrap(Ldr)->dlopen(Path, Mode);
}

int orc_rt_DlfcnLoaderAdapter_dlclose(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                      void *Handle) noexcept {
  return unwrap(Ldr)->dlclose(Handle);
}

void *orc_rt_DlfcnLoaderAdapter_dlsym(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                      void *Handle,
                                      const char *Symbol) noexcept {
  return unwrap(Ldr)->dlsym(Handle, Symbol);
}

int orc_rt_DlfcnLoaderAdapter_dladdr(orc_rt_DlfcnLoaderAdapterRef Ldr,
                                     const void *Addr,
                                     orc_rt_Dl_info *Info) noexcept {
  return unwrap(Ldr)->dladdr(Addr, Info);
}

char *
orc_rt_DlfcnLoaderAdapter_dlerror(orc_rt_DlfcnLoaderAdapterRef Ldr) noexcept {
  return unwrap(Ldr)->dlerror();
}

} // extern "C"

namespace {

// dlerror state: one error message per thread, shared by all adapters.
//
// ErrMsg holds the message (Buffer) and whether dlerror has returned it
// (Checked). It is initialized on first use, and destroyed when its thread
// exits. Thread-exit destructors that run after it (e.g. those of
// thread_locals constructed earlier) may still call adapter APIs, so every
// access to ErrMsg is guarded by ErrMsgState:
//
//   Valid              - ErrMsg is alive, or not yet constructed: use it.
//   DestroyedNoError   - ErrMsg has been destroyed, and there is no error to
//                        report.
//   DestroyedWithError - ErrMsg has been destroyed, and an error whose message
//                        couldn't be kept is waiting to be reported: dlerror
//                        returns ErrorDuringThreadExitMessage once.
//
// ErrMsgState must be a separate, constant-initialized, trivially
// destructible thread_local: that keeps it alive until every dynamically
// initialized thread_local on its thread has been destroyed.

enum class ErrorMessageState { Valid, DestroyedNoError, DestroyedWithError };
thread_local ErrorMessageState ErrMsgState = ErrorMessageState::Valid;

struct ErrorMessage {
  ~ErrorMessage() {
    // A message that has already been read is not a pending error.
    ErrMsgState = (!Buffer.empty() && !Checked)
                      ? ErrorMessageState::DestroyedWithError
                      : ErrorMessageState::DestroyedNoError;
  }

  std::string Buffer;
  bool Checked = false;
};

thread_local ErrorMessage ErrMsg;

// Returned by dlerror for errors whose messages could not be stored: those
// raised after ErrMsg was destroyed, and those still unread when it was. The
// messages themselves are logged.
const char ErrorDuringThreadExitMessage[] =
    "dlfcn error during thread exit (see log for error message)";

} // namespace

DlfcnLoaderAdapter::DlfcnLoaderAdapter(SimpleSymbolTable &ST,
                                       Error &Err) noexcept {
  ErrorAsOutParameter _(Err);
  std::pair<SymbolNameSpec, const void *> InstanceSym[] = {
      {SymbolNameSpec::c(
           ORC_RT_STRINGIFY(ORC_RT_DLFCN_LOADER_ADAPTER_INSTANCE)),
       static_cast<const void *>(this)}};
  Err = ST.addUnique(InstanceSym);
}

DlfcnLoaderAdapter::~DlfcnLoaderAdapter() = default;

void *DlfcnLoaderAdapter::dlopen(const char *Path, int Mode) noexcept {
  return updateErrorState<void *>(dlopenImpl(Path, Mode), nullptr, [&] {
    return (StringOutputStream()
            << "dlopen(" << quote(Path) << ", " << hex(Mode) << ")")
        .str();
  });
}

int DlfcnLoaderAdapter::dlclose(void *Handle) noexcept {
  bool Succeeded = updateErrorState(dlcloseImpl(Handle), [&] {
    return (StringOutputStream() << "dlclose(" << Handle << ")").str();
  });
  return Succeeded ? 0 : -1;
}

void *DlfcnLoaderAdapter::dlsym(void *Handle, const char *Symbol) noexcept {
  return updateErrorState<void *>(dlsymImpl(Handle, Symbol), nullptr, [&] {
    return (StringOutputStream()
            << "dlsym(" << Handle << ", " << quote(Symbol) << ")")
        .str();
  });
}

int DlfcnLoaderAdapter::dladdr(const void *Addr,
                               orc_rt_Dl_info *Info) noexcept {
  // Unlike the other operations, a completed lookup (whether or not Addr was
  // found) leaves the dlerror state alone, as native dladdr does.
  auto Found = dladdrImpl(Addr, Info);
  if (Found)
    return *Found;
  updateErrorState(Found.takeError(), [&] {
    return (StringOutputStream() << "dladdr(" << Addr << ")").str();
  });
  return 0;
}

char *DlfcnLoaderAdapter::dlerror() noexcept {
  switch (ErrMsgState) {
  case ErrorMessageState::Valid:
    if (ErrMsg.Checked)
      ErrMsg.Buffer.clear();
    ErrMsg.Checked = true;
    return ErrMsg.Buffer.empty() ? nullptr : ErrMsg.Buffer.data();
  case ErrorMessageState::DestroyedNoError:
    return nullptr;
  case ErrorMessageState::DestroyedWithError:
    ErrMsgState = ErrorMessageState::DestroyedNoError; // Report it once.
    return const_cast<char *>(ErrorDuringThreadExitMessage);
  }
  ORC_RT_UNREACHABLE("Unknown ErrorMessageState");
}

template <typename DescribeOpFn>
bool DlfcnLoaderAdapter::updateErrorState(Error Err,
                                          DescribeOpFn &&DescribeOp) noexcept {
  // Reset error state.
  if (ErrMsgState == ErrorMessageState::Valid) {
    ErrMsg.Buffer.clear();
    ErrMsg.Checked = false;
  } else {
    ErrMsgState = ErrorMessageState::DestroyedNoError;
  }

  if (!Err)
    return true;

  // Otherwise log the failure and update the error state. dlerror must report
  // something after a failure (and an empty Buffer means "no error"), so
  // substitute a message for an empty one.
  auto Msg = toString(std::move(Err));
  if (Msg.empty())
    Msg = "unknown error";

  // Failures are often routine (e.g. probing for an optional symbol), so log
  // at Debug level when dlerror can report them, and at Error level only when
  // it can't because the thread is exiting: the log is then the only record
  // of the message.
  if (ErrMsgState == ErrorMessageState::Valid) {
    ORC_RT_LOG(Debug, LoaderAPIs,
               "DlfcnLoaderAdapter %p: " ORC_RT_LOG_PUB_S
               " failed: " ORC_RT_LOG_PUB_S,
               static_cast<void *>(this), DescribeOp().c_str(), Msg.c_str());
    ErrMsg.Buffer = std::move(Msg);
  } else {
    ORC_RT_LOG(Error, LoaderAPIs,
               "DlfcnLoaderAdapter %p: " ORC_RT_LOG_PUB_S
               " failed: " ORC_RT_LOG_PUB_S
               " (not available from dlerror: thread exiting)",
               static_cast<void *>(this), DescribeOp().c_str(), Msg.c_str());
    ErrMsgState = ErrorMessageState::DestroyedWithError;
  }

  return false;
}

template <typename T, typename DescribeOpFn>
T DlfcnLoaderAdapter::updateErrorState(Expected<T> Result, T ResultOnError,
                                       DescribeOpFn &&DescribeOp) noexcept {
  if (updateErrorState(Result.takeError(),
                       std::forward<DescribeOpFn>(DescribeOp)))
    return std::move(*Result);
  return ResultOnError;
}

} // namespace orc_rt
