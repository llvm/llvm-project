#ifndef LLVM_SUPPORT_IOSANDBOXINTERNAL_H
#define LLVM_SUPPORT_IOSANDBOXINTERNAL_H

#include "llvm/Support/IOSandbox.h"

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif
#include <dirent.h>

extern thread_local bool IOSandboxEnabled;

namespace llvm {
namespace detail {
template <class FnTy> struct Interposed;

template <class RetTy, class... ArgTy> struct Interposed<RetTy (*)(ArgTy...)> {
  RetTy (*Fn)(ArgTy...);

  RetTy operator()(ArgTy... Arg) const {
    sys::sandbox_violation_if_enabled();
    return Fn(std::forward<ArgTy>(Arg)...);
  }
};

template <class RetTy, class... ArgTy>
struct Interposed<RetTy (*)(ArgTy..., ...)> {
  RetTy (*Fn)(ArgTy..., ...);

  template <class... CVarArgTy>
  RetTy operator()(ArgTy... Arg, CVarArgTy... CVarArg) const {
    sys::sandbox_violation_if_enabled();
    return Fn(std::forward<ArgTy>(Arg)..., std::forward<CVarArgTy>(CVarArg)...);
  }
};

template <class FnTy> constexpr auto interpose(FnTy Fn) {
  return Interposed<FnTy>{Fn};
}
} // namespace detail

static constexpr auto read = detail::interpose(::read);
static constexpr auto pread = detail::interpose(::pread);
static constexpr auto mmap = detail::interpose(::mmap);
static constexpr auto readdir = detail::interpose(::readdir);
} // namespace llvm

#endif
