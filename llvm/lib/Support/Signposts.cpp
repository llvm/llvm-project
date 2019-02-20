//===-- Signposts.cpp - Interval debug annotations ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Signposts.h"
#include "llvm/Support/Timer.h"

#include "llvm/Config/config.h"
#if LLVM_SUPPORT_XCODE_SIGNPOSTS
#include "llvm/ADT/DenseMap.h"
#include <os/signpost.h>
#endif // if LLVM_SUPPORT_XCODE_SIGNPOSTS

using namespace llvm;

#if LLVM_SUPPORT_XCODE_SIGNPOSTS
namespace {
os_log_t *LogCreator() {
  os_log_t *X = new os_log_t;
  *X = os_log_create("org.llvm.signposts", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
  return X;
}
void LogDeleter(os_log_t *X) {
  os_release(*X);
  delete X;
}
} // end anonymous namespace

namespace llvm {
class SignpostEmitterImpl {
  using LogPtrTy =
      std::unique_ptr<os_log_t, std::function<void(os_log_t *)>>;
  using LogTy = LogPtrTy::element_type;

  LogPtrTy SignpostLog;
  DenseMap<const Timer *, os_signpost_id_t> Signposts;

  LogTy &getLogger() const { return *SignpostLog; }
  os_signpost_id_t getSignpostForTimer(const Timer *T) {
    const auto &I = Signposts.find(T);
    if (I != Signposts.end())
      return I->second;

    const auto &Inserted = Signposts.insert(
        std::make_pair(T, os_signpost_id_make_with_pointer(getLogger(), T)));
    return Inserted.first->second;
  }

public:
  SignpostEmitterImpl() : SignpostLog(LogCreator(), LogDeleter), Signposts() {}

  bool isEnabled() const { return os_signpost_enabled(*SignpostLog); }

  void startTimerInterval(Timer *T) {
    if (isEnabled()) {
      // Both strings used here are required to be constant literal strings
      os_signpost_interval_begin(getLogger(), getSignpostForTimer(T),
                                 "Pass Timers", "Begin %s",
                                 T->getName().c_str());
    }
  }

  void endTimerInterval(Timer *T) {
    if (isEnabled()) {
      // Both strings used here are required to be constant literal strings
      os_signpost_interval_end(getLogger(), getSignpostForTimer(T),
                               "Pass Timers", "End %s", T->getName().c_str());
    }
  }
};
} // end namespace llvm
#endif // if LLVM_SUPPORT_XCODE_SIGNPOSTS

#if LLVM_SUPPORT_XCODE_SIGNPOSTS
#define HAVE_ANY_SIGNPOST_IMPL 1
#endif

SignpostEmitter::SignpostEmitter() {
#if HAVE_ANY_SIGNPOST_IMPL
  Impl = new SignpostEmitterImpl();
#else // if HAVE_ANY_SIGNPOST_IMPL
  Impl = nullptr;
#endif // if !HAVE_ANY_SIGNPOST_IMPL
}

SignpostEmitter::~SignpostEmitter() {
#if HAVE_ANY_SIGNPOST_IMPL
  delete Impl;
#endif // if HAVE_ANY_SIGNPOST_IMPL
}

bool SignpostEmitter::isEnabled() const {
#if HAVE_ANY_SIGNPOST_IMPL
  return Impl->isEnabled();
#else
  return false;
#endif // if !HAVE_ANY_SIGNPOST_IMPL
}

void SignpostEmitter::startTimerInterval(Timer *T) {
#if HAVE_ANY_SIGNPOST_IMPL
  if (Impl == nullptr)
    return;
  return Impl->startTimerInterval(T);
#endif // if !HAVE_ANY_SIGNPOST_IMPL
}

void SignpostEmitter::endTimerInterval(Timer *T) {
#if HAVE_ANY_SIGNPOST_IMPL
  if (Impl == nullptr)
    return;
  Impl->endTimerInterval(T);
#endif // if !HAVE_ANY_SIGNPOST_IMPL
}
