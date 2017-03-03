//===- Unix/Threading.inc - Unix Threading Implementation ----- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Unix specific implementation of Threading functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

#if defined(__APPLE__)
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#endif

#include <pthread.h>

#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

#if defined(__NetBSD__)
#include <lwp.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

#if defined(__NetBSD__) || defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
#include <sys/sysctl.h>
#include <sys/user.h>
#endif

namespace {
  struct ThreadInfo {
    void(*UserFn)(void *);
    void *UserData;
  };
}

static void *ExecuteOnThread_Dispatch(void *Arg) {
  ThreadInfo *TI = reinterpret_cast<ThreadInfo*>(Arg);
  TI->UserFn(TI->UserData);
  return nullptr;
}

void llvm::llvm_execute_on_thread(void(*Fn)(void*), void *UserData,
  unsigned RequestedStackSize) {
  ThreadInfo Info = { Fn, UserData };
  pthread_attr_t Attr;
  pthread_t Thread;

  // Construct the attributes object.
  if (::pthread_attr_init(&Attr) != 0)
    return;

  // Set the requested stack size, if given.
  if (RequestedStackSize != 0) {
    if (::pthread_attr_setstacksize(&Attr, RequestedStackSize) != 0)
      goto error;
  }

  // Construct and execute the thread.
  if (::pthread_create(&Thread, &Attr, ExecuteOnThread_Dispatch, &Info) != 0)
    goto error;

  // Wait for the thread and clean up.
  ::pthread_join(Thread, nullptr);

error:
  ::pthread_attr_destroy(&Attr);
}


uint64_t llvm::get_threadid() {
#if defined(__APPLE__)
  // Calling "mach_thread_self()" bumps the reference count on the thread
  // port, so we need to deallocate it. mach_task_self() doesn't bump the ref
  // count.
  thread_port_t Self = mach_thread_self();
  mach_port_deallocate(mach_task_self(), Self);
  return Self;
#elif defined(__FreeBSD__)
  return uint64_t(pthread_getthreadid_np());
#elif defined(__NetBSD__)
  return uint64_t(_lwp_self());
#elif defined(__ANDROID__)
  return uint64_t(gettid());
#elif defined(__linux__)
  return uint64_t(syscall(SYS_gettid));
#elif defined(LLVM_ON_WIN32)
  return uint64_t(::GetCurrentThreadId());
#else
  return uint64_t(pthread_self());
#endif
}


void llvm::set_thread_name(const Twine &Name) {
  // Make sure the input is null terminated.
  SmallString<64> Storage;
  StringRef NameStr = Name.toNullTerminatedStringRef(Storage);
#if defined(__linux__)
#if (defined(__GLIBC__) && defined(_GNU_SOURCE)) || defined(__ANDROID__)
  ::pthread_setname_np(::pthread_self(), NameStr.data());
#endif
#elif defined(__FreeBSD__)
  ::pthread_set_name_np(::pthread_self(), NameStr.data());
#elif defined(__NetBSD__)
  ::pthread_setname_np(::pthread_self(), "%s",
    const_cast<char *>(NameStr.data()));
#elif defined(__APPLE__)
  ::pthread_setname_np(NameStr.data());
#endif
}

void llvm::get_thread_name(SmallVectorImpl<char> &Name) {
  Name.clear();

#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
#if defined(__FreeBSD_kernel__)
  auto pid = ::pthread_self();
#else
  auto pid = ::getpid();
#endif

  int tid = ::pthread_getthreadid_np();

  struct kinfo_proc *kp = nullptr, *nkp;
  size_t len = 0;
  int error;
  int ctl[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID | KERN_PROC_INC_THREAD,
    (int)pid };

  while (1) {
    error = sysctl(ctl, 4, kp, &len, nullptr, 0);
    if (kp == nullptr || (error != 0 && errno == ENOMEM)) {
      // Add extra space in case threads are added before next call.
      len += sizeof(*kp) + len / 10;
      nkp = (struct kinfo_proc *)realloc(kp, len);
      if (nkp == nullptr) {
        free(kp);
        return;
      }
      kp = nkp;
      continue;
    }
    if (error != 0)
      len = 0;
    break;
  }

  for (size_t i = 0; i < len / sizeof(*kp); i++) {
    if (kp[i].ki_tid == (lwpid_t)tid) {
      Name.append(kp[i].ki_tdname, kp[i].ki_tdname + strlen(kp[i].ki_tdname));
      break;
    }
  }
  free(kp);
  return;
#elif defined(__NetBSD__)
  char buf[PTHREAD_MAX_NAMELEN_NP];
  ::pthread_getname_np(::pthread_self(), buf, PTHREAD_MAX_NAMELEN_NP);

  Name.append(buf, buf + strlen(buf));
#elif defined(__linux__)
#if (defined(__GLIBC__) && defined(_GNU_SOURCE)) || defined(__ANDROID__)
  constexpr int MAXNAMELEN = 16;
  char Buffer[MAXNAMELEN];
  if (0 == ::pthread_getname_np(::pthread_self(), Buffer, MAXNAMELEN))
    Name.append(Buffer, Buffer + strlen(Buffer));
#endif
#endif
}
