//===-- LibcGlue.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file adds functions missing from libc on older versions of linux

#include <cerrno>
#include <lldb/Host/linux/Uio.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef HAVE_PROCESS_VM_READV // If the syscall wrapper is not available,
                              // provide one.
ssize_t process_vm_readv(::pid_t pid, const struct iovec *local_iov,
                         unsigned long liovcnt, const struct iovec *remote_iov,
                         unsigned long riovcnt, unsigned long flags) {
#ifdef HAVE_NR_PROCESS_VM_READV // If we have the syscall number, we can issue
                                // the syscall ourselves.
  return syscall(__NR_process_vm_readv, pid, local_iov, liovcnt, remote_iov,
                 riovcnt, flags);
#else // If not, let's pretend the syscall is not present.
  errno = ENOSYS;
  return -1;
#endif
}
#endif
