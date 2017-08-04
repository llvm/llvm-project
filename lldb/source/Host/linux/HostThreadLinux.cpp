//===-- HostThreadLinux.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/linux/HostThreadLinux.h"
#include "Plugins/Process/Linux/ProcFileReader.h"
#include "lldb/Core/DataBuffer.h"

#include "llvm/ADT/SmallVector.h"

#include <pthread.h>

using namespace lldb_private;

HostThreadLinux::HostThreadLinux() : HostThreadPosix() {}

HostThreadLinux::HostThreadLinux(lldb::thread_t thread)
    : HostThreadPosix(thread) {}

void HostThreadLinux::SetName(lldb::thread_t thread, llvm::StringRef name) {
#if (defined(__GLIBC__) && defined(_GNU_SOURCE)) || defined(__ANDROID__)
  ::pthread_setname_np(thread, name.data());
#else
  (void)thread;
  (void)name;
#endif
}

void HostThreadLinux::GetName(lldb::thread_t thread,
                              llvm::SmallVectorImpl<char> &name) {
  // Read /proc/$TID/comm file.
  lldb::DataBufferSP buf_sp =
      process_linux::ProcFileReader::ReadIntoDataBuffer(thread, "comm");
  const char *comm_str = (const char *)buf_sp->GetBytes();
  const char *cr_str = ::strchr(comm_str, '\n');
  size_t length = cr_str ? (cr_str - comm_str) : strlen(comm_str);

  name.clear();
  name.append(comm_str, comm_str + length);
}
