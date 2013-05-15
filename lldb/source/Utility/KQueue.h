//===--------------------- KQueue.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_KQueue_h_
#define utility_KQueue_h_

#if defined(__APPLE__)
#define LLDB_USE_KQUEUES
#endif

#ifdef LLDB_USE_KQUEUES

#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>

#include "lldb/lldb-defines.h"

namespace lldb_private {

class KQueue
{
public:
    KQueue() :
        m_fd(-1)
    {
    }

    ~KQueue()
    {
        Close();
    }
    
    bool
    IsValid () const
    {
        return m_fd >= 0;
    }

    int
    GetFD (bool can_create);

    int
    Close ();

    bool
    AddFDEvent (int fd,
                bool read,
                bool write,
                bool vnode);

    int
    WaitForEvents (struct kevent *events,
                   int num_events,
                   Error &error,
                   uint32_t timeout_usec = UINT32_MAX); // UINT32_MAX means infinite timeout

protected:
    int m_fd; // The kqueue fd
};

} // namespace lldb_private

#endif // #ifdef LLDB_USE_KQUEUES

#endif // #ifndef utility_KQueue_h_
