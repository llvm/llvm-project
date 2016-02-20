//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ThisThread.h"

#include <pthread.h>

using namespace lldb_private;

void
ThisThread::SetName(llvm::StringRef name)
{
#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
    ::pthread_setname_np(name);
#endif
}

void
ThisThread::GetName(llvm::SmallVectorImpl<char> &name)
{
#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
    char pthread_name[1024];
    dispatch_queue_t current_queue = ::dispatch_get_current_queue();
    if (current_queue != NULL)
    {
        const char *queue_name = dispatch_queue_get_label(current_queue);
        if (queue_name && queue_name[0])
        {
            name = queue_name;
        }
    }
#endif
}
