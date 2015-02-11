//===-- HostThreadFreeBSD.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// lldb Includes
#include "lldb/Host/freebsd/HostThreadFreeBSD.h"
#include "lldb/Host/Host.h"

// C includes
#include <errno.h>
#include <pthread.h>
#if defined (__FreeBSD__)
#include <pthread_np.h>
#endif
#include <stdlib.h>
#include <sys/sysctl.h>
#include <sys/user.h>

// C++ includes
#include <string>

using namespace lldb_private;

HostThreadFreeBSD::HostThreadFreeBSD()
{
}

HostThreadFreeBSD::HostThreadFreeBSD(lldb::thread_t thread)
    : HostThreadPosix(thread)
{
}

void
HostThreadFreeBSD::GetName(lldb::tid_t tid, llvm::SmallVectorImpl<char> &name)
{
    name.clear();
    int pid = Host::GetCurrentProcessID();

    struct kinfo_proc *kp = nullptr, *nkp;
    size_t len = 0;
    int error;
    int ctl[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID | KERN_PROC_INC_THREAD, (int)pid};

    while (1)
    {
        error = sysctl(ctl, 4, kp, &len, nullptr, 0);
        if (kp == nullptr || (error != 0 && errno == ENOMEM))
        {
            // Add extra space in case threads are added before next call.
            len += sizeof(*kp) + len / 10;
            nkp = (struct kinfo_proc *)realloc(kp, len);
            if (nkp == nullptr)
            {
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

    for (size_t i = 0; i < len / sizeof(*kp); i++)
    {
        if (kp[i].ki_tid == (lwpid_t)tid)
        {
            name.append(kp[i].ki_tdname, kp[i].ki_tdname + strlen(kp[i].ki_tdname));
            break;
        }
    }
    free(kp);
}
