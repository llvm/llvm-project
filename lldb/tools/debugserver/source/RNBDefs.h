//===-- RNBDefs.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 12/14/07.
//
//===----------------------------------------------------------------------===//

#ifndef __RNBDefs_h__
#define __RNBDefs_h__

#include "DNBDefs.h"
#include <memory>

extern "C" const unsigned char debugserverVersionString[];
extern "C" const double debugserverVersionNumber;
#define DEBUGSERVER_PROGRAM_NAME "debugserver"
#define DEBUGSERVER_VERSION_STR debugserverVersionString
#define DEBUGSERVER_VERSION_NUM debugserverVersionNumber

#if defined (__i386__)

#define RNB_ARCH    "i386"

#elif defined (__x86_64__)

#define RNB_ARCH    "x86_64"

#elif defined (__ppc64__)

#define RNB_ARCH    "ppc64"

#elif defined (__powerpc__) || defined (__ppc__)

#define RNB_ARCH    "ppc"

#elif defined (__arm__)

#define RNB_ARCH    "armv7"

#else

#error undefined architecture

#endif

class RNBRemote;
typedef std::shared_ptr<RNBRemote> RNBRemoteSP;

typedef enum
{
    rnb_success = 0,
    rnb_err = 1,
    rnb_not_connected = 2
} rnb_err_t;

// Log bits
// reserve low bits for DNB
#define LOG_RNB_MINIMAL     ((LOG_LO_USER) << 0)  // Minimal logging    (min verbosity)
#define LOG_RNB_MEDIUM      ((LOG_LO_USER) << 1)  // Medium logging     (med verbosity)
#define LOG_RNB_MAX         ((LOG_LO_USER) << 2)  // Max logging        (max verbosity)
#define LOG_RNB_COMM        ((LOG_LO_USER) << 3)  // Log communications (RNBSocket)
#define LOG_RNB_REMOTE      ((LOG_LO_USER) << 4)  // Log remote         (RNBRemote)
#define LOG_RNB_EVENTS      ((LOG_LO_USER) << 5)  // Log events         (PThreadEvents)
#define LOG_RNB_PROC        ((LOG_LO_USER) << 6)  // Log process state  (Process thread)
#define LOG_RNB_PACKETS     ((LOG_LO_USER) << 7)  // Log gdb remote packets
#define LOG_RNB_ALL         (~((LOG_LO_USER) - 1))
#define LOG_RNB_DEFAULT     (LOG_RNB_ALL)

extern RNBRemoteSP g_remoteSP;

#endif // #ifndef __RNBDefs_h__
