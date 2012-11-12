//===-- LinuxSignals.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LinuxSignals_H_
#define liblldb_LinuxSignals_H_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/UnixSignals.h"

/// Linux specific set of Unix signals.
class LinuxSignals
    : public lldb_private::UnixSignals
{
public:
    LinuxSignals();

private:
    void
    Reset();
};

#endif
