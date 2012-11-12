//===-- RegisterContextPOSIX.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextPOSIX_H_
#define liblldb_RegisterContextPOSIX_H_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Target/RegisterContext.h"

//------------------------------------------------------------------------------
/// @class RegisterContextPOSIX
///
/// @brief Extends RegisterClass with a few virtual operations useful on POSIX.
class RegisterContextPOSIX
    : public lldb_private::RegisterContext
{
public:
    RegisterContextPOSIX(lldb_private::Thread &thread,
                         uint32_t concrete_frame_idx)
        : RegisterContext(thread, concrete_frame_idx) { }

    /// Updates the register state of the associated thread after hitting a
    /// breakpoint (if that make sense for the architecture).  Default
    /// implementation simply returns true for architectures which do not
    /// require any update.
    ///
    /// @return
    ///    True if the operation succeeded and false otherwise.
    virtual bool UpdateAfterBreakpoint() { return true; }
};

#endif // #ifndef liblldb_RegisterContextPOSIX_H_
