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
        : RegisterContext(thread, concrete_frame_idx)
        { m_watchpoints_initialized = false; }

    /// Updates the register state of the associated thread after hitting a
    /// breakpoint (if that make sense for the architecture).  Default
    /// implementation simply returns true for architectures which do not
    /// require any update.
    ///
    /// @return
    ///    True if the operation succeeded and false otherwise.
    virtual bool UpdateAfterBreakpoint() { return true; }

    /// Determines the index in lldb's register file given a kernel byte offset.
    virtual unsigned
    GetRegisterIndexFromOffset(unsigned offset) { return LLDB_INVALID_REGNUM; }

    // Checks to see if a watchpoint specified by hw_index caused the inferior
    // to stop.
    virtual bool
    IsWatchpointHit (uint32_t hw_index) { return false; }

    // Resets any watchpoints that have been hit.
    virtual bool
    ClearWatchpointHits () { return false; }

    // Returns the watchpoint address associated with a watchpoint hardware
    // index.
    virtual lldb::addr_t
    GetWatchpointAddress (uint32_t hw_index) { return LLDB_INVALID_ADDRESS; }

    virtual bool
    IsWatchpointVacant (uint32_t hw_index) { return false; }

    virtual bool
    SetHardwareWatchpointWithIndex (lldb::addr_t addr, size_t size,
                                    bool read, bool write,
                                    uint32_t hw_index) { return false; }

protected:
    bool m_watchpoints_initialized;
};

#endif // #ifndef liblldb_RegisterContextPOSIX_H_
