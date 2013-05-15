//===-- DNBArch.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/24/07.
//
//===----------------------------------------------------------------------===//

#ifndef __DebugNubArch_h__
#define __DebugNubArch_h__

#include "DNBDefs.h"
#include "MacOSX/MachException.h"

#include <mach/mach.h>
#include <stdio.h>

struct DNBRegisterValue;
struct DNBRegisterSetInfo;
class DNBArchProtocol;
class MachThread;

typedef DNBArchProtocol * (* DNBArchCallbackCreate)(MachThread *thread);
typedef const DNBRegisterSetInfo * (* DNBArchCallbackGetRegisterSetInfo)(nub_size_t *num_reg_sets);
typedef const uint8_t * const (* DNBArchCallbackGetBreakpointOpcode)(nub_size_t byte_size);

typedef struct DNBArchPluginInfoTag
{
    uint32_t cpu_type;
    DNBArchCallbackCreate               Create;
    DNBArchCallbackGetRegisterSetInfo   GetRegisterSetInfo;
    DNBArchCallbackGetBreakpointOpcode  GetBreakpointOpcode;
} DNBArchPluginInfo;

class DNBArchProtocol
{
public:
    static DNBArchProtocol *
    Create (MachThread *thread);

    static const DNBRegisterSetInfo * 
    GetRegisterSetInfo (nub_size_t *num_reg_sets);

    static const uint8_t * const 
    GetBreakpointOpcode (nub_size_t byte_size);

    static void
    RegisterArchPlugin (const DNBArchPluginInfo &arch_info);

    static uint32_t
    GetArchitecture ();

    static bool
    SetArchitecture (uint32_t cpu_type);
    
    virtual bool            GetRegisterValue (int set, int reg, DNBRegisterValue *value) = 0;
    virtual bool            SetRegisterValue (int set, int reg, const DNBRegisterValue *value) = 0;
    virtual nub_size_t      GetRegisterContext (void *buf, nub_size_t buf_len) = 0;
    virtual nub_size_t      SetRegisterContext (const void *buf, nub_size_t buf_len) = 0;

    virtual kern_return_t   GetRegisterState (int set, bool force) = 0;
    virtual kern_return_t   SetRegisterState (int set) = 0;
    virtual bool            RegisterSetStateIsValid (int set) const = 0;

    virtual uint64_t        GetPC (uint64_t failValue) = 0;    // Get program counter
    virtual kern_return_t   SetPC (uint64_t value) = 0;
    virtual uint64_t        GetSP (uint64_t failValue) = 0;    // Get stack pointer
    virtual void            ThreadWillResume () = 0;
    virtual bool            ThreadDidStop () = 0;
    virtual bool            NotifyException (MachException::Data& exc) { return false; }
    virtual uint32_t        NumSupportedHardwareBreakpoints() { return 0; }
    virtual uint32_t        NumSupportedHardwareWatchpoints() { return 0; }
    virtual uint32_t        EnableHardwareBreakpoint (nub_addr_t addr, nub_size_t size) { return INVALID_NUB_HW_INDEX; }
    virtual uint32_t        EnableHardwareWatchpoint (nub_addr_t addr, nub_size_t size, bool read, bool write) { return INVALID_NUB_HW_INDEX; }
    virtual void            HardwareWatchpointStateChanged () { ; }
    virtual bool            DisableHardwareBreakpoint (uint32_t hw_index) { return false; }
    virtual bool            DisableHardwareWatchpoint (uint32_t hw_index) { return false; }
    virtual uint32_t        GetHardwareWatchpointHit() { return INVALID_NUB_HW_INDEX; }
    virtual bool            StepNotComplete () { return false; }

protected:
    friend class MachThread;

    enum
    {
        Trans_Pending = 0,      // Transaction is pending, and checkpoint state has been snapshotted.
        Trans_Done = 1,         // Transaction is done, the current state is committed, and checkpoint state is irrelevant.
        Trans_Rolled_Back = 2   // Transaction is done, the current state has been rolled back to the checkpoint state.
    };
    virtual bool StartTransForHWP() { return true; }
    virtual bool RollbackTransForHWP() { return true; }
    virtual bool FinishTransForHWP() { return true; }
};


#include "MacOSX/arm/DNBArchImpl.h"
#include "MacOSX/i386/DNBArchImplI386.h"
#include "MacOSX/x86_64/DNBArchImplX86_64.h"
#include "MacOSX/ppc/DNBArchImpl.h"

#endif
