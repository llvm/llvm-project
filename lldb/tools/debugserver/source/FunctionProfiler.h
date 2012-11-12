//===-- FunctionProfiler.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 10/8/08.
//
//===----------------------------------------------------------------------===//

#ifndef __FunctionProfiler_h__
#define __FunctionProfiler_h__

// C Includes

// C++ Includes
#include <map>
#include <vector>
#include <string>

// Other libraries and framework includes

// Project includes
#include "DNBDefs.h"
#include "DNBRuntimeAction.h"
#include "PThreadMutex.h"

class DNBBreakpoint;
class MachProcess;

class FunctionProfiler : public DNBRuntimeAction
{
public:
    FunctionProfiler (nub_addr_t start_addr, nub_addr_t stop_addr);
    virtual ~FunctionProfiler ();

    //------------------------------------------------------------------
    // DNBRuntimeAction required functions
    //------------------------------------------------------------------
    virtual void        Initialize(nub_process_t pid);
    virtual void        ProcessStateChanged(nub_state_t state);
    virtual void        SharedLibraryStateChanged(DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos) {}

    nub_bool_t          BreakpointHit(nub_process_t pid, nub_thread_t tid, nub_break_t breakID);
    bool                ShouldStepProcess() const
                        {
                            return m_func_entered_count > 0;
                        }
protected:
    static  nub_bool_t  BreakpointHitCallback (nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *baton);
    void                Clear();
    void                SetBreakpoints();

    nub_process_t       m_pid;
    nub_addr_t          m_start_addr;
    nub_addr_t          m_stop_addr;
    nub_break_t         m_start_break_id;
    nub_break_t         m_stop_break_id;
    uint32_t            m_func_entered_count;
    nub_addr_t          m_last_pc;
    uint32_t            m_last_flags;
    uint32_t            m_consecutive_opcode_count;
    uint32_t            m_total_opcode_count;
};


#endif  // __FunctionProfiler_h__
