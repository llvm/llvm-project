//===-- ProfileObjectiveC.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 10/4/07.
//
//===----------------------------------------------------------------------===//

#ifndef __ProfileObjectiveC_h__
#define __ProfileObjectiveC_h__

#include "DNB.h"
#include "DNBRuntimeAction.h"
#include <map>
#include <sys/time.h>

class ProfileObjectiveC : public DNBRuntimeAction
{
public:
    ProfileObjectiveC();
    virtual ~ProfileObjectiveC();
    //------------------------------------------------------------------
    // DNBRuntimeAction required functions
    //------------------------------------------------------------------
    virtual void Initialize(nub_process_t pid);
    virtual void ProcessStateChanged(nub_state_t state);
    virtual void SharedLibraryStateChanged(DNBExecutableImageInfo *image_infos, nub_size_t num_image_infos);

protected:
    typedef uint32_t objc_selector_t;
    typedef uint32_t objc_class_ptr_t;
    void Clear();
    static nub_bool_t MessageSendBreakpointCallback(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *userData);
    static nub_bool_t MessageSendSuperBreakpointCallback(nub_process_t pid, nub_thread_t tid, nub_break_t breakID, void *userData);
    void DumpStats(nub_process_t pid, FILE *f);
    void SetStartTime();
    void SelectorHit(objc_class_ptr_t isa, objc_selector_t sel);
    typedef std::map<objc_selector_t, uint32_t> SelectorHitCount;
    typedef std::map<objc_class_ptr_t, SelectorHitCount> ClassStatsMap;
    typedef struct Probe
    {
        nub_addr_t        addr;
        nub_break_t        breakID;
        Probe() : addr(INVALID_NUB_ADDRESS), breakID(INVALID_NUB_BREAK_ID) {}
        void Clear()
        {
            addr = INVALID_NUB_ADDRESS;
            breakID = INVALID_NUB_BREAK_ID;
        }
        bool IsValid() const
        {
            return (addr != INVALID_NUB_ADDRESS) && (NUB_BREAK_ID_IS_VALID(breakID));
        }
    };

    uint32_t IncrementHitCount() { return ++m_hit_count; }
    bool ShouldDumpStats() const {     return m_dump_count && (m_hit_count % m_dump_count) == 0; }

    nub_process_t m_pid;
    Probe m_objc_msgSend;
    Probe m_objc_msgSendSuper;
    uint32_t m_hit_count;    // Number of times we have gotten one of our breakpoints hit
    uint32_t m_dump_count;    // Dump stats every time the hit count reaches a multiple of this value
#if defined (__powerpc__) || defined (__ppc__)
    enum
    {
        k_opcode_size = 4
    };
    uint8_t            m_objc_msgSend_opcode[k_opcode_size];        // Saved copy of first opcode in objc_msgSend
    Probe            m_objc_msgSend_rtp;                            // COMM page probe info for objc_msgSend
#endif
    struct timeval    m_begin_time;
    ClassStatsMap    m_objcStats;
};


#endif    // #ifndef __ProfileObjectiveC_h__
