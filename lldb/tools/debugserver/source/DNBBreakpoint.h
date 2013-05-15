//===-- DNBBreakpoint.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/29/07.
//
//===----------------------------------------------------------------------===//

#ifndef __DNBBreakpoint_h__
#define __DNBBreakpoint_h__

#include <list>

#include "DNBDefs.h"

class DNBBreakpoint
{
public:
    DNBBreakpoint(nub_addr_t m_addr, nub_size_t byte_size, nub_thread_t tid, bool hardware);
    ~DNBBreakpoint();

    nub_break_t GetID() const { return m_breakID; }
    nub_size_t  ByteSize() const { return m_byte_size; }
    uint8_t *   SavedOpcodeBytes() { return &m_opcode[0]; }
    const uint8_t *
                SavedOpcodeBytes() const { return &m_opcode[0]; }
    nub_addr_t  Address() const { return m_addr; }
    nub_thread_t ThreadID() const { return m_tid; }
    bool        IsEnabled() const { return m_enabled; }
    bool        IntersectsRange(nub_addr_t addr, nub_size_t size, nub_addr_t *intersect_addr, nub_size_t *intersect_size, nub_size_t *opcode_offset) const
                {
                    // We only use software traps for software breakpoints
                    if (IsBreakpoint() && IsEnabled() && !IsHardware())
                    {
                        if (m_byte_size > 0)
                        {
                            const nub_addr_t bp_end_addr = m_addr + m_byte_size;
                            const nub_addr_t end_addr = addr + size;
                            // Is the breakpoint end address before the passed in start address?
                            if (bp_end_addr <= addr)
                                return false;
                            // Is the breakpoint start address after passed in end address?
                            if (end_addr <= m_addr)
                                return false;
                            if (intersect_addr || intersect_size || opcode_offset)
                            {
                                if (m_addr < addr)
                                {
                                    if (intersect_addr)
                                        *intersect_addr = addr;
                                    if (intersect_size)
                                        *intersect_size = std::min<nub_addr_t>(bp_end_addr, end_addr) - addr;
                                    if (opcode_offset)
                                        *opcode_offset = addr - m_addr;
                                }
                                else
                                {
                                    if (intersect_addr)
                                        *intersect_addr = m_addr;
                                    if (intersect_size)
                                        *intersect_size = std::min<nub_addr_t>(bp_end_addr, end_addr) - m_addr;
                                    if (opcode_offset)
                                        *opcode_offset = 0;
                                }
                            }
                            return true;
                        }
                    }
                    return false;
                }
    void        SetEnabled(bool enabled)
                {
                    if (!enabled)
                        SetHardwareIndex(INVALID_NUB_HW_INDEX);
                    m_enabled = enabled;
                }
    void        SetIsWatchpoint (uint32_t type)
                {
                    m_is_watchpoint = 1;
                    m_watch_read = (type & WATCH_TYPE_READ) != 0;
                    m_watch_write = (type & WATCH_TYPE_WRITE) != 0;
                }
    bool        IsBreakpoint() const { return m_is_watchpoint == 0; }
    bool        IsWatchpoint() const { return m_is_watchpoint == 1; }
    bool        WatchpointRead() const { return m_watch_read != 0; }
    bool        WatchpointWrite() const { return m_watch_write != 0; }
    bool        HardwarePreferred() const { return m_hw_preferred; }
    bool        IsHardware() const { return m_hw_index != INVALID_NUB_HW_INDEX; }
    uint32_t    GetHardwareIndex() const { return m_hw_index; }
    void        SetHardwareIndex(uint32_t hw_index) { m_hw_index = hw_index; }
//  StateType   GetState() const { return m_state; }
//  void        SetState(StateType newState) { m_state = newState; }
    int32_t     GetHitCount() const { return m_hit_count; }
    int32_t     GetIgnoreCount() const { return m_ignore_count; }
    void        SetIgnoreCount(int32_t n) { m_ignore_count = n; }
    bool        BreakpointHit(nub_process_t pid, nub_thread_t tid);
    void        SetCallback(DNBCallbackBreakpointHit callback, void *callback_baton);
    void        Dump() const;

private:
    nub_break_t m_breakID;          // The unique identifier for this breakpoint
    nub_thread_t m_tid;             // Thread ID for the breakpoint (can be INVALID_NUB_THREAD for all threads)
    nub_size_t  m_byte_size;        // Length in bytes of the breakpoint if set in memory
    uint8_t     m_opcode[8];        // Saved opcode bytes
    nub_addr_t  m_addr;             // Address of this breakpoint
    uint32_t    m_enabled:1,        // Flags for this breakpoint
                m_hw_preferred:1,   // 1 if this point has been requested to be set using hardware (which may fail due to lack of resources)
                m_is_watchpoint:1,  // 1 if this is a watchpoint
                m_watch_read:1,     // 1 if we stop when the watched data is read from
                m_watch_write:1;    // 1 if we stop when the watched data is written to
    uint32_t    m_hw_index;         // The hardware resource index for this breakpoint/watchpoint
    int32_t     m_hit_count;        // Number of times this breakpoint has been hit
    int32_t     m_ignore_count;     // Number of times to ignore this breakpoint
    DNBCallbackBreakpointHit
                m_callback;         // Callback to call when this breakpoint gets hit
    void *      m_callback_baton;   // Callback user data to pass to callback

    static nub_break_t GetNextID();

};


class DNBBreakpointList
{
public:
                                DNBBreakpointList();
                                ~DNBBreakpointList();

            nub_break_t         Add (const DNBBreakpoint& bp);
            nub_break_t         FindIDByAddress (nub_addr_t addr);
            bool                ShouldStop (nub_process_t pid, nub_thread_t tid, nub_break_t breakID);
            bool                Remove (nub_break_t breakID);
            bool                SetCallback (nub_break_t breakID, DNBCallbackBreakpointHit callback, void *callback_baton);
            DNBBreakpoint *     FindByAddress (nub_addr_t addr);
    const   DNBBreakpoint *     FindByAddress (nub_addr_t addr) const;
            DNBBreakpoint *     FindByID (nub_break_t breakID);
    const   DNBBreakpoint *     FindByID (nub_break_t breakID) const;
            void                Dump () const;

            size_t              Size() const { return m_breakpoints.size(); }
            DNBBreakpoint *     GetByIndex (uint32_t i);
    const   DNBBreakpoint *     GetByIndex (uint32_t i) const;

protected:
    typedef std::list<DNBBreakpoint>    collection;
    typedef collection::iterator        iterator;
    typedef collection::const_iterator  const_iterator;
            iterator                    GetBreakIDIterator(nub_break_t breakID);
            const_iterator              GetBreakIDConstIterator(nub_break_t breakID) const;
            collection                  m_breakpoints;
};

#endif

