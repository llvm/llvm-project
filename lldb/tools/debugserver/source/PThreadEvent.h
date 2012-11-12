//===-- PThreadEvent.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/16/07.
//
//===----------------------------------------------------------------------===//

#ifndef __PThreadEvent_h__
#define __PThreadEvent_h__
#include "PThreadMutex.h"
#include "PThreadCondition.h"
#include <stdint.h>
#include <time.h>

class PThreadEvent
{
public:
                PThreadEvent        (uint32_t bits = 0, uint32_t validBits = 0);
                ~PThreadEvent       ();

    uint32_t    NewEventBit         ();
    void        FreeEventBits       (const uint32_t mask);

    void        ReplaceEventBits    (const uint32_t bits);
    uint32_t    GetEventBits        () const;
    void        SetEvents           (const uint32_t mask);
    void        ResetEvents         (const uint32_t mask);
    // Wait for events to be set or reset. These functions take an optional
    // timeout value. If timeout is NULL an infinite timeout will be used.
    uint32_t    WaitForSetEvents    (const uint32_t mask, const struct timespec *timeout_abstime = NULL) const;
    uint32_t    WaitForEventsToReset(const uint32_t mask, const struct timespec *timeout_abstime = NULL) const;

    uint32_t    GetResetAckMask () const { return m_reset_ack_mask; }
    uint32_t    SetResetAckMask (uint32_t mask) { return m_reset_ack_mask = mask; }
    uint32_t    WaitForResetAck (const uint32_t mask, const struct timespec *timeout_abstime = NULL) const;
protected:
    //----------------------------------------------------------------------
    // pthread condition and mutex variable to controll access and allow
    // blocking between the main thread and the spotlight index thread.
    //----------------------------------------------------------------------
    mutable PThreadMutex        m_mutex;
    mutable PThreadCondition    m_set_condition;
    mutable PThreadCondition    m_reset_condition;
    uint32_t            m_bits;
    uint32_t            m_validBits;
    uint32_t            m_reset_ack_mask;
private:
    PThreadEvent(const PThreadEvent&);  // Outlaw copy contructor
    PThreadEvent& operator=(const PThreadEvent& rhs);

};

#endif // #ifndef __PThreadEvent_h__
