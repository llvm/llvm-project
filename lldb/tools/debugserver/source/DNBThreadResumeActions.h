//===-- DNBThreadResumeActions.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 03/13/2010
//
//===----------------------------------------------------------------------===//


#ifndef __DNBThreadResumeActions_h__
#define __DNBThreadResumeActions_h__

#include <vector>

#include "DNBDefs.h"


class DNBThreadResumeActions
{
public:
    DNBThreadResumeActions ();

    DNBThreadResumeActions (nub_state_t default_action, int signal);

    DNBThreadResumeActions (const DNBThreadResumeAction *actions, size_t num_actions);

    bool
    IsEmpty() const
    {
        return m_actions.empty();
    }

    void
    Append (const DNBThreadResumeAction &action);

    void
    AppendAction (nub_thread_t tid,
                  nub_state_t state,
                  int signal = 0,
                  nub_addr_t addr = INVALID_NUB_ADDRESS);

    void
    AppendResumeAll ()
    {
        AppendAction (INVALID_NUB_THREAD, eStateRunning);
    }

    void
    AppendSuspendAll ()
    {
        AppendAction (INVALID_NUB_THREAD, eStateStopped);
    }

    void
    AppendStepAll ()
    {
        AppendAction (INVALID_NUB_THREAD, eStateStepping);
    }

    const DNBThreadResumeAction *
    GetActionForThread (nub_thread_t tid, bool default_ok) const;

    size_t
    NumActionsWithState (nub_state_t state) const;

    bool
    SetDefaultThreadActionIfNeeded (nub_state_t action, int signal);

    void
    SetSignalHandledForThread (nub_thread_t tid) const;

    const DNBThreadResumeAction *
    GetFirst() const
    {
        return m_actions.data();
    }

    size_t
    GetSize () const
    {
        return m_actions.size();
    }
    
    void
    Clear()
    {
        m_actions.clear();
        m_signal_handled.clear();
    }

protected:
    std::vector<DNBThreadResumeAction> m_actions;
    mutable std::vector<bool> m_signal_handled;
};


#endif    // #ifndef __DNBThreadResumeActions_h__
