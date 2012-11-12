//===-- PThreadCondition.h --------------------------------------*- C++ -*-===//
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

#ifndef __PThreadCondition_h__
#define __PThreadCondition_h__

#include <pthread.h>

class PThreadCondition
{
public:

    PThreadCondition()
    {
        ::pthread_cond_init (&m_condition, NULL);
    }

    ~PThreadCondition()
    {
        ::pthread_cond_destroy (&m_condition);
    }

    pthread_cond_t *Condition()
    {
        return &m_condition;
    }

    int Broadcast()
    {
        return ::pthread_cond_broadcast (&m_condition);
    }

    int Signal()
    {
        return ::pthread_cond_signal (&m_condition);
    }

protected:
    pthread_cond_t        m_condition;
};

#endif

