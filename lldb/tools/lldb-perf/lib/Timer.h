//===-- Timer.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PerfTestDriver__Timer__
#define __PerfTestDriver__Timer__

#include "Gauge.h"

#include <chrono>

using namespace std::chrono;

namespace lldb_perf
{
class TimeGauge : public Gauge<double>
{
public:
    TimeGauge ();
    
    virtual
    ~TimeGauge ()
    {
    }
    
    void
    Start ();
    
    double
    Stop ();
    
    virtual double
    GetStartValue () const;
    
    virtual double
    GetStopValue () const;

    virtual double
    GetDeltaValue () const;

private:
    enum class State
    {
        eNeverUsed,
        eCounting,
        eStopped
    };
    
    typedef high_resolution_clock::time_point TimeType;
    TimeType m_start;
    TimeType m_stop;
    double m_delta;
    State m_state;
    
    TimeType
    Now ();
    
};
}

#endif /* defined(__PerfTestDriver__Timer__) */
