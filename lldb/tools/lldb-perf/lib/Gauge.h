//===-- Gauge.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PerfTestDriver_Gauge_h
#define PerfTestDriver_Gauge_h

#include <functional>
#include <string>

#include "Results.h"

namespace lldb_perf {

template <class T>
class Gauge
{
public:
    typedef T ValueType;

    Gauge ()
    {}
    
    virtual
    ~Gauge ()
    {}
    
    virtual void
    Start () = 0;
    
    virtual ValueType
    Stop () = 0;

    virtual ValueType
    GetStartValue () const = 0;

    virtual ValueType
    GetStopValue () const = 0;

    virtual ValueType
    GetDeltaValue () const = 0;

};

template <class T>
Results::ResultSP GetResult (const char *description, T value);

template <>
Results::ResultSP GetResult (const char *description, double value);

template <>
Results::ResultSP GetResult (const char *description, uint64_t value);

template <>
Results::ResultSP GetResult (const char *description, std::string value);

}

#endif
