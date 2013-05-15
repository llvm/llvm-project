//===-- Timer.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Timer.h"
#include <assert.h>

using namespace lldb_perf;

TimeGauge::TimeType
TimeGauge::Now ()
{
	return high_resolution_clock::now();
}

TimeGauge::TimeGauge () :
    m_start(),
    m_state(TimeGauge::State::eNeverUsed)
{
}

void
TimeGauge::Start ()
{
	m_state = TimeGauge::State::eCounting;
	m_start = Now();
}

double
TimeGauge::Stop ()
{
	m_stop = Now();
	assert(m_state == TimeGauge::State::eCounting && "cannot stop a non-started clock");
	m_state = TimeGauge::State::eStopped;
    m_delta = duration_cast<duration<double>>(m_stop-m_start).count();
	return m_delta;
}

double
TimeGauge::GetStartValue () const
{
    return (double)m_start.time_since_epoch().count() * (double)system_clock::period::num / (double)system_clock::period::den;
}

double
TimeGauge::GetStopValue () const
{
    return (double)m_stop.time_since_epoch().count() * (double)system_clock::period::num / (double)system_clock::period::den;
}

double
TimeGauge::GetDeltaValue () const
{
	assert(m_state == TimeGauge::State::eStopped && "clock must be used before you can evaluate it");
	return m_delta;
}
