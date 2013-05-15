//===-- Metric.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Metric.h"
#include "MemoryGauge.h"
#include <cmath>

using namespace lldb_perf;

template <class T>
Metric<T>::Metric () : Metric ("")
{
}

template <class T>
Metric<T>::Metric (const char* n, const char* d) :
    m_name(n ? n : ""),
    m_description(d ? d : ""),
    m_dataset ()
{
}

template <class T>
void
Metric<T>::Append (T v)
{
    m_dataset.push_back(v);
}

template <class T>
size_t
Metric<T>::GetCount () const
{
    return m_dataset.size();
}

template <class T>
T
Metric<T>::GetSum () const
{
    T sum = 0;
    for (auto v : m_dataset)
        sum += v;
    return sum;
}

template <class T>
T
Metric<T>::GetAverage () const
{
    return GetSum()/GetCount();
}


// Knuth's algorithm for stddev - massive cancellation resistant
template <class T>
T
Metric<T>::GetStandardDeviation (StandardDeviationMode mode) const
{
    size_t n = 0;
    T mean = 0;
    T M2 = 0;
    for (auto x : m_dataset)
    {
        n = n + 1;
        T delta = x - mean;
        mean = mean + delta/n;
        M2 = M2+delta*(x-mean);
    }
    T variance;
    if (mode == StandardDeviationMode::ePopulation || n == 1)
        variance = M2 / n;
    else
        variance = M2 / (n - 1);
    return sqrt(variance);
}

template class lldb_perf::Metric<double>;
template class lldb_perf::Metric<MemoryStats>;
