//  Copyright John Maddock 2016.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_CUDA_STOPWATCH_HPP
#define BOOST_MATH_CUDA_STOPWATCH_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <chrono>

template <class Clock>
struct stopwatch
{
    typedef typename Clock::duration duration;
    stopwatch()
    {
        m_start = Clock::now();
    }
    double elapsed()
    {
        duration t = Clock::now() - m_start;
        return std::chrono::duration_cast<std::chrono::duration<double>>(t).count();
    }
    void reset()
    {
        m_start = Clock::now();
    }

private:
    typename Clock::time_point m_start;
};

typedef stopwatch<std::chrono::high_resolution_clock> watch;

#endif
