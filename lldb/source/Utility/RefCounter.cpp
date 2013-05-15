//===---------------------RefCounter.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RefCounter.h"

namespace lldb_utility {

RefCounter::RefCounter(RefCounter::value_type* ctr):
m_counter(ctr)
{
    increment(m_counter);
}

RefCounter::~RefCounter()
{
    decrement(m_counter);
}

} // namespace lldb_utility
