//===--------------------- Range.cpp -----------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Range.h"

using namespace lldb_utility;

Range::Range(const Range &rng) : m_low(rng.m_low), m_high(rng.m_high) {
  InitRange();
}

Range::Range(Range::ValueType low, Range::ValueType high)
    : m_low(low), m_high(high) {
  InitRange();
}

void Range::InitRange() {
  if (m_low == OPEN_END) {
    if (m_high == OPEN_END)
      m_low = 0;
    else {
      // make an empty range
      m_low = 1;
      m_high = 0;
    }
  }
}

Range &Range::operator=(const Range &rhs) {
  if (&rhs != this) {
    this->m_low = rhs.m_low;
    this->m_high = rhs.m_high;
  }
  return *this;
}

void Range::Flip() { std::swap(m_high, m_low); }

void Range::Intersection(const Range &other) {
  m_low = std::max(m_low, other.m_low);
  m_high = std::min(m_high, other.m_high);
}

void Range::Union(const Range &other) {
  m_low = std::min(m_low, other.m_low);
  m_high = std::max(m_high, other.m_high);
}

void Range::Iterate(RangeCallback callback) {
  ValueType counter = m_low;
  while (counter <= m_high) {
    bool should_continue = callback(counter);
    if (!should_continue)
      return;
    counter++;
  }
}

bool Range::IsEmpty() { return (m_low > m_high); }

Range::ValueType Range::GetSize() {
  if (m_high == OPEN_END)
    return OPEN_END;
  if (m_high >= m_low)
    return m_high - m_low + 1;
  return 0;
}
