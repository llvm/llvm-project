//===-- SPSMemoryFlags.h - SPS-serialization for MemoryFlags.h --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPSSerialization for relevant types in MemoryFlags.h.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPSMEMORYFLAGS_H
#define ORC_RT_SPSMEMORYFLAGS_H

#include "orc-rt/MemoryFlags.h"
#include "orc-rt/SimplePackedSerialization.h"

namespace orc_rt {

struct SPSAllocGroup;

template <> class SPSSerializationTraits<SPSAllocGroup, AllocGroup> {
private:
  typedef detail::AllocGroupInternals::underlying_type UT;

public:
  static size_t size(const AllocGroup &AG) {
    return SPSSerializationTraits<UT, UT>::size(
        detail::AllocGroupInternals::getId(AG));
  }

  static bool serialize(SPSOutputBuffer &OB, const AllocGroup &AG) {
    return SPSSerializationTraits<UT, UT>::serialize(
        OB, detail::AllocGroupInternals::getId(AG));
  }

  static bool deserialize(SPSInputBuffer &IB, AllocGroup &AG) {
    UT Id = 0;
    if (!SPSSerializationTraits<UT, UT>::deserialize(IB, Id))
      return false;
    AG = detail::AllocGroupInternals::fromId(Id);
    return true;
  }
};

} // namespace orc_rt

#endif // ORC_RT_SPSMEMORYFLAGS_H
