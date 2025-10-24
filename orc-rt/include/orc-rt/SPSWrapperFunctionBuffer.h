//===-- SPSWrapperFunctionBuffer.h - SPS serialization for WFB --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS serialization for WrapperFunctionBuffer.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SPSWRAPPERFUNCTIONBUFFER_H
#define ORC_RT_SPSWRAPPERFUNCTIONBUFFER_H

#include "orc-rt/SimplePackedSerialization.h"
#include "orc-rt/WrapperFunction.h"

namespace orc_rt {

struct SPSWrapperFunctionBuffer;

template <>
class SPSSerializationTraits<SPSWrapperFunctionBuffer, WrapperFunctionBuffer> {
public:
  static size_t size(const WrapperFunctionBuffer &WFB) {
    return SPSArgList<uint64_t>::size(static_cast<uint64_t>(WFB.size())) +
           WFB.size();
  }

  static bool serialize(SPSOutputBuffer &OB, const WrapperFunctionBuffer &WFB) {
    if (!SPSArgList<uint64_t>::serialize(OB, static_cast<uint64_t>(WFB.size())))
      return false;
    return OB.write(WFB.data(), WFB.size());
  }

  static bool deserialize(SPSInputBuffer &IB, WrapperFunctionBuffer &WFB) {
    uint64_t Size;
    if (!SPSArgList<uint64_t>::deserialize(IB, Size))
      return false;
    WFB = WrapperFunctionBuffer::allocate(Size);
    return IB.read(WFB.data(), WFB.size());
  }
};

} // namespace orc_rt

#endif // ORC_RT_SPSWRAPPERFUNCTIONBUFFER_H
