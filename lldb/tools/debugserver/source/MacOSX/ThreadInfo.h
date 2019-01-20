//===-- ThreadInfo.h -----------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __ThreadInfo_h__
#define __ThreadInfo_h__

namespace ThreadInfo {

class QoS {
public:
  QoS() : constant_name(), printable_name(), enum_value(UINT32_MAX) {}
  bool IsValid() { return enum_value != UINT32_MAX; }
  std::string constant_name;
  std::string printable_name;
  uint32_t enum_value;
};
};

#endif // __ThreadInfo_h__
