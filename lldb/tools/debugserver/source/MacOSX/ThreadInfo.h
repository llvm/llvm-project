//===-- ThreadInfo.h -----------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
