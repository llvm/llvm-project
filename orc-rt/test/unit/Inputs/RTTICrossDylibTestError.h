//===- RTTICrossDylibTestError.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An ErrorInfoBase subclass shared between CoreTests and
// RTTICrossDylibTestLib, so that RTTICrossDylibTest.cpp can construct an
// instance in one library, and check/cast it via RTTI in another -- a real
// cross-library boundary, not a same-binary simulation of one.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_TEST_RTTICROSSDYLIBTESTERROR_H
#define ORC_RT_TEST_RTTICROSSDYLIBTESTERROR_H

#include "orc-rt/support/Error.h"

namespace orc_rt_test {

class CrossDylibTestError
    : public orc_rt::ErrorExtends<CrossDylibTestError, orc_rt::ErrorInfoBase> {
public:
  static constexpr const char *RTTIName = "orc_rt_test::CrossDylibTestError";

  explicit CrossDylibTestError(int Code) noexcept : Code(Code) {}

  std::string toString() const noexcept override {
    return "CrossDylibTestError(" + std::to_string(Code) + ")";
  }

  int getCode() const noexcept { return Code; }

private:
  int Code;
};

} // namespace orc_rt_test

#endif // ORC_RT_TEST_RTTICROSSDYLIBTESTERROR_H
