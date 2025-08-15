//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the InputGenerator class, which defines
/// the abstract interface for classes that generate math test inputs.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_INPUTGENERATOR_HPP
#define MATHTEST_INPUTGENERATOR_HPP

#include "llvm/ADT/ArrayRef.h"

namespace mathtest {

template <typename... InTypes> class InputGenerator {
public:
  virtual ~InputGenerator() noexcept = default;

  virtual void reset() noexcept = 0;

  [[nodiscard]] virtual size_t
  fill(llvm::MutableArrayRef<InTypes>... Buffers) noexcept = 0;
};
} // namespace mathtest

#endif // MATHTEST_INPUTGENERATOR_HPP
