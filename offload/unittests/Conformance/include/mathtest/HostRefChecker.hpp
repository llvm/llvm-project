//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the HostRefChecker class, which
/// verifies the results of a device computation against a reference
/// implementation on the host.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_HOSTREFCHECKER_HPP
#define MATHTEST_HOSTREFCHECKER_HPP

#include "mathtest/Numerics.hpp"
#include "mathtest/Support.hpp"
#include "mathtest/TestResult.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Parallel.h"

#include <cstddef>
#include <tuple>
#include <utility>

namespace mathtest {

template <auto Func> class HostRefChecker {
  using FunctionTraits = FunctionTraits<Func>;
  using InTypesTuple = typename FunctionTraits::ArgTypesTuple;

  using FunctionConfig = FunctionConfig<Func>;

  template <typename... Ts>
  using BuffersTupleType = std::tuple<llvm::ArrayRef<Ts>...>;

public:
  using OutType = typename FunctionTraits::ReturnType;

private:
  template <typename... Ts>
  using PartialResultType = TestResult<OutType, Ts...>;

public:
  using ResultType = ApplyTupleTypes_t<InTypesTuple, PartialResultType>;
  using InBuffersTupleType = ApplyTupleTypes_t<InTypesTuple, BuffersTupleType>;

  HostRefChecker() = delete;

  static ResultType check(InBuffersTupleType InBuffersTuple,
                          llvm::ArrayRef<OutType> OutBuffer) noexcept {
    const std::size_t BufferSize = OutBuffer.size();
    std::apply(
        [&](const auto &...InBuffers) {
          assert(
              ((InBuffers.size() == BufferSize) && ...) &&
              "All input buffers must have the same size as the output buffer");
        },
        InBuffersTuple);

    assert((BufferSize != 0) && "Buffer size cannot be zero");

    ResultType Init;

    auto Transform = [&](std::size_t Index) {
      auto CurrentInputsTuple = std::apply(
          [&](const auto &...InBuffers) {
            return std::make_tuple(InBuffers[Index]...);
          },
          InBuffersTuple);

      const OutType Actual = OutBuffer[Index];
      const OutType Expected = std::apply(Func, CurrentInputsTuple);

      const auto UlpDistance = computeUlpDistance(Actual, Expected);
      const bool IsFailure = UlpDistance > FunctionConfig::UlpTolerance;

      return ResultType(UlpDistance, IsFailure,
                        typename ResultType::TestCase(
                            std::move(CurrentInputsTuple), Actual, Expected));
    };

    auto Reduce = [](ResultType A, const ResultType &B) {
      A.accumulate(B);
      return A;
    };

    const auto Indexes = llvm::seq(BufferSize);
    return llvm::parallelTransformReduce(Indexes.begin(), Indexes.end(), Init,
                                         Reduce, Transform);
  }
};
} // namespace mathtest

#endif // MATHTEST_HOSTREFCHECKER_HPP
