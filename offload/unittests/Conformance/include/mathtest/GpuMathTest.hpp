//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the GpuMathTest class, a test harness
/// that orchestrates running a math function on a device (GPU) and verifying
/// its results.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_GPUMATHTEST_HPP
#define MATHTEST_GPUMATHTEST_HPP

#include "mathtest/DeviceContext.hpp"
#include "mathtest/DeviceResources.hpp"
#include "mathtest/HostRefChecker.hpp"
#include "mathtest/InputGenerator.hpp"
#include "mathtest/Support.hpp"
#include "mathtest/TestResult.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace mathtest {

template <auto Func, typename Checker = HostRefChecker<Func>>
class [[nodiscard]] GpuMathTest final {
  using FunctionTraits = FunctionTraits<Func>;
  using OutType = typename FunctionTraits::ReturnType;
  using InTypesTuple = typename FunctionTraits::ArgTypesTuple;

  template <typename... Ts>
  using PartialResultType = TestResult<OutType, Ts...>;
  using KernelSignature = KernelSignatureOf_t<Func>;

  template <typename... Ts>
  using TypeIdentitiesTuple = std::tuple<TypeIdentityOf<Ts>...>;
  using InTypeIdentitiesTuple =
      ApplyTupleTypes_t<InTypesTuple, TypeIdentitiesTuple>;

  static constexpr std::size_t DefaultBufferSize =
      DefaultBufferSizeFor_v<OutType, InTypesTuple>;
  static constexpr uint32_t DefaultGroupSize = 512;

public:
  using FunctionConfig = FunctionConfig<Func>;
  using ResultType = ApplyTupleTypes_t<InTypesTuple, PartialResultType>;
  using GeneratorType = ApplyTupleTypes_t<InTypesTuple, InputGenerator>;

  [[nodiscard]] static llvm::Expected<GpuMathTest>
  create(std::shared_ptr<DeviceContext> Context, llvm::StringRef Provider,
         llvm::StringRef DeviceBinaryDir) {
    assert(Context && "Context must not be null");

    auto ExpectedKernel = getKernel(*Context, Provider, DeviceBinaryDir);
    if (!ExpectedKernel)
      return ExpectedKernel.takeError();

    return GpuMathTest(std::move(Context), Provider, *ExpectedKernel);
  }

  ResultType run(GeneratorType &Generator,
                 std::size_t BufferSize = DefaultBufferSize,
                 uint32_t GroupSize = DefaultGroupSize) const noexcept {
    assert(BufferSize > 0 && "Buffer size must be a positive value");
    assert(GroupSize > 0 && "Group size must be a positive value");

    auto [InBuffersTuple, OutBuffer] = createBuffers(BufferSize);
    ResultType FinalResult;

    while (true) {
      const std::size_t BatchSize = std::apply(
          [&](auto &...Buffers) { return Generator.fill(Buffers...); },
          InBuffersTuple);

      if (BatchSize == 0)
        break;

      const auto BatchResult =
          processBatch(InBuffersTuple, OutBuffer, BatchSize, GroupSize);

      FinalResult.accumulate(BatchResult);
    }

    return FinalResult;
  }

  [[nodiscard]] std::shared_ptr<DeviceContext> getContext() const noexcept {
    return Context;
  }

  [[nodiscard]] std::string getProvider() const noexcept { return Provider; }

private:
  explicit GpuMathTest(std::shared_ptr<DeviceContext> Context,
                       llvm::StringRef Provider,
                       DeviceKernel<KernelSignature> Kernel)
      : Context(std::move(Context)), Provider(Provider), Kernel(Kernel) {}

  static llvm::Expected<DeviceKernel<KernelSignature>>
  getKernel(const DeviceContext &Context, llvm::StringRef Provider,
            llvm::StringRef DeviceBinaryDir) {
    llvm::StringRef BinaryName = Provider;

    auto ExpectedImage = Context.loadBinary(DeviceBinaryDir, BinaryName);
    if (!ExpectedImage)
      return ExpectedImage.takeError();

    auto ExpectedKernel = Context.getKernel<KernelSignature>(
        *ExpectedImage, FunctionConfig::KernelName);
    if (!ExpectedKernel)
      return ExpectedKernel.takeError();

    return *ExpectedKernel;
  }

  [[nodiscard]] auto createBuffers(std::size_t BufferSize) const {
    auto InBuffersTuple = std::apply(
        [&](auto... InTypeIdentities) {
          return std::make_tuple(
              Context->createManagedBuffer<
                  typename decltype(InTypeIdentities)::type>(BufferSize)...);
        },
        InTypeIdentitiesTuple{});
    auto OutBuffer = Context->createManagedBuffer<OutType>(BufferSize);

    return std::make_pair(std::move(InBuffersTuple), std::move(OutBuffer));
  }

  template <typename InBuffersTupleType>
  [[nodiscard]] ResultType
  processBatch(const InBuffersTupleType &InBuffersTuple,
               ManagedBuffer<OutType> &OutBuffer, std::size_t BatchSize,
               uint32_t GroupSize) const noexcept {
    const uint32_t NumGroups = (BatchSize + GroupSize - 1) / GroupSize;
    const auto KernelArgsTuple = std::apply(
        [&](const auto &...InBuffers) {
          return std::make_tuple(InBuffers.data()..., OutBuffer.data(),
                                 BatchSize);
        },
        InBuffersTuple);

    std::apply(
        [&](const auto &...KernelArgs) {
          Context->launchKernel(Kernel, NumGroups, GroupSize, KernelArgs...);
        },
        KernelArgsTuple);

    return check(InBuffersTuple, OutBuffer, BatchSize);
  }

  template <typename InBuffersTupleType>
  [[nodiscard]] static ResultType
  check(const InBuffersTupleType &InBuffersTuple,
        const ManagedBuffer<OutType> &OutBuffer,
        std::size_t BatchSize) noexcept {
    const auto InViewsTuple = std::apply(
        [&](auto &...InBuffers) {
          return std::make_tuple(
              llvm::ArrayRef(InBuffers.data(), BatchSize)...);
        },
        InBuffersTuple);
    const auto OutView = llvm::ArrayRef<OutType>(OutBuffer.data(), BatchSize);

    return Checker::check(InViewsTuple, OutView);
  }

  std::shared_ptr<DeviceContext> Context;
  std::string Provider;
  DeviceKernel<KernelSignature> Kernel;
};
} // namespace mathtest

#endif // MATHTEST_GPUMATHTEST_HPP
