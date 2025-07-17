#pragma once

#include "mathtest/DeviceContext.hpp"
#include "mathtest/DeviceResources.hpp"
#include "mathtest/HostRefChecker.hpp"
#include "mathtest/InputGenerator.hpp"
#include "mathtest/Support.hpp"
#include "mathtest/TestResult.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
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

  explicit GpuMathTest(std::shared_ptr<DeviceContext> Context,
                       llvm::StringRef Provider,
                       llvm::StringRef DeviceBinsDirectory)
      : Context(std::move(Context)),
        Kernel(getKernel(this->Context, Provider, DeviceBinsDirectory)) {
    assert(this->Context && "Context must not be null");
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

  [[nodiscard]] const DeviceContext &getContext() const noexcept {
    assert(Context && "Context must not be null");
    return *Context;
  }

private:
  static DeviceKernel<KernelSignature>
  getKernel(const std::shared_ptr<DeviceContext> &Context,
            llvm::StringRef Provider,
            llvm::StringRef DeviceBinsDirectory) noexcept {
    llvm::StringRef BinaryName = llvm::StringSwitch<llvm::StringRef>(Provider)
                                     .Case("llvm-libm", "LLVMLibm")
                                     .Default("");

    if (BinaryName.empty()) {
      FATAL_ERROR(llvm::Twine("Unsupported provider: '") + Provider + "'");
    }

    const auto Image = Context->loadBinary(DeviceBinsDirectory, BinaryName);

    return Context->getKernel<KernelSignature>(Image,
                                               FunctionConfig::KernelName);
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
  DeviceKernel<KernelSignature> Kernel;
};
} // namespace mathtest
