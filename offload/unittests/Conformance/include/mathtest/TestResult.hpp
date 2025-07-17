#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace mathtest {

template <typename OutType, typename... InTypes>
class [[nodiscard]] TestResult {
public:
  struct [[nodiscard]] TestCase {
    std::tuple<InTypes...> Inputs;
    OutType Actual;
    OutType Expected;

    explicit constexpr TestCase(std::tuple<InTypes...> &&Inputs, OutType Actual,
                                OutType Expected) noexcept
        : Inputs(std::move(Inputs)), Actual(std::move(Actual)),
          Expected(std::move(Expected)) {}
  };

  TestResult() = default;

  explicit TestResult(uint64_t UlpDistance, bool IsFailure,
                      TestCase &&Case) noexcept
      : MaxUlpDistance(UlpDistance), FailureCount(IsFailure ? 1 : 0),
        TestCaseCount(1) {
    if (IsFailure) {
      WorstFailingCase.emplace(std::move(Case));
    }
  }

  void accumulate(const TestResult &Other) noexcept {
    if (Other.MaxUlpDistance > MaxUlpDistance) {
      MaxUlpDistance = Other.MaxUlpDistance;
      WorstFailingCase = Other.WorstFailingCase;
    }

    FailureCount += Other.FailureCount;
    TestCaseCount += Other.TestCaseCount;
  }

  [[nodiscard]] bool hasPassed() const noexcept { return FailureCount == 0; }

  [[nodiscard]] uint64_t getMaxUlpDistance() const noexcept {
    return MaxUlpDistance;
  }

  [[nodiscard]] uint64_t getFailureCount() const noexcept {
    return FailureCount;
  }

  [[nodiscard]] uint64_t getTestCaseCount() const noexcept {
    return TestCaseCount;
  }

  [[nodiscard]] const std::optional<TestCase> &
  getWorstFailingCase() const noexcept {
    return WorstFailingCase;
  }

private:
  uint64_t MaxUlpDistance = 0;
  uint64_t FailureCount = 0;
  uint64_t TestCaseCount = 0;
  std::optional<TestCase> WorstFailingCase;
};
} // namespace mathtest
