//===------------------- InsightBase.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

// Closed enum of built-in insight kinds. Adding a new built-in insight
// requires updating this enum and calling registerInsight() in the registry.
// Out-of-tree insights can bypass the enum by using a string-based kind
// identifier if the registry is queried by name rather than by kind.
enum InsightKind {
  FunctionComplexity,
  PassImpact,
  CompilationFlow,
  OptimizationDelta,
  MetricTrends,
  DiagnosticDelta,
  CallFrequency,
  LoopNesting,
  SectionSizes,
  DebugInfo,
  HeaderDepth
};

// Pre-fetched capability data passed to Insight::analyze(). CoreClient
// resolves the required capability and populates these before calling analyze.
struct InsightInput {
  std::string UnitId;
  std::optional<std::string> SnapshotId;
  std::optional<std::string> BaselineSnapshotId;
  const json::Object *PrimaryData = nullptr;
  const json::Object *BaselineData = nullptr;
};

struct InsightOutput {
  InsightKind Kind;
  std::string Name;
  json::Object Data;
  SmallVector<std::string, 4> Warnings;
};

class Insight {
public:
  virtual ~Insight() = default;

  virtual InsightKind getKind() const = 0;
  virtual StringRef getName() const = 0;
  virtual StringRef getDescription() const = 0;
  virtual StringRef getRequiredCapability() const = 0;
  virtual bool requiresBaseline() const { return false; }

  bool supportsInput(const InsightInput &Input) const {
    return Input.PrimaryData != nullptr &&
           (!requiresBaseline() || Input.BaselineData != nullptr);
  }

  virtual Expected<InsightOutput> analyze(const InsightInput &Input) const = 0;

protected:
  static Error noDataError() {
    return createStringError(inconvertibleErrorCode(),
                             "no capability data available");
  }
  static int64_t getInt(const json::Object &O, StringRef Key, int64_t Def = 0) {
    return O.getInteger(Key).value_or(Def);
  }
  static double getDouble(const json::Object &O, StringRef Key,
                          double Def = 0.0) {
    if (auto V = O.getNumber(Key))
      return *V;
    return Def;
  }
};

// Shared insight constants.
constexpr int DefaultTopN = 10;

// Round a percentage to one decimal place.
inline double roundToOneDecimal(double V) {
  return std::round(V * 10.0) / 10.0;
}

// Convert a range of strings into a JSON array.
template <typename Range>
json::Array toJSONArray(Range &&R) {
  json::Array A;
  for (auto &S : R)
    A.push_back(S);
  return A;
}

using InsightPtr = std::unique_ptr<Insight>;
using InsightMap = StringMap<Insight *>;

class InsightRegistry {
public:
  static InsightRegistry &instance();
  static void registerBuiltinInsights();

  void registerInsight(InsightPtr I);
  Insight *get(StringRef Name) const;
  SmallVector<Insight *, 16> all() const;
  SmallVector<Insight *, 16> getByKind(InsightKind Kind) const;
  bool isAvailable(StringRef Name, const InsightInput &Input) const;

private:
  InsightRegistry() = default;
  InsightMap Insights;
  SmallVector<InsightPtr, 16> Owned;
};

} // namespace llvm::advisor
