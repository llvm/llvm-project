//===- RemarkSummary.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Specialized tool to summarize remarks
//
//===----------------------------------------------------------------------===//

#include "RemarkUtilHelpers.h"
#include "RemarkUtilRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/WithColor.h"
#include <memory>

using namespace llvm;
using namespace remarks;
using namespace llvm::remarkutil;

namespace summary {

static cl::SubCommand
    SummarySub("summary", "Summarize remarks using different strategies.");

INPUT_FORMAT_COMMAND_LINE_OPTIONS(SummarySub)
OUTPUT_FORMAT_COMMAND_LINE_OPTIONS(SummarySub)
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(SummarySub)

static cl::OptionCategory SummaryStrategyCat("Strategy options");

enum class KeepMode { None, Used, All };

static cl::opt<KeepMode> KeepInputOpt(
    "keep", cl::desc("Keep input remarks in output"), cl::init(KeepMode::None),
    cl::values(clEnumValN(KeepMode::None, "none",
                          "Don't keep input remarks (default)"),
               clEnumValN(KeepMode::Used, "used",
                          "Keep only remarks used for summary"),
               clEnumValN(KeepMode::All, "all", "Keep all input remarks")),
    cl::sub(SummarySub));

static cl::opt<bool>
    IgnoreMalformedOpt("ignore-malformed",
                       cl::desc("Ignore remarks that fail to process"),
                       cl::init(false), cl::Hidden, cl::sub(SummarySub));

// Use one cl::opt per Strategy, because future strategies might need to take
// per-strategy parameters.
static cl::opt<bool> EnableInlineSummaryOpt(
    "inline-callees", cl::desc("Summarize per-callee inling statistics"),
    cl::cat(SummaryStrategyCat), cl::init(false), cl::sub(SummarySub));

/// An interface to implement different strategies for creating remark
/// summaries. Override this class to develop new strategies.
class SummaryStrategy {
public:
  virtual ~SummaryStrategy() = default;

  /// Strategy should return true if it wants to process the remark \p R.
  virtual bool filter(Remark &R) = 0;

  /// Hook to process the remark \p R (i.e. collect the necessary data for
  /// producing summary remarks). This will only be called with remarks
  /// accepted by filter(). Can return an error if \p R is malformed or
  /// unexpected.
  virtual Error process(Remark &R) = 0;

  /// Hook to emit new remarks based on the collected data.
  virtual void emit(RemarkSerializer &Serializer) = 0;
};

/// Check if any summary strategy options are explicitly enabled.
static bool isAnyStrategyRequested() {
  for (auto &[_, Opt] : cl::getRegisteredOptions(SummarySub)) {
    if (!is_contained(Opt->Categories, &SummaryStrategyCat))
      continue;
    if (!Opt->getNumOccurrences())
      continue;
    return true;
  }
  return false;
}

class InlineCalleeSummary : public SummaryStrategy {
  struct CallsiteCost {
    int Cost = 0;
    int Threshold = 0;
    std::optional<RemarkLocation> Loc;

    int getProfit() const { return Threshold - Cost; }

    friend bool operator==(const CallsiteCost &A, const CallsiteCost &B) {
      return A.Cost == B.Cost && A.Threshold == B.Threshold && A.Loc == B.Loc;
    }

    friend bool operator!=(const CallsiteCost &A, const CallsiteCost &B) {
      return !(A == B);
    }
  };

  struct CalleeSummary {
    SmallDenseMap<StringRef, size_t> Stats;
    std::optional<RemarkLocation> Loc;
    std::optional<CallsiteCost> LeastProfit;
    std::optional<CallsiteCost> MostProfit;

    void updateCost(CallsiteCost NewCost) {
      if (!LeastProfit || NewCost.getProfit() < LeastProfit->getProfit())
        LeastProfit = NewCost;
      if (!MostProfit || NewCost.getProfit() > MostProfit->getProfit())
        MostProfit = NewCost;
    }
  };

  DenseMap<StringRef, CalleeSummary> Callees;

  Error malformed() { return createStringError("Malformed inline remark."); }

  bool filter(Remark &R) override {
    return R.PassName == "inline" && R.RemarkName != "Summary";
  }

  Error process(Remark &R) override {
    auto *CalleeArg = R.getArgByKey("Callee");
    if (!CalleeArg)
      return Error::success();
    auto &Callee = Callees[CalleeArg->Val];
    ++Callee.Stats[R.RemarkName];
    if (!Callee.Loc)
      Callee.Loc = CalleeArg->Loc;

    Argument *CostArg = R.getArgByKey("Cost");
    Argument *ThresholdArg = R.getArgByKey("Threshold");
    if (!CostArg || !ThresholdArg)
      return Error::success();
    auto CostVal = CostArg->getValAsInt<int>();
    auto ThresholdVal = ThresholdArg->getValAsInt<int>();
    if (!CostVal || !ThresholdVal)
      return malformed();
    Callee.updateCost({*CostVal, *ThresholdVal, R.Loc});
    return Error::success();
  }

  void emit(RemarkSerializer &Serializer) override {
    SmallVector<StringRef> SortedKeys(Callees.keys());
    llvm::sort(SortedKeys);
    for (StringRef K : SortedKeys) {
      auto &V = Callees[K];
      RemarkBuilder RB(Type::Analysis, "inline", "Summary", K);
      if (V.Stats.empty())
        continue;
      RB.R.Loc = V.Loc;
      RB << "Incoming Calls (";
      SmallVector<StringRef> StatKeys(V.Stats.keys());
      llvm::sort(StatKeys);
      bool First = true;
      for (StringRef StatK : StatKeys) {
        if (!First)
          RB << ", ";
        RB << StatK << ": " << NV(StatK, V.Stats[StatK]);
        First = false;
      }
      RB << ")";
      if (V.LeastProfit && V.MostProfit != V.LeastProfit) {
        RB << "\nLeast profitable (cost="
           << NV("LeastProfitCost", V.LeastProfit->Cost, V.LeastProfit->Loc)
           << ", threshold="
           << NV("LeastProfitThreshold", V.LeastProfit->Threshold) << ")";
      }
      if (V.MostProfit) {
        RB << "\nMost profitable (cost="
           << NV("MostProfitCost", V.MostProfit->Cost, V.MostProfit->Loc)
           << ", threshold="
           << NV("MostProfitThreshold", V.MostProfit->Threshold) << ")";
      }
      Serializer.emit(RB.R);
    }
  }
};

static Error trySummary() {
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;

  Format SerializerFormat =
      getSerializerFormat(OutputFileName, OutputFormat, Parser.ParserFormat);

  auto MaybeOF = getOutputFileForRemarks(OutputFileName, SerializerFormat);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);

  auto MaybeSerializer = createRemarkSerializer(SerializerFormat, OF->os());
  if (!MaybeSerializer)
    return MaybeSerializer.takeError();
  auto &Serializer = **MaybeSerializer;

  bool UseDefaultStrategies = !isAnyStrategyRequested();
  SmallVector<std::unique_ptr<SummaryStrategy>> Strategies;
  if (EnableInlineSummaryOpt || UseDefaultStrategies)
    Strategies.push_back(std::make_unique<InlineCalleeSummary>());

  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    Remark &Remark = **MaybeRemark;
    bool UsedRemark = false;
    for (auto &Strategy : Strategies) {
      if (!Strategy->filter(Remark))
        continue;
      UsedRemark = true;
      if (auto E = Strategy->process(Remark)) {
        if (IgnoreMalformedOpt) {
          WithColor::warning() << "Ignored error: " << E << "\n";
          consumeError(std::move(E));
          continue;
        }
        return E;
      }
    }
    if (KeepInputOpt == KeepMode::All ||
        (KeepInputOpt == KeepMode::Used && UsedRemark))
      Serializer.emit(Remark);
  }

  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));

  for (auto &Strategy : Strategies)
    Strategy->emit(Serializer);

  OF->keep();
  return Error::success();
}

static CommandRegistration SummaryReg(&SummarySub, trySummary);

} // namespace summary
