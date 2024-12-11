/* TO_UPSTREAM(BoundsSafety) ON */
//===- BoundsSafetyCount.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collect total count information for instructions annotated by -fbounds-safety
//
//===----------------------------------------------------------------------===//
#include "RemarkUtilHelpers.h"
#include "RemarkUtilRegistry.h"
#include "llvm/ADT/MapVector.h"

using namespace llvm;
using namespace remarks;
using namespace llvm::remarkutil;

static cl::SubCommand
    BoundsSafetyCount("bounds-safety-count",
                   "Collect total count information for instructions annotated "
                   "with -fbounds-safety remarks. By default this will return the "
                   "count for the given binary. (uses AnnotationRemarksPass)");

namespace boundssafetycount {
static const std::string BoundsSafetySummaryRemark = "bounds-safety-total-summary";
static cl::opt<bool>
    CollectPerSource("collect-per-source",
                     cl::desc("collect -fbounds-safety remark count per source file"),
                     cl::init(false), cl::sub(BoundsSafetyCount));

INPUT_FORMAT_COMMAND_LINE_OPTIONS(BoundsSafetyCount)
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(BoundsSafetyCount)

static bool shouldSkipRemark(bool UseDebugLoc, Remark &Remark) {
  return UseDebugLoc && !Remark.Loc.has_value();
}

static Error tryBoundsSafetyCount() {
  // Create the output buffer.
  auto MaybeOF = getOutputFileWithFlags(OutputFileName,
                                        /*Flags = */ sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);
  // Create a parser for the user-specified input format.
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  // Emit CSV header.
  if (CollectPerSource)
    OF->os() << "Source,Count\n";
  else
    OF->os() << "Count\n";
  // Parse all remarks. When we see the specified remark collect the count
  // information.
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  // collecting count per source path when `--collect-per-source` is passed.
  MapVector<StringRef, unsigned> SourceCountMap;
  // collect the total count for the given binary.
  unsigned TotalCount = 0;
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    auto &Remark = **MaybeRemark;
    if (Remark.RemarkName != "AnnotationSummary")
      continue;
    // Skip remark if it doesn't contain DebugLoc info.
    if (shouldSkipRemark(CollectPerSource, Remark))
      continue;
    auto *RemarkNameArg = find_if(Remark.Args, [](const Argument &Arg) {
      return Arg.Key == "type" && Arg.Val == BoundsSafetySummaryRemark;
    });
    if (RemarkNameArg == Remark.Args.end())
      continue;
    auto *CountArg = find_if(
        Remark.Args, [](const Argument &Arg) { return Arg.Key == "count"; });
    assert(CountArg != Remark.Args.end() &&
           "Expected annotation-type remark to have a count key?");
    unsigned CountInt = std::stoi(CountArg->Val.str());
    if (CollectPerSource) {
      auto SourceCountMapIter =
          SourceCountMap.insert({Remark.Loc->SourceFilePath, 0});
      SourceCountMapIter.first->second += CountInt;
    } else
      TotalCount += CountInt;
  }
  if (CollectPerSource)
    for (auto [Source, Count] : SourceCountMap)
      OF->os() << Source << "," << Count << "\n";
  else if (TotalCount != 0)
    OF->os() << TotalCount << "\n";
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  OF->keep();
  return Error::success();
}
} // namespace boundssafetycount

static CommandRegistration BoundsSafetyCountReg(&BoundsSafetyCount,
                                             boundssafetycount::tryBoundsSafetyCount);

/* TO_UPSTREAM(BoundsSafety) OFF */
