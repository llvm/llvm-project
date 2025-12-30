//===- RemarkCounter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic tool to count remarks based on properties
//
//===----------------------------------------------------------------------===//

#include "RemarkCounter.h"
#include "RemarkUtilRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace remarks;
using namespace llvm::remarkutil;

static cl::SubCommand CountSub("count",
                               "Collect remarks based on specified criteria.");

INPUT_FORMAT_COMMAND_LINE_OPTIONS(CountSub)
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(CountSub)
REMARK_FILTER_COMMAND_LINE_OPTIONS(CountSub)

REMARK_FILTER_SETUP_FUNC()

static cl::list<std::string>
    Keys("args", cl::desc("Specify remark argument/s to count by."),
         cl::value_desc("arguments"), cl::sub(CountSub), cl::ValueOptional);
static cl::list<std::string> RKeys(
    "rargs",
    cl::desc(
        "Specify remark argument/s to count (accepts regular expressions)."),
    cl::value_desc("arguments"), cl::sub(CountSub), cl::ValueOptional);

static cl::opt<CountBy> CountByOpt(
    "count-by", cl::desc("Specify the property to collect remarks by."),
    cl::values(
        clEnumValN(CountBy::REMARK, "remark-name",
                   "Counts individual remarks based on how many of the remark "
                   "exists."),
        clEnumValN(CountBy::ARGUMENT, "arg",
                   "Counts based on the value each specified argument has. The "
                   "argument has to have a number value to be considered.")),
    cl::init(CountBy::REMARK), cl::sub(CountSub));
static cl::opt<GroupBy> GroupByOpt(
    "group-by", cl::desc("Specify the property to group remarks by."),
    cl::values(
        clEnumValN(
            GroupBy::PER_SOURCE, "source",
            "Display the count broken down by the filepath of each remark "
            "emitted. Requires remarks to have DebugLoc information."),
        clEnumValN(GroupBy::PER_FUNCTION, "function",
                   "Breakdown the count by function name."),
        clEnumValN(
            GroupBy::PER_FUNCTION_WITH_DEBUG_LOC, "function-with-loc",
            "Breakdown the count by function name taking into consideration "
            "the filepath info from the DebugLoc of the remark."),
        clEnumValN(GroupBy::TOTAL, "total",
                   "Output the total number corresponding to the count for the "
                   "provided input file.")),
    cl::init(GroupBy::PER_SOURCE), cl::sub(CountSub));

/// Look for matching argument with \p Key in \p Remark and return the parsed
/// integer value or 0 if it is has no integer value.
static unsigned getValForKey(StringRef Key, const Remark &Remark) {
  auto *RemarkArg = find_if(Remark.Args, [&Key](const Argument &Arg) {
    return Arg.Key == Key && Arg.getValAsInt<unsigned>();
  });
  if (RemarkArg == Remark.Args.end())
    return 0;
  return *RemarkArg->getValAsInt<unsigned>();
}

Error ArgumentCounter::getAllMatchingArgumentsInRemark(
    StringRef Buffer, ArrayRef<FilterMatcher> Arguments, Filters &Filter) {
  auto MaybeParser = createRemarkParser(InputFormat, Buffer);
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    auto &Remark = **MaybeRemark;
    // Only collect keys from remarks included in the filter.
    if (!Filter.filterRemark(Remark))
      continue;
    for (auto &Key : Arguments) {
      for (Argument Arg : Remark.Args)
        if (Key.match(Arg.Key) && Arg.getValAsInt<unsigned>())
          ArgumentSetIdxMap.insert({Arg.Key, ArgumentSetIdxMap.size()});
    }
  }

  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  return Error::success();
}

std::optional<std::string> Counter::getGroupByKey(const Remark &Remark) {
  switch (Group) {
  case GroupBy::PER_FUNCTION:
    return Remark.FunctionName.str();
  case GroupBy::TOTAL:
    return "Total";
  case GroupBy::PER_SOURCE:
  case GroupBy::PER_FUNCTION_WITH_DEBUG_LOC:
    if (!Remark.Loc.has_value())
      return std::nullopt;

    if (Group == GroupBy::PER_FUNCTION_WITH_DEBUG_LOC)
      return Remark.Loc->SourceFilePath.str() + ":" + Remark.FunctionName.str();
    return Remark.Loc->SourceFilePath.str();
  }
  llvm_unreachable("Fully covered switch above!");
}

void ArgumentCounter::collect(const Remark &Remark) {
  SmallVector<unsigned, 4> Row(ArgumentSetIdxMap.size());
  std::optional<std::string> GroupByKey = getGroupByKey(Remark);
  // Early return if we don't have a value
  if (!GroupByKey)
    return;
  auto GroupVal = *GroupByKey;
  CountByKeysMap.insert({GroupVal, Row});
  for (auto [Key, Idx] : ArgumentSetIdxMap) {
    auto Count = getValForKey(Key, Remark);
    CountByKeysMap[GroupVal][Idx] += Count;
  }
}

void RemarkCounter::collect(const Remark &Remark) {
  if (std::optional<std::string> Key = getGroupByKey(Remark))
    ++CountedByRemarksMap[*Key];
}

Error ArgumentCounter::print(StringRef OutputFileName) {
  auto MaybeOF =
      getOutputFileWithFlags(OutputFileName, sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();

  auto OF = std::move(*MaybeOF);
  OF->os() << groupByToStr(Group) << ",";
  OF->os() << llvm::interleaved(llvm::make_first_range(ArgumentSetIdxMap), ",");
  OF->os() << "\n";
  for (auto [Header, CountVector] : CountByKeysMap) {
    OF->os() << Header << ",";
    OF->os() << llvm::interleaved(CountVector, ",");
    OF->os() << "\n";
  }
  return Error::success();
}

Error RemarkCounter::print(StringRef OutputFileName) {
  auto MaybeOF =
      getOutputFileWithFlags(OutputFileName, sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();

  auto OF = std::move(*MaybeOF);
  OF->os() << groupByToStr(Group) << ","
           << "Count\n";
  for (auto [Key, Count] : CountedByRemarksMap)
    OF->os() << Key << "," << Count << "\n";
  OF->keep();
  return Error::success();
}

Error useCollectRemark(StringRef Buffer, Counter &Counter, Filters &Filter) {
  // Create Parser.
  auto MaybeParser = createRemarkParser(InputFormat, Buffer);
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;
  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    const Remark &Remark = **MaybeRemark;
    if (Filter.filterRemark(Remark))
      Counter.collect(Remark);
  }

  if (auto E = Counter.print(OutputFileName))
    return E;
  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  return Error::success();
}

static Error collectRemarks() {
  // Create a parser for the user-specified input format.
  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  StringRef Buffer = (*MaybeBuf)->getBuffer();
  auto MaybeFilter = getRemarkFilters();
  if (!MaybeFilter)
    return MaybeFilter.takeError();
  auto &Filter = *MaybeFilter;
  if (CountByOpt == CountBy::REMARK) {
    RemarkCounter RC(GroupByOpt);
    if (auto E = useCollectRemark(Buffer, RC, Filter))
      return E;
  } else if (CountByOpt == CountBy::ARGUMENT) {
    SmallVector<FilterMatcher, 4> ArgumentsVector;
    if (!Keys.empty()) {
      for (auto &Key : Keys)
        ArgumentsVector.push_back(FilterMatcher::createExact(Key));
    } else if (!RKeys.empty())
      for (auto Key : RKeys) {
        auto FM = FilterMatcher::createRE(Key, RKeys);
        if (!FM)
          return FM.takeError();
        ArgumentsVector.push_back(std::move(*FM));
      }
    else
      ArgumentsVector.push_back(FilterMatcher::createAny());

    Expected<ArgumentCounter> AC = ArgumentCounter::createArgumentCounter(
        GroupByOpt, ArgumentsVector, Buffer, Filter);
    if (!AC)
      return AC.takeError();
    if (auto E = useCollectRemark(Buffer, *AC, Filter))
      return E;
  }
  return Error::success();
}

static CommandRegistration CountReg(&CountSub, collectRemarks);
