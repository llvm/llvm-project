//===-------------- RemarkDiff.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Diffs remarks between two remark files.
/// The tool offers different modes for comparing two versions of remarks.
/// 1. Look through common remarks between two files.
/// 2. Compare the remark type. This is useful to check if an optimzation
/// changed from passing to failing.
/// 3. Compare remark arguments. This is useful to check if a remark argument
/// changed after some compiler change.
///
/// The results are presented as a json file.
///
//===----------------------------------------------------------------------===//

#include "RemarkDiff.h"
#include "llvm/Support/JSON.h"
#include <utility>

using namespace llvm;
using namespace remarks;
using namespace llvm::remarkutil;

static cl::SubCommand DiffSub("diff",
                              "diff remarks based on specified properties.");
static cl::opt<std::string> RemarkFileA(cl::Positional,
                                        cl::desc("<remarka_file>"),
                                        cl::Required, cl::sub(DiffSub));
static cl::opt<std::string> RemarkFileB(cl::Positional,
                                        cl::desc("<remarkb_file>"),
                                        cl::Required, cl::sub(DiffSub));

static cl::opt<bool> Verbose(
    "v", cl::init(false),
    cl::desc("Output detailed difference for remarks. By default the tool will "
             "only show the remark name, type and location. If the flag is "
             "added we display the arguments that are different."),
    cl::sub(DiffSub));
static cl::opt<bool>
    ShowArgDiffOnly("show-arg-diff-only", cl::init(false),
                    cl::desc("Show only the remarks that have the same header "
                             "and differ in arguments"),
                    cl::sub(DiffSub));
static cl::opt<bool> OnlyShowCommonRemarks(
    "only-show-common-remarks", cl::init(false),
    cl::desc("Ignore any remarks that don't exist in both <remarka_file> and "
             "<remarkb_file>."),
    cl::sub(DiffSub));
static cl::opt<bool> ShowOnlyDifferentRemarks(
    "only-show-different-remarks", cl::init(false),
    cl::desc("Show remarks that are exclusively at either A or B"),
    cl::sub(DiffSub));
static cl::opt<bool> ShowOnlyA("only-show-a", cl::init(false),
                               cl::desc("Show remarks that are only in A"),
                               cl::sub(DiffSub));

static cl::opt<bool> ShowOnlyB("only-show-b", cl::init(false),
                               cl::desc("Show remarks that are only in B"),
                               cl::sub(DiffSub));
static cl::opt<bool> ShowRemarkTypeDiffOnly(
    "show-remark-type-diff-only", cl::init(false),
    cl::desc("Only show diff if remarks have the same header but different "
             "type"),
    cl::sub(DiffSub));

static cl::opt<Format> InputFormat(
    "parser", cl::desc("Input remark format to parse"),
    cl::values(clEnumValN(Format::YAML, "yaml", "YAML"),
               clEnumValN(Format::Bitstream, "bitstream", "Bitstream")),
    cl::sub(DiffSub));
static cl::opt<ReportStyleOptions> ReportStyle(
    "report_style", cl::sub(DiffSub),
    cl::init(ReportStyleOptions::human_output),
    cl::desc("Choose the report output format:"),
    cl::values(clEnumValN(human_output, "human", "Human-readable format"),
               clEnumValN(json_output, "json", "JSON format")));
static cl::opt<std::string> OutputFileName("o", cl::init("-"), cl::sub(DiffSub),
                                           cl::desc("Output"),
                                           cl::value_desc("file"));
FILTER_COMMAND_LINE_OPTIONS(DiffSub)

void RemarkArgInfo::print(raw_ostream &OS) const {
  OS << Key << ": " << Val << "\n";
}

void RemarkInfo::printHeader(raw_ostream &OS) const {
  OS << "Name: " << RemarkName << "\n";
  OS << "FunctionName: " << FunctionName << "\n";
  OS << "PassName: " << PassName << "\n";
}

void RemarkInfo::print(raw_ostream &OS) const {
  printHeader(OS);
  OS << "Type: " << typeToStr(RemarkType) << "\n";
  if (!Args.empty()) {
    OS << "Args:\n";
    for (auto Arg : Args)
      OS << "\t" << Arg;
  }
}

void DiffAtRemark::print(raw_ostream &OS) const {
  BaseRemark.printHeader(OS);
  if (RemarkTypeDiff) {
    OS << "Only at A >>>>\n";
    OS << "Type: " << typeToStr(RemarkTypeDiff->first) << "\n";
    OS << "=====\n";
    OS << "Only at B <<<<\n";
    OS << "Type: " << typeToStr(RemarkTypeDiff->second) << "\n";
    OS << "=====\n";
  }

  if (!OnlyA.empty()) {
    OS << "Only at A >>>>\n";
    unsigned Idx = 0;
    for (auto &R : OnlyA) {
      OS << R;
      if (Idx < OnlyA.size() - 1)
        OS << "\n";
      Idx++;
    }
    OS << "=====\n";
  }
  if (!OnlyB.empty()) {
    OS << "Only at B <<<<\n";
    unsigned Idx = 0;
    for (auto &R : OnlyB) {
      OS << R;
      if (Idx < OnlyB.size() - 1)
        OS << "\n";
      Idx++;
    }
    OS << "=====\n";
  }
  if (Verbose)
    for (auto &R : InBoth)
      OS << R << "\n";
}

void DiffAtLoc::print(raw_ostream &OS) {
  if (!OnlyA.empty()) {
    OS << "Only at A >>>>\n";
    unsigned Idx = 0;
    for (auto &R : OnlyA) {
      OS << R;
      if (Idx < OnlyA.size() - 1)
        OS << "\n";
      Idx++;
    }
    OS << "=====\n";
  }

  if (!OnlyB.empty()) {
    OS << "Only at B <<<<\n";
    unsigned Idx = 0;
    for (auto &R : OnlyB) {
      OS << R;
      if (Idx < OnlyB.size() - 1)
        OS << "\n";
      Idx++;
    }
    OS << "=====\n";
  }

  if (!HasTheSameHeader.empty()) {
    OS << "--- Has the same header ---\n";
    for (auto &R : HasTheSameHeader)
      R.print(OS);
  }
}

/// \returns json array representation of a vecotor of remark arguments.
static json::Array remarkArgsToJson(SmallVectorImpl<RemarkArgInfo> &Args) {
  json::Array ArgArray;
  for (auto Arg : Args) {
    json::Object ArgPair({{Arg.Key, Arg.Val}});
    ArgArray.push_back(std::move(ArgPair));
  }
  return ArgArray;
}

/// \returns remark representation as a json object.
static json::Object remarkToJSON(RemarkInfo &Remark) {
  json::Object RemarkJSON;
  RemarkJSON["RemarkName"] = Remark.RemarkName;
  RemarkJSON["FunctionName"] = Remark.FunctionName;
  RemarkJSON["PassName"] = Remark.PassName;
  RemarkJSON["RemarkType"] = typeToStr(Remark.RemarkType);
  if (Verbose)
    RemarkJSON["Args"] = remarkArgsToJson(Remark.Args);
  return RemarkJSON;
}

json::Object DiffAtRemark::toJson() {
  json::Object Object;
  Object["FunctionName"] = BaseRemark.FunctionName;
  Object["PassName"] = BaseRemark.PassName;
  Object["RemarkName"] = BaseRemark.RemarkName;
  // display remark type if it is the same between the two remarks.
  if (!RemarkTypeDiff)
    Object["RemarkType"] = typeToStr(BaseRemark.RemarkType);
  json::Array InBothJSON;
  json::Array OnlyAJson;
  json::Array OnlyBJson;
  for (auto Arg : InBoth) {
    json::Object ArgPair({{Arg.Key, Arg.Val}});
    InBothJSON.push_back(std::move(ArgPair));
  }
  for (auto Arg : OnlyA) {
    json::Object ArgPair({{Arg.Key, Arg.Val}});
    OnlyAJson.push_back(std::move(ArgPair));
  }
  for (auto Arg : OnlyB) {
    json::Object ArgPair({{Arg.Key, Arg.Val}});
    OnlyBJson.push_back(std::move(ArgPair));
  }
  json::Object Diff;
  if (RemarkTypeDiff) {
    Diff["RemarkTypeA"] = typeToStr(RemarkTypeDiff->first);
    Diff["RemarkTypeB"] = typeToStr(RemarkTypeDiff->second);
  }

  // Only display common remark arguments if verbose is passed.
  if (Verbose)
    Object["ArgsInBoth"] = remarkArgsToJson(InBoth);
  if (!OnlyAJson.empty())
    Diff["ArgsAtA"] = remarkArgsToJson(OnlyA);
  if (!OnlyBJson.empty())
    Diff["ArgsAtB"] = remarkArgsToJson(OnlyB);
  Object["Diff"] = std::move(Diff);
  return Object;
}
json::Object DiffAtLoc::toJson() {
  json::Object Obj;
  json::Array DiffObj;
  json::Array OnlyAObj;
  json::Array OnlyBObj;
  json::Array HasSameHeaderObj;
  for (auto R : OnlyA)
    OnlyAObj.push_back(remarkToJSON(R));
  for (auto R : OnlyB)
    OnlyBObj.push_back(remarkToJSON(R));
  for (auto R : HasTheSameHeader)
    HasSameHeaderObj.push_back(R.toJson());
  if (!OnlyShowCommonRemarks) {
    Obj["OnlyA"] = std::move(OnlyAObj);
    Obj["OnlyB"] = std::move(OnlyBObj);
  }
  Obj["HasSameHeaderObj"] = std::move(HasSameHeaderObj);
  return Obj;
}

/// Parse the a remark buffer and generate a set of remarks ordered by the debug
/// location.
static Error parseRemarkFile(
    std::unique_ptr<RemarkParser> &Parser,
    MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> &DebugLoc2RemarkMap,
    Filters &Filter) {
  auto MaybeRemark = Parser->next();
  // Use a set vector to remove duplicate entries in the remark file.
  MapVector<DebugLocation, SmallSet<RemarkInfo, 4>> DebugLoc2RemarkMapSet;
  for (; MaybeRemark; MaybeRemark = Parser->next()) {
    auto &Remark = **MaybeRemark;
    if (!Filter.filterRemark(Remark))
      continue;
    std::string SourceFilePath = "";
    unsigned SourceLine = 0;
    unsigned SourceColumn = 0;
    if (Remark.Loc.has_value()) {
      SourceFilePath = Remark.Loc->SourceFilePath.str();
      SourceLine = Remark.Loc->SourceLine;
      SourceColumn = Remark.Loc->SourceColumn;
    }

    DebugLocation Key(SourceFilePath, Remark.FunctionName, SourceLine,
                      SourceColumn);
    auto Iter = DebugLoc2RemarkMapSet.insert({Key, {}});
    Iter.first->second.insert(Remark);
  }
  for (auto [DebugLocation, RemarkSet] : DebugLoc2RemarkMapSet)
    DebugLoc2RemarkMap[DebugLocation] = {RemarkSet.begin(), RemarkSet.end()};

  auto E = MaybeRemark.takeError();
  if (!E.isA<remarks::EndOfFileError>())
    return E;
  consumeError(std::move(E));
  return Error::success();
}

void Diff::computeDiffAtLoc(DebugLocation Loc, ArrayRef<RemarkInfo> RemarksA,
                            ArrayRef<RemarkInfo> RemarksB) {

  // A set of remarks where either they have a remark at the other file
  // equaling them or share the same header. This is used to reduce the
  // duplicates when looking at a location. If a remark has a counterpart in
  // the other file then we aren't interested if it shares the same header
  // with another remark.
  DiffAtLoc DiffLoc(Loc);
  SmallSet<RemarkInfo, 4> FoundRemarks;
  SmallVector<std::pair<RemarkInfo, RemarkInfo>, 4> HasSameHeader;
  // First look through the remarks that are exactly equal in the two files.
  for (auto &RA : RemarksA)
    for (auto &RB : RemarksB)
      if (RA == RB)
        FoundRemarks.insert(RA);
  for (auto &RA : RemarksA) {
    // skip
    if (FoundRemarks.contains(RA))
      continue;
    for (auto &RB : RemarksB) {
      if (FoundRemarks.contains(RB))
        continue;
      if (RA.hasSameHeader(RB)) {
        HasSameHeader.push_back({RA, RB});
        FoundRemarks.insert(RA);
        FoundRemarks.insert(RB);
      }
    }
  }

  for (auto &RA : RemarksA) {
    if (!FoundRemarks.contains(RA) && DiffConfig.AddRemarksFromA)
      DiffLoc.OnlyA.push_back(RA);
  }
  for (auto &RB : RemarksB) {
    if (!FoundRemarks.contains(RB) && DiffConfig.AddRemarksFromB)
      DiffLoc.OnlyB.push_back(RB);
  }
  // Discard any shared remarks and only display uniquly different remarks
  // between A and B.
  if (!DiffConfig.ShowCommonRemarks) {
    DiffAtLocs.push_back(DiffLoc);
    return;
  }

  // calculate the diff at each shared remark.
  for (auto &[RA, RB] : HasSameHeader)
    DiffLoc.HasTheSameHeader.push_back({RA, RB, DiffConfig});
  DiffAtLocs.push_back(DiffLoc);
}

void Diff::computeDiff(
    SetVector<DebugLocation> &DebugLocs,
    MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> &DebugLoc2RemarkA,
    MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> &DebugLoc2RemarkB) {
  // Add all debug locs from file a and file b to a unique set of Locations.
  for (const DebugLocation &Loc : DebugLocs) {
    SmallVector<RemarkInfo, 4> RemarksLocAIt = DebugLoc2RemarkA.lookup(Loc);
    SmallVector<RemarkInfo, 4> RemarksLocBIt = DebugLoc2RemarkB.lookup(Loc);
    computeDiffAtLoc(Loc, RemarksLocAIt, RemarksLocBIt);
  }
}

Error Diff::printDiff(StringRef InputFileNameA, StringRef InputFileNameB) {
  // Create the output buffer.
  auto MaybeOF = getOutputFileWithFlags(OutputFileName,
                                        /*Flags = */ sys::fs::OF_TextWithCRLF);
  if (!MaybeOF)
    return MaybeOF.takeError();
  std::unique_ptr<ToolOutputFile> OF = std::move(*MaybeOF);
  if (ReportStyle == ReportStyleOptions::human_output) {
    OF->os() << "File A: " << InputFileNameA << "\n";
    OF->os() << "File B: " << InputFileNameB << "\n";
    for (auto LocDiff : DiffAtLocs) {
      if (LocDiff.isEmpty())
        continue;
      OF->os() << "----------\n";
      OF->os() << LocDiff.Loc.SourceFilePath << ":" << LocDiff.Loc.FunctionName
               << "  Ln: " << LocDiff.Loc.SourceLine
               << " Col: " << LocDiff.Loc.SourceColumn << "\n";
      LocDiff.print(OF->os());
    }
  } else {

    json::Object Output;
    json::Object Files(
        {{"A", InputFileNameA.str()}, {"B", InputFileNameB.str()}});
    Output["Files"] = std::move(Files);
    SmallVector<json::Object> Locs;
    for (auto LocDiff : DiffAtLocs) {
      Output[LocDiff.Loc.SourceFilePath] = json::Object(
          {{LocDiff.Loc.FunctionName,
            json::Object({{LocDiff.Loc.toString(), LocDiff.toJson()}})}});
    }

    json::OStream JOS(OF->os(), 2);
    JOS.value(std::move(Output));
    OF->os() << '\n';
  }
  OF->keep();
  return Error::success();
}

static Error createRemarkDiff() {
  // Get memory buffer for file a and file b.
  auto RemarkAMaybeBuf = getInputMemoryBuffer(RemarkFileA);
  if (!RemarkAMaybeBuf)
    return RemarkAMaybeBuf.takeError();
  auto RemarkBMaybeBuf = getInputMemoryBuffer(RemarkFileB);
  if (!RemarkBMaybeBuf)
    return RemarkBMaybeBuf.takeError();
  StringRef BufferA = (*RemarkAMaybeBuf)->getBuffer();
  StringRef BufferB = (*RemarkBMaybeBuf)->getBuffer();
  // Create parsers for file a and file b remarks.
  auto MaybeParser1 = createRemarkParserFromMeta(InputFormat, BufferA);
  if (!MaybeParser1)
    return MaybeParser1.takeError();
  auto MaybeParser2 = createRemarkParserFromMeta(InputFormat, BufferB);
  if (!MaybeParser2)
    return MaybeParser2.takeError();
  auto MaybeFilter = getRemarkFilter(
      RemarkNameOpt, RemarkNameOptRE, PassNameOpt, PassNameOptRE, RemarkTypeOpt,
      RemarkFilterArgByOpt, RemarkArgFilterOptRE);
  if (!MaybeFilter)
    return MaybeFilter.takeError();
  auto &Filter = *MaybeFilter;
  // Order the remarks based on their debug location and function name.
  MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> DebugLoc2RemarkA;
  MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> DebugLoc2RemarkB;
  if (auto E = parseRemarkFile(*MaybeParser1, DebugLoc2RemarkA, Filter))
    return E;
  if (auto E = parseRemarkFile(*MaybeParser2, DebugLoc2RemarkB, Filter))
    return E;
  SetVector<DebugLocation> DebugLocs;
  for (const auto &[Loc, _] : DebugLoc2RemarkA)
    DebugLocs.insert(Loc);
  for (const auto &[Loc, _] : DebugLoc2RemarkB)
    DebugLocs.insert(Loc);
  DiffConfigurator DiffConfig(ShowArgDiffOnly, OnlyShowCommonRemarks,
                              ShowOnlyDifferentRemarks, ShowOnlyA, ShowOnlyB,
                              ShowRemarkTypeDiffOnly);
  Diff Diff(Filter, DiffConfig);
  Diff.computeDiff(DebugLocs, DebugLoc2RemarkA, DebugLoc2RemarkB);
  if (auto E = Diff.printDiff(RemarkFileA, RemarkFileB))
    return E;
  return Error::success();
}

static CommandRegistration DiffReg(&DiffSub, createRemarkDiff);