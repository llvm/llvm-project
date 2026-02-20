//===---------------- CoverageProcessor.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoverageProcessor.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/tools/llvm-cov/CoverageExporterJson.h"
#include "llvm/tools/llvm-cov/CoverageFilters.h"
#include "llvm/tools/llvm-cov/CoverageViewOptions.h"
#include <vector>

using namespace llvm;

namespace llvm::advisor {

static CoverageViewOptions buildCoverageViewOptions() {
  CoverageViewOptions Opts;
  Opts.Debug = false;
  Opts.Colors = false;
  Opts.ShowLineNumbers = true;
  Opts.ShowLineStats = true;
  Opts.ShowRegionMarkers = false;
  Opts.ShowMCDC = true;
  Opts.ShowBranchCounts = true;
  Opts.ShowBranchPercents = true;
  Opts.ShowExpandedRegions = true;
  Opts.ShowFunctionInstantiations = true;
  Opts.UnifyFunctionInstantiations = true;
  Opts.ShowFullFilenames = true;
  Opts.ShowBranchSummary = true;
  Opts.ShowMCDCSummary = true;
  Opts.ShowRegionSummary = true;
  Opts.ShowInstantiationSummary = true;
  Opts.ShowDirectoryCoverage = false;
  Opts.ExportSummaryOnly = false;
  Opts.SkipExpansions = false;
  Opts.SkipFunctions = false;
  Opts.SkipBranches = false;
  Opts.BinaryCounters = false;
  Opts.Format = CoverageViewOptions::OutputFormat::Text;
  Opts.ShowBranches = CoverageViewOptions::BranchOutputType::Count;
  Opts.TabSize = 2;
  Opts.NumThreads = 0;
  return Opts;
}

Error CoverageProcessor::mergeRawProfile(StringRef RawProfile,
                                         StringRef IndexedProfile) {
  auto &FS = *vfs::getRealFileSystem();
  auto ReaderOrErr = InstrProfReader::create(RawProfile, FS);
  if (!ReaderOrErr)
    return ReaderOrErr.takeError();

  auto Reader = std::move(*ReaderOrErr);
  InstrProfWriter Writer(/*Sparse=*/false);
  if (Error E = Writer.mergeProfileKind(Reader->getProfileKind()))
    return E;

  std::vector<std::string> Warnings;
  auto Warn = [&](Error E) {
    Warnings.push_back(toString(std::move(E)));
  };

  for (auto &Record : *Reader) {
    Writer.addRecord(std::move(Record), /*Weight=*/1, [&](Error E) {
      Warn(std::move(E));
    });
  }

  if (Reader->hasError())
    return Reader->takeError();

  std::error_code EC;
  raw_fd_ostream OS(IndexedProfile, EC, sys::fs::OF_None);
  if (EC)
    return createStringError(EC, "failed to open %s", IndexedProfile.str().c_str());

  if (Error E = Writer.write(OS))
    return E;

  if (!Warnings.empty()) {
    for (const auto &Message : Warnings)
      errs() << "warning: " << Message << "\n";
  }
  return Error::success();
}

Error CoverageProcessor::exportCoverageReport(StringRef InstrumentedBinary,
                                              StringRef IndexedProfile,
                                              StringRef ReportPath,
                                              StringRef CompilationDir) {
  auto &FS = *vfs::getRealFileSystem();
  auto CoverageOrErr = coverage::CoverageMapping::load(
      ArrayRef<StringRef>{InstrumentedBinary}, IndexedProfile, FS,
      /*Arches=*/{}, CompilationDir,
      /*BIDFetcher=*/nullptr, /*CheckBinaryIDs=*/false);
  if (!CoverageOrErr)
    return CoverageOrErr.takeError();

  auto Coverage = std::move(*CoverageOrErr);
  CoverageViewOptions ViewOpts = buildCoverageViewOptions();

  std::error_code EC;
  raw_fd_ostream OS(ReportPath, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "failed to open %s", ReportPath.str().c_str());

  CoverageFiltersMatchAll IgnoreFilters;
CoverageExporterJson Exporter(*Coverage, ViewOpts, OS);
Exporter.renderRoot(IgnoreFilters);
return Error::success();
}

Error CoverageProcessor::summarizeProfile(StringRef IndexedProfile,
                                          StringRef TextSummaryPath,
                                          StringRef JsonSummaryPath) {
  auto &FS = *vfs::getRealFileSystem();
  auto ReaderOrErr = InstrProfReader::create(IndexedProfile, FS);
  if (!ReaderOrErr)
    return ReaderOrErr.takeError();

  auto Reader = std::move(*ReaderOrErr);

  std::error_code EC;
  raw_fd_ostream TextOS(TextSummaryPath, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "failed to open %s",
                             TextSummaryPath.str().c_str());

  TextOS << "PGO Profile Data Summary\n";
  TextOS << "========================\n";
  TextOS << "Profile: " << IndexedProfile << "\n\n";

  json::Array Functions;
  uint64_t TotalFunctions = 0;
  uint64_t TotalCounts = 0;

  for (auto &Record : *Reader) {
    ++TotalFunctions;
    TextOS << "Function: " << Record.Name << " (hash: 0x"
           << format_hex(Record.Hash, 10) << ")\n";
    json::Array CountsJson;
    for (uint64_t Count : Record.Counts) {
      CountsJson.push_back(static_cast<int64_t>(Count));
      TotalCounts += Count;
      TextOS << "  Count: " << Count << "\n";
    }
    TextOS << "\n";
    Functions.push_back(json::Object{
        {"name", Record.Name},
        {"hash", static_cast<int64_t>(Record.Hash)},
        {"counts", std::move(CountsJson)},
    });
  }

  std::error_code JEC;
  raw_fd_ostream JsonOS(JsonSummaryPath, JEC, sys::fs::OF_Text);
  if (JEC)
    return createStringError(JEC, "failed to open %s",
                             JsonSummaryPath.str().c_str());

  json::Object Root{
      {"num-functions", static_cast<int64_t>(TotalFunctions)},
      {"total-count", static_cast<int64_t>(TotalCounts)},
      {"functions", std::move(Functions)},
  };
  JsonOS << formatv("{0:2}\n", json::Value(std::move(Root)));
  return Error::success();
}

} // namespace llvm::advisor
