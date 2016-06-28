//===- SourceCoverageView.cpp - Code coverage view for source code --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file This class implements rendering for code coverage of source code.
///
//===----------------------------------------------------------------------===//

#include "SourceCoverageView.h"
#include "SourceCoverageViewText.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Path.h"

using namespace llvm;

std::string SourceCoverageView::formatCount(uint64_t N) {
  std::string Number = utostr(N);
  int Len = Number.size();
  if (Len <= 3)
    return Number;
  int IntLen = Len % 3 == 0 ? 3 : Len % 3;
  std::string Result(Number.data(), IntLen);
  if (IntLen != 3) {
    Result.push_back('.');
    Result += Number.substr(IntLen, 3 - IntLen);
  }
  Result.push_back(" kMGTPEZY"[(Len - 1) / 3]);
  return Result;
}

void SourceCoverageView::StreamDestructor::operator()(raw_ostream *OS) const {
  if (OS == &outs())
    return;
  delete OS;
}

/// \brief Create a file at ``Dir/ToplevelDir/@Path.Extension``. If
/// \p ToplevelDir is empty, its path component is skipped.
static Expected<SourceCoverageView::OwnedStream>
createFileInDirectory(StringRef Dir, StringRef ToplevelDir, StringRef Path,
                      StringRef Extension) {
  assert(Extension.size() && "The file extension may not be empty");

  SmallString<256> FullPath(Dir);
  if (!ToplevelDir.empty())
    sys::path::append(FullPath, ToplevelDir);

  auto PathBaseDir = sys::path::relative_path(sys::path::parent_path(Path));
  sys::path::append(FullPath, PathBaseDir);

  if (auto E = sys::fs::create_directories(FullPath))
    return errorCodeToError(E);

  auto PathFilename = (sys::path::filename(Path) + "." + Extension).str();
  sys::path::append(FullPath, PathFilename);

  std::error_code E;
  auto OS = SourceCoverageView::OwnedStream(
      new raw_fd_ostream(FullPath, E, sys::fs::F_RW));
  if (E)
    return errorCodeToError(E);
  return std::move(OS);
}

Expected<SourceCoverageView::OwnedStream>
SourceCoverageView::createOutputStream(const CoverageViewOptions &Opts,
                                       StringRef Path, StringRef Extension,
                                       bool InToplevel) {
  if (Opts.ShowOutputDirectory == "")
    return OwnedStream(&outs());

  return createFileInDirectory(Opts.ShowOutputDirectory,
                               InToplevel ? "" : "coverage", Path, Extension);
}

void SourceCoverageView::addExpansion(
    const coverage::CounterMappingRegion &Region,
    std::unique_ptr<SourceCoverageView> View) {
  ExpansionSubViews.emplace_back(Region, std::move(View));
}

void SourceCoverageView::addInstantiation(
    StringRef FunctionName, unsigned Line,
    std::unique_ptr<SourceCoverageView> View) {
  InstantiationSubViews.emplace_back(FunctionName, Line, std::move(View));
}

std::unique_ptr<SourceCoverageView>
SourceCoverageView::create(StringRef SourceName, const MemoryBuffer &File,
                           const CoverageViewOptions &Options,
                           coverage::CoverageData &&CoverageInfo) {
  switch (Options.ShowFormat) {
  case CoverageViewOptions::OutputFormat::Text:
    return llvm::make_unique<SourceCoverageViewText>(SourceName, File, Options,
                                                     std::move(CoverageInfo));
  }
}

void SourceCoverageView::print(raw_ostream &OS, bool WholeFile,
                               bool ShowSourceName, unsigned ViewDepth) {
  if (ShowSourceName)
    renderSourceName(OS);

  // We need the expansions and instantiations sorted so we can go through them
  // while we iterate lines.
  std::sort(ExpansionSubViews.begin(), ExpansionSubViews.end());
  std::sort(InstantiationSubViews.begin(), InstantiationSubViews.end());
  auto NextESV = ExpansionSubViews.begin();
  auto EndESV = ExpansionSubViews.end();
  auto NextISV = InstantiationSubViews.begin();
  auto EndISV = InstantiationSubViews.end();

  // Get the coverage information for the file.
  auto NextSegment = CoverageInfo.begin();
  auto EndSegment = CoverageInfo.end();

  unsigned FirstLine = NextSegment != EndSegment ? NextSegment->Line : 0;
  const coverage::CoverageSegment *WrappedSegment = nullptr;
  SmallVector<const coverage::CoverageSegment *, 8> LineSegments;
  for (line_iterator LI(File, /*SkipBlanks=*/false); !LI.is_at_eof(); ++LI) {
    // If we aren't rendering the whole file, we need to filter out the prologue
    // and epilogue.
    if (!WholeFile) {
      if (NextSegment == EndSegment)
        break;
      else if (LI.line_number() < FirstLine)
        continue;
    }

    // Collect the coverage information relevant to this line.
    if (LineSegments.size())
      WrappedSegment = LineSegments.back();
    LineSegments.clear();
    while (NextSegment != EndSegment && NextSegment->Line == LI.line_number())
      LineSegments.push_back(&*NextSegment++);

    // Calculate a count to be for the line as a whole.
    LineCoverageStats LineCount;
    if (WrappedSegment && WrappedSegment->HasCount)
      LineCount.addRegionCount(WrappedSegment->Count);
    for (const auto *S : LineSegments)
      if (S->HasCount && S->IsRegionEntry)
        LineCount.addRegionStartCount(S->Count);

    renderLinePrefix(OS, ViewDepth);
    if (getOptions().ShowLineStats)
      renderLineCoverageColumn(OS, LineCount);
    if (getOptions().ShowLineNumbers)
      renderLineNumberColumn(OS, LI.line_number());

    // If there are expansion subviews, we want to highlight the first one.
    unsigned ExpansionColumn = 0;
    if (NextESV != EndESV && NextESV->getLine() == LI.line_number() &&
        getOptions().Colors)
      ExpansionColumn = NextESV->getStartCol();

    // Display the source code for the current line.
    renderLine(OS, {*LI, LI.line_number()}, WrappedSegment, LineSegments,
               ExpansionColumn, ViewDepth);

    // Show the region markers.
    if (getOptions().ShowRegionMarkers &&
        (!getOptions().ShowLineStatsOrRegionMarkers ||
         LineCount.hasMultipleRegions()) &&
        !LineSegments.empty()) {
      renderRegionMarkers(OS, LineSegments, ViewDepth);
    }

    // Show the expansions and instantiations for this line.
    bool RenderedSubView = false;
    for (; NextESV != EndESV && NextESV->getLine() == LI.line_number();
         ++NextESV) {
      renderViewDivider(OS, ViewDepth + 1);

      // Re-render the current line and highlight the expansion range for
      // this subview.
      if (RenderedSubView) {
        ExpansionColumn = NextESV->getStartCol();
        renderExpansionSite(
            OS, *NextESV, {*LI, LI.line_number()}, WrappedSegment, LineSegments,
            ExpansionColumn, ViewDepth);
        renderViewDivider(OS, ViewDepth + 1);
      }

      renderExpansionView(OS, *NextESV, ViewDepth + 1);
      RenderedSubView = true;
    }
    for (; NextISV != EndISV && NextISV->Line == LI.line_number(); ++NextISV) {
      renderViewDivider(OS, ViewDepth + 1);
      renderInstantiationView(OS, *NextISV, ViewDepth + 1);
      RenderedSubView = true;
    }
    if (RenderedSubView)
      renderViewDivider(OS, ViewDepth + 1);
  }
}
