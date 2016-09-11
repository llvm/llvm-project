//===- SourceCoverageViewText.cpp - A text-based code coverage view -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements the text-based coverage renderer.
///
//===----------------------------------------------------------------------===//

#include "CoverageReport.h"
#include "SourceCoverageViewText.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

Expected<CoveragePrinter::OwnedStream>
CoveragePrinterText::createViewFile(StringRef Path, bool InToplevel) {
  return createOutputStream(Path, "txt", InToplevel);
}

void CoveragePrinterText::closeViewFile(OwnedStream OS) {
  OS->operator<<('\n');
}

Error CoveragePrinterText::createIndexFile(
    ArrayRef<StringRef> SourceFiles,
    const coverage::CoverageMapping &Coverage) {
  auto OSOrErr = createOutputStream("index", "txt", /*InToplevel=*/true);
  if (Error E = OSOrErr.takeError())
    return E;
  auto OS = std::move(OSOrErr.get());
  raw_ostream &OSRef = *OS.get();

  CoverageReport Report(Opts, Coverage);
  Report.renderFileReports(OSRef);

  return Error::success();
}

namespace {

static const unsigned LineCoverageColumnWidth = 7;
static const unsigned LineNumberColumnWidth = 5;

/// \brief Get the width of the leading columns.
unsigned getCombinedColumnWidth(const CoverageViewOptions &Opts) {
  return (Opts.ShowLineStats ? LineCoverageColumnWidth + 1 : 0) +
         (Opts.ShowLineNumbers ? LineNumberColumnWidth + 1 : 0);
}

/// \brief The width of the line that is used to divide between the view and
/// the subviews.
unsigned getDividerWidth(const CoverageViewOptions &Opts) {
  return getCombinedColumnWidth(Opts) + 4;
}

} // anonymous namespace

void SourceCoverageViewText::renderViewHeader(raw_ostream &) {}

void SourceCoverageViewText::renderViewFooter(raw_ostream &) {}

void SourceCoverageViewText::renderSourceName(raw_ostream &OS, bool WholeFile) {
  std::string ViewInfo = WholeFile ? getVerboseSourceName() : getSourceName();
  getOptions().colored_ostream(OS, raw_ostream::CYAN) << ViewInfo << ":\n";
}

void SourceCoverageViewText::renderLinePrefix(raw_ostream &OS,
                                              unsigned ViewDepth) {
  for (unsigned I = 0; I < ViewDepth; ++I)
    OS << "  |";
}

void SourceCoverageViewText::renderLineSuffix(raw_ostream &, unsigned) {}

void SourceCoverageViewText::renderViewDivider(raw_ostream &OS,
                                               unsigned ViewDepth) {
  assert(ViewDepth != 0 && "Cannot render divider at top level");
  renderLinePrefix(OS, ViewDepth - 1);
  OS.indent(2);
  unsigned Length = getDividerWidth(getOptions());
  for (unsigned I = 0; I < Length; ++I)
    OS << '-';
  OS << '\n';
}

void SourceCoverageViewText::renderLine(
    raw_ostream &OS, LineRef L,
    const coverage::CoverageSegment *WrappedSegment,
    CoverageSegmentArray Segments, unsigned ExpansionCol, unsigned ViewDepth) {
  StringRef Line = L.Line;
  unsigned LineNumber = L.LineNo;

  Optional<raw_ostream::Colors> Highlight;
  SmallVector<std::pair<unsigned, unsigned>, 2> HighlightedRanges;

  // The first segment overlaps from a previous line, so we treat it specially.
  if (WrappedSegment && WrappedSegment->HasCount && WrappedSegment->Count == 0)
    Highlight = raw_ostream::RED;

  // Output each segment of the line, possibly highlighted.
  unsigned Col = 1;
  for (const auto *S : Segments) {
    unsigned End = std::min(S->Col, static_cast<unsigned>(Line.size()) + 1);
    colored_ostream(OS, Highlight ? *Highlight : raw_ostream::SAVEDCOLOR,
                    getOptions().Colors && Highlight, /*Bold=*/false,
                    /*BG=*/true)
        << Line.substr(Col - 1, End - Col);
    if (getOptions().Debug && Highlight)
      HighlightedRanges.push_back(std::make_pair(Col, End));
    Col = End;
    if (Col == ExpansionCol)
      Highlight = raw_ostream::CYAN;
    else if (S->HasCount && S->Count == 0)
      Highlight = raw_ostream::RED;
    else
      Highlight = None;
  }

  // Show the rest of the line.
  colored_ostream(OS, Highlight ? *Highlight : raw_ostream::SAVEDCOLOR,
                  getOptions().Colors && Highlight, /*Bold=*/false, /*BG=*/true)
      << Line.substr(Col - 1, Line.size() - Col + 1);
  OS << '\n';

  if (getOptions().Debug) {
    for (const auto &Range : HighlightedRanges)
      errs() << "Highlighted line " << LineNumber << ", " << Range.first
             << " -> " << Range.second << '\n';
    if (Highlight)
      errs() << "Highlighted line " << LineNumber << ", " << Col << " -> ?\n";
  }
}

void SourceCoverageViewText::renderLineCoverageColumn(
    raw_ostream &OS, const LineCoverageStats &Line) {
  if (!Line.isMapped()) {
    OS.indent(LineCoverageColumnWidth) << '|';
    return;
  }
  std::string C = formatCount(Line.ExecutionCount);
  OS.indent(LineCoverageColumnWidth - C.size());
  colored_ostream(OS, raw_ostream::MAGENTA,
                  Line.hasMultipleRegions() && getOptions().Colors)
      << C;
  OS << '|';
}

void SourceCoverageViewText::renderLineNumberColumn(raw_ostream &OS,
                                                    unsigned LineNo) {
  SmallString<32> Buffer;
  raw_svector_ostream BufferOS(Buffer);
  BufferOS << LineNo;
  auto Str = BufferOS.str();
  // Trim and align to the right.
  Str = Str.substr(0, std::min(Str.size(), (size_t)LineNumberColumnWidth));
  OS.indent(LineNumberColumnWidth - Str.size()) << Str << '|';
}

void SourceCoverageViewText::renderRegionMarkers(
    raw_ostream &OS, CoverageSegmentArray Segments, unsigned ViewDepth) {
  renderLinePrefix(OS, ViewDepth);
  OS.indent(getCombinedColumnWidth(getOptions()));

  unsigned PrevColumn = 1;
  for (const auto *S : Segments) {
    if (!S->IsRegionEntry)
      continue;
    // Skip to the new region.
    if (S->Col > PrevColumn)
      OS.indent(S->Col - PrevColumn);
    PrevColumn = S->Col + 1;
    std::string C = formatCount(S->Count);
    PrevColumn += C.size();
    OS << '^' << C;
  }
  OS << '\n';

  if (getOptions().Debug)
    for (const auto *S : Segments)
      errs() << "Marker at " << S->Line << ":" << S->Col << " = "
             << formatCount(S->Count) << (S->IsRegionEntry ? "\n" : " (pop)\n");
}

void SourceCoverageViewText::renderExpansionSite(
    raw_ostream &OS, LineRef L, const coverage::CoverageSegment *WrappedSegment,
    CoverageSegmentArray Segments, unsigned ExpansionCol, unsigned ViewDepth) {
  renderLinePrefix(OS, ViewDepth);
  OS.indent(getCombinedColumnWidth(getOptions()) + (ViewDepth == 0 ? 0 : 1));
  renderLine(OS, L, WrappedSegment, Segments, ExpansionCol, ViewDepth);
}

void SourceCoverageViewText::renderExpansionView(raw_ostream &OS,
                                                 ExpansionView &ESV,
                                                 unsigned ViewDepth) {
  // Render the child subview.
  if (getOptions().Debug)
    errs() << "Expansion at line " << ESV.getLine() << ", " << ESV.getStartCol()
           << " -> " << ESV.getEndCol() << '\n';
  ESV.View->print(OS, /*WholeFile=*/false, /*ShowSourceName=*/false,
                  ViewDepth + 1);
}

void SourceCoverageViewText::renderInstantiationView(raw_ostream &OS,
                                                     InstantiationView &ISV,
                                                     unsigned ViewDepth) {
  renderLinePrefix(OS, ViewDepth);
  OS << ' ';
  ISV.View->print(OS, /*WholeFile=*/false, /*ShowSourceName=*/true, ViewDepth);
}

void SourceCoverageViewText::renderCellInTitle(raw_ostream &OS,
                                               StringRef CellText) {
  if (getOptions().hasProjectTitle())
    getOptions().colored_ostream(OS, raw_ostream::CYAN)
        << getOptions().ProjectTitle << "\n";

  getOptions().colored_ostream(OS, raw_ostream::CYAN) << CellText << "\n";

  if (getOptions().hasCreatedTime())
    getOptions().colored_ostream(OS, raw_ostream::CYAN)
        << getOptions().CreatedTimeStr << "\n";
}

void SourceCoverageViewText::renderTableHeader(raw_ostream &, unsigned,
                                               unsigned) {}
