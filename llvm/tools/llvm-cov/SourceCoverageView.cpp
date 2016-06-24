//===- SourceCoverageView.cpp - Code coverage view for source code --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements rendering for code coverage of source code.
//
//===----------------------------------------------------------------------===//

#include "SourceCoverageView.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;

void SourceCoverageView::renderLine(
    raw_ostream &OS, StringRef Line, int64_t LineNumber,
    const coverage::CoverageSegment *WrappedSegment,
    ArrayRef<const coverage::CoverageSegment *> Segments,
    unsigned ExpansionCol) {
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

  // Show the rest of the line
  colored_ostream(OS, Highlight ? *Highlight : raw_ostream::SAVEDCOLOR,
                  getOptions().Colors && Highlight, /*Bold=*/false, /*BG=*/true)
      << Line.substr(Col - 1, Line.size() - Col + 1);
  OS << "\n";

  if (getOptions().Debug) {
    for (const auto &Range : HighlightedRanges)
      errs() << "Highlighted line " << LineNumber << ", " << Range.first
             << " -> " << Range.second << "\n";
    if (Highlight)
      errs() << "Highlighted line " << LineNumber << ", " << Col << " -> ?\n";
  }
}

void SourceCoverageView::renderIndent(raw_ostream &OS, unsigned Level) {
  for (unsigned I = 0; I < Level; ++I)
    OS << "  |";
}

void SourceCoverageView::renderViewDivider(unsigned Level, unsigned Length,
                                           raw_ostream &OS) {
  assert(Level != 0 && "Cannot render divider at top level");
  renderIndent(OS, Level - 1);
  OS.indent(2);
  for (unsigned I = 0; I < Length; ++I)
    OS << "-";
}

/// Format a count using engineering notation with 3 significant digits.
static std::string formatCount(uint64_t N) {
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

void
SourceCoverageView::renderLineCoverageColumn(raw_ostream &OS,
                                             const LineCoverageStats &Line) {
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

void SourceCoverageView::renderLineNumberColumn(raw_ostream &OS,
                                                unsigned LineNo) {
  SmallString<32> Buffer;
  raw_svector_ostream BufferOS(Buffer);
  BufferOS << LineNo;
  auto Str = BufferOS.str();
  // Trim and align to the right
  Str = Str.substr(0, std::min(Str.size(), (size_t)LineNumberColumnWidth));
  OS.indent(LineNumberColumnWidth - Str.size()) << Str << '|';
}

void SourceCoverageView::renderRegionMarkers(
    raw_ostream &OS, ArrayRef<const coverage::CoverageSegment *> Segments) {
  unsigned PrevColumn = 1;
  for (const auto *S : Segments) {
    if (!S->IsRegionEntry)
      continue;
    // Skip to the new region
    if (S->Col > PrevColumn)
      OS.indent(S->Col - PrevColumn);
    PrevColumn = S->Col + 1;
    std::string C = formatCount(S->Count);
    PrevColumn += C.size();
    OS << '^' << C;
  }
  OS << "\n";

  if (getOptions().Debug)
    for (const auto *S : Segments)
      errs() << "Marker at " << S->Line << ":" << S->Col << " = "
             << formatCount(S->Count) << (S->IsRegionEntry ? "\n" : " (pop)\n");
}

void SourceCoverageView::render(raw_ostream &OS, bool WholeFile,
                                unsigned IndentLevel) {
  // The width of the leading columns
  unsigned CombinedColumnWidth =
      (getOptions().ShowLineStats ? LineCoverageColumnWidth + 1 : 0) +
      (getOptions().ShowLineNumbers ? LineNumberColumnWidth + 1 : 0);
  // The width of the line that is used to divide between the view and the
  // subviews.
  unsigned DividerWidth = CombinedColumnWidth + 4;

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

    // Render the line prefix.
    renderIndent(OS, IndentLevel);
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
    renderLine(OS, *LI, LI.line_number(), WrappedSegment, LineSegments,
               ExpansionColumn);

    // Show the region markers.
    if (getOptions().ShowRegionMarkers &&
        (!getOptions().ShowLineStatsOrRegionMarkers ||
         LineCount.hasMultipleRegions()) &&
        !LineSegments.empty()) {
      renderIndent(OS, IndentLevel);
      OS.indent(CombinedColumnWidth);
      renderRegionMarkers(OS, LineSegments);
    }

    // Show the expansions and instantiations for this line.
    unsigned NestedIndent = IndentLevel + 1;
    bool RenderedSubView = false;
    for (; NextESV != EndESV && NextESV->getLine() == LI.line_number();
         ++NextESV) {
      renderViewDivider(NestedIndent, DividerWidth, OS);
      OS << "\n";
      if (RenderedSubView) {
        // Re-render the current line and highlight the expansion range for
        // this subview.
        ExpansionColumn = NextESV->getStartCol();
        renderIndent(OS, IndentLevel);
        OS.indent(CombinedColumnWidth + (IndentLevel == 0 ? 0 : 1));
        renderLine(OS, *LI, LI.line_number(), WrappedSegment, LineSegments,
                   ExpansionColumn);
        renderViewDivider(NestedIndent, DividerWidth, OS);
        OS << "\n";
      }
      // Render the child subview
      if (getOptions().Debug)
        errs() << "Expansion at line " << NextESV->getLine() << ", "
               << NextESV->getStartCol() << " -> " << NextESV->getEndCol()
               << "\n";
      NextESV->View->render(OS, false, NestedIndent);
      RenderedSubView = true;
    }
    for (; NextISV != EndISV && NextISV->Line == LI.line_number(); ++NextISV) {
      renderViewDivider(NestedIndent, DividerWidth, OS);
      OS << "\n";
      renderIndent(OS, NestedIndent);
      OS << ' ';
      NextISV->View->renderSourceName(OS);
      NextISV->View->render(OS, false, NestedIndent);
      RenderedSubView = true;
    }
    if (RenderedSubView) {
      renderViewDivider(NestedIndent, DividerWidth, OS);
      OS << "\n";
    }
  }
}

void SourceCoverageView::renderSourceName(raw_ostream &OS) {
  getOptions().colored_ostream(OS, raw_ostream::CYAN) << getSourceName()
                                                      << ":\n";
}
