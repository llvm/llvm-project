//===- SourceCoverageViewHTML.cpp - A html code coverage view -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements the html coverage renderer.
///
//===----------------------------------------------------------------------===//

#include "SourceCoverageViewHTML.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace {

// Return a string with the special characters in \p Str escaped.
std::string escape(StringRef Str, const CoverageViewOptions &Opts) {
  std::string Result;
  unsigned ColNum = 0; // Record the column number.
  for (char C : Str) {
    ++ColNum;
    if (C == '&')
      Result += "&amp;";
    else if (C == '<')
      Result += "&lt;";
    else if (C == '>')
      Result += "&gt;";
    else if (C == '\"')
      Result += "&quot;";
    else if (C == '\n' || C == '\r') {
      Result += C;
      ColNum = 0;
    } else if (C == '\t') {
      // Replace '\t' with TabSize spaces.
      unsigned NumSpaces = Opts.TabSize - (--ColNum % Opts.TabSize);
      for (unsigned I = 0; I < NumSpaces; ++I)
        Result += "&nbsp;";
      ColNum += NumSpaces;
    } else
      Result += C;
  }
  return Result;
}

// Create a \p Name tag around \p Str, and optionally set its \p ClassName.
std::string tag(const std::string &Name, const std::string &Str,
                const std::string &ClassName = "") {
  std::string Tag = "<" + Name;
  if (ClassName != "")
    Tag += " class='" + ClassName + "'";
  return Tag + ">" + Str + "</" + Name + ">";
}

// Create an anchor to \p Link with the label \p Str.
std::string a(const std::string &Link, const std::string &Str,
              const std::string &TargetType = "href") {
  return "<a " + TargetType + "='" + Link + "'>" + Str + "</a>";
}

const char *BeginHeader =
  "<head>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<meta charset='UTF-8'>";

const char *CSSForCoverage =
    R"(.red {
  background-color: #FFD0D0;
}
.cyan {
  background-color: cyan;
}
body {
  font-family: -apple-system, sans-serif;
}
pre {
  margin-top: 0px !important;
  margin-bottom: 0px !important;
}
.source-name-title {
  padding: 5px 10px;
  border-bottom: 1px solid #dbdbdb;
  background-color: #eee;
}
.centered {
  display: table;
  margin-left: auto;
  margin-right: auto;
  border: 1px solid #dbdbdb;
  border-radius: 3px;
}
.expansion-view {
  background-color: rgba(0, 0, 0, 0);
  margin-left: 0px;
  margin-top: 5px;
  margin-right: 5px;
  margin-bottom: 5px;
  border: 1px solid #dbdbdb;
  border-radius: 3px;
}
table {
  border-collapse: collapse;
}
.line-number {
  text-align: right;
  color: #aaa;
}
.covered-line {
  text-align: right;
  color: #0080ff;
}
.uncovered-line {
  text-align: right;
  color: #ff3300;
}
.tooltip {
  position: relative;
  display: inline;
  background-color: #b3e6ff;
  text-decoration: none;
}
.tooltip span.tooltip-content {
  position: absolute;
  width: 100px;
  margin-left: -50px;
  color: #FFFFFF;
  background: #000000;
  height: 30px;
  line-height: 30px;
  text-align: center;
  visibility: hidden;
  border-radius: 6px;
}
.tooltip span.tooltip-content:after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  margin-left: -8px;
  width: 0; height: 0;
  border-top: 8px solid #000000;
  border-right: 8px solid transparent;
  border-left: 8px solid transparent;
}
:hover.tooltip span.tooltip-content {
  visibility: visible;
  opacity: 0.8;
  bottom: 30px;
  left: 50%;
  z-index: 999;
}
th, td {
  vertical-align: top;
  padding: 2px 5px;
  border-collapse: collapse;
  border-right: solid 1px #eee;
  border-left: solid 1px #eee;
}
td:first-child {
  border-left: none;
}
td:last-child {
  border-right: none;
}
)";

const char *EndHeader = "</head>";

const char *BeginCenteredDiv = "<div class='centered'>";

const char *EndCenteredDiv = "</div>";

const char *BeginSourceNameDiv = "<div class='source-name-title'>";

const char *EndSourceNameDiv = "</div>";

const char *BeginCodeTD = "<td class='code'>";

const char *EndCodeTD = "</td>";

const char *BeginPre = "<pre>";

const char *EndPre = "</pre>";

const char *BeginExpansionDiv = "<div class='expansion-view'>";

const char *EndExpansionDiv = "</div>";

const char *BeginTable = "<table>";

const char *EndTable = "</table>";

std::string getPathToStyle(StringRef ViewPath) {
  std::string PathToStyle = "";
  std::string PathSep = sys::path::get_separator();
  unsigned NumSeps = ViewPath.count(PathSep);
  for (unsigned I = 0, E = NumSeps; I < E; ++I)
    PathToStyle += ".." + PathSep;
  return PathToStyle + "style.css";
}

void emitPrelude(raw_ostream &OS, const CoverageViewOptions &Opts,
                 const std::string &PathToStyle = "") {
  OS << "<!doctype html>"
        "<html>"
     << BeginHeader;

  // Link to a stylesheet if one is available. Otherwise, use the default style.
  if (PathToStyle.empty())
    OS << "<style>" << CSSForCoverage << "</style>";
  else
    OS << "<link rel='stylesheet' type='text/css' href='"
       << escape(PathToStyle, Opts) << "'>";

  OS << EndHeader << "<body>" << BeginCenteredDiv;
}

void emitEpilog(raw_ostream &OS) {
  OS << EndCenteredDiv << "</body>"
                          "</html>";
}

} // anonymous namespace

Expected<CoveragePrinter::OwnedStream>
CoveragePrinterHTML::createViewFile(StringRef Path, bool InToplevel) {
  auto OSOrErr = createOutputStream(Path, "html", InToplevel);
  if (!OSOrErr)
    return OSOrErr;

  OwnedStream OS = std::move(OSOrErr.get());

  if (!Opts.hasOutputDirectory()) {
    emitPrelude(*OS.get(), Opts);
  } else {
    std::string ViewPath = getOutputPath(Path, "html", InToplevel);
    emitPrelude(*OS.get(), Opts, getPathToStyle(ViewPath));
  }

  return std::move(OS);
}

void CoveragePrinterHTML::closeViewFile(OwnedStream OS) {
  emitEpilog(*OS.get());
}

Error CoveragePrinterHTML::createIndexFile(ArrayRef<StringRef> SourceFiles) {
  auto OSOrErr = createOutputStream("index", "html", /*InToplevel=*/true);
  if (Error E = OSOrErr.takeError())
    return E;
  auto OS = std::move(OSOrErr.get());
  raw_ostream &OSRef = *OS.get();

  // Emit a table containing links to reports for each file in the covmapping.
  assert(Opts.hasOutputDirectory() && "No output directory for index file");
  emitPrelude(OSRef, Opts, getPathToStyle(""));
  OSRef << BeginSourceNameDiv << "Index" << EndSourceNameDiv;
  OSRef << BeginTable;
  for (StringRef SF : SourceFiles) {
    std::string LinkText = escape(sys::path::relative_path(SF), Opts);
    std::string LinkTarget =
        escape(getOutputPath(SF, "html", /*InToplevel=*/false), Opts);
    OSRef << tag("tr", tag("td", tag("pre", a(LinkTarget, LinkText), "code")));
  }
  OSRef << EndTable;
  emitEpilog(OSRef);

  // Emit the default stylesheet.
  auto CSSOrErr = createOutputStream("style", "css", /*InToplevel=*/true);
  if (Error E = CSSOrErr.takeError())
    return E;

  OwnedStream CSS = std::move(CSSOrErr.get());
  CSS->operator<<(CSSForCoverage);

  return Error::success();
}

void SourceCoverageViewHTML::renderViewHeader(raw_ostream &OS) {
  OS << BeginTable;
}

void SourceCoverageViewHTML::renderViewFooter(raw_ostream &OS) {
  OS << EndTable;
}

void SourceCoverageViewHTML::renderSourceName(raw_ostream &OS) {
  OS << BeginSourceNameDiv << tag("pre", escape(getSourceName(), getOptions()))
     << EndSourceNameDiv;
}

void SourceCoverageViewHTML::renderLinePrefix(raw_ostream &OS, unsigned) {
  OS << "<tr>";
}

void SourceCoverageViewHTML::renderLineSuffix(raw_ostream &OS, unsigned) {
  // If this view has sub-views, renderLine() cannot close the view's cell.
  // Take care of it here, after all sub-views have been rendered.
  if (hasSubViews())
    OS << EndCodeTD;
  OS << "</tr>";
}

void SourceCoverageViewHTML::renderViewDivider(raw_ostream &, unsigned) {
  // The table-based output makes view dividers unnecessary.
}

void SourceCoverageViewHTML::renderLine(
    raw_ostream &OS, LineRef L, const coverage::CoverageSegment *WrappedSegment,
    CoverageSegmentArray Segments, unsigned ExpansionCol, unsigned) {
  StringRef Line = L.Line;
  unsigned LineNo = L.LineNo;

  // Steps for handling text-escaping, highlighting, and tooltip creation:
  //
  // 1. Split the line into N+1 snippets, where N = |Segments|. The first
  //    snippet starts from Col=1 and ends at the start of the first segment.
  //    The last snippet starts at the last mapped column in the line and ends
  //    at the end of the line. Both are required but may be empty.

  SmallVector<std::string, 8> Snippets;

  unsigned LCol = 1;
  auto Snip = [&](unsigned Start, unsigned Len) {
    assert(Start + Len <= Line.size() && "Snippet extends past the EOL");
    Snippets.push_back(Line.substr(Start, Len));
    LCol += Len;
  };

  Snip(LCol - 1, Segments.empty() ? 0 : (Segments.front()->Col - 1));

  for (unsigned I = 1, E = Segments.size(); I < E; ++I) {
    assert(LCol == Segments[I - 1]->Col && "Snippet start position is wrong");
    Snip(LCol - 1, Segments[I]->Col - LCol);
  }

  // |Line| + 1 is needed to avoid underflow when, e.g |Line| = 0 and LCol = 1.
  Snip(LCol - 1, Line.size() + 1 - LCol);
  assert(LCol == Line.size() + 1 && "Final snippet doesn't reach the EOL");

  // 2. Escape all of the snippets.

  for (unsigned I = 0, E = Snippets.size(); I < E; ++I)
    Snippets[I] = escape(Snippets[I], getOptions());

  // 3. Use \p WrappedSegment to set the highlight for snippet 0. Use segment
  //    1 to set the highlight for snippet 2, segment 2 to set the highlight for
  //    snippet 3, and so on.

  Optional<std::string> Color;
  SmallVector<std::pair<unsigned, unsigned>, 4> HighlightedRanges;
  auto Highlight = [&](const std::string &Snippet, unsigned LC, unsigned RC) {
    if (getOptions().Debug) {
      if (!HighlightedRanges.empty() &&
          HighlightedRanges.back().second == LC - 1) {
        HighlightedRanges.back().second = RC;
      } else
        HighlightedRanges.emplace_back(LC, RC);
    }
    return tag("span", Snippet, Color.getValue());
  };

  auto CheckIfUncovered = [](const coverage::CoverageSegment *S) {
    return S && (S->HasCount && S->Count == 0);
  };

  if (CheckIfUncovered(WrappedSegment) ||
      CheckIfUncovered(Segments.empty() ? nullptr : Segments.front())) {
    Color = "red";
    Snippets[0] = Highlight(Snippets[0], 0, Snippets[0].size());
  }

  for (unsigned I = 0, E = Segments.size(); I < E; ++I) {
    const auto *CurSeg = Segments[I];
    if (CurSeg->Col == ExpansionCol)
      Color = "cyan";
    else if (CheckIfUncovered(CurSeg))
      Color = "red";
    else
      Color = None;

    if (Color.hasValue())
      Snippets[I + 1] = Highlight(Snippets[I + 1], CurSeg->Col,
                                  CurSeg->Col + Snippets[I + 1].size());
  }

  if (Color.hasValue() && Segments.empty())
    Snippets.back() = Highlight(Snippets.back(), Snippets[0].size(), 0);

  if (getOptions().Debug) {
    for (const auto &Range : HighlightedRanges) {
      errs() << "Highlighted line " << LineNo << ", " << Range.first + 1
             << " -> ";
      if (Range.second == 0)
        errs() << "?";
      else
        errs() << Range.second + 1;
      errs() << "\n";
    }
  }

  // 4. Snippets[1:N+1] correspond to \p Segments[0:N]: use these to generate
  //    sub-line region count tooltips if needed.

  bool HasMultipleRegions = [&] {
    unsigned RegionCount = 0;
    for (const auto *S : Segments)
      if (S->HasCount && S->IsRegionEntry)
        if (++RegionCount > 1)
          return true;
    return false;
  }();

  if (shouldRenderRegionMarkers(HasMultipleRegions)) {
    for (unsigned I = 0, E = Segments.size(); I < E; ++I) {
      const auto *CurSeg = Segments[I];
      if (!CurSeg->IsRegionEntry || !CurSeg->HasCount)
        continue;

      Snippets[I + 1] =
          tag("div", Snippets[I + 1] + tag("span", formatCount(CurSeg->Count),
                                           "tooltip-content"),
              "tooltip");
    }
  }

  OS << BeginCodeTD;
  OS << BeginPre;
  for (const auto &Snippet : Snippets)
    OS << Snippet;
  OS << EndPre;

  // If there are no sub-views left to attach to this cell, end the cell.
  // Otherwise, end it after the sub-views are rendered (renderLineSuffix()).
  if (!hasSubViews())
    OS << EndCodeTD;
}

void SourceCoverageViewHTML::renderLineCoverageColumn(
    raw_ostream &OS, const LineCoverageStats &Line) {
  std::string Count = "";
  if (Line.isMapped())
    Count = tag("pre", formatCount(Line.ExecutionCount));
  std::string CoverageClass =
      (Line.ExecutionCount > 0) ? "covered-line" : "uncovered-line";
  OS << tag("td", Count, CoverageClass);
}

void SourceCoverageViewHTML::renderLineNumberColumn(raw_ostream &OS,
                                                    unsigned LineNo) {
  std::string LineNoStr = utostr(uint64_t(LineNo));
  OS << tag("td", a("L" + LineNoStr, tag("pre", LineNoStr), "name"),
            "line-number");
}

void SourceCoverageViewHTML::renderRegionMarkers(raw_ostream &,
                                                 CoverageSegmentArray,
                                                 unsigned) {
  // Region markers are rendered in-line using tooltips.
}

void SourceCoverageViewHTML::renderExpansionSite(
    raw_ostream &OS, LineRef L, const coverage::CoverageSegment *WrappedSegment,
    CoverageSegmentArray Segments, unsigned ExpansionCol, unsigned ViewDepth) {
  // Render the line containing the expansion site. No extra formatting needed.
  renderLine(OS, L, WrappedSegment, Segments, ExpansionCol, ViewDepth);
}

void SourceCoverageViewHTML::renderExpansionView(raw_ostream &OS,
                                                 ExpansionView &ESV,
                                                 unsigned ViewDepth) {
  OS << BeginExpansionDiv;
  ESV.View->print(OS, /*WholeFile=*/false, /*ShowSourceName=*/false,
                  ViewDepth + 1);
  OS << EndExpansionDiv;
}

void SourceCoverageViewHTML::renderInstantiationView(raw_ostream &OS,
                                                     InstantiationView &ISV,
                                                     unsigned ViewDepth) {
  OS << BeginExpansionDiv;
  ISV.View->print(OS, /*WholeFile=*/false, /*ShowSourceName=*/true, ViewDepth);
  OS << EndExpansionDiv;
}
