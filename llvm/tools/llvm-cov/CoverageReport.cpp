//===- CoverageReport.cpp - Code coverage report -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements rendering of a code coverage report.
//
//===----------------------------------------------------------------------===//

#include "CoverageReport.h"
#include "RenderingSupport.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include <numeric>

using namespace llvm;

namespace {

/// \brief Helper struct which prints trimmed and aligned columns.
struct Column {
  enum TrimKind { NoTrim, WidthTrim, RightTrim };

  enum AlignmentKind { LeftAlignment, RightAlignment };

  StringRef Str;
  unsigned Width;
  TrimKind Trim;
  AlignmentKind Alignment;

  Column(StringRef Str, unsigned Width)
      : Str(Str), Width(Width), Trim(WidthTrim), Alignment(LeftAlignment) {}

  Column &set(TrimKind Value) {
    Trim = Value;
    return *this;
  }

  Column &set(AlignmentKind Value) {
    Alignment = Value;
    return *this;
  }

  void render(raw_ostream &OS) const {
    if (Str.size() <= Width) {
      if (Alignment == RightAlignment) {
        OS.indent(Width - Str.size());
        OS << Str;
        return;
      }
      OS << Str;
      OS.indent(Width - Str.size());
      return;
    }

    switch (Trim) {
    case NoTrim:
      OS << Str;
      break;
    case WidthTrim:
      OS << Str.substr(0, Width);
      break;
    case RightTrim:
      OS << Str.substr(0, Width - 3) << "...";
      break;
    }
  }
};

raw_ostream &operator<<(raw_ostream &OS, const Column &Value) {
  Value.render(OS);
  return OS;
}

Column column(StringRef Str, unsigned Width) { return Column(Str, Width); }

template <typename T>
Column column(StringRef Str, unsigned Width, const T &Value) {
  return Column(Str, Width).set(Value);
}

// Specify the default column widths.
size_t FileReportColumns[] = {25, 12, 18, 10, 12, 18, 10, 12, 18, 10};
size_t FunctionReportColumns[] = {25, 10, 8, 8, 10, 8, 8};

/// \brief Adjust column widths to fit long file paths and function names.
void adjustColumnWidths(ArrayRef<StringRef> Files,
                        ArrayRef<StringRef> Functions) {
  for (StringRef Filename : Files)
    FileReportColumns[0] = std::max(FileReportColumns[0], Filename.size());
  for (StringRef Funcname : Functions)
    FunctionReportColumns[0] =
        std::max(FunctionReportColumns[0], Funcname.size());
}

/// \brief Prints a horizontal divider long enough to cover the given column
/// widths.
void renderDivider(ArrayRef<size_t> ColumnWidths, raw_ostream &OS) {
  size_t Length = std::accumulate(ColumnWidths.begin(), ColumnWidths.end(), 0);
  for (size_t I = 0; I < Length; ++I)
    OS << '-';
}

/// \brief Return the color which correponds to the coverage percentage of a
/// certain metric.
template <typename T>
raw_ostream::Colors determineCoveragePercentageColor(const T &Info) {
  if (Info.isFullyCovered())
    return raw_ostream::GREEN;
  return Info.getPercentCovered() >= 80.0 ? raw_ostream::YELLOW
                                          : raw_ostream::RED;
}

/// \brief Determine the length of the longest common prefix of the strings in
/// \p Strings.
unsigned getLongestCommonPrefixLen(ArrayRef<StringRef> Strings) {
  unsigned LCP = Strings[0].size();
  for (unsigned I = 1, E = Strings.size(); LCP > 0 && I < E; ++I) {
    auto Mismatch =
        std::mismatch(Strings[0].begin(), Strings[0].end(), Strings[I].begin())
            .first;
    LCP = std::min(LCP, (unsigned)std::distance(Strings[0].begin(), Mismatch));
  }
  return LCP;
}

} // end anonymous namespace

namespace llvm {

void CoverageReport::render(const FileCoverageSummary &File,
                            raw_ostream &OS) const {
  auto FileCoverageColor =
      determineCoveragePercentageColor(File.RegionCoverage);
  auto FuncCoverageColor =
      determineCoveragePercentageColor(File.FunctionCoverage);
  auto LineCoverageColor = determineCoveragePercentageColor(File.LineCoverage);
  SmallString<256> FileName = File.Name;
  sys::path::remove_dots(FileName, /*remove_dot_dots=*/true);
  sys::path::native(FileName);
  OS << column(FileName, FileReportColumns[0], Column::NoTrim)
     << format("%*u", FileReportColumns[1],
               (unsigned)File.RegionCoverage.NumRegions);
  Options.colored_ostream(OS, FileCoverageColor) << format(
      "%*u", FileReportColumns[2], (unsigned)File.RegionCoverage.NotCovered);
  Options.colored_ostream(OS, FileCoverageColor)
      << format("%*.2f", FileReportColumns[3] - 1,
                File.RegionCoverage.getPercentCovered())
      << '%';
  OS << format("%*u", FileReportColumns[4],
               (unsigned)File.FunctionCoverage.NumFunctions);
  OS << format("%*u", FileReportColumns[5],
               (unsigned)(File.FunctionCoverage.NumFunctions -
                          File.FunctionCoverage.Executed));
  Options.colored_ostream(OS, FuncCoverageColor)
      << format("%*.2f", FileReportColumns[6] - 1,
                File.FunctionCoverage.getPercentCovered())
      << '%';
  OS << format("%*u", FileReportColumns[7],
               (unsigned)File.LineCoverage.NumLines);
  Options.colored_ostream(OS, LineCoverageColor) << format(
      "%*u", FileReportColumns[8], (unsigned)File.LineCoverage.NotCovered);
  Options.colored_ostream(OS, LineCoverageColor)
      << format("%*.2f", FileReportColumns[9] - 1,
                File.LineCoverage.getPercentCovered())
      << '%';
  OS << "\n";
}

void CoverageReport::render(const FunctionCoverageSummary &Function,
                            raw_ostream &OS) const {
  auto FuncCoverageColor =
      determineCoveragePercentageColor(Function.RegionCoverage);
  auto LineCoverageColor =
      determineCoveragePercentageColor(Function.LineCoverage);
  OS << column(Function.Name, FunctionReportColumns[0], Column::RightTrim)
     << format("%*u", FunctionReportColumns[1],
               (unsigned)Function.RegionCoverage.NumRegions);
  Options.colored_ostream(OS, FuncCoverageColor)
      << format("%*u", FunctionReportColumns[2],
                (unsigned)Function.RegionCoverage.NotCovered);
  Options.colored_ostream(
      OS, determineCoveragePercentageColor(Function.RegionCoverage))
      << format("%*.2f", FunctionReportColumns[3] - 1,
                Function.RegionCoverage.getPercentCovered())
      << '%';
  OS << format("%*u", FunctionReportColumns[4],
               (unsigned)Function.LineCoverage.NumLines);
  Options.colored_ostream(OS, LineCoverageColor)
      << format("%*u", FunctionReportColumns[5],
                (unsigned)Function.LineCoverage.NotCovered);
  Options.colored_ostream(
      OS, determineCoveragePercentageColor(Function.LineCoverage))
      << format("%*.2f", FunctionReportColumns[6] - 1,
                Function.LineCoverage.getPercentCovered())
      << '%';
  OS << "\n";
}

void CoverageReport::renderFunctionReports(ArrayRef<StringRef> Files,
                                           raw_ostream &OS) {
  bool isFirst = true;
  for (StringRef Filename : Files) {
    auto Functions = Coverage.getCoveredFunctions(Filename);

    if (isFirst)
      isFirst = false;
    else
      OS << "\n";

    std::vector<StringRef> Funcnames;
    for (const auto &F : Functions)
      Funcnames.emplace_back(F.Name);
    adjustColumnWidths({}, Funcnames);

    OS << "File '" << Filename << "':\n";
    OS << column("Name", FunctionReportColumns[0])
       << column("Regions", FunctionReportColumns[1], Column::RightAlignment)
       << column("Miss", FunctionReportColumns[2], Column::RightAlignment)
       << column("Cover", FunctionReportColumns[3], Column::RightAlignment)
       << column("Lines", FunctionReportColumns[4], Column::RightAlignment)
       << column("Miss", FunctionReportColumns[5], Column::RightAlignment)
       << column("Cover", FunctionReportColumns[6], Column::RightAlignment);
    OS << "\n";
    renderDivider(FunctionReportColumns, OS);
    OS << "\n";
    FunctionCoverageSummary Totals("TOTAL");
    for (const auto &F : Functions) {
      FunctionCoverageSummary Function = FunctionCoverageSummary::get(F);
      ++Totals.ExecutionCount;
      Totals.RegionCoverage += Function.RegionCoverage;
      Totals.LineCoverage += Function.LineCoverage;
      render(Function, OS);
    }
    if (Totals.ExecutionCount) {
      renderDivider(FunctionReportColumns, OS);
      OS << "\n";
      render(Totals, OS);
    }
  }
}

std::vector<FileCoverageSummary>
CoverageReport::prepareFileReports(FileCoverageSummary &Totals,
                                   ArrayRef<StringRef> Files) const {
  std::vector<FileCoverageSummary> FileReports;
  unsigned LCP = 0;
  if (Files.size() > 1)
    LCP = getLongestCommonPrefixLen(Files);

  for (StringRef Filename : Files) {
    FileCoverageSummary Summary(Filename.drop_front(LCP));
    for (const auto &F : Coverage.getCoveredFunctions(Filename)) {
      FunctionCoverageSummary Function = FunctionCoverageSummary::get(F);
      Summary.addFunction(Function);
      Totals.addFunction(Function);
    }
    FileReports.push_back(Summary);
  }

  return FileReports;
}

void CoverageReport::renderFileReports(raw_ostream &OS) const {
  std::vector<StringRef> UniqueSourceFiles = Coverage.getUniqueSourceFiles();
  renderFileReports(OS, UniqueSourceFiles);
}

void CoverageReport::renderFileReports(raw_ostream &OS,
                                       ArrayRef<StringRef> Files) const {
  FileCoverageSummary Totals("TOTAL");
  auto FileReports = prepareFileReports(Totals, Files);

  std::vector<StringRef> Filenames;
  for (const FileCoverageSummary &FCS : FileReports)
    Filenames.emplace_back(FCS.Name);
  adjustColumnWidths(Filenames, {});

  OS << column("Filename", FileReportColumns[0])
     << column("Regions", FileReportColumns[1], Column::RightAlignment)
     << column("Missed Regions", FileReportColumns[2], Column::RightAlignment)
     << column("Cover", FileReportColumns[3], Column::RightAlignment)
     << column("Functions", FileReportColumns[4], Column::RightAlignment)
     << column("Missed Functions", FileReportColumns[5], Column::RightAlignment)
     << column("Executed", FileReportColumns[6], Column::RightAlignment)
     << column("Lines", FileReportColumns[7], Column::RightAlignment)
     << column("Missed Lines", FileReportColumns[8], Column::RightAlignment)
     << column("Cover", FileReportColumns[9], Column::RightAlignment) << "\n";
  renderDivider(FileReportColumns, OS);
  OS << "\n";

  for (const FileCoverageSummary &FCS : FileReports)
    render(FCS, OS);

  renderDivider(FileReportColumns, OS);
  OS << "\n";
  render(Totals, OS);
}

} // end namespace llvm
