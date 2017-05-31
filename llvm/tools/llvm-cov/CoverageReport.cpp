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

using namespace llvm;
namespace {
/// \brief Helper struct which prints trimmed and aligned columns.
struct Column {
  enum TrimKind { NoTrim, WidthTrim, LeftTrim, RightTrim };

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

  void render(raw_ostream &OS) const;
};

raw_ostream &operator<<(raw_ostream &OS, const Column &Value) {
  Value.render(OS);
  return OS;
}
}

void Column::render(raw_ostream &OS) const {
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
  case LeftTrim:
    OS << "..." << Str.substr(Str.size() - Width + 3);
    break;
  case RightTrim:
    OS << Str.substr(0, Width - 3) << "...";
    break;
  }
}

static Column column(StringRef Str, unsigned Width) {
  return Column(Str, Width);
}

template <typename T>
static Column column(StringRef Str, unsigned Width, const T &Value) {
  return Column(Str, Width).set(Value);
}

// Specify the default column widths.
static size_t FileReportColumns[] = {25, 12, 18, 10, 12, 18, 10, 12, 18, 10};
static size_t FunctionReportColumns[] = {25, 10, 8, 8, 10, 8, 8};

/// \brief  Adjust column widths to fit long file paths and function names.
static void adjustColumnWidths(coverage::CoverageMapping *CM) {
  for (StringRef Filename : CM->getUniqueSourceFiles()) {
    FileReportColumns[0] = std::max(FileReportColumns[0], Filename.size());
    for (const auto &F : CM->getCoveredFunctions(Filename)) {
      FunctionReportColumns[0] =
          std::max(FunctionReportColumns[0], F.Name.size());
    }
  }
}

/// \brief Prints a horizontal divider which spans across the given columns.
template <typename T, size_t N>
static void renderDivider(T (&Columns)[N], raw_ostream &OS) {
  unsigned Length = 0;
  for (unsigned I = 0; I < N; ++I)
    Length += Columns[I];
  for (unsigned I = 0; I < Length; ++I)
    OS << '-';
}

/// \brief Return the color which correponds to the coverage
/// percentage of a certain metric.
template <typename T>
static raw_ostream::Colors determineCoveragePercentageColor(const T &Info) {
  if (Info.isFullyCovered())
    return raw_ostream::GREEN;
  return Info.getPercentCovered() >= 80.0 ? raw_ostream::YELLOW
                                          : raw_ostream::RED;
}

void CoverageReport::render(const FileCoverageSummary &File, raw_ostream &OS) {
  OS << column(File.Name, FileReportColumns[0], Column::NoTrim)
     << format("%*u", FileReportColumns[1],
               (unsigned)File.RegionCoverage.NumRegions);
  Options.colored_ostream(OS, File.RegionCoverage.isFullyCovered()
                                  ? raw_ostream::GREEN
                                  : raw_ostream::RED)
      << format("%*u", FileReportColumns[2], (unsigned)File.RegionCoverage.NotCovered);
  Options.colored_ostream(OS,
                          determineCoveragePercentageColor(File.RegionCoverage))
      << format("%*.2f", FileReportColumns[3] - 1,
                File.RegionCoverage.getPercentCovered()) << '%';
  OS << format("%*u", FileReportColumns[4],
               (unsigned)File.FunctionCoverage.NumFunctions);
  OS << format("%*u", FileReportColumns[5],
               (unsigned)(File.FunctionCoverage.NumFunctions -
                          File.FunctionCoverage.Executed));
  Options.colored_ostream(
      OS, determineCoveragePercentageColor(File.FunctionCoverage))
      << format("%*.2f", FileReportColumns[6] - 1,
                File.FunctionCoverage.getPercentCovered()) << '%';
  OS << format("%*u", FileReportColumns[7],
               (unsigned)File.LineCoverage.NumLines);
  Options.colored_ostream(OS, File.LineCoverage.isFullyCovered()
                                  ? raw_ostream::GREEN
                                  : raw_ostream::RED)
      << format("%*u", FileReportColumns[8],
                (unsigned)File.LineCoverage.NotCovered);
  Options.colored_ostream(OS,
                          determineCoveragePercentageColor(File.LineCoverage))
      << format("%*.2f", FileReportColumns[9] - 1,
                File.LineCoverage.getPercentCovered()) << '%';
  OS << "\n";
}

void CoverageReport::render(const FunctionCoverageSummary &Function,
                            raw_ostream &OS) {
  OS << column(Function.Name, FunctionReportColumns[0], Column::RightTrim)
     << format("%*u", FunctionReportColumns[1],
               (unsigned)Function.RegionCoverage.NumRegions);
  Options.colored_ostream(OS, Function.RegionCoverage.isFullyCovered()
                                  ? raw_ostream::GREEN
                                  : raw_ostream::RED)
      << format("%*u", FunctionReportColumns[2],
                (unsigned)Function.RegionCoverage.NotCovered);
  Options.colored_ostream(
      OS, determineCoveragePercentageColor(Function.RegionCoverage))
      << format("%*.2f", FunctionReportColumns[3] - 1,
                Function.RegionCoverage.getPercentCovered()) << '%';
  OS << format("%*u", FunctionReportColumns[4],
               (unsigned)Function.LineCoverage.NumLines);
  Options.colored_ostream(OS, Function.LineCoverage.isFullyCovered()
                                  ? raw_ostream::GREEN
                                  : raw_ostream::RED)
      << format("%*u", FunctionReportColumns[5],
                (unsigned)Function.LineCoverage.NotCovered);
  Options.colored_ostream(
      OS, determineCoveragePercentageColor(Function.LineCoverage))
      << format("%*.2f", FunctionReportColumns[6] - 1,
                Function.LineCoverage.getPercentCovered()) << '%';
  OS << "\n";
}

void CoverageReport::renderFunctionReports(ArrayRef<StringRef> Files,
                                           raw_ostream &OS) {
  adjustColumnWidths(Coverage.get());
  bool isFirst = true;
  for (StringRef Filename : Files) {
    if (isFirst)
      isFirst = false;
    else
      OS << "\n";
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
    for (const auto &F : Coverage->getCoveredFunctions(Filename)) {
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

void CoverageReport::renderFileReports(raw_ostream &OS) {
  adjustColumnWidths(Coverage.get());
  OS << column("Filename", FileReportColumns[0])
     << column("Regions", FileReportColumns[1], Column::RightAlignment)
     << column("Missed Regions", FileReportColumns[2], Column::RightAlignment)
     << column("Cover", FileReportColumns[3], Column::RightAlignment)
     << column("Functions", FileReportColumns[4], Column::RightAlignment)
     << column("Missed Functions", FileReportColumns[5],
               Column::RightAlignment)
     << column("Executed", FileReportColumns[6], Column::RightAlignment)
     << column("Lines", FileReportColumns[7], Column::RightAlignment)
     << column("Missed Lines", FileReportColumns[8], Column::RightAlignment)
     << column("Cover", FileReportColumns[9], Column::RightAlignment)
     << "\n";
  renderDivider(FileReportColumns, OS);
  OS << "\n";

  FileCoverageSummary Totals("TOTAL");
  for (StringRef Filename : Coverage->getUniqueSourceFiles()) {
    FileCoverageSummary Summary(Filename);
    for (const auto &F : Coverage->getCoveredFunctions(Filename)) {
      FunctionCoverageSummary Function = FunctionCoverageSummary::get(F);
      Summary.addFunction(Function);
      Totals.addFunction(Function);
    }
    render(Summary, OS);
  }
  renderDivider(FileReportColumns, OS);
  OS << "\n";
  render(Totals, OS);
}
