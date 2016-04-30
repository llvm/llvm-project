//===- SourceCoverageView.h - Code coverage view for source code ----------===//
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

#ifndef LLVM_COV_SOURCECOVERAGEVIEW_H
#define LLVM_COV_SOURCECOVERAGEVIEW_H

#include "CoverageViewOptions.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace llvm {

class SourceCoverageView;

/// \brief A view that represents a macro or include expansion
struct ExpansionView {
  coverage::CounterMappingRegion Region;
  std::unique_ptr<SourceCoverageView> View;

  ExpansionView(const coverage::CounterMappingRegion &Region,
                std::unique_ptr<SourceCoverageView> View)
      : Region(Region), View(std::move(View)) {}
  ExpansionView(ExpansionView &&RHS)
      : Region(std::move(RHS.Region)), View(std::move(RHS.View)) {}
  ExpansionView &operator=(ExpansionView &&RHS) {
    Region = std::move(RHS.Region);
    View = std::move(RHS.View);
    return *this;
  }

  unsigned getLine() const { return Region.LineStart; }
  unsigned getStartCol() const { return Region.ColumnStart; }
  unsigned getEndCol() const { return Region.ColumnEnd; }

  friend bool operator<(const ExpansionView &LHS, const ExpansionView &RHS) {
    return LHS.Region.startLoc() < RHS.Region.startLoc();
  }
};

/// \brief A view that represents a function instantiation
struct InstantiationView {
  StringRef FunctionName;
  unsigned Line;
  std::unique_ptr<SourceCoverageView> View;

  InstantiationView(StringRef FunctionName, unsigned Line,
                    std::unique_ptr<SourceCoverageView> View)
      : FunctionName(FunctionName), Line(Line), View(std::move(View)) {}
  InstantiationView(InstantiationView &&RHS)
      : FunctionName(std::move(RHS.FunctionName)), Line(std::move(RHS.Line)),
        View(std::move(RHS.View)) {}
  InstantiationView &operator=(InstantiationView &&RHS) {
    FunctionName = std::move(RHS.FunctionName);
    Line = std::move(RHS.Line);
    View = std::move(RHS.View);
    return *this;
  }

  friend bool operator<(const InstantiationView &LHS,
                        const InstantiationView &RHS) {
    return LHS.Line < RHS.Line;
  }
};

/// \brief A code coverage view of a specific source file.
/// It can have embedded coverage views.
class SourceCoverageView {
private:
  /// \brief Coverage information for a single line.
  struct LineCoverageInfo {
    uint64_t ExecutionCount;
    unsigned RegionCount;
    bool Mapped;

    LineCoverageInfo() : ExecutionCount(0), RegionCount(0), Mapped(false) {}

    bool isMapped() const { return Mapped; }

    bool hasMultipleRegions() const { return RegionCount > 1; }

    void addRegionStartCount(uint64_t Count) {
      // The max of all region starts is the most interesting value.
      addRegionCount(RegionCount ? std::max(ExecutionCount, Count) : Count);
      ++RegionCount;
    }

    void addRegionCount(uint64_t Count) {
      Mapped = true;
      ExecutionCount = Count;
    }
  };

  const MemoryBuffer &File;
  const CoverageViewOptions &Options;
  coverage::CoverageData CoverageInfo;
  std::vector<ExpansionView> ExpansionSubViews;
  std::vector<InstantiationView> InstantiationSubViews;

  /// \brief Render a source line with highlighting.
  void renderLine(raw_ostream &OS, StringRef Line, int64_t LineNumber,
                  const coverage::CoverageSegment *WrappedSegment,
                  ArrayRef<const coverage::CoverageSegment *> Segments,
                  unsigned ExpansionCol);

  void renderIndent(raw_ostream &OS, unsigned Level);

  void renderViewDivider(unsigned Offset, unsigned Length, raw_ostream &OS);

  /// \brief Render the line's execution count column.
  void renderLineCoverageColumn(raw_ostream &OS, const LineCoverageInfo &Line);

  /// \brief Render the line number column.
  void renderLineNumberColumn(raw_ostream &OS, unsigned LineNo);

  /// \brief Render all the region's execution counts on a line.
  void
  renderRegionMarkers(raw_ostream &OS,
                      ArrayRef<const coverage::CoverageSegment *> Segments);

  static const unsigned LineCoverageColumnWidth = 7;
  static const unsigned LineNumberColumnWidth = 5;

public:
  SourceCoverageView(const MemoryBuffer &File,
                     const CoverageViewOptions &Options,
                     coverage::CoverageData &&CoverageInfo)
      : File(File), Options(Options), CoverageInfo(std::move(CoverageInfo)) {}

  const CoverageViewOptions &getOptions() const { return Options; }

  /// \brief Add an expansion subview to this view.
  void addExpansion(const coverage::CounterMappingRegion &Region,
                    std::unique_ptr<SourceCoverageView> View) {
    ExpansionSubViews.emplace_back(Region, std::move(View));
  }

  /// \brief Add a function instantiation subview to this view.
  void addInstantiation(StringRef FunctionName, unsigned Line,
                        std::unique_ptr<SourceCoverageView> View) {
    InstantiationSubViews.emplace_back(FunctionName, Line, std::move(View));
  }

  /// \brief Print the code coverage information for a specific
  /// portion of a source file to the output stream.
  void render(raw_ostream &OS, bool WholeFile, unsigned IndentLevel = 0);
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEVIEW_H
