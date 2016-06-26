//===- SourceCoverageView.h - Code coverage view for source code ----------===//
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

#ifndef LLVM_COV_SOURCECOVERAGEVIEW_H
#define LLVM_COV_SOURCECOVERAGEVIEW_H

#include "CoverageViewOptions.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/Support/MemoryBuffer.h"
#include <vector>

namespace llvm {

class SourceCoverageView;

/// \brief A view that represents a macro or include expansion.
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

/// \brief A view that represents a function instantiation.
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

/// \brief Coverage statistics for a single line.
struct LineCoverageStats {
  uint64_t ExecutionCount;
  unsigned RegionCount;
  bool Mapped;

  LineCoverageStats() : ExecutionCount(0), RegionCount(0), Mapped(false) {}

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

/// \brief A code coverage view of a source file or function.
///
/// A source coverage view and its nested sub-views form a file-oriented
/// representation of code coverage data. This view can be printed out by a
/// renderer which implements the Rendering Interface.
class SourceCoverageView {
  /// A function or file name.
  StringRef SourceName;

  /// A memory buffer backing the source on display.
  const MemoryBuffer &File;

  /// Various options to guide the coverage renderer.
  const CoverageViewOptions &Options;

  /// Complete coverage information about the source on display.
  coverage::CoverageData CoverageInfo;

  /// A container for all expansions (e.g macros) in the source on display.
  std::vector<ExpansionView> ExpansionSubViews;

  /// A container for all instantiations (e.g template functions) in the source
  /// on display.
  std::vector<InstantiationView> InstantiationSubViews;

protected:
  struct LineRef {
    StringRef Line;
    int64_t LineNo;

    LineRef(StringRef Line, int64_t LineNo) : Line(Line), LineNo(LineNo) {}
  };

  using CoverageSegmentArray = ArrayRef<const coverage::CoverageSegment *>;

  /// @name Rendering Interface
  /// @{

  /// \brief Render the source name for the view.
  virtual void renderSourceName(raw_ostream &OS) = 0;

  /// \brief Render the line prefix at the given \p ViewDepth.
  virtual void renderLinePrefix(raw_ostream &OS, unsigned ViewDepth) = 0;

  /// \brief Render a view divider at the given \p ViewDepth.
  virtual void renderViewDivider(raw_ostream &OS, unsigned ViewDepth) = 0;

  /// \brief Render a source line with highlighting.
  virtual void renderLine(raw_ostream &OS, LineRef L,
                          const coverage::CoverageSegment *WrappedSegment,
                          CoverageSegmentArray Segments, unsigned ExpansionCol,
                          unsigned ViewDepth) = 0;

  /// \brief Render the line's execution count column.
  virtual void renderLineCoverageColumn(raw_ostream &OS,
                                        const LineCoverageStats &Line) = 0;

  /// \brief Render the line number column.
  virtual void renderLineNumberColumn(raw_ostream &OS, unsigned LineNo) = 0;

  /// \brief Render all the region's execution counts on a line.
  virtual void renderRegionMarkers(raw_ostream &OS,
                                   CoverageSegmentArray Segments,
                                   unsigned ViewDepth) = 0;

  /// \brief Render the site of an expansion.
  virtual void
  renderExpansionSite(raw_ostream &OS, ExpansionView &ESV, LineRef L,
                      const coverage::CoverageSegment *WrappedSegment,
                      CoverageSegmentArray Segments, unsigned ExpansionCol,
                      unsigned ViewDepth) = 0;

  /// \brief Render an expansion view and any nested views.
  virtual void renderExpansionView(raw_ostream &OS, ExpansionView &ESV,
                                   unsigned ViewDepth) = 0;

  /// \brief Render an instantiation view and any nested views.
  virtual void renderInstantiationView(raw_ostream &OS, InstantiationView &ISV,
                                       unsigned ViewDepth) = 0;

  /// @}

  /// \brief Format a count using engineering notation with 3 significant
  /// digits.
  static std::string formatCount(uint64_t N);

  SourceCoverageView(StringRef SourceName, const MemoryBuffer &File,
                     const CoverageViewOptions &Options,
                     coverage::CoverageData &&CoverageInfo)
      : SourceName(SourceName), File(File), Options(Options),
        CoverageInfo(std::move(CoverageInfo)) {}

public:
  static std::unique_ptr<SourceCoverageView>
  create(StringRef SourceName, const MemoryBuffer &File,
         const CoverageViewOptions &Options,
         coverage::CoverageData &&CoverageInfo);

  virtual ~SourceCoverageView() {}

  StringRef getSourceName() const { return SourceName; }

  const CoverageViewOptions &getOptions() const { return Options; }

  /// \brief Add an expansion subview to this view.
  void addExpansion(const coverage::CounterMappingRegion &Region,
                    std::unique_ptr<SourceCoverageView> View);

  /// \brief Add a function instantiation subview to this view.
  void addInstantiation(StringRef FunctionName, unsigned Line,
                        std::unique_ptr<SourceCoverageView> View);

  /// \brief Print the code coverage information for a specific portion of a
  /// source file to the output stream.
  void print(raw_ostream &OS, bool WholeFile, bool ShowSourceName,
             unsigned ViewDepth = 0);
};

} // namespace llvm

#endif // LLVM_COV_SOURCECOVERAGEVIEW_H
