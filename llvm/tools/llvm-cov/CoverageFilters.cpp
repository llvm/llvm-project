//===- CoverageFilters.cpp - Function coverage mapping filters ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes provide filtering for function coverage mapping records.
//
//===----------------------------------------------------------------------===//

#include "CoverageFilters.h"
#include "CoverageSummaryInfo.h"
#include "llvm/Support/Regex.h"

using namespace llvm;

bool NameCoverageFilter::matches(const coverage::FunctionRecord &Function) {
  StringRef FuncName = Function.Name;
  return FuncName.find(Name) != StringRef::npos;
}

bool
NameRegexCoverageFilter::matches(const coverage::FunctionRecord &Function) {
  return llvm::Regex(Regex).match(Function.Name);
}

bool RegionCoverageFilter::matches(const coverage::FunctionRecord &Function) {
  return PassesThreshold(FunctionCoverageSummary::get(Function)
                             .RegionCoverage.getPercentCovered());
}

bool LineCoverageFilter::matches(const coverage::FunctionRecord &Function) {
  return PassesThreshold(
      FunctionCoverageSummary::get(Function).LineCoverage.getPercentCovered());
}

void CoverageFilters::push_back(std::unique_ptr<CoverageFilter> Filter) {
  Filters.push_back(std::move(Filter));
}

bool CoverageFilters::matches(const coverage::FunctionRecord &Function) {
  for (const auto &Filter : Filters) {
    if (Filter->matches(Function))
      return true;
  }
  return false;
}

bool
CoverageFiltersMatchAll::matches(const coverage::FunctionRecord &Function) {
  for (const auto &Filter : Filters) {
    if (!Filter->matches(Function))
      return false;
  }
  return true;
}
