//===- CoverageExclusions.h - Source coverage exclusions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares the source exclusion ranges and filtered coverage queries used
/// by llvm-cov reports and exports.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEEXCLUSIONS_H
#define LLVM_COV_COVERAGEEXCLUSIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/Support/Error.h"
#include <utility>
#include <vector>

namespace llvm {

/// Holds source exclusion ranges keyed by the filename in the coverage map.
class CoverageExclusions {
public:
  /// An inclusive range of one-based source line numbers.
  using LineRange = std::pair<unsigned, unsigned>;

  /// Read exclusion markers from the source for a coverage-mapped filename.
  /// Return an error if the block markers are unmatched or overlapping.
  Error parse(StringRef Filename, StringRef Source);

  /// Check whether the region is excluded in its own source file. Code
  /// regions must be fully contained, while other regions use their start line.
  bool isRegionExcluded(const coverage::FunctionRecord &Function,
                        const coverage::CounterMappingRegion &Region) const;

  /// Check whether all the function's code regions in the requested file are
  /// excluded. With no filename, consider code regions in all its files.
  /// A function with no matching code regions is kept.
  bool isFunctionExcluded(const coverage::FunctionRecord &Function,
                          StringRef Filename = {}) const;

  /// Retrieve a file's combined coverage with source exclusions applied.
  coverage::CoverageData
  getCoverageForFile(const coverage::CoverageMapping &Mapping,
                     StringRef Filename) const;

  /// Retrieve coverage for the function's primary source file with exclusions
  /// applied. The function's original counted regions remain available.
  coverage::CoverageData
  getCoverageForFunction(const coverage::CoverageMapping &Mapping,
                         const coverage::FunctionRecord &Function) const;

  /// Retrieve one macro expansion's coverage with exclusions from the
  /// expansion's source file applied.
  coverage::CoverageData
  getCoverageForExpansion(const coverage::CoverageMapping &Mapping,
                          const coverage::ExpansionRecord &Expansion) const;

private:
  /// Return the sorted, merged ranges for the file, if any.
  ArrayRef<LineRange> getRanges(StringRef Filename) const;

  /// Mask the excluded lines and filter records anchored on them.
  coverage::CoverageData apply(coverage::CoverageData Coverage) const;

  StringMap<std::vector<LineRange>> ExcludedLineRanges;
};

} // namespace llvm

#endif // LLVM_COV_COVERAGEEXCLUSIONS_H
