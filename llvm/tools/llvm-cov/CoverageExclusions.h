//===- CoverageExclusions.h - Source coverage exclusions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

class CoverageExclusions {
public:
  using LineRange = std::pair<unsigned, unsigned>;

  Error parse(StringRef Filename, StringRef Source);

  bool isLineExcluded(StringRef Filename, unsigned Line) const;

  bool isRegionExcluded(const coverage::FunctionRecord &Function,
                        const coverage::CounterMappingRegion &Region) const;

  bool isFunctionExcluded(const coverage::FunctionRecord &Function,
                          StringRef Filename = {}) const;

  coverage::CoverageData apply(coverage::CoverageData Coverage) const;

private:
  ArrayRef<LineRange> getRanges(StringRef Filename) const;

  StringMap<std::vector<LineRange>> ExcludedLineRanges;
};

/// Apply \p Exclusions when present, otherwise return \p Coverage unchanged.
coverage::CoverageData
applyCoverageExclusions(const CoverageExclusions *Exclusions,
                        coverage::CoverageData Coverage);

} // namespace llvm

#endif // LLVM_COV_COVERAGEEXCLUSIONS_H
