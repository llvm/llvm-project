//===- ExecutorProcessInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the
// orc-rt/bedrock/ExecutorProcessInfo.h header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ExecutorProcessInfo.h"
#include "orc-rt-internal/bedrock/sys/CPUFeatures.h"
#include "orc-rt-internal/bedrock/sys/PageSize.h"
#include "orc-rt-internal/bedrock/sys/TargetTriple.h"
#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt/support/bit.h"

#include <cassert>

namespace orc_rt {

ExecutorProcessInfo::ExecutorProcessInfo(std::string Triple, size_t PageSize,
                                         std::string CPUFeatures) noexcept
    : Triple(std::move(Triple)), PageSize(PageSize),
      CPUFeatures(std::move(CPUFeatures)) {
  assert(!this->Triple.empty() && "triple cannot be empty");
  assert(has_single_bit(this->PageSize) && "page-size is not a power of two");
}

/// Create an ExecutorProcessInfo, auto-detecting property values.
Expected<ExecutorProcessInfo> ExecutorProcessInfo::Detect() noexcept {
  auto Features = sys::detectTargetCPUFeatures();
  std::string CPUFeatures;
  if (!Features.empty()) {
    // Every feature is emitted with a '+' prefix, so the separator carries
    // the prefix for all but the first.
    CPUFeatures = "+" + join(Features, ",+");
  }
  auto Triple = sys::detectTargetTriple();
  auto PageSize = sys::detectPageSize();
  if (!PageSize)
    return PageSize.takeError();
  return ExecutorProcessInfo(std::move(Triple), std::move(*PageSize),
                             std::move(CPUFeatures));
}

} // namespace orc_rt
