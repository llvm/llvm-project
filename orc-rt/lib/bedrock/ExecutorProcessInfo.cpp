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
#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt-internal/support/sys/Errno.h"
#include "orc-rt/support/bit.h"

#include <cassert>
#include <cstring>
#include <unistd.h>

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
  auto CPUFeatures = detectCPUFeatures();
  auto Triple = detectTargetTriple();
  auto PageSize = detectPageSize();
  if (!PageSize)
    return PageSize.takeError();
  return ExecutorProcessInfo(std::move(Triple), std::move(*PageSize),
                             std::move(CPUFeatures));
}

std::string ExecutorProcessInfo::formatCPUFeatures(
    const std::vector<std::string_view> &Features) {
  if (Features.empty())
    return {};

  // Every feature is emitted with a '+' prefix, so the separator carries the
  // prefix for all but the first.
  return "+" + join(Features, ",+");
}

std::string ExecutorProcessInfo::detectCPUFeatures() noexcept {
  // Detection involves system calls, so cache the result. Function-local
  // static initialization is thread safe.
  static const std::string Cache = formatCPUFeatures(detectTargetCPUFeatures());
  return Cache;
}

std::string ExecutorProcessInfo::makeTargetTriple(
    std::initializer_list<std::string_view> Components) {
  const std::string_view *First = Components.begin();
  const std::string_view *Last = Components.end();

  // Trailing empty components have to be dropped
  // Empty parts in the middle results in "--"
  // this matches llvm behaviour
  while (Last != First && (Last - 1)->empty())
    --Last;

  return join(First, Last, "-");
}

Expected<size_t> ExecutorProcessInfo::detectPageSize() noexcept {
  long PageSize = sysconf(_SC_PAGESIZE);
  if (PageSize == -1)
    return make_error<StringError>((StringOutputStream()
                                    << "sysconf did not return a page size: "
                                    << sys::strError(errno))
                                       .str());
  if (PageSize <= 0 || !has_single_bit(static_cast<size_t>(PageSize)))
    return make_error<StringError>((StringOutputStream()
                                    << "reported page size " << PageSize
                                    << " is not a power of two")
                                       .str());
  return static_cast<size_t>(PageSize);
}

} // namespace orc_rt
