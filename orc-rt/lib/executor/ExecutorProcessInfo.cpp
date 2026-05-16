//===- ExecutorProcessInfo.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/ExecutorProcessInfo.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ExecutorProcessInfo.h"
#include "orc-rt/Math.h"

#include <cassert>
#include <cstring>
#include <unistd.h>

namespace orc_rt {

ExecutorProcessInfo::ExecutorProcessInfo(std::string Triple,
                                         size_t PageSize) noexcept
    : Triple(std::move(Triple)), PageSize(PageSize) {
  assert(!this->Triple.empty() && "triple cannot be empty");
  assert(isPowerOf2(this->PageSize) && "page-size is not a power of two");
}

/// Create an ExecutorProcessInfo, auto-detecting property values.
Expected<ExecutorProcessInfo> ExecutorProcessInfo::Detect() noexcept {
  auto Triple = detectTargetTriple();
  auto PageSize = detectPageSize();
  if (!PageSize)
    return PageSize.takeError();
  return ExecutorProcessInfo(std::move(Triple), std::move(*PageSize));
}

std::string ExecutorProcessInfo::detectTargetTriple() noexcept {
  std::string Triple;

// Arch
#if defined(__x86_64__) || defined(_M_X64)
  Triple += "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  Triple += "aarch64";
#else
#error "Unsupported architecture"
#endif

  // Vendor
#if defined(__APPLE__)
  Triple += "-apple";
#else
  Triple += "-unknown";
#endif

  // OS
#if defined(__APPLE__)
  Triple += "-darwin";
#elif defined(__linux__)
  Triple += "-linux";
#else
#error "Unsupported OS"
#endif

  return Triple;
}

Expected<size_t> ExecutorProcessInfo::detectPageSize() noexcept {
  long PageSize = sysconf(_SC_PAGESIZE);
  if (PageSize == -1)
    return make_error<StringError>(strerror(errno));
  if (!isPowerOf2(PageSize))
    return make_error<StringError>("reported page size " +
                                   std::to_string(PageSize) +
                                   " is not a power of two");
  return static_cast<size_t>(PageSize);
}

} // namespace orc_rt
