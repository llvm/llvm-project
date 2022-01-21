//===-- Instrumentation.cpp -----------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Instrumentation.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

using namespace lldb_private;
using namespace lldb_private::instrumentation;

// Whether we're currently across the API boundary.
static thread_local bool g_global_boundary = false;

Instrumenter::Instrumenter(llvm::StringRef pretty_func,
                           std::string &&pretty_args)
    : m_local_boundary(false) {
  if (!g_global_boundary) {
    g_global_boundary = true;
    m_local_boundary = true;
  }
  LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API), "[{0}] {1} ({2})",
           m_local_boundary ? "external" : "internal", pretty_func,
           pretty_args);
}

Instrumenter::~Instrumenter() { UpdateBoundary(); }

void Instrumenter::UpdateBoundary() {
  if (m_local_boundary)
    g_global_boundary = false;
}
