//===--------------------- LLDBAssert.cpp ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/LLDBAssert.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace lldb_private;

void lldb_private::lldb_assert(bool expression, const char *expr_text,
                               const char *func, const char *file,
                               unsigned int line) {
  if (LLVM_LIKELY(expression))
    return;

  errs() << format("Assertion failed: (%s), function %s, file %s, line %u\n",
                   expr_text, func, file, line);
  errs() << "backtrace leading to the failure:\n";
  llvm::sys::PrintStackTrace(errs());
  errs() << "please file a bug report against lldb reporting this failure "
            "log, and as many details as possible\n";
  abort();
}
