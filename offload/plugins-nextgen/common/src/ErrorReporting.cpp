//===- ErrorReporting.cpp - Diagnostic dispatch for the plugin interface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic dispatch for the detection side of error reporting, plus the
// minimal built-in reporting used for diagnostics that come with no handler.
//
//===----------------------------------------------------------------------===//

#include "OffloadAPI.h"

#include "ErrorReporting.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace llvm;
using namespace omp;
using namespace target;
using namespace plugin;

/// Simple default reporting of diagnostics.
static void reportDefault(const ol_diagnostic_t &Diag) {
  auto &OS = errs();
  OS << "OFFLOAD ERROR: " << Diag.Summary << "\n";

  if (*Diag.OffendingStackTrace)
    OS << Diag.OffendingStackTrace;

  if (Diag.Pointer && Diag.Pointer->Allocation) {
    const ol_allocation_info_t &Alloc = *Diag.Pointer->Allocation;
    if (*Alloc.DeallocationTrace)
      OS << "Last deallocation:\n" << Alloc.DeallocationTrace;
    OS << "Last allocation of size " << Alloc.Size << " -> device pointer "
       << Alloc.DevicePtr << ":\n"
       << Alloc.AllocationTrace;
  }

  for (size_t I = 0, E = Diag.NumKernelTraces; I < E; ++I) {
    const ol_kernel_launch_info_t &KTI = Diag.KernelTraces[I];
    OS << "Kernel '" << KTI.Name << "'\n";
    if (*KTI.LaunchTrace)
      OS << KTI.LaunchTrace;
  }
}

void llvm::omp::target::plugin::emitDiagnostic(
    const ol_diagnostic_handler_t *Handler, const ol_diagnostic_t &Diag) {
  // The handler only has to override what it cares about.
  ol_diagnostic_response_t Response = {/*Reported=*/0, /*Abort=*/Diag.IsFatal};
  if (Handler && Handler->Callback)
    Handler->Callback(&Diag, Handler->UserData, &Response);

  if (!Response.Reported)
    reportDefault(Diag);

  if (Response.Abort)
    abort();
}
