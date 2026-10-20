//===-------- DiagnosticPrinter.h --------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DIAGNOSTIC_PRINTER_H
#define OMPTARGET_DIAGNOSTIC_PRINTER_H

#include "OffloadAPI.h"

namespace llvm::omp::target {

/// Return the handler that prints the diagnostic error messages.
const ol_diagnostic_handler_t *getDiagnosticPrinter();

} // namespace llvm::omp::target

#endif // OMPTARGET_DIAGNOSTIC_PRINTER_H
