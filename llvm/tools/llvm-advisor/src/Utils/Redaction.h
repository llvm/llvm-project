//===------------------- Redaction.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sensitive argument and path redaction for secure logging.
// Removes secrets from stored command lines.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

/// Redact a single string if it contains sensitive keywords.
std::string redactString(StringRef Value);

/// Redact sensitive values from a command line argument list.
SmallVector<std::string, 16> redactCommand(ArrayRef<std::string> Arguments);

} // namespace llvm::advisor
