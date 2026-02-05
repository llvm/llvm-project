//===-- Error.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX

#ifdef LLVM_ON_UNIX
#include <string.h>
#endif // LLVM_ON_UNIX

namespace llvm {
namespace exegesis {

const char ClusteringError::ID = 0;

void ClusteringError::log(raw_ostream &OS) const { OS << Msg; }

std::error_code ClusteringError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

const char SnippetExecutionFailure::ID = 0;

std::error_code SnippetExecutionFailure::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

const char SnippetSegmentationFault::ID = 0;

void SnippetSegmentationFault::log(raw_ostream &OS) const {
  OS << "The snippet encountered a segmentation fault at address "
     << Twine::utohexstr(Address);
}

const char SnippetSignal::ID = 0;

void SnippetSignal::log(raw_ostream &OS) const {
  OS << "snippet crashed while running";
#ifdef LLVM_ON_UNIX
  OS << ": " << strsignal(SignalNumber);
#else
  (void)SignalNumber;
#endif // LLVM_ON_UNIX
}

const char PerfCounterNotFullyEnabled::ID = 0;

std::error_code PerfCounterNotFullyEnabled::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

void PerfCounterNotFullyEnabled::log(raw_ostream &OS) const {
  OS << "The perf counter was not scheduled on the CPU the entire time.";
}

} // namespace exegesis
} // namespace llvm
