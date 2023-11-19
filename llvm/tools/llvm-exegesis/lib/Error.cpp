//===-- Error.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"

#ifdef LLVM_ON_UNIX
#include <string.h>
#endif // LLVM_ON_UNIX

namespace llvm {
namespace exegesis {

char ClusteringError::ID;

void ClusteringError::log(raw_ostream &OS) const { OS << Msg; }

std::error_code ClusteringError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

char SnippetCrash::ID;

void SnippetCrash::log(raw_ostream &OS) const {
  if (SISignalNumber == -1) {
    OS << Msg;
    return;
  }
#ifdef LLVM_ON_UNIX
  OS << "The snippet crashed with signal " << strsignal(SISignalNumber)
     << " at address " << Twine::utohexstr(SIAddress);
#else
  OS << "The snippet crashed with a signal";
#endif // LLVM_ON_UNIX
}

std::error_code SnippetCrash::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

} // namespace exegesis
} // namespace llvm
