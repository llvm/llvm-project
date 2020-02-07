//===-- Error.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_ERROR_H
#define LLVM_TOOLS_LLVM_EXEGESIS_ERROR_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace exegesis {

// A class representing failures that happened within llvm-exegesis, they are
// used to report informations to the user.
class Failure : public StringError {
public:
  Failure(const Twine &S) : StringError(S, inconvertibleErrorCode()) {}
};

// A class representing failures that happened during clustering calculations.
class ClusteringError : public ErrorInfo<ClusteringError> {
public:
  static char ID;
  ClusteringError(const Twine &S) : Msg(S.str()) {}

  void log(raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override;

private:
  std::string Msg;
};

// A class representing failures that happened during snippet execution.
// Instead of terminating the program crashes are logged into the output.
class SnippetCrash : public ErrorInfo<SnippetCrash> {
public:
  static char ID;
  SnippetCrash(const Twine &S) : Msg(S.str()) {}

  void log(raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override;

private:
  std::string Msg;
};

} // namespace exegesis
} // namespace llvm

#endif
