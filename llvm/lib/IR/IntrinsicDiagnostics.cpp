//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicDiagnostics.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void IntrinsicDiagnosticsProvider::querySignatureMismatch(StringRef IntrName,
                                                          FunctionType *DeclFTy,
                                                          FunctionType *CallFTy,
                                                          raw_ostream &OS) {
  OS << ": expected signature: ";
  DeclFTy->print(OS);
  OS << ", got: ";
  CallFTy->print(OS);
}

void IntrinsicDiagnosticsProvider::queryParserMismatch(
    StringRef IntrName, FunctionType *CallFTy, FunctionType *ExpectedFTy,
    raw_ostream &OS) {
  OS << "for '" << IntrName << "': got ";
  CallFTy->print(OS);
  if (ExpectedFTy) {
    OS << ", expected ";
    ExpectedFTy->print(OS);
  }
}
