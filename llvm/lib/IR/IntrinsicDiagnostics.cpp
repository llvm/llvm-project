//===-- IntrinsicDiagnostics.cpp - Intrinsic diagnostic hooks -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicDiagnostics.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

static SmallVector<IntrinsicDiagnosticsProvider *, 4> &getProviders() {
  static SmallVector<IntrinsicDiagnosticsProvider *, 4> Providers;
  return Providers;
}

void IntrinsicDiagnosticsProvider::registerProvider(
    IntrinsicDiagnosticsProvider *P) {
  getProviders().push_back(P);
}

void IntrinsicDiagnosticsProvider::querySignatureMismatch(StringRef IntrName,
                                                          FunctionType *DeclFTy,
                                                          FunctionType *CallFTy,
                                                          raw_ostream &OS) {
  for (auto *P : getProviders())
    P->getSignatureMismatch(IntrName, DeclFTy, CallFTy, OS);
}

void IntrinsicDiagnosticsProvider::queryReturnTypeMismatch(StringRef IntrName,
                                                           FunctionType *IFTy,
                                                           raw_ostream &OS) {
  for (auto *P : getProviders())
    P->getReturnTypeMismatch(IntrName, IFTy, OS);
}

void IntrinsicDiagnosticsProvider::queryArgTypeMismatch(StringRef IntrName,
                                                        FunctionType *IFTy,
                                                        raw_ostream &OS) {
  for (auto *P : getProviders())
    P->getArgTypeMismatch(IntrName, IFTy, OS);
}

void IntrinsicDiagnosticsProvider::queryParserMismatch(
    StringRef IntrName, FunctionType *CallFTy, FunctionType *ExpectedFTy,
    raw_ostream &OS) {
  for (auto *P : getProviders())
    P->getParserMismatch(IntrName, CallFTy, ExpectedFTy, OS);
}
