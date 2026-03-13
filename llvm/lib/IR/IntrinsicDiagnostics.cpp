//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicDiagnostics.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Returns the canonical FunctionType for a non-overloaded intrinsic, or null.
static FunctionType *getCanonicalType(StringRef Name, LLVMContext &Ctx) {
  Intrinsic::ID ID = Intrinsic::lookupIntrinsicID(Name);
  if (ID == Intrinsic::not_intrinsic || Intrinsic::isOverloaded(ID))
    return nullptr;
  return Intrinsic::getType(Ctx, ID);
}

void IntrinsicDiagnosticsProvider::querySignatureMismatch(StringRef IntrName,
                                                          FunctionType *DeclFTy,
                                                          FunctionType *CallFTy,
                                                          raw_ostream &OS) {
  OS << ": expected signature: ";
  DeclFTy->print(OS);
  OS << ", got: ";
  CallFTy->print(OS);
}

void IntrinsicDiagnosticsProvider::queryReturnTypeMismatch(StringRef IntrName,
                                                           FunctionType *IFTy,
                                                           raw_ostream &OS) {
  OS << " declared return type is '";
  IFTy->getReturnType()->print(OS);
  OS << "'";
  if (FunctionType *ExpFTy = getCanonicalType(IntrName, IFTy->getContext())) {
    OS << ", expected '";
    ExpFTy->getReturnType()->print(OS);
    OS << "' in canonical signature '";
    ExpFTy->print(OS);
    OS << "'";
  }
}

void IntrinsicDiagnosticsProvider::queryArgTypeMismatch(StringRef IntrName,
                                                        FunctionType *IFTy,
                                                        raw_ostream &OS) {
  OS << " declared signature is '";
  IFTy->print(OS);
  OS << "'";
  if (FunctionType *ExpFTy = getCanonicalType(IntrName, IFTy->getContext())) {
    OS << ", canonical signature is '";
    ExpFTy->print(OS);
    OS << "'";
  }
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
