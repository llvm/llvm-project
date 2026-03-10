//===-- NVVMIntrinsicDiagnostics.cpp - Detailed NVVM diags ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers a diagnostics provider that appends detailed information to error
// messages for NVVM intrinsic signature mismatches.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicDiagnostics.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static bool isNVVMIntrinsic(StringRef Name) {
  return Name.starts_with("llvm.nvvm.");
}

/// Returns the canonical FunctionType for a non-overloaded intrinsic, or null.
static FunctionType *getCanonicalType(StringRef Name, LLVMContext &Ctx) {
  Intrinsic::ID ID = Intrinsic::lookupIntrinsicID(Name);
  if (ID == Intrinsic::not_intrinsic || Intrinsic::isOverloaded(ID))
    return nullptr;
  SmallVector<Type *, 4> OverloadTys;
  return Intrinsic::getType(Ctx, ID, OverloadTys);
}

namespace {

class NVVMIntrinsicDiagnosticsProvider : public IntrinsicDiagnosticsProvider {
public:
  void getSignatureMismatch(StringRef Name, FunctionType *DeclFTy,
                            FunctionType *CallFTy,
                            raw_ostream &OS) const override {
    if (!isNVVMIntrinsic(Name))
      return;
    if (DeclFTy->getReturnType() != CallFTy->getReturnType()) {
      OS << "\nreturn type mismatch (expected ";
      DeclFTy->getReturnType()->print(OS);
      OS << ", got ";
      CallFTy->getReturnType()->print(OS);
      OS << ")";
    } else if (DeclFTy->getNumParams() != CallFTy->getNumParams()) {
      OS << "\nwrong number of arguments (expected " << DeclFTy->getNumParams()
         << ", got " << CallFTy->getNumParams() << "), expected signature: ";
      DeclFTy->print(OS);
      OS << ", got signature: ";
      CallFTy->print(OS);
    } else {
      for (unsigned I = 0, E = DeclFTy->getNumParams(); I < E; ++I) {
        if (DeclFTy->getParamType(I) != CallFTy->getParamType(I)) {
          OS << "\nargument " << (I + 1) << " type mismatch (expected ";
          DeclFTy->getParamType(I)->print(OS);
          OS << ", got ";
          CallFTy->getParamType(I)->print(OS);
          OS << ")";
          break;
        }
      }
    }
  }

  void getReturnTypeMismatch(StringRef Name, FunctionType *IFTy,
                             raw_ostream &OS) const override {
    if (!isNVVMIntrinsic(Name))
      return;
    OS << "\ndeclared return type is '";
    IFTy->getReturnType()->print(OS);
    OS << "'";
    if (FunctionType *ExpFTy = getCanonicalType(Name, IFTy->getContext())) {
      OS << ", expected '";
      ExpFTy->getReturnType()->print(OS);
      OS << "' in canonical signature '";
      ExpFTy->print(OS);
      OS << "'";
    }
  }

  void getArgTypeMismatch(StringRef Name, FunctionType *IFTy,
                          raw_ostream &OS) const override {
    if (!isNVVMIntrinsic(Name))
      return;
    OS << "\ndeclared signature is '";
    IFTy->print(OS);
    OS << "'";
    if (FunctionType *ExpFTy = getCanonicalType(Name, IFTy->getContext())) {
      OS << ", canonical signature is '";
      ExpFTy->print(OS);
      OS << "'";
    }
  }

  void getParserMismatch(StringRef Name, FunctionType *CallFTy,
                         FunctionType *ExpectedFTy,
                         raw_ostream &OS) const override {
    if (!isNVVMIntrinsic(Name))
      return;
    OS << "\nfor '" << Name << "': got ";
    CallFTy->print(OS);
    if (ExpectedFTy) {
      OS << ", expected ";
      ExpectedFTy->print(OS);
    }
  }
};

NVVMIntrinsicDiagnosticsProvider TheProvider;

struct ProviderRegistrar {
  ProviderRegistrar() {
    IntrinsicDiagnosticsProvider::registerProvider(&TheProvider);
  }
};
static ProviderRegistrar TheRegistrar;

} // anonymous namespace
