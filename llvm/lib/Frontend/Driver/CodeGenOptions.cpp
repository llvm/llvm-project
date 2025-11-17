//===--- CodeGenOptions.cpp - Shared codegen option handling --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/SystemLibraries.h"
#include "llvm/ProfileData/InstrProfCorrelator.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
extern llvm::cl::opt<llvm::InstrProfCorrelator::ProfCorrelatorKind>
    ProfileCorrelate;
} // namespace llvm

namespace llvm::driver {

TargetLibraryInfoImpl *createTLII(const llvm::Triple &TargetTriple,
                                  driver::VectorLibrary Veclib) {
  TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(TargetTriple);

  using VectorLibrary = llvm::driver::VectorLibrary;
  switch (Veclib) {
  case VectorLibrary::Accelerate:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::Accelerate,
                                             TargetTriple);
    break;
  case VectorLibrary::LIBMVEC:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::LIBMVEC,
                                             TargetTriple);
    break;
  case VectorLibrary::MASSV:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::MASSV,
                                             TargetTriple);
    break;
  case VectorLibrary::SVML:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::SVML,
                                             TargetTriple);
    break;
  case VectorLibrary::SLEEF:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::SLEEFGNUABI,
                                             TargetTriple);
    break;
  case VectorLibrary::Darwin_libsystem_m:
    TLII->addVectorizableFunctionsFromVecLib(
        llvm::VectorLibrary::DarwinLibSystemM, TargetTriple);
    break;
  case VectorLibrary::ArmPL:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::ArmPL,
                                             TargetTriple);
    break;
  case VectorLibrary::AMDLIBM:
    TLII->addVectorizableFunctionsFromVecLib(llvm::VectorLibrary::AMDLIBM,
                                             TargetTriple);
    break;
  default:
    break;
  }
  return TLII;
}

std::string getDefaultProfileGenName() {
  return llvm::ProfileCorrelate != InstrProfCorrelator::NONE
             ? "default_%m.proflite"
             : "default_%m.profraw";
}
} // namespace llvm::driver
