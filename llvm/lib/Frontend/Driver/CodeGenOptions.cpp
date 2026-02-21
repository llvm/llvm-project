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

llvm::VectorLibrary
convertDriverVectorLibraryToVectorLibrary(llvm::driver::VectorLibrary VecLib) {
  switch (VecLib) {
  case llvm::driver::VectorLibrary::NoLibrary:
    return llvm::VectorLibrary::NoLibrary;
  case llvm::driver::VectorLibrary::Accelerate:
    return llvm::VectorLibrary::Accelerate;
  case llvm::driver::VectorLibrary::Darwin_libsystem_m:
    return llvm::VectorLibrary::DarwinLibSystemM;
  case llvm::driver::VectorLibrary::LIBMVEC:
    return llvm::VectorLibrary::LIBMVEC;
  case llvm::driver::VectorLibrary::MASSV:
    return llvm::VectorLibrary::MASSV;
  case llvm::driver::VectorLibrary::SVML:
    return llvm::VectorLibrary::SVML;
  case llvm::driver::VectorLibrary::SLEEF:
    return llvm::VectorLibrary::SLEEFGNUABI;
  case llvm::driver::VectorLibrary::ArmPL:
    return llvm::VectorLibrary::ArmPL;
  case llvm::driver::VectorLibrary::AMDLIBM:
    return llvm::VectorLibrary::AMDLIBM;
  }
  llvm_unreachable("Unexpected driver::VectorLibrary");
}

TargetLibraryInfoImpl *createTLII(const llvm::Triple &TargetTriple,
                                  driver::VectorLibrary Veclib) {
  return new TargetLibraryInfoImpl(
      TargetTriple, convertDriverVectorLibraryToVectorLibrary(Veclib));
}

std::string getDefaultProfileGenName() {
  return llvm::ProfileCorrelate != InstrProfCorrelator::NONE
             ? "default_%m.proflite"
             : "default_%m.profraw";
}
} // namespace llvm::driver
