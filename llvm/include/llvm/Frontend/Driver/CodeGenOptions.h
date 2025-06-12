//===--- CodeGenOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines frontend codegen options common to clang and flang
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_DRIVER_CODEGENOPTIONS_H
#define LLVM_FRONTEND_DRIVER_CODEGENOPTIONS_H

#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {
class Triple;
class TargetLibraryInfoImpl;
} // namespace llvm

namespace llvm::driver {
// The current supported vector libraries in enum \VectorLibrary are 9(including
// the NoLibrary). Changing the bitcount from 3 to 4 so that more than 8 values
// can be supported. Now the maximum number of vector libraries supported
// increase from 8(2^3) to 16(2^4).
//
// ENUM_CODEGENOPT(VecLib, llvm::driver::VectorLibrary,
// <bitcount>4</bitcount>, llvm::driver::VectorLibrary::NoLibrary) is the
// currently defined in clang/include/clang/Basic/CodeGenOptions.def
// bitcount is the number of bits used to represent the enum value.
//
// IMPORTANT NOTE: When adding a new vector library support, and if count of
// supported vector libraries crosses the current max limit. Please increment
// the bitcount value.

/// Vector library option used with -fveclib=
enum class VectorLibrary {
  NoLibrary,          // Don't use any vector library.
  Accelerate,         // Use the Accelerate framework.
  LIBMVEC,            // GLIBC vector math library.
  MASSV,              // IBM MASS vector library.
  SVML,               // Intel short vector math library.
  SLEEF,              // SLEEF SIMD Library for Evaluating Elementary Functions.
  Darwin_libsystem_m, // Use Darwin's libsystem_m vector functions.
  ArmPL,              // Arm Performance Libraries.
  AMDLIBM             // AMD vector math library.
};

LLVM_ABI TargetLibraryInfoImpl *createTLII(const llvm::Triple &TargetTriple,
                                           VectorLibrary Veclib);

enum ProfileInstrKind {
  ProfileNone,       // Profile instrumentation is turned off.
  ProfileClangInstr, // Clang instrumentation to generate execution counts
                     // to use with PGO.
  ProfileIRInstr,    // IR level PGO instrumentation in LLVM.
  ProfileCSIRInstr,  // IR level PGO context sensitive instrumentation in LLVM.
  ProfileIRSampleColdCov, // IR level sample pgo based cold function coverage
                          // instrumentation in LLVM.
};

// Default filename used for profile generation.
std::string getDefaultProfileGenName();
} // end namespace llvm::driver

#endif
