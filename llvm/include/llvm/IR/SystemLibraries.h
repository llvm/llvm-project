//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SYSTEMLIBRARIES_H
#define LLVM_IR_SYSTEMLIBRARIES_H

namespace llvm {
/// List of known vector-functions libraries.
///
/// The vector-functions library defines, which functions are vectorizable
/// and with which factor. The library can be specified by either frontend,
/// or a commandline option, and then used by
/// addVectorizableFunctionsFromVecLib for filling up the tables of
/// vectorizable functions.
enum class VectorLibrary {
  NoLibrary,        // Don't use any vector library.
  Accelerate,       // Use Accelerate framework.
  DarwinLibSystemM, // Use Darwin's libsystem_m.
  LIBMVEC,          // GLIBC Vector Math library.
  MASSV,            // IBM MASS vector library.
  SVML,             // Intel short vector math library.
  SLEEFGNUABI,      // SLEEF - SIMD Library for Evaluating Elementary Functions.
  ArmPL,            // Arm Performance Libraries.
  AMDLIBM           // AMD Math Vector library.
};

} // namespace llvm

#endif // LLVM_IR_SYSTEMLIBRARIES_H
