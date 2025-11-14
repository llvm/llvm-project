//===-----------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/SystemLibraries.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

VectorLibrary llvm::ClVectorLibrary;

static cl::opt<VectorLibrary, true> ClVectorLibraryOpt(
    "vector-library", cl::Hidden, cl::desc("Vector functions library"),
    cl::location(llvm::ClVectorLibrary), cl::init(VectorLibrary::NoLibrary),
    cl::values(
        clEnumValN(VectorLibrary::NoLibrary, "none",
                   "No vector functions library"),
        clEnumValN(VectorLibrary::Accelerate, "Accelerate",
                   "Accelerate framework"),
        clEnumValN(VectorLibrary::DarwinLibSystemM, "Darwin_libsystem_m",
                   "Darwin libsystem_m"),
        clEnumValN(VectorLibrary::LIBMVEC, "LIBMVEC",
                   "GLIBC Vector Math library"),
        clEnumValN(VectorLibrary::MASSV, "MASSV", "IBM MASS vector library"),
        clEnumValN(VectorLibrary::SVML, "SVML", "Intel SVML library"),
        clEnumValN(VectorLibrary::SLEEFGNUABI, "sleefgnuabi",
                   "SIMD Library for Evaluating Elementary Functions"),
        clEnumValN(VectorLibrary::ArmPL, "ArmPL", "Arm Performance Libraries"),
        clEnumValN(VectorLibrary::AMDLIBM, "AMDLIBM",
                   "AMD vector math library")));
