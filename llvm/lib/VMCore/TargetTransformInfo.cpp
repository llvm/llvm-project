//===- llvm/VMCore/TargetTransformInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetTransformInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

/// Default ctor.
///
/// @note This has to exist, because this is a pass, but it should never be
/// used.
TargetTransformInfo::TargetTransformInfo() : ImmutablePass(ID) {
  /// You are seeing this error because your pass required the TTI
  /// using a call to "getAnalysis<TargetTransformInfo>()", and you did
  /// not initialize a machine target which can provide the TTI.
  /// You should use "getAnalysisIfAvailable<TargetTransformInfo>()" instead.
  report_fatal_error("Bad TargetTransformInfo ctor used.  "
                     "Tool did not specify a TargetTransformInfo to use?");
}

INITIALIZE_PASS(TargetTransformInfo, "targettransforminfo",
                "Target Transform Info", false, true)
char TargetTransformInfo::ID = 0;

