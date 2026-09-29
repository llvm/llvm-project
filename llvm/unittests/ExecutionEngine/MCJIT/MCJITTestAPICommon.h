//===- MCJITTestBase.h - Common base class for MCJIT Unit tests  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements functionality shared by both MCJIT C API tests, and
// the C++ API tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_MCJIT_MCJITTESTAPICOMMON_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_MCJIT_MCJITTESTAPICOMMON_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

// Used to skip tests on unsupported architectures and operating systems.
// To skip a test, add this macro at the top of a test-case in a suite that
// inherits from MCJITTestBase. See MCJITTest.cpp for examples.
#define SKIP_UNSUPPORTED_PLATFORM \
  do \
    if (!ArchSupportsMCJIT() || !OSSupportsMCJIT() || !HostCanBeTargeted()) \
      GTEST_SKIP(); \
  while(0)

namespace llvm {

class MCJITTestAPICommon {
protected:
  MCJITTestAPICommon() : HostTripleName(sys::getProcessTriple()) {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // FIXME: It isn't at all clear why this is necesasry, but without it we
    // fail to initialize the AssumptionCacheTracker.
    initializeAssumptionCacheTrackerPass(*PassRegistry::getPassRegistry());

#ifdef _WIN32
    // On Windows, generate ELF objects by specifying "-elf" in triple
    HostTripleName += "-elf";
#endif // _WIN32
    HostTripleName = Triple::normalize(HostTripleName);
    HostTriple = Triple(HostTripleName);
  }

  bool HostCanBeTargeted() {
    std::string Error;
    return TargetRegistry::lookupTarget(HostTriple, Error) != nullptr;
  }

  /// Returns true if the host architecture is known to support MCJIT
  bool ArchSupportsMCJIT() {
    Triple Host(HostTriple);
    // If ARCH is not supported, bail
    if (!is_contained(SupportedArchs, Host.getArch()))
      return false;

    // If ARCH is supported and has no specific sub-arch support
    if (!is_contained(HasSubArchs, Host.getArch()))
      return true;

    // If ARCH has sub-arch support, find it
    SmallVectorImpl<std::string>::const_iterator I = SupportedSubArchs.begin();
    for(; I != SupportedSubArchs.end(); ++I)
      if (Host.getArchName().starts_with(*I))
        return true;

    return false;
  }

  /// Returns true if the host OS is known to support MCJIT
  bool OSSupportsMCJIT() {
    if (find(UnsupportedEnvironments, HostTriple.getEnvironment()) !=
        UnsupportedEnvironments.end())
      return false;

    if (!is_contained(UnsupportedOSs, HostTriple.getOS()))
      return true;

    return false;
  }

  std::string HostTripleName;
  Triple HostTriple;
  SmallVector<Triple::ArchType, 4> SupportedArchs;
  SmallVector<Triple::ArchType, 1> HasSubArchs;
  SmallVector<std::string, 2> SupportedSubArchs; // We need to own the memory
  SmallVector<Triple::OSType, 4> UnsupportedOSs;
  SmallVector<Triple::EnvironmentType, 1> UnsupportedEnvironments;
};

} // namespace llvm

#endif

