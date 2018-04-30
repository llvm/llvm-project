//===- MCJITTestBase.h - Common base class for MCJIT Unit tests  ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

// Used to skip tests on unsupported architectures and operating systems.
// To skip a test, add this macro at the top of a test-case in a suite that
// inherits from MCJITTestBase. See MCJITTest.cpp for examples.
#define SKIP_UNSUPPORTED_PLATFORM \
  do \
    if (!ArchSupportsMCJIT() || !OSSupportsMCJIT()) \
      return; \
  while(0)

namespace llvm {

class MCJITTestAPICommon {
protected:
  MCJITTestAPICommon()
    : HostTriple(sys::getProcessTriple())
  {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // FIXME: It isn't at all clear why this is necesasry, but without it we
    // fail to initialize the AssumptionCacheTracker.
    initializeAssumptionCacheTrackerPass(*PassRegistry::getPassRegistry());

#ifdef _WIN32
    // On Windows, generate ELF objects by specifying "-elf" in triple
    HostTriple += "-elf";
#endif // _WIN32
    HostTriple = Triple::normalize(HostTriple);
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
      if (Host.getArchName().startswith(*I))
        return true;

    return false;
  }

  /// Returns true if the host OS is known to support MCJIT
  bool OSSupportsMCJIT() {
    Triple Host(HostTriple);

    if (find(UnsupportedEnvironments, Host.getEnvironment()) !=
        UnsupportedEnvironments.end())
      return false;

    if (!is_contained(UnsupportedOSs, Host.getOS()))
      return true;

    return false;
  }

  std::string HostTriple;
  SmallVector<Triple::ArchType, 4> SupportedArchs;
  SmallVector<Triple::ArchType, 1> HasSubArchs;
  SmallVector<std::string, 2> SupportedSubArchs; // We need to own the memory
  SmallVector<Triple::OSType, 4> UnsupportedOSs;
  SmallVector<Triple::EnvironmentType, 1> UnsupportedEnvironments;
};

} // namespace llvm

#endif

