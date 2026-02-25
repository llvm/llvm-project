//===- llvm/TargetParser/Host.h - Host machine detection  -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Methods for querying the nature of the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_HOST_H
#define LLVM_TARGETPARSER_HOST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {
class MallocAllocator;
class StringRef;
template <typename ValueTy, typename AllocatorTy> class StringMap;
class raw_ostream;

namespace sys {

/// getDefaultTargetTriple() - Return the default target triple the compiler
/// has been configured to produce code for.
///
/// The target triple is a string in the format of:
///   CPU_TYPE-VENDOR-OPERATING_SYSTEM
/// or
///   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM
LLVM_ABI std::string getDefaultTargetTriple();

/// getProcessTriple() - Return an appropriate target triple for generating
/// code to be loaded into the current process, e.g. when using the JIT.
LLVM_ABI std::string getProcessTriple();

/// getHostCPUName - Get the LLVM name for the host CPU. The particular format
/// of the name is target dependent, and suitable for passing as -mcpu to the
/// target which matches the host.
///
/// \return - The host CPU name, or empty if the CPU could not be determined.
LLVM_ABI StringRef getHostCPUName();

/// getHostCPUFeatures - Get the LLVM names for the host CPU features.
/// The particular format of the names are target dependent, and suitable for
/// passing as -mattr to the target which matches the host.
///
/// \return - A string map mapping feature names to either true (if enabled)
/// or false (if disabled). This routine makes no guarantees about exactly
/// which features may appear in this map, except that they are all valid LLVM
/// feature names. The map can be empty, for example if feature detection
/// fails.
LLVM_ABI StringMap<bool, MallocAllocator> getHostCPUFeatures();

/// This is a function compatible with cl::AddExtraVersionPrinter, which adds
/// info about the current target triple and detected CPU.
LLVM_ABI void printDefaultTargetAndDetectedCPU(raw_ostream &OS);

namespace detail {
/// Helper functions to extract HostCPUName from /proc/cpuinfo on linux.
LLVM_ABI StringRef getHostCPUNameForPowerPC(StringRef ProcCpuinfoContent);
LLVM_ABI StringRef getHostCPUNameForARM(StringRef ProcCpuinfoContent);
LLVM_ABI StringRef getHostCPUNameForARM(uint64_t PrimaryCpuInfo,
                                        ArrayRef<uint64_t> UniqueCpuInfos);
LLVM_ABI StringRef getHostCPUNameForS390x(StringRef ProcCpuinfoContent);
LLVM_ABI StringRef getHostCPUNameForRISCV(StringRef ProcCpuinfoContent);
LLVM_ABI StringRef getHostCPUNameForSPARC(StringRef ProcCpuinfoContent);
LLVM_ABI StringRef getHostCPUNameForBPF();

/// Helper functions to extract CPU details from CPUID on x86.
namespace x86 {
enum class VendorSignatures {
  UNKNOWN,
  GENUINE_INTEL,
  AUTHENTIC_AMD,
};

/// Returns the host CPU's vendor.
/// MaxLeaf: if a non-nullptr pointer is specified, the EAX value will be
/// assigned to its pointee.
LLVM_ABI VendorSignatures getVendorSignature(unsigned *MaxLeaf = nullptr);
} // namespace x86
} // namespace detail
} // namespace sys
} // namespace llvm

#endif
