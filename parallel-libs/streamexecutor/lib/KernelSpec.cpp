//===-- KernelSpec.cpp - General kernel spec implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation details for kernel loader specs.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/KernelSpec.h"

#include "llvm/ADT/STLExtras.h"

namespace streamexecutor {

KernelLoaderSpec::KernelLoaderSpec(llvm::StringRef KernelName)
    : KernelName(KernelName) {}

CUDAPTXInMemorySpec::CUDAPTXInMemorySpec(
    llvm::StringRef KernelName,
    const llvm::ArrayRef<CUDAPTXInMemorySpec::PTXSpec> SpecList)
    : KernelLoaderSpec(KernelName) {
  for (const auto &Spec : SpecList)
    PTXByComputeCapability.emplace(Spec.TheComputeCapability, Spec.PTXCode);
}

const char *CUDAPTXInMemorySpec::getCode(int ComputeCapabilityMajor,
                                         int ComputeCapabilityMinor) const {
  auto Iterator =
      PTXByComputeCapability.upper_bound(CUDAPTXInMemorySpec::ComputeCapability{
          ComputeCapabilityMajor, ComputeCapabilityMinor});
  if (Iterator == PTXByComputeCapability.begin())
    return nullptr;
  --Iterator;
  return Iterator->second;
}

CUDAFatbinInMemorySpec::CUDAFatbinInMemorySpec(llvm::StringRef KernelName,
                                               const void *Bytes)
    : KernelLoaderSpec(KernelName), Bytes(Bytes) {}

OpenCLTextInMemorySpec::OpenCLTextInMemorySpec(llvm::StringRef KernelName,
                                               const char *Text)
    : KernelLoaderSpec(KernelName), Text(Text) {}

void MultiKernelLoaderSpec::setKernelName(llvm::StringRef KernelName) {
  if (TheKernelName)
    assert(KernelName.equals(*TheKernelName) &&
           "different kernel names in one MultiKernelLoaderSpec");
  else
    TheKernelName = llvm::make_unique<std::string>(KernelName);
}

MultiKernelLoaderSpec &MultiKernelLoaderSpec::addCUDAPTXInMemory(
    llvm::StringRef KernelName,
    llvm::ArrayRef<CUDAPTXInMemorySpec::PTXSpec> SpecList) {
  assert((TheCUDAPTXInMemorySpec == nullptr) &&
         "illegal loader spec overwrite");
  setKernelName(KernelName);
  TheCUDAPTXInMemorySpec =
      llvm::make_unique<CUDAPTXInMemorySpec>(KernelName, SpecList);
  return *this;
}

MultiKernelLoaderSpec &
MultiKernelLoaderSpec::addCUDAFatbinInMemory(llvm::StringRef KernelName,
                                             const void *Bytes) {
  assert((TheCUDAFatbinInMemorySpec == nullptr) &&
         "illegal loader spec overwrite");
  setKernelName(KernelName);
  TheCUDAFatbinInMemorySpec =
      llvm::make_unique<CUDAFatbinInMemorySpec>(KernelName, Bytes);
  return *this;
}

MultiKernelLoaderSpec &
MultiKernelLoaderSpec::addOpenCLTextInMemory(llvm::StringRef KernelName,
                                             const char *OpenCLText) {
  assert((TheOpenCLTextInMemorySpec == nullptr) &&
         "illegal loader spec overwrite");
  setKernelName(KernelName);
  TheOpenCLTextInMemorySpec =
      llvm::make_unique<OpenCLTextInMemorySpec>(KernelName, OpenCLText);
  return *this;
}

} // namespace streamexecutor
