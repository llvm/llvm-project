//===-- Kernel.cpp - General kernel implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation details for kernel types.
///
//===----------------------------------------------------------------------===//

#include <cassert>

#include "streamexecutor/Device.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/PlatformDevice.h"

#include "llvm/DebugInfo/Symbolize/Symbolize.h"

namespace streamexecutor {

KernelBase::KernelBase(PlatformDevice *D, const void *PlatformKernelHandle,
                       llvm::StringRef Name)
    : PDevice(D), PlatformKernelHandle(PlatformKernelHandle), Name(Name),
      DemangledName(
          llvm::symbolize::LLVMSymbolizer::DemangleName(Name, nullptr)) {
  assert(D != nullptr &&
         "cannot construct a kernel object with a null platform device");
  assert(PlatformKernelHandle != nullptr &&
         "cannot construct a kernel object with a null platform kernel handle");
}

KernelBase::KernelBase(KernelBase &&Other)
    : PDevice(Other.PDevice), PlatformKernelHandle(Other.PlatformKernelHandle),
      Name(std::move(Other.Name)),
      DemangledName(std::move(Other.DemangledName)) {
  Other.PDevice = nullptr;
  Other.PlatformKernelHandle = nullptr;
}

KernelBase &KernelBase::operator=(KernelBase &&Other) {
  PDevice = Other.PDevice;
  PlatformKernelHandle = Other.PlatformKernelHandle;
  Name = std::move(Other.Name);
  DemangledName = std::move(Other.DemangledName);
  Other.PDevice = nullptr;
  Other.PlatformKernelHandle = nullptr;
  return *this;
}

KernelBase::~KernelBase() {
  if (PlatformKernelHandle)
    // TODO(jhen): Handle the error here.
    consumeError(PDevice->destroyKernel(PlatformKernelHandle));
}

} // namespace streamexecutor
