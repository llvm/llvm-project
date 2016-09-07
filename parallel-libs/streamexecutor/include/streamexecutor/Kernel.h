//===-- Kernel.h - StreamExecutor kernel types ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Types to represent device kernels (code compiled to run on GPU or other
/// accelerator).
///
/// See the \ref index "main page" for an example of how a compiler-generated
/// specialization of the Kernel class template can be used along with the
/// streamexecutor::Stream::thenLaunch method to create a typesafe interface for
/// kernel launches.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_KERNEL_H
#define STREAMEXECUTOR_KERNEL_H

#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/Utils/Error.h"

#include <memory>

namespace streamexecutor {

class PlatformDevice;

/// The base class for all kernel types.
///
/// Stores the name of the kernel in both mangled and demangled forms.
class KernelBase {
public:
  KernelBase(PlatformDevice *D, const void *PlatformKernelHandle,
             llvm::StringRef Name);

  KernelBase(const KernelBase &Other) = delete;
  KernelBase &operator=(const KernelBase &Other) = delete;

  KernelBase(KernelBase &&Other);
  KernelBase &operator=(KernelBase &&Other);

  ~KernelBase();

  const void *getPlatformHandle() const { return PlatformKernelHandle; }
  const std::string &getName() const { return Name; }
  const std::string &getDemangledName() const { return DemangledName; }

private:
  PlatformDevice *PDevice;
  const void *PlatformKernelHandle;

  std::string Name;
  std::string DemangledName;
};

/// A StreamExecutor kernel.
///
/// The template parameters are the types of the parameters to the kernel
/// function.
template <typename... ParameterTs> class Kernel : public KernelBase {
public:
  Kernel(PlatformDevice *D, const void *PlatformKernelHandle,
         llvm::StringRef Name)
      : KernelBase(D, PlatformKernelHandle, Name) {}

  Kernel(Kernel &&Other) = default;
  Kernel &operator=(Kernel &&Other) = default;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_KERNEL_H
