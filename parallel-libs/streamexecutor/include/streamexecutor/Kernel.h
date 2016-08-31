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
/// With the kernel parameter types recorded in the Kernel template parameters,
/// type-safe kernel launch functions can be written with signatures like the
/// following:
/// \code
///     template <typename... ParameterTs>
///     void Launch(
///       const Kernel<ParameterTs...> &Kernel, ParamterTs... Arguments);
/// \endcode
/// and the compiler will check that the user passes in arguments with types
/// matching the corresponding kernel parameters.
///
/// A problem is that a Kernel template specialization with the right parameter
/// types must be passed as the first argument to the Launch function, and it's
/// just as hard to get the types right in that template specialization as it is
/// to get them right for the kernel arguments.
///
/// With this problem in mind, it is not recommended for users to specialize the
/// Kernel template class themselves, but instead to let the compiler do it for
/// them. When the compiler encounters a device kernel function, it can create a
/// Kernel template specialization in the host code that has the right parameter
/// types for that kernel and which has a type name based on the name of the
/// kernel function.
///
/// \anchor CompilerGeneratedKernelExample
/// For example, if a CUDA device kernel function with the following signature
/// has been defined:
/// \code
///     void Saxpy(float A, float *X, float *Y);
/// \endcode
/// the compiler can insert the following declaration in the host code:
/// \code
///     namespace compiler_cuda_namespace {
///     namespace se = streamexecutor;
///     using SaxpyKernel =
///         se::Kernel<
///             float,
///             se::GlobalDeviceMemory<float>,
///             se::GlobalDeviceMemory<float>>;
///     } // namespace compiler_cuda_namespace
/// \endcode
/// and then the user can launch the kernel by calling the StreamExecutor launch
/// function as follows:
/// \code
///     namespace ccn = compiler_cuda_namespace;
///     using KernelPtr = std::unique_ptr<cnn::SaxpyKernel>;
///     // Assumes Device is a pointer to the Device on which to launch the
///     // kernel.
///     //
///     // See KernelSpec.h for details on how the compiler can create a
///     // MultiKernelLoaderSpec instance like SaxpyKernelLoaderSpec below.
///     Expected<KernelPtr> MaybeKernel =
///         Device->createKernel<ccn::SaxpyKernel>(ccn::SaxpyKernelLoaderSpec);
///     if (!MaybeKernel) { /* Handle error */ }
///     KernelPtr SaxpyKernel = std::move(*MaybeKernel);
///     Launch(*SaxpyKernel, A, X, Y);
/// \endcode
///
/// With the compiler's help in specializing Kernel for each device kernel
/// function (and generating a MultiKernelLoaderSpec instance for each kernel),
/// the user can safely launch the device kernel from the host and get an error
/// message at compile time if the argument types don't match the kernel
/// parameter types.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_KERNEL_H
#define STREAMEXECUTOR_KERNEL_H

#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/Utils/Error.h"

#include <memory>

namespace streamexecutor {

class PlatformKernelHandle;

/// The base class for all kernel types.
///
/// Stores the name of the kernel in both mangled and demangled forms.
class KernelBase {
public:
  KernelBase(llvm::StringRef Name);

  const std::string &getName() const { return Name; }
  const std::string &getDemangledName() const { return DemangledName; }

private:
  std::string Name;
  std::string DemangledName;
};

/// A StreamExecutor kernel.
///
/// The template parameters are the types of the parameters to the kernel
/// function.
template <typename... ParameterTs> class Kernel : public KernelBase {
public:
  Kernel(llvm::StringRef Name, std::unique_ptr<PlatformKernelHandle> PHandle)
      : KernelBase(Name), PHandle(std::move(PHandle)) {}

  /// Gets the underlying platform-specific handle for this kernel.
  PlatformKernelHandle *getPlatformHandle() const { return PHandle.get(); }

private:
  std::unique_ptr<PlatformKernelHandle> PHandle;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_KERNEL_H
