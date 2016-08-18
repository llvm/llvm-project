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
/// The TypedKernel class is used to provide type safety to the user API's
/// launch functions, and the KernelBase class is used like a void* function
/// pointer to perform type-unsafe operations inside StreamExecutor.
///
/// With the kernel parameter types recorded in the TypedKernel template
/// parameters, type-safe kernel launch functions can be written with signatures
/// like the following:
/// \code
///     template <typename... ParameterTs>
///     void Launch(
///       const TypedKernel<ParameterTs...> &Kernel, ParamterTs... Arguments);
/// \endcode
/// and the compiler will check that the user passes in arguments with types
/// matching the corresponding kernel parameters.
///
/// A problem is that a TypedKernel template specialization with the right
/// parameter types must be passed as the first argument to the Launch function,
/// and it's just as hard to get the types right in that template specialization
/// as it is to get them right for the kernel arguments.
///
/// With this problem in mind, it is not recommended for users to specialize the
/// TypedKernel template class themselves, but instead to let the compiler do it
/// for them. When the compiler encounters a device kernel function, it can
/// create a TypedKernel template specialization in the host code that has the
/// right parameter types for that kernel and which has a type name based on the
/// name of the kernel function.
///
/// For example, if a CUDA device kernel function with the following signature
/// has been defined:
/// \code
///     void Saxpy(float *A, float *X, float *Y);
/// \endcode
/// the compiler can insert the following declaration in the host code:
/// \code
///     namespace compiler_cuda_namespace {
///     using SaxpyKernel =
///         streamexecutor::TypedKernel<float *, float *, float *>;
///     } // namespace compiler_cuda_namespace
/// \endcode
/// and then the user can launch the kernel by calling the StreamExecutor launch
/// function as follows:
/// \code
///     namespace ccn = compiler_cuda_namespace;
///     // Assumes Executor is a pointer to the StreamExecutor on which to
///     // launch the kernel.
///     //
///     // See KernelSpec.h for details on how the compiler can create a
///     // MultiKernelLoaderSpec instance like SaxpyKernelLoaderSpec below.
///     Expected<ccn::SaxpyKernel> MaybeKernel =
///         ccn::SaxpyKernel::create(Executor, ccn::SaxpyKernelLoaderSpec);
///     if (!MaybeKernel) { /* Handle error */ }
///     ccn::SaxpyKernel SaxpyKernel = *MaybeKernel;
///     Launch(SaxpyKernel, A, X, Y);
/// \endcode
///
/// With the compiler's help in specializing TypedKernel for each device kernel
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

class Executor;
class KernelInterface;

/// The base class for device kernel functions.
///
/// This class has no information about the types of the parameters taken by the
/// kernel, so it is analogous to a void* pointer to a device function.
///
/// See the TypedKernel class below for the subclass which does have information
/// about parameter types.
class KernelBase {
public:
  KernelBase(KernelBase &&) = default;
  KernelBase &operator=(KernelBase &&) = default;
  ~KernelBase();

  /// Creates a kernel object from an Executor and a MultiKernelLoaderSpec.
  ///
  /// The Executor knows which platform it belongs to and the
  /// MultiKernelLoaderSpec knows how to find the kernel code for different
  /// platforms, so the combined information is enough to get the kernel code
  /// for the appropriate platform.
  static Expected<KernelBase> create(Executor *ParentExecutor,
                                     const MultiKernelLoaderSpec &Spec);

  const std::string &getName() const { return Name; }
  const std::string &getDemangledName() const { return DemangledName; }

  /// Gets a pointer to the platform-specific implementation of this kernel.
  KernelInterface *getImplementation() { return Implementation.get(); }

private:
  KernelBase(Executor *ParentExecutor, const std::string &Name,
             const std::string &DemangledName,
             std::unique_ptr<KernelInterface> Implementation);

  Executor *ParentExecutor;
  std::string Name;
  std::string DemangledName;
  std::unique_ptr<KernelInterface> Implementation;

  KernelBase(const KernelBase &) = delete;
  KernelBase &operator=(const KernelBase &) = delete;
};

/// A device kernel function with specified parameter types.
template <typename... ParameterTs> class TypedKernel : public KernelBase {
public:
  TypedKernel(TypedKernel &&) = default;
  TypedKernel &operator=(TypedKernel &&) = default;

  /// Parameters here have the same meaning as in KernelBase::create.
  static Expected<TypedKernel> create(Executor *ParentExecutor,
                                      const MultiKernelLoaderSpec &Spec) {
    auto MaybeBase = KernelBase::create(ParentExecutor, Spec);
    if (!MaybeBase) {
      return MaybeBase.takeError();
    }
    TypedKernel Instance(std::move(*MaybeBase));
    return std::move(Instance);
  }

private:
  TypedKernel(KernelBase &&Base) : KernelBase(std::move(Base)) {}

  TypedKernel(const TypedKernel &) = delete;
  TypedKernel &operator=(const TypedKernel &) = delete;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_KERNEL_H
