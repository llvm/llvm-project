//===-- KernelSpec.h - Kernel loader spec types -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// KernelLoaderSpec is the base class for types that know where to find the
/// code for a data-parallel kernel in a particular format on a particular
/// platform. So, for example, there will be one subclass that deals with CUDA
/// PTX code, another subclass that deals with CUDA fatbin code, and yet another
/// subclass that deals with OpenCL text code.
///
/// A MultiKernelLoaderSpec is basically a collection of KernelLoaderSpec
/// instances. This is useful when code is available for the same kernel in
/// several different formats or targeted for several different platforms. All
/// the various KernelLoaderSpec instances for this kernel can be combined
/// together in one MultiKernelLoaderSpec and the specific platform consumer can
/// decide which instance of the code it wants to use.
///
/// MultiKernelLoaderSpec provides several helper functions to build and
/// register KernelLoaderSpec instances all in a single operation. For example,
/// MultiKernelLoaderSpec::addCUDAPTXInMemory can be used to construct and
/// register a CUDAPTXInMemorySpec KernelLoaderSpec.
///
/// The loader spec classes declared here are designed primarily to be
/// instantiated by the compiler, but they can also be instantiated directly by
/// the user. A simplified example workflow which a compiler might follow in the
/// case of a CUDA kernel that is compiled to CUDA fatbin code is as follows:
///
/// 1. The user defines a kernel function called UserKernel.
/// 2. The compiler compiles the kernel code into CUDA fatbin data and embeds
///    that data into the host code at address __UserKernelFatbinAddress.
/// 3. The compiler adds code at the beginning of the host code to instantiate a
///    MultiKernelLoaderSpec:
///    \code
///    namespace compiler_cuda_namespace {
///      MultiKernelLoaderSpec UserKernelLoaderSpec;
///    } // namespace compiler_cuda_namespace
///    \endcode
/// 4. The compiler then adds code to the host code to add the fatbin data to
///    the new MultiKernelLoaderSpec, and to associate that data with the kernel
///    name "UserKernel":
///    \code
///    namespace compiler_cuda_namespace {
///      UserKernelLoaderSpec.addCUDAFatbinInMemory(
///        __UserKernelFatbinAddress, "UserKernel");
///    } // namespace compiler_cuda_namespace
///    \encode
/// 5. The host code, having known beforehand that the compiler would initialize
///    a MultiKernelLoaderSpec based on the name of the CUDA kernel, makes use
///    of the symbol cudanamespace::UserKernelLoaderSpec without defining it.
///
/// In the example above, the MultiKernelLoaderSpec instance created by the
/// compiler can be used by the host code to create StreamExecutor kernel
/// objects. In turn, those StreamExecutor kernel objects can be used by the
/// host code to launch the kernel on the device as desired.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_KERNELSPEC_H
#define STREAMEXECUTOR_KERNELSPEC_H

#include <cassert>
#include <map>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace streamexecutor {

/// An object that knows how to find the code for a device kernel.
///
/// This is the base class for the hierarchy of loader specs. The different
/// subclasses know how to find code in different formats (e.g. CUDA PTX, OpenCL
/// binary).
///
/// This base class has functionality for storing and getting the name of the
/// kernel as a string.
class KernelLoaderSpec {
public:
  /// Returns the name of the kernel this spec loads.
  const std::string &getKernelName() const { return KernelName; }

protected:
  explicit KernelLoaderSpec(llvm::StringRef KernelName);

private:
  std::string KernelName;

  KernelLoaderSpec(const KernelLoaderSpec &) = delete;
  KernelLoaderSpec &operator=(const KernelLoaderSpec &) = delete;
};

/// A KernelLoaderSpec for CUDA PTX code that resides in memory as a
/// null-terminated string.
class CUDAPTXInMemorySpec : public KernelLoaderSpec {
public:
  /// First component is major version, second component is minor version.
  using ComputeCapability = std::pair<int, int>;

  /// PTX code combined with its compute capability.
  struct PTXSpec {
    ComputeCapability TheComputeCapability;
    const char *PTXCode;
  };

  /// Creates a CUDAPTXInMemorySpec from an array of PTXSpec objects.
  ///
  /// Adds each item in SpecList to this object.
  ///
  /// Does not take ownership of the PTXCode pointers in the SpecList elements.
  CUDAPTXInMemorySpec(llvm::StringRef KernelName,
                      const llvm::ArrayRef<PTXSpec> SpecList);

  /// Returns a pointer to the PTX code for the requested compute capability.
  ///
  /// Returns nullptr on failed lookup (if the requested compute capability is
  /// not available). Matches exactly the specified compute capability. Doesn't
  /// try to do anything smart like finding the next best compute capability if
  /// the specified capability cannot be found.
  const char *getCode(int ComputeCapabilityMajor,
                      int ComputeCapabilityMinor) const;

private:
  /// PTX code contents in memory.
  ///
  /// The key is a pair (cc_major, cc_minor), i.e., (2, 0), (3, 0), (3, 5).
  std::map<ComputeCapability, const char *> PTXByComputeCapability;

  CUDAPTXInMemorySpec(const CUDAPTXInMemorySpec &) = delete;
  CUDAPTXInMemorySpec &operator=(const CUDAPTXInMemorySpec &) = delete;
};

/// A KernelLoaderSpec for CUDA fatbin code that resides in memory.
class CUDAFatbinInMemorySpec : public KernelLoaderSpec {
public:
  /// Creates a CUDAFatbinInMemorySpec with a reference to the given fatbin
  /// bytes.
  ///
  /// Does not take ownership of the Bytes pointer.
  CUDAFatbinInMemorySpec(llvm::StringRef KernelName, const void *Bytes);

  /// Gets the fatbin data bytes.
  const void *getBytes() const { return Bytes; }

private:
  const void *Bytes;

  CUDAFatbinInMemorySpec(const CUDAFatbinInMemorySpec &) = delete;
  CUDAFatbinInMemorySpec &operator=(const CUDAFatbinInMemorySpec &) = delete;
};

/// A KernelLoaderSpec for OpenCL text that resides in memory as a
/// null-terminated string.
class OpenCLTextInMemorySpec : public KernelLoaderSpec {
public:
  /// Creates a OpenCLTextInMemorySpec with a reference to the given OpenCL text
  /// code bytes.
  ///
  /// Does not take ownership of the Text pointer.
  OpenCLTextInMemorySpec(llvm::StringRef KernelName, const char *Text);

  /// Returns the OpenCL text contents.
  const char *getText() const { return Text; }

private:
  const char *Text;

  OpenCLTextInMemorySpec(const OpenCLTextInMemorySpec &) = delete;
  OpenCLTextInMemorySpec &operator=(const OpenCLTextInMemorySpec &) = delete;
};

/// An object to store several different KernelLoaderSpecs for the same kernel.
///
/// This allows code in different formats and for different platforms to be
/// stored all together for a single kernel.
///
/// Various methods are available to add a new KernelLoaderSpec to a
/// MultiKernelLoaderSpec. There are also methods to query which formats and
/// platforms are supported by the currently added KernelLoaderSpec objects, and
/// methods to get the KernelLoaderSpec objects for each format and platform.
///
/// Since all stored KernelLoaderSpecs are supposed to reference the same
/// kernel, they are all assumed to take the same number and type of parameters,
/// but no checking is done to enforce this. In debug mode, all
/// KernelLoaderSpecs are checked to make sure they have the same kernel name,
/// so passing in specs with different kernel names can cause the program to
/// abort.
///
/// This interface is prone to errors, so it is better to leave
/// MultiKernelLoaderSpec creation and initialization to the compiler rather
/// than doing it by hand.
class MultiKernelLoaderSpec {
public:
  std::string getKernelName() const {
    if (TheKernelName) {
      return *TheKernelName;
    }
    return "";
  }

  // Convenience getters for testing whether these platform variants have
  // kernel loader specifications available.

  bool hasCUDAPTXInMemory() const { return TheCUDAPTXInMemorySpec != nullptr; }
  bool hasCUDAFatbinInMemory() const {
    return TheCUDAFatbinInMemorySpec != nullptr;
  }
  bool hasOpenCLTextInMemory() const {
    return TheOpenCLTextInMemorySpec != nullptr;
  }

  // Accessors for platform variant kernel load specifications.
  //
  // Precondition: corresponding has* method returns true.

  const CUDAPTXInMemorySpec &getCUDAPTXInMemory() const {
    assert(hasCUDAPTXInMemory() && "getting spec that is not present");
    return *TheCUDAPTXInMemorySpec;
  }
  const CUDAFatbinInMemorySpec &getCUDAFatbinInMemory() const {
    assert(hasCUDAFatbinInMemory() && "getting spec that is not present");
    return *TheCUDAFatbinInMemorySpec;
  }
  const OpenCLTextInMemorySpec &getOpenCLTextInMemory() const {
    assert(hasOpenCLTextInMemory() && "getting spec that is not present");
    return *TheOpenCLTextInMemorySpec;
  }

  // Builder-pattern-like methods for use in initializing a
  // MultiKernelLoaderSpec.
  //
  // Each of these should be used at most once for a single
  // MultiKernelLoaderSpec object. See file comment for example usage.
  //
  // Note that the KernelName parameter must be consistent with the kernel in
  // the PTX or OpenCL being loaded. Also be aware that in CUDA C++ the kernel
  // name may be mangled by the compiler if it is not declared extern "C".

  /// Does not take ownership of the PTXCode pointers in the SpecList elements.
  MultiKernelLoaderSpec &
  addCUDAPTXInMemory(llvm::StringRef KernelName,
                     llvm::ArrayRef<CUDAPTXInMemorySpec::PTXSpec> SpecList);

  /// Does not take ownership of the FatbinBytes pointer.
  MultiKernelLoaderSpec &addCUDAFatbinInMemory(llvm::StringRef KernelName,
                                               const void *FatbinBytes);

  /// Does not take ownership of the OpenCLText pointer.
  MultiKernelLoaderSpec &addOpenCLTextInMemory(llvm::StringRef KernelName,
                                               const char *OpenCLText);

private:
  void setKernelName(llvm::StringRef KernelName);

  std::unique_ptr<std::string> TheKernelName;
  std::unique_ptr<CUDAPTXInMemorySpec> TheCUDAPTXInMemorySpec;
  std::unique_ptr<CUDAFatbinInMemorySpec> TheCUDAFatbinInMemorySpec;
  std::unique_ptr<OpenCLTextInMemorySpec> TheOpenCLTextInMemorySpec;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_KERNELSPEC_H
