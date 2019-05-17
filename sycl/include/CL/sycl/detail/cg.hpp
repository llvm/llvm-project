//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

using namespace cl;

// The structure represents kernel argument.
class ArgDesc {
public:
  ArgDesc(sycl::detail::kernel_param_kind_t Type, void *Ptr, int Size,
          int Index)
      : MType(Type), MPtr(Ptr), MSize(Size), MIndex(Index) {}

  sycl::detail::kernel_param_kind_t MType;
  void *MPtr;
  int MSize;
  int MIndex;
};

// The structure represents NDRange - global, local sizes, global offset and
// number of dimensions.
class NDRDescT {
  // The method initializes all sizes for dimensions greater than the passed one
  // to the default values, so they will not affect execution.
  template <int Dims_> void setNDRangeLeftover() {
    for (int I = Dims_; I < 3; ++I) {
      GlobalSize[I] = 1;
      LocalSize[I] = LocalSize[0] ? 1 : 0;
      GlobalOffset[I] = 0;
    }
  }

public:
  NDRDescT() = default;

  template <int Dims_> void set(sycl::range<Dims_> NumWorkItems) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = NumWorkItems[I];
      LocalSize[I] = 0;
      GlobalOffset[I] = 0;
    }

    setNDRangeLeftover<Dims_>();
    Dims = Dims_;
  }

  template <int Dims_> void set(sycl::nd_range<Dims_> ExecutionRange) {
    for (int I = 0; I < Dims_; ++I) {
      GlobalSize[I] = ExecutionRange.get_global_range()[I];
      LocalSize[I] = ExecutionRange.get_local_range()[I];
      GlobalOffset[I] = ExecutionRange.get_offset()[I];
    }
    setNDRangeLeftover<Dims_>();
    Dims = Dims_;
  }

  sycl::range<3> GlobalSize;
  sycl::range<3> LocalSize;
  sycl::id<3> GlobalOffset;
  size_t Dims;
};

// The pure virtual class aimed to store lambda/functors of any type.
class HostKernelBase {
public:
  // The method executes lambda stored using NDRange passed.
  virtual void call(const NDRDescT &NDRDesc) = 0;
  // Return pointer to the lambda object.
  // Used to extract captured variables.
  virtual char *getPtr() = 0;
  virtual ~HostKernelBase() = default;
};

// Class which stores specific lambda object.
template <class KernelType, class KernelArgType, int Dims>
class HostKernel : public HostKernelBase {
  using IDBuilder = sycl::detail::Builder;
  KernelType MKernel;

public:
  HostKernel(KernelType Kernel) : MKernel(Kernel) {}
  void call(const NDRDescT &NDRDesc) override { runOnHost(NDRDesc); }

  char *getPtr() override { return reinterpret_cast<char *>(&MKernel); }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, void>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    MKernel();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<Dims>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    size_t XYZ[3] = {0};
    sycl::id<Dims> ID;
    for (; XYZ[2] < NDRDesc.GlobalSize[2]; ++XYZ[2]) {
      XYZ[1] = 0;
      for (; XYZ[1] < NDRDesc.GlobalSize[1]; ++XYZ[1]) {
        XYZ[0] = 0;
        for (; XYZ[0] < NDRDesc.GlobalSize[0]; ++XYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            ID[I] = XYZ[I];
          MKernel(ID);
        }
      }
    }
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<
      (std::is_same<ArgT, item<Dims, /*Offset=*/false>>::value ||
       std::is_same<ArgT, item<Dims, /*Offset=*/true>>::value)>::type
  runOnHost(const NDRDescT &NDRDesc) {
    size_t XYZ[3] = {0};
    sycl::id<Dims> ID;
    sycl::range<Dims> Range;
    for (int I = 0; I < Dims; ++I)
      Range[I] = NDRDesc.GlobalSize[I];

    for (; XYZ[2] < NDRDesc.GlobalSize[2]; ++XYZ[2]) {
      XYZ[1] = 0;
      for (; XYZ[1] < NDRDesc.GlobalSize[1]; ++XYZ[1]) {
        XYZ[0] = 0;
        for (; XYZ[0] < NDRDesc.GlobalSize[0]; ++XYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            ID[I] = XYZ[I];

          sycl::item<Dims, /*Offset=*/false> Item =
              IDBuilder::createItem<Dims, false>(Range, ID);
          MKernel(Item);
        }
      }
    }
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, nd_item<Dims>>::value>::type
  runOnHost(const NDRDescT &NDRDesc) {
    // TODO add offset logic

    sycl::id<3> GroupSize;
    for (int I = 0; I < 3; ++I) {
      GroupSize[I] = NDRDesc.GlobalSize[I] / NDRDesc.LocalSize[I];
    }

    sycl::range<Dims> GlobalSize;
    sycl::range<Dims> LocalSize;
    sycl::id<Dims> GlobalOffset;
    for (int I = 0; I < Dims; ++I) {
      GlobalOffset[I] = NDRDesc.GlobalOffset[I];
      LocalSize[I] = NDRDesc.LocalSize[I];
      GlobalSize[I] = NDRDesc.GlobalSize[I];
    }

    sycl::id<Dims> GlobalID;
    sycl::id<Dims> LocalID;

    size_t GroupXYZ[3] = {0};
    sycl::id<Dims> GroupID;
    for (; GroupXYZ[2] < GroupSize[2]; ++GroupXYZ[2]) {
      GroupXYZ[1] = 0;
      for (; GroupXYZ[1] < GroupSize[1]; ++GroupXYZ[1]) {
        GroupXYZ[0] = 0;
        for (; GroupXYZ[0] < GroupSize[0]; ++GroupXYZ[0]) {
          for (int I = 0; I < Dims; ++I)
            GroupID[I] = GroupXYZ[I];

          sycl::group<Dims> Group =
              IDBuilder::createGroup<Dims>(GlobalSize, LocalSize, GroupID);
          size_t LocalXYZ[3] = {0};
          for (; LocalXYZ[2] < NDRDesc.LocalSize[2]; ++LocalXYZ[2]) {
            LocalXYZ[1] = 0;
            for (; LocalXYZ[1] < NDRDesc.LocalSize[1]; ++LocalXYZ[1]) {
              LocalXYZ[0] = 0;
              for (; LocalXYZ[0] < NDRDesc.LocalSize[0]; ++LocalXYZ[0]) {

                for (int I = 0; I < Dims; ++I) {
                  GlobalID[I] = GroupXYZ[I] * LocalSize[I] + LocalXYZ[I];
                  LocalID[I] = LocalXYZ[I];
                }
                const sycl::item<Dims, /*Offset=*/true> GlobalItem =
                    IDBuilder::createItem<Dims, true>(GlobalSize, GlobalID,
                                                      GlobalOffset);
                const sycl::item<Dims, /*Offset=*/false> LocalItem =
                    IDBuilder::createItem<Dims, false>(LocalSize, LocalID);
                const sycl::nd_item<Dims> NDItem =
                    IDBuilder::createNDItem<Dims>(GlobalItem, LocalItem, Group);
                MKernel(NDItem);
              }
            }
          }
        }
      }
    }
  }
  ~HostKernel() = default;
};

class stream_impl;
// The base class for all types of command groups.
class CG {
public:
  // Type of the command group.
  enum CGTYPE {
    KERNEL,
    COPY_ACC_TO_PTR,
    COPY_PTR_TO_ACC,
    COPY_ACC_TO_ACC,
    FILL,
    UPDATE_HOST
  };

  CG(CGTYPE Type, std::vector<std::vector<char>> ArgsStorage,
     std::vector<detail::AccessorImplPtr> AccStorage,
     std::vector<std::shared_ptr<void>> SharedPtrStorage,
     std::vector<Requirement *> Requirements)
      : MType(Type), MArgsStorage(std::move(ArgsStorage)),
        MAccStorage(std::move(AccStorage)),
        MSharedPtrStorage(std::move(SharedPtrStorage)),
        MRequirements(std::move(Requirements)) {}

  CG(CG &&CommandGroup) = default;

  std::vector<Requirement *> getRequirements() const { return MRequirements; }

  CGTYPE getType() { return MType; }

private:
  CGTYPE MType;
  // The following storages needed to ensure that arguments won't die while
  // we are using them.
  // Storage for standard layout arguments.
  std::vector<std::vector<char>> MArgsStorage;
  // Storage for accessors.
  std::vector<detail::AccessorImplPtr> MAccStorage;
  // Storage for shared_ptrs.
  std::vector<std::shared_ptr<void>> MSharedPtrStorage;
  // List of requirements that specify which memory is needed for the command
  // group to be executed.
  std::vector<Requirement *> MRequirements;
};

// The class which represents "execute kernel" command group.
class CGExecKernel : public CG {
public:
  NDRDescT MNDRDesc;
  std::unique_ptr<HostKernelBase> MHostKernel;
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  std::vector<ArgDesc> MArgs;
  std::string MKernelName;
  detail::OSModuleHandle MOSModuleHandle;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreams;

  CGExecKernel(NDRDescT NDRDesc, std::unique_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements,
               std::vector<ArgDesc> Args, std::string KernelName,
               detail::OSModuleHandle OSModuleHandle,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams)
      : CG(KERNEL, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MOSModuleHandle(OSModuleHandle),
        MStreams(std::move(Streams)) {}

  std::vector<ArgDesc> getArguments() const { return MArgs; }
  std::string getKernelName() const { return MKernelName; }
  std::vector<std::shared_ptr<detail::stream_impl>> getStreams() const {
    return MStreams;
  }
};

// The class which represents "copy" command group.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements)
      : CG(CopyType, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements)),
        MSrc(Src), MDst(Dst) {}
  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
};

// The class which represents "fill" command group.
class CGFill : public CG {
public:
  std::vector<char> MPattern;
  Requirement *MPtr;

  CGFill(std::vector<char> Pattern, void *Ptr,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements)
      : CG(FILL, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements)),
        MPattern(std::move(Pattern)), MPtr((Requirement *)Ptr) {}
  Requirement *getReqToFill() { return MPtr; }
};

// The class which represents "update host" command group.
class CGUpdateHost : public CG {
  Requirement *MPtr;

public:
  CGUpdateHost(void *Ptr, std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements)
      : CG(UPDATE_HOST, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements)),
        MPtr((Requirement *)Ptr) {}

  Requirement *getReqToUpdate() { return MPtr; }
};

} // namespace cl
} // namespace sycl
} // namespace detail
