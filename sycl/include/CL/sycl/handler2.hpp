//==-------- handler.hpp --- SYCL command group handler --------*- C++ -*---==//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive property
// of Intel Corporation and may not be disclosed, examined or reproduced in
// whole or in part without explicit written authorization from the company.
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/cg.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/sampler.hpp>

#include <CL/sycl/stl.hpp>

#include <functional>
#include <memory>
#include <type_traits>

template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __fill;

template <typename T_Src, typename T_Dst, int Dims,
          cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __copyAcc2Ptr;

template <typename T_Src, typename T_Dst, int Dims,
          cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder IsPlaceholder>
class __copyPtr2Acc;

template <typename T_Src, int Dims_Src, cl::sycl::access::mode AccessMode_Src,
          cl::sycl::access::target AccessTarget_Src, typename T_Dst,
          int Dims_Dst, cl::sycl::access::mode AccessMode_Dst,
          cl::sycl::access::target AccessTarget_Dst,
          cl::sycl::access::placeholder IsPlaceholder_Src,
          cl::sycl::access::placeholder IsPlaceholder_Dst>
class __copyAcc2Acc;

namespace cl {
namespace sycl {

namespace csd = cl::sycl::detail;

// Forward declaration

template <typename T, int Dimensions, typename AllocatorT> class buffer;
namespace detail {

#ifdef __SYCL_DEVICE_ONLY__

#define DEFINE_INIT_SIZES(POSTFIX)                                             \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[0] = cl::__spirv::get##POSTFIX<0>();                                 \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[1] = cl::__spirv::get##POSTFIX<1>();                                 \
      InitSizesST##POSTFIX<1, DstT>::initSize(Dst);                            \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[2] = cl::__spirv::get##POSTFIX<2>();                                 \
      InitSizesST##POSTFIX<2, DstT>::initSize(Dst);                            \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> static void init##POSTFIX(DstT &Dst) {       \
    InitSizesST##POSTFIX<Dims, DstT>::initSize(Dst);                           \
  }

DEFINE_INIT_SIZES(GlobalSize);
DEFINE_INIT_SIZES(GlobalInvocationId)
DEFINE_INIT_SIZES(WorkgroupSize)
DEFINE_INIT_SIZES(LocalInvocationId)
DEFINE_INIT_SIZES(WorkgroupId)
DEFINE_INIT_SIZES(GlobalOffset)

#undef DEFINE_INIT_SIZES

#endif //__SYCL_DEVICE_ONLY__

class queue_impl;
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

// Non-const version of the above template to match functors whose 'operator()'
// is declared w/o the 'const' qualifier.
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg));

//template <typename RetType, typename Func>
//static void member_ptr_helper(RetType (Func::*)() const);

//template <typename RetType, typename Func>
//static void member_ptr_helper(RetType (Func::*)());

template <typename F>
decltype(member_ptr_helper(&F::operator())) argument_helper(F);

template <typename T>
using lambda_arg_type = decltype(argument_helper(std::declval<T>()));
} // namespace detail

// Objects of the handler class collect information about command group, such as
// kernel, requirements to the memory, arguments for the kernel.
//
// sycl::queue::submit([](handler &CGH){
//   CGH.require(Accessor1);   // Adds a requirement to the memory object.
//   CGH.setArg(0, Accessor2); // Registers accessor given as an argument to the
//                             // kernel + adds a requirement to the memory
//                             // object.
//   CGH.setArg(1, N);         // Registers value given as an argument to the
//                             // kernel.
//   // The following registers KernelFunctor to be a kernel that will be
//   // executed in case of queue is bound to the host device, SyclKernel - for
//   // an OpenCL device. This function clearly indicates that command group
//   // represents kernel execution.
//   CGH.parallel_for(KernelFunctor, SyclKernel);
//  });
//
// The command group can represent absolutely different operations. Depending
// on the operation we need to store different data. But, in most cases, it's
// impossible to say what kind of operation we need to perform until the very
// end. So, handler class contains all fields simultaneously, then during
// "finalization" it constructs CG object, that represents specific operation,
// passing fields that are required only.

// 4.8.3 Command group handler class
class handler {
  std::shared_ptr<detail::queue_impl> MQueue;
  // The storage for the arguments passed.
  // We need to store a copy of values that are passed explicitly through
  // set_arg, require and so on, because we need them to be alive after
  // we exit the method they are passed in.
  std::vector<std::vector<char>> MArgsStorage;
  std::vector<detail::AccessorImplPtr> MAccStorage;
  std::vector<std::shared_ptr<void>> MSharedPtrStorage;
  // The list of arguments for the kernel.
  std::vector<detail::ArgDesc> MArgs;
  // The list of requirements to the memory objects for the scheduling.
  std::vector<detail::Requirement *> MRequirements;
  // Struct that encodes global size, local size, ...
  detail::NDRDescT MNDRDesc;
  std::string MKernelName;
  // Storage for a sycl::kernel object.
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  // Type of the command group, e.g. kernel, fill.
  detail::CG::CGTYPE MCGType;
  // Pointer to the source host memory or accessor(depending on command type).
  void *MSrcPtr = nullptr;
  // Pointer to the dest host memory or accessor(depends on command type).
  void *MDstPtr = nullptr;
  // Pattern that is used to fill memory object in case command type is fill.
  std::vector<char> MPattern;
  // Storage for a lambda or function object.
  std::unique_ptr<detail::HostKernelBase> MHostKernel;
  detail::OSModuleHandle MOSModuleHandle;

  bool MIsHost = false;

private:
  handler(std::shared_ptr<detail::queue_impl> Queue, bool IsHost)
      : MQueue(std::move(Queue)), MIsHost(IsHost) {}

  // Method stores copy of Arg passed to the MArgsStorage.
  template <typename T, typename F = typename std::remove_reference<T>::type>
  F *storePlainArg(T &&Arg) {
    MArgsStorage.emplace_back(sizeof(T));
    F *Storage = (F *)MArgsStorage.back().data();
    *Storage = Arg;
    return Storage;
  }

  // Method extracts kernel arguments and requirements from the lambda using
  // integration header.
  void
  extractArgsAndReqsFromLambda(char *LambdaPtr, size_t KernelArgsNum,
                               const detail::kernel_param_desc_t *KernelArgs) {
    unsigned NextArgId = 0;
    for (unsigned I = 0; I < KernelArgsNum; ++I, ++NextArgId) {
      void *Ptr = LambdaPtr + KernelArgs[I].offset;
      const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;

      switch (Kind) {
      case detail::kernel_param_kind_t::kind_std_layout: {
        const size_t Size = KernelArgs[I].info;
        MArgs.emplace_back(detail::ArgDesc(Kind, Ptr, Size, NextArgId));
        break;
      }
      case detail::kernel_param_kind_t::kind_accessor: {

        const int AccTarget = KernelArgs[I].info & 0x7ff;
        switch (static_cast<access::target>(AccTarget)) {
        case access::target::global_buffer:
        case access::target::constant_buffer: {

          detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)Ptr;

          detail::Requirement *AccImpl = detail::getSyclObjImpl(*AccBase).get();

          MArgs.emplace_back(
              detail::ArgDesc(Kind, AccImpl, /*Size=*/0, NextArgId));
          MRequirements.emplace_back(AccImpl);

          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &(AccImpl->MAccessRange[0]),
                              sizeof(size_t) * AccImpl->MDims, NextArgId + 1));
          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &AccImpl->MMemoryRange[0],
                              sizeof(size_t) * AccImpl->MDims, NextArgId + 2));
          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &AccImpl->MOffset[0],
                              sizeof(size_t) * AccImpl->MDims, NextArgId + 3));
          NextArgId += 3;
          break;
        }
        case access::target::local: {

          detail::LocalAccessorBaseHost *LAcc =
              (detail::LocalAccessorBaseHost *)Ptr;
          range<3> &Size = LAcc->getSize();
          const int Dims = LAcc->getNumOfDims();
          int SizeInBytes = LAcc->getElementSize();
          for (int I = 0; I < Dims; ++I)
            SizeInBytes *= Size[I];
          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              nullptr, SizeInBytes, NextArgId));

          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &Size, Dims * sizeof(Size[0]), NextArgId + 1));
          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &Size, Dims * sizeof(Size[0]), NextArgId + 2));
          MArgs.emplace_back(
              detail::ArgDesc(detail::kernel_param_kind_t::kind_std_layout,
                              &Size, Dims * sizeof(Size[0]), NextArgId + 3));
          NextArgId += 3;
          break;
        }

        case access::target::image:
        case access::target::host_buffer:
        case access::target::host_image:
        case access::target::image_array: {
          assert(0);
          break;
        }
        }
        break;
      }
      case detail::kernel_param_kind_t::kind_sampler: {
        MArgs.emplace_back(
            detail::ArgDesc(detail::kernel_param_kind_t::kind_sampler, Ptr,
                            sizeof(sampler), NextArgId));
        NextArgId++;
        break;
      }
      }
    }
  }

  // The method constructs CG object of specific type, pass it to Scheduler and
  // returns sycl::event object representing the command group.
  // It's expected that the method is the latest method executed before
  // object destruction.
  event finalize() {
    sycl::event EventRet;
    std::unique_ptr<detail::CG> CommandGroup;
    switch (MCGType) {
    case detail::CG::KERNEL:
      CommandGroup.reset(new detail::CGExecKernel(
          std::move(MNDRDesc), std::move(MHostKernel), std::move(MSyclKernel),
          std::move(MArgsStorage), std::move(MAccStorage),
          std::move(MSharedPtrStorage), std::move(MRequirements),
          std::move(MArgs), std::move(MKernelName),
          std::move(MOSModuleHandle)));
      break;
    case detail::CG::COPY_ACC_TO_PTR:
    case detail::CG::COPY_PTR_TO_ACC:
    case detail::CG::COPY_ACC_TO_ACC:
      CommandGroup.reset(new detail::CGCopy(
          MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements)));
      break;
    case detail::CG::FILL:
      CommandGroup.reset(new detail::CGFill(
          std::move(MPattern), MDstPtr, std::move(MArgsStorage),
          std::move(MAccStorage), std::move(MSharedPtrStorage),
          std::move(MRequirements)));
      break;
    case detail::CG::UPDATE_HOST:
      CommandGroup.reset(new detail::CGUpdateHost(
          MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
          std::move(MSharedPtrStorage), std::move(MRequirements)));
      break;
    default:
      throw runtime_error("Unhandled type of command group");
    }

    detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
        std::move(CommandGroup), std::move(MQueue));

    EventRet = detail::createSyclObjFromImpl<event>(Event);
    return EventRet;
  }

  ~handler() = default;

  bool is_host() { return MIsHost; }

  // Recursively calls itself until arguments pack is fully processed.
  // The version for regular(standard layout) argument.
  template <typename T, typename... Ts>
  void setArgsHelper(int ArgIndex, T &&Arg, Ts &&... Args) {
    set_arg(ArgIndex, std::move(Arg));
    setArgsHelper(++ArgIndex, std::move(Args)...);
  }

  void setArgsHelper(int ArgIndex) {}

  // setArgHelper version for accessor argument.
  template <typename DataT, int Dims, access::mode AccessMode,
            access::target AccessTarget, access::placeholder IsPlaceholder>
  void setArgHelper(
      int ArgIndex,
      accessor<DataT, Dims, AccessMode, AccessTarget, IsPlaceholder> &&Arg) {
    // TODO: Handle local accessor in separate method.

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Arg;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    // Add accessor to the list of arguments.
    MRequirements.push_back(AccImpl.get());
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                       AccImpl.get(),
                       /*size=*/0, ArgIndex);
    // TODO: offset, ranges...

    // Store copy of the accessor.
    MAccStorage.push_back(std::move(AccImpl));
  }

  template <typename T> void setArgHelper(int ArgIndex, T &&Arg) {
    void *StoredArg = (void *)storePlainArg(Arg);
    MArgs.emplace_back(detail::kernel_param_kind_t::kind_std_layout, StoredArg,
                       sizeof(T), ArgIndex);
  }

  //  TODO: implement when sampler class is ready
  //  void setArgHelper(int argIndex, sampler &&arg) {}

  void verifySyclKernelInvoc(const kernel &SyclKernel) {
    if (is_host()) {
      throw invalid_object_error(
          "This kernel invocation method cannot be used on the host");
    }
    if (SyclKernel.is_host()) {
      throw invalid_object_error("Invalid kernel type, OpenCL expected");
    }
  }

  // Make queue_impl class friend to be able to call finalize method.
  friend class detail::queue_impl;

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  // The method registers requirement to the memory. So, the command group has a
  // requirement to gain access to the given memory object before executing.
  template <typename DataT, int Dims, access::mode AccMode,
            access::target AccTarget>
  void
  require(accessor<DataT, Dims, AccMode, AccTarget, access::placeholder::true_t>
              Acc) {
    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
    // Add accessor to the list of requirements.
    MRequirements.emplace_back(AccImpl.get());
    // Store copy of the accessor.
    MAccStorage.push_back(std::move(AccImpl));
  }

  // OpenCL interoperability interface
  // Registers Arg passed as argument # ArgIndex.
  template <typename T> void set_arg(int ArgIndex, T &&Arg) {
    setArgHelper(ArgIndex, std::move(Arg));
  }

  // Registers pack of arguments(Args) with indexes starting from 0.
  template <typename... Ts> void set_args(Ts &&... Args) {
    setArgsHelper(0, std::move(Args)...);
  }

#ifdef __SYCL_DEVICE_ONLY__
  template <typename KernelName, typename KernelType>
  __attribute__((sycl_kernel)) void kernel_single_task(KernelType KernelFunc) {
    KernelFunc();
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           id<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type KernelFunc) {
    id<dimensions> global_id;

    detail::initGlobalInvocationId<dimensions>(global_id);

    KernelFunc(global_id);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           item<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type KernelFunc) {
    id<dimensions> global_id;
    range<dimensions> global_size;

    detail::initGlobalInvocationId<dimensions>(global_id);
    detail::initGlobalSize<dimensions>(global_size);

    item<dimensions, false> Item =
        detail::Builder::createItem<dimensions, false>(global_size, global_id);
    KernelFunc(Item);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           nd_item<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type KernelFunc) {
    range<dimensions> global_size;
    range<dimensions> local_size;
    id<dimensions> group_id;
    id<dimensions> global_id;
    id<dimensions> local_id;
    id<dimensions> global_offset;

    detail::initGlobalSize<dimensions>(global_size);
    detail::initWorkgroupSize<dimensions>(local_size);
    detail::initWorkgroupId<dimensions>(group_id);
    detail::initGlobalInvocationId<dimensions>(global_id);
    detail::initLocalInvocationId<dimensions>(local_id);
    detail::initGlobalOffset<dimensions>(global_offset);

    group<dimensions> Group = detail::Builder::createGroup<dimensions>(
        global_size, local_size, group_id);
    item<dimensions, true> globalItem =
        detail::Builder::createItem<dimensions, true>(global_size, global_id,
                                                      global_offset);
    item<dimensions, false> localItem =
        detail::Builder::createItem<dimensions, false>(local_size, local_id);
    nd_item<dimensions> Nd_item =
        detail::Builder::createNDItem<dimensions>(globalItem, localItem, Group);

    KernelFunc(Nd_item);
  }
#endif

  // The method stores lambda to the template-free object and initializes
  // kernel name, list of arguments and requirements using information from
  // integration header.
  template <typename KernelName, typename KernelType, int Dims,
            typename LambdaArgType = sycl::detail::lambda_arg_type<KernelType>>
  void StoreLambda(KernelType KernelFunc) {
    MHostKernel.reset(
        new detail::HostKernel<KernelType, LambdaArgType, Dims>(KernelFunc));

    using KI = sycl::detail::KernelInfo<KernelName>;
    // Empty name indicates that the compilation happens without integration
    // header, so don't perform things that require it.
    if (KI::getName() != "") {
      MArgs.clear();
      extractArgsAndReqsFromLambda(MHostKernel->getPtr(), KI::getNumParams(),
                                   &KI::getParamDesc(0));
      MKernelName = KI::getName();
      MOSModuleHandle = csd::OSUtil::getOSModuleHandle(KI::getName());
    }
  }

  // single_task version with a kernel represented as a lambda.
  template <typename KernelName, typename KernelType>
  void single_task(KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<KernelName>(KernelFunc);
#else
    MNDRDesc.set(range<1>{1});

    StoreLambda<KernelName, KernelType, /*Dims*/ 0, void>(KernelFunc);
    MCGType = detail::CG::KERNEL;
#endif
  }

  // single_task version with a kernel represented as a functor. Simply redirect
  // to the lambda-based form of invocation, setting kernel name type to the
  // functor type.
  template <typename KernelFunctorType>
  void single_task(KernelFunctorType KernelFunctor) {
    single_task<KernelFunctorType, KernelFunctorType>(KernelFunctor);
  }

  // parallel_for version with a kernel represented as a lambda + range that
  // specifies global size only.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version with a kernel represented as a functor + range that
  // specifies global size only. Simply redirect to the lambda-based form of
  // invocation, setting kernel name type to the functor type.
  template <typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, KernelType KernelFunc) {
    parallel_for<KernelType, KernelType, Dims>(NumWorkItems, KernelFunc);
  }

  // parallel_for version with a kernel represented as a lambda + range and
  // offset that specify global size and global offset correspondingly.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version with a kernel represented as a lambda + nd_range that
  // specifies global, local sizes and offset.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(ExecutionRange));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version with a kernel represented as a functor + nd_range that
  // specifies global, local sizes and offset. Simply redirect to the
  // lambda-based form of invocation, setting kernel name type to the functor
  // type.
  template <typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange, KernelType KernelFunc) {
    parallel_for<KernelType, KernelType, Dims>(ExecutionRange, KernelFunc);
  }

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> numWorkGroups,
  //                              WorkgroupFunctionType KernelFunc);

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> numWorkGroups,
  //                              range<dimensions> workGroupSize,
  //                              WorkgroupFunctionType KernelFunc);

  // single_task version with a kernel represented as a sycl::kernel.
  // The kernel invocation method has no functors and cannot be called on host.
  void single_task(kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MNDRDesc.set(range<1>{1});
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MCGType = detail::CG::KERNEL;
  }

  // parallel_for version with a kernel represented as a sycl::kernel + range
  // that specifies global size only. The kernel invocation method has no
  // functors and cannot be called on host.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NumWorkItems));
    MCGType = detail::CG::KERNEL;
  }

  // parallel_for version with a kernel represented as a sycl::kernel + range
  // and offset that specify global size and global offset correspondingly.
  // The kernel invocation method has no functors and cannot be called on host.
  template <int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> workItemOffset,
                    kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NumWorkItems), std::move(workItemOffset));
    MCGType = detail::CG::KERNEL;
  }

  // parallel_for version with a kernel represented as a sycl::kernel + nd_range
  // that specifies global, local sizes and offset. The kernel invocation
  // method has no functors and cannot be called on host.
  template <int Dims>
  void parallel_for(nd_range<Dims> NDRange, kernel SyclKernel) {
    verifySyclKernelInvoc(SyclKernel);
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    MNDRDesc.set(std::move(NDRange));
    MCGType = detail::CG::KERNEL;
  }

  // Note: the kernel invocation methods below are only planned to be added
  // to the spec as of v1.2.1 rev. 3, despite already being present in SYCL
  // conformance tests.

  // single_task version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise.
  template <typename KernelName, typename KernelType>
  void single_task(kernel SyclKernel, KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<KernelName>(KernelFunc);
#else
    MNDRDesc.set(range<1>{1});
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<KernelName, KernelType, /*Dims*/ 0, void>(
        std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // single_task version which takes two "kernels". One is a functor which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. Simply redirect to the lambda-based form of
  // invocation, setting kernel name type to the functor type.
  template <typename KernelType>
  void single_task(kernel SyclKernel, KernelType KernelFunc) {
    single_task<KernelType, KernelType>(SyclKernel, KernelFunc);
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. range argument specifies global size.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, kernel SyclKernel,
                    KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version which takes two "kernels". One is a functor which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. range argument specifies global size. Simply
  // redirect to the lambda-based form of invocation, setting kernel name type
  // to the functor type.
  template <typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, kernel SyclKernel,
                    KernelType KernelFunc) {
    parallel_for<KernelType, KernelType, Dims>(NumWorkItems, SyclKernel,
                                               KernelFunc);
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. range and id specify global size and offset.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(range<Dims> NumWorkItems, id<Dims> WorkItemOffset,
                    kernel SyclKernel, KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NumWorkItems), std::move(WorkItemOffset));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. nd_range specifies global, local size and offset.
  template <typename KernelName, typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> NDRange, kernel SyclKernel,
                    KernelType KernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, Dims>(KernelFunc);
#else
    MNDRDesc.set(std::move(NDRange));
    MSyclKernel = detail::getSyclObjImpl(std::move(SyclKernel));
    StoreLambda<KernelName, KernelType, Dims>(std::move(KernelFunc));
    MCGType = detail::CG::KERNEL;
#endif
  }

  // parallel_for version which takes two "kernels". One is a lambda which is
  // used if device, queue is bound to, is host device. Second is a sycl::kernel
  // which is used otherwise. nd_range specifies global, local size and offset.
  // Simply redirects to the lambda-based form of invocation, setting kernel
  // name type to the functor type.
  template <typename KernelType, int Dims>
  void parallel_for(nd_range<Dims> NDRange, kernel SyclKernel,
                    KernelType KernelFunc) {
    parallel_for<KernelType, KernelType, Dims>(NDRange, SyclKernel, KernelFunc);
  }

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> num_work_groups, kernel
  // SyclKernel, WorkgroupFunctionType KernelFunc);

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> num_work_groups,
  // range<dimensions> work_group_size, kernel SyclKernel, WorkgroupFunctionType
  // KernelFunc);

  // Explicit copy operations API

  // copy memory pointed by accessor to host memory pointed by shared_ptr
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  copy(accessor<T_Src, Dims, AccessMode, AccessTarget, IsPlaceholder> Src,
       shared_ptr_class<T_Dst> Dst) {
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Dst);
    T_Dst *RawDstPtr = Dst.get();
    copy(Src, RawDstPtr);
  }

  // copy memory pointer by shared_ptr to host memory pointed by accessor
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  copy(shared_ptr_class<T_Src> Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {
    // Make sure data shared_ptr points to is not released until we finish
    // work with it.
    MSharedPtrStorage.push_back(Src);
    T_Dst *RawSrcPtr = Src.get();
    copy(RawSrcPtr, Dst);
  }

  // copy memory pointed by accessor to host memory pointed by raw pointer
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  copy(accessor<T_Src, Dims, AccessMode, AccessTarget, IsPlaceholder> Src,
       T_Dst *Dst) {
    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Src.get_range();
      parallel_for< class __copyAcc2Ptr< T_Src, T_Dst, Dims, AccessMode,
                                         AccessTarget, IsPlaceholder>>
                                         (Range, [=](id<Dims> Index) {
        size_t LinearIndex = Index[0];
        for (int I = 1; I < Dims; ++I)
          LinearIndex += Range[I] * Index[I];
        ((T_Src *)Dst)[LinearIndex] = Src[Index];
      });

      return;
    }
    MCGType = detail::CG::COPY_ACC_TO_PTR;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = (void *)AccImpl.get();
    MDstPtr = (void *)Dst;
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  // copy memory pointed by raw pointer to host memory pointed by accessor
  template <typename T_Src, typename T_Dst, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  copy(const T_Src *Src,
       accessor<T_Dst, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst) {

    if (MIsHost) {
      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Dst.get_range();
      parallel_for< class __copyPtr2Acc< T_Src, T_Dst, Dims, AccessMode,
                                         AccessTarget, IsPlaceholder>>
                                         (Range, [=](id<Dims> Index) {
        size_t LinearIndex = Index[0];
        for (int I = 1; I < Dims; ++I)
          LinearIndex += Range[I] * Index[I];

        Dst[Index] = ((T_Dst *)Src)[LinearIndex];
      });

      return;
    }
    MCGType = detail::CG::COPY_PTR_TO_ACC;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MRequirements.push_back(AccImpl.get());
    MSrcPtr = (void *)Src;
    MDstPtr = (void *)AccImpl.get();
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImpl));
  }

  template <access::target AccessTarget>
  constexpr static bool isConstOrGlobal() {
    return AccessTarget == access::target::global_buffer ||
           AccessTarget == access::target::constant_buffer;
  }

  // copy memory pointed by accessor to the memory pointed by another accessor
  template <
      typename T_Src, int Dims_Src, access::mode AccessMode_Src,
      access::target AccessTarget_Src, typename T_Dst, int Dims_Dst,
      access::mode AccessMode_Dst, access::target AccessTarget_Dst,
      access::placeholder IsPlaceholder_Src = access::placeholder::false_t,
      access::placeholder IsPlaceholder_Dst = access::placeholder::false_t>

  typename std::enable_if<(isConstOrGlobal<AccessTarget_Src>() ||
                           isConstOrGlobal<AccessTarget_Dst>()),
                          void>::type
  copy(accessor<T_Src, Dims_Src, AccessMode_Src, AccessTarget_Src,
                IsPlaceholder_Src> Src,
       accessor<T_Dst, Dims_Dst, AccessMode_Dst, AccessTarget_Dst,
                IsPlaceholder_Dst> Dst) {

    if (MIsHost) {
      range<Dims_Src> Range = Dst.get_range();
      parallel_for< class __copyAcc2Acc< T_Src, Dims_Src, AccessMode_Src,
                                         AccessTarget_Src, T_Dst, Dims_Dst,
                                         AccessMode_Dst, AccessTarget_Dst,
                                         IsPlaceholder_Src,
                                         IsPlaceholder_Dst>>
                                         (Range, [=](id<Dims_Src> Index) {
        Dst[Index] = Src[Index];
      });

      return;
    }
    MCGType = detail::CG::COPY_ACC_TO_ACC;

    detail::AccessorBaseHost *AccBaseSrc = (detail::AccessorBaseHost *)&Src;
    detail::AccessorImplPtr AccImplSrc = detail::getSyclObjImpl(*AccBaseSrc);

    detail::AccessorBaseHost *AccBaseDst = (detail::AccessorBaseHost *)&Dst;
    detail::AccessorImplPtr AccImplDst = detail::getSyclObjImpl(*AccBaseDst);

    MRequirements.push_back(AccImplSrc.get());
    MRequirements.push_back(AccImplDst.get());
    MSrcPtr = AccImplSrc.get();
    MDstPtr = AccImplDst.get();
    // Store copy of accessor to the local storage to make sure it is alive
    // until we finish
    MAccStorage.push_back(std::move(AccImplSrc));
    MAccStorage.push_back(std::move(AccImplDst));
  }

  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  update_host(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder> Acc) {
    MCGType = detail::CG::UPDATE_HOST;

    detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Acc;
    detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

    MDstPtr = (void *)AccImpl.get();
    MRequirements.push_back(AccImpl.get());
    MAccStorage.push_back(std::move(AccImpl));
  }

  // Fill memory pointed by accessor with the pattern given.
  // If the operation is submitted to queue associated with OpenCL device and
  // accessor points to one dimensional memory object then use special type for
  // filling. Otherwise fill using regular kernel.
  template <typename T, int Dims, access::mode AccessMode,
            access::target AccessTarget,
            access::placeholder IsPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(AccessTarget == access::target::global_buffer ||
                           AccessTarget == access::target::constant_buffer),
                          void>::type
  fill(accessor<T, Dims, AccessMode, AccessTarget, IsPlaceholder> Dst,
       const T &Pattern) {
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    if (!MIsHost && Dims == 1) {
      MCGType = detail::CG::FILL;

      detail::AccessorBaseHost *AccBase = (detail::AccessorBaseHost *)&Dst;
      detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);

      MDstPtr = (void *)AccImpl.get();
      MRequirements.push_back(AccImpl.get());
      MAccStorage.push_back(std::move(AccImpl));

      MPattern.resize(sizeof(T));
      T *PatternPtr = (T *)MPattern.data();
      *PatternPtr = Pattern;
    } else {

      // TODO: Temporary implementation for host. Should be handled by memory
      // manger.
      range<Dims> Range = Dst.get_range();
      parallel_for<class __fill<T, Dims, AccessMode, AccessTarget,
                                IsPlaceholder>>(Range, [=](id<Dims> Index) {
        Dst[Index] = Pattern;
      });
    }
  }
};
} // namespace sycl
} // namespace cl
