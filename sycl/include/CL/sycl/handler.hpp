//==-------- handler.hpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/scheduler/scheduler.h>
#include <CL/sycl/event.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

#include <functional>
#include <memory>
#include <type_traits>

template <typename T_src, int dim_src, cl::sycl::access::mode mode_src,
          cl::sycl::access::target tgt_src, typename T_dest, int dim_dest,
          cl::sycl::access::mode mode_dest, cl::sycl::access::target tgt_dest,
          cl::sycl::access::placeholder isPlaceholder_src,
          cl::sycl::access::placeholder isPlaceholder_dest>
class __copy;

template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder isPlaceholder>
class __update_host;

template <typename DataT, int Dimensions, cl::sycl::access::mode AccessMode,
          cl::sycl::access::target AccessTarget,
          cl::sycl::access::placeholder isPlaceholder>
class __fill;

namespace cl {
namespace sycl {

namespace csd = cl::sycl::detail;

// Forward declaration
class queue;

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;
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
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder,
          typename voidT>
class accessor_impl;

template <typename AllocatorT> class buffer_impl;
// Type inference of first arg from a lambda
// auto fun = [&](item a) { a; };
// lambda_arg_type<decltype(fun)> value; # value type is item

// Templated static declaration of a function whose single parameter is a
// pointer to a member function of type 'Func'. The member function must have
// 'RetType' return type, single argument of type 'Arg' and be declared with
// the 'const' qualifier.
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg) const);

// Non-const version of the above template to match functors whose 'operator()'
// is declared w/o the 'const' qualifier.
template <typename RetType, typename Func, typename Arg>
static Arg member_ptr_helper(RetType (Func::*)(Arg));

template <typename F>
decltype(member_ptr_helper(&F::operator())) argument_helper(F);

template <typename T>
using lambda_arg_type = decltype(argument_helper(std::declval<T>()));

} // namespace detail

template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor_base;

template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;

// 4.8.3 Command group handler class
class handler {
  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  friend class accessor;

  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder,
            typename voidT>
  friend class detail::accessor_impl;

  template <typename AllocatorT> friend class detail::buffer_impl;

  friend class detail::queue_impl;

protected:
  simple_scheduler::Node m_Node;
  bool isHost = false;
  unique_ptr_class<event> m_Finalized;
  // TODO: Obtain is host information from Queue when we split queue_impl
  // interface and implementation.
  handler(std::shared_ptr<detail::queue_impl> Queue, bool host)
      : m_Node(std::move(Queue)), isHost(host) {}

  event finalize() {
    if (!m_Finalized) {
      event *Event =
          new event(simple_scheduler::Scheduler::getInstance().addNode(
              std::move(m_Node)));
      m_Finalized.reset(Event);
    }
    return *m_Finalized.get();
  }

  ~handler() = default;

  bool is_host() { return isHost; }

  template <access::mode mode, access::target target, typename AllocatorT>
  void AddBufDep(detail::buffer_impl<AllocatorT> &Buf) {
    m_Node.addBufRequirement<mode, target>(Buf);
  }

  template <typename T, typename... Ts>
  void setArgsHelper(int ArgIndex, T &&Arg, Ts &&... Args) {
    set_arg(ArgIndex, std::move(Arg));
    setArgsHelper(++ArgIndex, std::move(Args)...);
  }

  void setArgsHelper(int ArgIndex) {}

  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  void setArgHelper(int argIndex, accessor<dataT, dimensions, accessMode,
                                           accessTarget, isPlaceholder> &&arg) {
    m_Node.addAccRequirement<dataT, dimensions, accessMode, accessTarget,
                             isPlaceholder>(std::move(arg), argIndex);
  }

  template <typename T> void setArgHelper(int argIndex, T &&arg) {
    using Type = typename std::remove_reference<T>::type;
    shared_ptr_class<void> Ptr = std::make_shared<Type>(std::move(arg));
    m_Node.addInteropArg(Ptr, sizeof(T), argIndex);
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

  // This dummy functor is passed to Node::addKernel in SYCL kernel
  // parallel_for invocation with range.
  template <int dimensions> struct DummyFunctor {
    void operator()(id<dimensions>) {}
  };

  // Method provides unified getting of the range from an accessor, because
  // 1 dimension accessor has no get_range method according to the SYCL
  // specification
  template <typename T, int dim, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  struct getAccessorRangeHelper {
    static range<dim>
    getAccessorRange(const accessor<T, dim, mode, tgt, isPlaceholder> &Acc) {
      return Acc.get_range();
    }
  };

  template <typename T, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder>
  struct getAccessorRangeHelper<T, 1, mode, tgt, isPlaceholder> {
    static range<1>
    getAccessorRange(const accessor<T, 1, mode, tgt, isPlaceholder> &Acc) {
      return range<1>(Acc.get_count());
    }
  };

public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  // template <typename dataT, int dimensions, access::mode accessMode,
  //           access::target accessTarget>
  // void require(accessor<dataT, dimensions, accessMode, accessTarget,
  //                       placeholder::true_t> acc);

  // OpenCL interoperability interface
  template <typename T> void set_arg(int argIndex, T &&arg) {
    setArgHelper(argIndex, std::move(arg));
  }

  template <typename... Ts> void set_args(Ts &&... args) {
    setArgsHelper(0, std::move(args)...);
  }

#ifdef __SYCL_DEVICE_ONLY__
  template <typename KernelName, typename KernelType>
  __attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
    kernelFunc();
  }
#endif

  // Kernel dispatch API
  // Kernel is represented as a lambda.
  template <typename KernelName, typename KernelType>
  void single_task(KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<KernelName>(kernelFunc);
#else
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node.addKernel(csd::OSUtil::getOSModuleHandle(KI::getName()),
                     KI::getName(), KI::getNumParams(), &KI::getParamDesc(0),
                     std::move(kernelFunc));
#endif
  }

  // Kernel is represented as a functor - simply redirect to the lambda-based
  // form of invocation, setting kernel name type to the functor type.
  template <typename KernelFunctorType>
  void single_task(KernelFunctorType KernelFunctor) {
    single_task<KernelFunctorType, KernelFunctorType>(KernelFunctor);
  }

#ifdef __SYCL_DEVICE_ONLY__
  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           id<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type kernelFunc) {
    id<dimensions> global_id;

    detail::initGlobalInvocationId<dimensions>(global_id);

    kernelFunc(global_id);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           item<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type kernelFunc) {
    id<dimensions> global_id;
    range<dimensions> global_size;

    detail::initGlobalInvocationId<dimensions>(global_id);
    detail::initGlobalSize<dimensions>(global_size);

    item<dimensions, false> Item =
        detail::Builder::createItem<dimensions, false>(global_size, global_id);
    kernelFunc(Item);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  __attribute__((sycl_kernel)) void kernel_parallel_for(
      typename std::enable_if<std::is_same<detail::lambda_arg_type<KernelType>,
                                           nd_item<dimensions>>::value &&
                                  (dimensions > 0 && dimensions < 4),
                              KernelType>::type kernelFunc) {
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

    kernelFunc(Nd_item);
  }
#endif

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node
        .addKernel<KernelType, dimensions, detail::lambda_arg_type<KernelType>>(
            csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
            KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
            numWorkItems);
#endif
  }

  // The version for a functor kernel.
  template <typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc) {
    parallel_for<KernelType, KernelType, dimensions>(numWorkItems, kernelFunc);
  }

  // The version with an offset
  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node
        .addKernel<KernelType, dimensions, detail::lambda_arg_type<KernelType>>(
            csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
            KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
            numWorkItems, workItemOffset);
#endif
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange,
                    KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node.addKernel<KernelType, dimensions>(
        csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
        KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
        executionRange);
#endif
  }

  // The version for a functor kernel.
  template <typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange,
                    KernelType kernelFunc) {

    parallel_for<KernelType, KernelType, dimensions>(executionRange,
                                                     kernelFunc);
  }

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> numWorkGroups,
  //                              WorkgroupFunctionType kernelFunc);

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> numWorkGroups,
  //                              range<dimensions> workGroupSize,
  //                              WorkgroupFunctionType kernelFunc);

  // The kernel invocation methods below have no functors and cannot be
  // called on host.
  // TODO current workaround passes dummy functors to Node::addKernel.
  // A better way of adding kernels to scheduler if they cannot be run on host
  // would be preferrable.
  void single_task(kernel syclKernel) {
    verifySyclKernelInvoc(syclKernel);
    std::function<void()> DummyLambda = []() {};
    m_Node.addKernel(nullptr,
                     syclKernel.get_info<info::kernel::function_name>(), 0,
                     nullptr, std::move(DummyLambda), syclKernel.get());
  }

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel) {
    verifySyclKernelInvoc(syclKernel);
    m_Node.addKernel<DummyFunctor<dimensions>, dimensions, id<dimensions>>(
        nullptr, syclKernel.get_info<info::kernel::function_name>(), 0, nullptr,
        DummyFunctor<dimensions>(), numWorkItems, syclKernel.get());
  }

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, kernel syclKernel) {
    verifySyclKernelInvoc(syclKernel);
    m_Node.addKernel<DummyFunctor<dimensions>, dimensions, id<dimensions>>(
        nullptr, syclKernel.get_info<info::kernel::function_name>(), 0, nullptr,
        DummyFunctor<dimensions>(), numWorkItems, workItemOffset,
        syclKernel.get());
  }

  template <int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel) {
    verifySyclKernelInvoc(syclKernel);
    m_Node.addKernel(
        nullptr, syclKernel.get_info<info::kernel::function_name>(), 0, nullptr,
        [](nd_item<dimensions>) {}, ndRange, syclKernel.get());
  }

  // Note: the kernel invocation methods below are only planned to be added
  // to the spec as of v1.2.1 rev. 3, despite already being present in SYCL
  // conformance tests.

  template <typename KernelName, typename KernelType>
  void single_task(kernel syclKernel, KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<KernelName>(kernelFunc);
#else
    cl_kernel clKernel = nullptr;
    if (!is_host()) {
      clKernel = syclKernel.get();
    }
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node.addKernel(csd::OSUtil::getOSModuleHandle(KI::getName()),
                     KI::getName(), KI::getNumParams(), &KI::getParamDesc(0),
                     std::move(kernelFunc), clKernel);
#endif
  }

  // The version for a functor kernel.
  template <typename KernelType>
  void single_task(kernel syclKernel, KernelType kernelFunc) {
    single_task<KernelType, KernelType>(syclKernel, kernelFunc);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel,
                    KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    cl_kernel clKernel = nullptr;
    if (!is_host()) {
      clKernel = syclKernel.get();
    }
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node
        .addKernel<KernelType, dimensions, detail::lambda_arg_type<KernelType>>(
            csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
            KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
            numWorkItems, clKernel);
#endif
  }

  // The version for a functor kernel.
  template <typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel,
                    KernelType kernelFunc) {

    parallel_for<KernelType, KernelType, dimensions>(numWorkItems, syclKernel,
                                                     kernelFunc);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, kernel syclKernel,
                    KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    cl_kernel clKernel = nullptr;
    if (!is_host()) {
      clKernel = syclKernel.get();
    }
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node
        .addKernel<KernelType, dimensions, detail::lambda_arg_type<KernelType>>(
            csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
            KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
            numWorkItems, workItemOffset, clKernel);
#endif
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel,
                    KernelType kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<KernelName, KernelType, dimensions>(kernelFunc);
#else
    cl_kernel clKernel = nullptr;
    if (!is_host()) {
      clKernel = syclKernel.get();
    }
    using KI = cl::sycl::detail::KernelInfo<KernelName>;
    m_Node.addKernel<KernelType, dimensions>(
        csd::OSUtil::getOSModuleHandle(KI::getName()), KI::getName(),
        KI::getNumParams(), &KI::getParamDesc(0), std::move(kernelFunc),
        ndRange, clKernel);
#endif
  }

  // The version for a functor kernel.
  template <typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel,
                    KernelType kernelFunc) {
    parallel_for<KernelType, KernelType, dimensions>(ndRange, syclKernel,
                                                     kernelFunc);
  }

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> num_work_groups, kernel
  // syclKernel, WorkgroupFunctionType kernelFunc);

  // template <typename KernelName, typename WorkgroupFunctionType, int
  // dimensions>
  // void parallel_for_work_group(range<dimensions> num_work_groups,
  // range<dimensions> work_group_size, kernel syclKernel, WorkgroupFunctionType
  // kernelFunc);

  // Explicit copy operations API
  template <typename T_src, typename T_dest, int dim, access::mode mode,
            access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  copy(accessor<T_src, dim, mode, tgt, isPlaceholder> src,
       shared_ptr_class<T_dest> dest) {
    range<dim> Range =
        getAccessorRangeHelper<T_src, dim, mode, tgt,
                               isPlaceholder>::getAccessorRange(src);
    // TODO use buffer_allocator when it is possible
    buffer<T_src, dim, std::allocator<char>> Buffer(
        (shared_ptr_class<T_src>)dest, Range,
        {property::buffer::use_host_ptr()});
    accessor<T_src, dim, access::mode::write, access::target::global_buffer,
             access::placeholder::false_t>
        DestAcc(Buffer, *this);
    copy(src, DestAcc);
  }

  template <typename T_src, typename T_dest, int dim, access::mode mode,
            access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  copy(shared_ptr_class<T_src> src,
       accessor<T_dest, dim, mode, tgt, isPlaceholder> dest) {
    range<dim> Range =
        getAccessorRangeHelper<T_dest, dim, mode, tgt,
                               isPlaceholder>::getAccessorRange(dest);
    // TODO use buffer_allocator when it is possible
    buffer<T_dest, dim, std::allocator<char>> Buffer(
        (shared_ptr_class<T_dest>)src, Range,
        {property::buffer::use_host_ptr()});
    accessor<T_dest, dim, access::mode::read, access::target::global_buffer,
             access::placeholder::false_t>
        SrcAcc(Buffer, *this);
    copy(SrcAcc, dest);
  }

  template <typename T_src, typename T_dest, int dim, access::mode mode,
            access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  copy(accessor<T_src, dim, mode, tgt, isPlaceholder> src, T_dest *dest) {
    range<dim> Range =
        getAccessorRangeHelper<T_src, dim, mode, tgt,
                               isPlaceholder>::getAccessorRange(src);
    // TODO use buffer_allocator when it is possible
    buffer<T_src, dim, std::allocator<char>> Buffer(
        (T_src *)dest, Range, {property::buffer::use_host_ptr()});
    accessor<T_src, dim, access::mode::write, access::target::global_buffer,
             access::placeholder::false_t>
        DestAcc(Buffer, *this);
    copy(src, DestAcc);
  }

  template <typename T_src, typename T_dest, int dim, access::mode mode,
            access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  copy(const T_src *src, accessor<T_dest, dim, mode, tgt, isPlaceholder> dest) {
    range<dim> Range =
        getAccessorRangeHelper<T_dest, dim, mode, tgt,
                               isPlaceholder>::getAccessorRange(dest);
    // TODO use buffer_allocator when it is possible
    buffer<T_dest, dim, std::allocator<char>> Buffer(
        (T_dest *)src, Range, {property::buffer::use_host_ptr()});
    accessor<T_dest, dim, access::mode::read, access::target::global_buffer,
             access::placeholder::false_t>
        SrcAcc(Buffer, *this);
    copy(SrcAcc, dest);
  }

  template <
      typename T_src, int dim_src, access::mode mode_src,
      access::target tgt_src, typename T_dest, int dim_dest,
      access::mode mode_dest, access::target tgt_dest,
      access::placeholder isPlaceholder_src = access::placeholder::false_t,
      access::placeholder isPlaceholder_dest = access::placeholder::false_t>
  typename std::enable_if<((tgt_src == access::target::global_buffer ||
                            tgt_src == access::target::constant_buffer) &&
                           (tgt_dest == access::target::global_buffer ||
                            tgt_dest == access::target::constant_buffer)),
                          void>::type
  copy(accessor<T_src, dim_src, mode_src, tgt_src, isPlaceholder_src> src,
       accessor<T_dest, dim_dest, mode_dest, tgt_dest, isPlaceholder_dest>
           dest) {
    if (isHost) {
      range<dim_src> Range =
          getAccessorRangeHelper<T_src, dim_src, mode_src, tgt_src,
                                 isPlaceholder_src>::getAccessorRange(src);
      parallel_for<
        class __copy<
          T_src, dim_src, mode_src, tgt_src, T_dest, dim_dest, mode_dest,
          tgt_dest, isPlaceholder_src, isPlaceholder_dest>
        >(Range, [=](id<dim_src> Index) {
        dest[Index] = src[Index];
      });
    } else {
#ifndef __SYCL_DEVICE_ONLY__
      m_Node.addExplicitMemOp<>(src, dest);
#endif
    }
    finalize();
    // force wait.
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  update_host(accessor<T, dim, mode, tgt, isPlaceholder> acc) {
#ifndef __SYCL_DEVICE_ONLY__
    assert(!m_Finalized && "The final event of this handler must not be set.");
    event *Event = new event;
    simple_scheduler::Scheduler::getInstance().updateHost(acc, *Event);
    m_Finalized.reset(Event);
#endif
  }

  template <typename T, int dim, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  typename std::enable_if<(tgt == access::target::global_buffer ||
                           tgt == access::target::constant_buffer),
                          void>::type
  fill(accessor<T, dim, mode, tgt, isPlaceholder> dest, const T &src) {
    // TODO add check:T must be an integral scalar value or a SYCL vector type
    if (!isHost && dim == 1) {
#ifndef __SYCL_DEVICE_ONLY__
      m_Node.addExplicitMemOp<>(dest, src);
#endif
    } else {
      // TODO multidimensional case with offset is not supported.
      // Fix it when parallel_for with offset is implemented
      range<dim> Range =
          getAccessorRangeHelper<T, dim, mode, tgt,
                                 isPlaceholder>::getAccessorRange(dest);
      parallel_for<class __fill<T, dim, mode, tgt, isPlaceholder>>(Range,
        [=](id<dim> Index) {
        dest[Index] = src;
      });
    }
  }
};
} // namespace sycl
} // namespace cl
