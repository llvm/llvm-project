//==-------------- group.hpp --- SYCL work group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/device_event.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/range.hpp>
#include <stdexcept>
#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {
struct Builder;
} // namespace detail

template <int dimensions = 1> class group {
public:
  group() = delete;

  id<dimensions> get_id() const { return index; }

  size_t get_id(int dimension) const { return index[dimension]; }

  range<dimensions> get_global_range() const { return globalRange; }

  size_t get_global_range(int dimension) const {
    return globalRange[dimension];
  }

  range<dimensions> get_local_range() const { return localRange; }

  size_t get_local_range(int dimension) const { return localRange[dimension]; }

  range<dimensions> get_group_range() const { return localRange; }

  size_t get_group_range(int dimension) const { return localRange[dimension]; }

  size_t operator[](int dimension) const { return index[dimension]; }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 1), size_t>::type get_linear() const {
    range<dimensions> groupNum = globalRange / localRange;
    return index[0];
  }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 2), size_t>::type get_linear() const {
    range<dimensions> groupNum = globalRange / localRange;
    return index[1] * groupNum[0] + index[0];
  }

  template <int dims = dimensions>
  typename std::enable_if<(dims == 3), size_t>::type get_linear() const {
    range<dimensions> groupNum = globalRange / localRange;
    return (index[2] * groupNum[1] * groupNum[0]) + (index[1] * groupNum[0]) +
           index[0];
  }

  // template<typename workItemFunctionT>
  // void parallel_for_work_item(workItemFunctionT func) const;

  // template<typename workItemFunctionT>
  // void parallel_for_work_item(range<dimensions> flexibleRange,
  // workItemFunctionT func) const;

  /// Executes a work-group mem-fence with memory ordering on the local address
  /// space, global address space or both based on the value of \p accessSpace.
  template <access::mode accessMode = access::mode::read_write>
  void mem_fence(typename std::enable_if<
                     accessMode == access::mode::read ||
                     accessMode == access::mode::write ||
                     accessMode == access::mode::read_write,
                     access::fence_space>::type accessSpace =
                     access::fence_space::global_and_local) const {
    uint32_t flags = ::cl::__spirv::MemorySemantics::SequentiallyConsistent;
    switch (accessSpace) {
    case access::fence_space::global_space:
      flags |= cl::__spirv::MemorySemantics::CrossWorkgroupMemory;
      break;
    case access::fence_space::local_space:
      flags |= cl::__spirv::MemorySemantics::WorkgroupMemory;
      break;
    case access::fence_space::global_and_local:
    default:
      flags |= cl::__spirv::MemorySemantics::CrossWorkgroupMemory |
               cl::__spirv::MemorySemantics::WorkgroupMemory;
      break;
    }
    // TODO: currently, there is no good way in SPIRV to set the memory
    // barrier only for load operations or only for store operations.
    // The full read-and-write barrier is used and the template parameter
    // 'accessMode' is ignored for now. Either SPIRV or SYCL spec may be
    // changed to address this discrepancy between SPIRV and SYCL,
    // or if we decide that 'accessMode' is the important feature then
    // we can fix this later, for example, by using OpenCL 1.2 functions
    // read_mem_fence() and write_mem_fence().
    cl::__spirv::OpMemoryBarrier(cl::__spirv::Scope::Workgroup, flags);
  }

  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements) const {
    cl::__spirv::OpTypeEvent *e =
        cl::__spirv::OpGroupAsyncCopyGlobalToLocal<dataT>(
            cl::__spirv::Scope::Workgroup,
            dest.get(), src.get(), numElements, 1, 0);
    return device_event(e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements) const {
    cl::__spirv::OpTypeEvent *e =
        cl::__spirv::OpGroupAsyncCopyLocalToGlobal<dataT>(
            cl::__spirv::Scope::Workgroup,
            dest.get(), src.get(), numElements, 1, 0);
    return device_event(e);
  }

  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src,
                                     size_t numElements,
                                     size_t srcStride) const {
    cl::__spirv::OpTypeEvent *e =
        cl::__spirv::OpGroupAsyncCopyGlobalToLocal<dataT>(
            cl::__spirv::Scope::Workgroup,
            dest.get(), src.get(), numElements, srcStride, 0);
    return device_event(e);
  }

  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src,
                                     size_t numElements,
                                     size_t destStride) const {
    cl::__spirv::OpTypeEvent *e =
        cl::__spirv::OpGroupAsyncCopyLocalToGlobal<dataT>(
            cl::__spirv::Scope::Workgroup,
            dest.get(), src.get(), numElements, destStride, 0);
    return device_event(e);
  }

  template <typename... eventTN>
  void wait_for(eventTN... Events) const {
    waitForHelper(Events...);
  }

  bool operator==(const group<dimensions> &rhs) const {
    return (rhs.globalRange == this->globalRange) &&
           (rhs.localRange == this->localRange) && (rhs.index == this->index);
  }

  bool operator!=(const group<dimensions> &rhs) const {
    return !((*this) == rhs);
  }

private:
  range<dimensions> globalRange;
  range<dimensions> localRange;
  id<dimensions> index;

  void waitForHelper() const {}

  void waitForHelper(device_event Event) const {
    Event.wait();
  }

  template <typename T, typename... Ts>
  void waitForHelper(T E, Ts... Es) const {
    waitForHelper(E);
    waitForHelper(Es...);
  }

protected:
  friend class detail::Builder;
  group(const range<dimensions> &G, const range<dimensions> &L,
        const id<dimensions> &I)
      : globalRange(G), localRange(L), index(I) {}
};

} // namespace sycl
} // namespace cl
