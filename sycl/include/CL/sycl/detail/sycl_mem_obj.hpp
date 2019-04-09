//==------------ sycl_mem_obj.hpp - SYCL standard header file --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>

#include <memory>

namespace cl {
namespace sycl {

namespace detail {

class event_impl;
class context_impl;

using EventImplPtr = std::shared_ptr<detail::event_impl>;
using ContextImplPtr = std::shared_ptr<detail::context_impl>;

// The class serves as a base for all SYCL memory objects.
class SYCLMemObjT {
public:
  enum MemObjType { BUFFER, IMAGE };

  virtual MemObjType getType() const = 0;

  // The method allocates memory for the SYCL memory object. The size of
  // allocation will be taken from the size of SYCL memory object.
  // If the memory returned cannot be used right away InteropEvent will
  // point to event that should be waited before using the memory.
  // InitFromUserData indicates that the returned memory should be intialized
  // with the data provided by user(if any). Usually it should happen on the
  // first allocation of memory for the buffer.
  // Method returns a pointer to host allocation if Context is host one and
  // cl_mem obect if not.
  virtual void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                            cl_event &InteropEvent) = 0;

  // Should be used for memory object created without use_host_ptr property.
  virtual void *allocateHostMem() = 0;

  // Ptr must be a pointer returned by allocateMem for the same context.
  // If Context is a device context and Ptr is a host pointer exception will be
  // thrown. And it's undefined behaviour if Context is a host context and Ptr
  // is a device pointer.
  virtual void releaseMem(ContextImplPtr Context, void *Ptr) = 0;

  // Ptr must be a pointer returned by allocateHostMem.
  virtual void releaseHostMem(void *Ptr) = 0;
};

} // namespace detail
} // namespace sycl
} // namespace detail
