//==----------- requirements.h ---------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/event.hpp>

namespace cl {
namespace sycl {
namespace detail {
template <typename T, int Dimensions, typename AllocatorT> class buffer_impl;
} // namespace detail

namespace detail {
class queue_impl;
class event_impl;
} // namespace detail
namespace simple_scheduler {

using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<cl::sycl::detail::event_impl>;

class BufferRequirement;
using BufferReqPtr = std::shared_ptr<BufferRequirement>;

class BufferRequirement {
public:
  BufferRequirement(void *UniqID, access::mode AccessMode,
                    access::target TargetType)
      : m_UniqID(UniqID), m_AccessMode(AccessMode), m_TargetType(TargetType) {}

  virtual ~BufferRequirement() = default;

  bool isBigger(const std::shared_ptr<BufferRequirement> &RHS) const {
    return m_UniqID > RHS->m_UniqID;
  }

  bool isSame(const std::shared_ptr<BufferRequirement> &RHS) const {
    return m_UniqID == RHS->m_UniqID;
  }

  void *getUniqID() const { return m_UniqID; }

  access::mode getAccessModeType() const { return m_AccessMode; }

  virtual cl_mem getCLMemObject() = 0;

  virtual void allocate(QueueImplPtr Queue,
                        std::vector<cl::sycl::event> DepEvents,
                        EventImplPtr Event) = 0;

  virtual void moveMemoryTo(QueueImplPtr Queue,
                            std::vector<cl::sycl::event> DepEvents,
                            EventImplPtr Event) = 0;

  virtual void fill(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                    EventImplPtr Event, void *Pattern, size_t PatternSize,
                    int Dim, size_t *Offset, size_t *Range) = 0;

  virtual void copy(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                    EventImplPtr Event, BufferReqPtr SrcReq, const int DimSrc,
                    const size_t *const SrcRange, const size_t *const SrcOffset,
                    const size_t *const DestOffset, const size_t SizeTySrc,
                    const size_t SizeSrc, const size_t *const BuffSrcRange) = 0;

  access::target getTargetType() const { return m_TargetType; }

  void addAccessMode(const access::mode AccessMode) {
    if (access::mode::read == m_AccessMode &&
        access::mode::read != AccessMode) {
      m_AccessMode = access::mode::read_write;
    } else if (access::mode::write == m_AccessMode &&
               (AccessMode != access::mode::write &&
                AccessMode != access::mode::discard_write)) {
      m_AccessMode = access::mode::read_write;
    }
  }

protected:
  void *m_UniqID;
  access::mode m_AccessMode;
  access::target m_TargetType;
};

template <typename T, int Dimensions, typename AllocatorT, access::mode Mode,
          access::target Target>
class BufferStorage : public BufferRequirement {
public:
  BufferStorage(
      typename cl::sycl::detail::buffer_impl<T, Dimensions, AllocatorT> &Buffer)
      : BufferRequirement(&Buffer, Mode, Target), m_Buffer(&Buffer) {}

  void allocate(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                EventImplPtr Event) override {
    assert(m_Buffer != nullptr && "BufferStorage::m_Buffer is nullptr");
    m_Buffer->allocate(std::move(Queue), std::move(DepEvents), std::move(Event),
                       Mode);
  }

  void moveMemoryTo(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
                    EventImplPtr Event) override {
    assert(m_Buffer != nullptr && "BufferStorage::m_Buffer is nullptr");
    m_Buffer->moveMemoryTo(std::move(Queue), std::move(DepEvents),
                           std::move(Event));
  }

  void fill(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, void *Pattern, size_t PatternSize, int Dim,
            size_t *Offset, size_t *Range) override {
    assert(m_Buffer != nullptr && "BufferStorage::m_Buffer is nullptr");
    m_Buffer->fill(std::move(Queue), std::move(DepEvents), std::move(Event),
                   std::move(Pattern), PatternSize, Dim, Offset, Range);
  }

  void copy(QueueImplPtr Queue, std::vector<cl::sycl::event> DepEvents,
            EventImplPtr Event, BufferReqPtr SrcReq, const int DimSrc,
            const size_t *const SrcRange, const size_t *const SrcOffset,
            const size_t *const DestOffset, const size_t SizeTySrc,
            const size_t SizeSrc, const size_t *const BuffSrcRange) override {
    assert(m_Buffer != nullptr && "BufferStorage::m_Buffer is nullptr");
    assert(SrcReq != nullptr && "BufferStorage::SrcReq is nullptr");

    m_Buffer->copy(std::move(Queue), std::move(DepEvents), std::move(Event),
                   std::move(SrcReq), DimSrc, SrcRange, SrcOffset, DestOffset,
                   SizeTySrc, SizeSrc, BuffSrcRange);
  }

  cl_mem getCLMemObject() override {
    assert(m_Buffer != nullptr && "BufferStorage::m_Buffer is nullptr");
    return m_Buffer->getOpenCLMem();
  }

private:
  cl::sycl::detail::buffer_impl<T, Dimensions, AllocatorT> *m_Buffer = nullptr;
};

struct classcomp {
  bool operator()(const BufferReqPtr &LHS, const BufferReqPtr &RHS) const {
    return LHS->isBigger(RHS);
  }
};

// Represents a call of set_arg made in the SYCL application
struct InteropArg {
  shared_ptr_class<void> m_Ptr;
  size_t m_Size;
  int m_ArgIndex;
  BufferReqPtr m_BufReq;

  InteropArg(shared_ptr_class<void> Ptr, size_t Size, int ArgIndex,
             BufferReqPtr BufReq)
      : m_Ptr(Ptr), m_Size(Size), m_ArgIndex(ArgIndex), m_BufReq(BufReq) {}
};

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
