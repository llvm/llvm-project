//==----------- commands.h -------------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cassert>
#include <climits>
#include <iostream>
#include <utility>

#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {
class queue_impl;
}
namespace simple_scheduler {
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;
using EventImplPtr = std::shared_ptr<cl::sycl::detail::event_impl>;
namespace csd = cl::sycl::detail;

class Command {
public:
  enum CommandType { RUN_KERNEL, MOVE_MEMORY, ALLOCA, COPY, FILL };

  Command(CommandType Type, QueueImplPtr Queue);

  CommandType getType() const { return m_Type; }

  size_t getID() const { return m_ID; }

  void addDep(std::shared_ptr<Command> Dep, BufferReqPtr Buf) {
    m_Deps.emplace_back(std::move(Dep), std::move(Buf));
  }

  void addInteropArg(InteropArg Arg) { m_InteropArgs.push_back(Arg); }

  cl::sycl::event enqueue(std::vector<cl::sycl::event> DepEvents) {
    bool Expected = false;
    if (m_Enqueued.compare_exchange_strong(Expected, true)) {
      enqueueImp(std::move(DepEvents), detail::getSyclObjImpl(m_Event));
    }
    return m_Event;
  }

  bool isEnqueued() const { return m_Enqueued; }

  virtual void dump() const = 0;

  virtual void print(std::ostream &Stream) const = 0;

  virtual void printDot(std::ostream &Stream) const = 0;

  QueueImplPtr getQueue() const { return m_Queue; }

  cl::sycl::event getEvent() const { return m_Event; }

  std::shared_ptr<Command> getDepCommandForReqBuf(const BufferReqPtr &Buf) {
    for (const auto &Dep : m_Deps) {
      if (Dep.second->isSame(Buf)) {
        return Dep.first;
      }
    }
    return nullptr;
  }

  cl::sycl::access::mode getAccessModeForReqBuf(const BufferReqPtr &Buf) const {
    for (const auto &Dep : m_Deps) {
      if (Dep.second->isSame(Buf)) {
        return Dep.second->getAccessModeType();
      }
    }
    throw cl::sycl::runtime_error("Buffer not found.");
  }

  void replaceDepCommandForReqBuf(const BufferReqPtr &Buf,
                                  std::shared_ptr<Command> NewCommand) {
    for (auto &Dep : m_Deps) {
      if (Dep.second->isSame(Buf)) {
        Dep.first = std::move(NewCommand);
        return;
      }
    }
    throw cl::sycl::runtime_error("Buffer not found.");
  }

  std::vector<std::pair<std::shared_ptr<Command>, BufferReqPtr>>
  getDependencies() {
    return m_Deps;
  }

  void removeAllDeps() { m_Deps.clear(); }

  virtual ~Command() = default;

private:
  virtual void enqueueImp(std::vector<cl::sycl::event> DepEvents,
                          EventImplPtr Event) = 0;

  CommandType m_Type;
  size_t m_ID;
  cl::sycl::event m_Event;
  std::atomic<bool> m_Enqueued;

protected:
  QueueImplPtr m_Queue;
  std::vector<std::pair<std::shared_ptr<Command>, BufferReqPtr>> m_Deps;
  std::vector<InteropArg> m_InteropArgs;
};

using CommandPtr = std::shared_ptr<Command>;

class MemMoveCommand : public Command {
public:
  MemMoveCommand(BufferReqPtr Buf, QueueImplPtr SrcQueue, QueueImplPtr DstQueue,
                 cl::sycl::access::mode mode)
      : Command(Command::MOVE_MEMORY, std::move(DstQueue)),
        m_Buf(std::move(Buf)), m_AccessMode(mode),
        m_SrcQueue(std::move(SrcQueue)) {}

  access::mode getAccessModeType() const { return m_Buf->getAccessModeType(); }
  void printDot(std::ostream &Stream) const override;
  void print(std::ostream &Stream) const override;
  void dump() const override { print(std::cout); }

private:
  void enqueueImp(std::vector<cl::sycl::event> DepEvents,
                  EventImplPtr Event) override;
  BufferReqPtr m_Buf = nullptr;
  cl::sycl::access::mode m_AccessMode;
  QueueImplPtr m_SrcQueue;
};

class AllocaCommand : public Command {
public:
  AllocaCommand(BufferReqPtr Buf, QueueImplPtr Queue,
                cl::sycl::access::mode mode)
      : Command(Command::ALLOCA, std::move(Queue)), m_Buf(std::move(Buf)),
        m_AccessMode(mode) {}

  access::mode getAccessModeType() const { return m_Buf->getAccessModeType(); }
  void printDot(std::ostream &Stream) const override;
  void print(std::ostream &Stream) const override;
  void dump() const override { print(std::cout); }

private:
  void enqueueImp(std::vector<cl::sycl::event> DepEvents,
                  EventImplPtr Event) override;
  BufferReqPtr m_Buf = nullptr;
  cl::sycl::access::mode m_AccessMode;
};

template <typename KernelType, int Dimensions, typename RangeType,
          typename KernelArgType, bool SingleTask = false>
class ExecuteKernelCommand : public Command {
public:
  ExecuteKernelCommand(KernelType &HostKernel, const std::string KernelName,
                       const unsigned int KernelArgsNum,
                       const detail::kernel_param_desc_t *KernelArgs,
                       RangeType workItemsRange, QueueImplPtr Queue,
                       cl_kernel ClKernel, id<Dimensions> workItemOffset = {})
      : Command(Command::RUN_KERNEL, std::move(Queue)),
        m_KernelName(KernelName), m_KernelArgsNum(KernelArgsNum),
        m_KernelArgs(KernelArgs), m_WorkItemsRange(workItemsRange),
        m_WorkItemsOffset(workItemOffset), m_HostKernel(HostKernel),
        m_ClKernel(ClKernel) {}

  void printDot(std::ostream &Stream) const override;
  void print(std::ostream &Stream) const override;
  void dump() const override { print(std::cout); }

private:
  cl_kernel createKernel(const std::string &KernelName,
                         cl_program Program) const;

  template <bool STask = SingleTask, int Dims = Dimensions,
            typename = typename std::enable_if<STask == true>::type>
  void runOnHost() {
    m_HostKernel();
  }

  template <bool STask = SingleTask, int Dims = Dimensions>
  typename std::enable_if<
      (STask == false) && (Dims > 0 && Dims < 4) &&
          std::is_same<RangeType, range<Dimensions>>::value &&
          std::is_same<KernelArgType, id<Dimensions>>::value,
      void>::type
  runOnHost() {
    const size_t ZMax = (Dims > 2) ? m_WorkItemsRange[2] : 1;
    const size_t YMax = (Dims > 1) ? m_WorkItemsRange[1] : 1;
    size_t XYZ[3];
    for (XYZ[2] = 0; XYZ[2] < ZMax; ++XYZ[2]) {
      for (XYZ[1] = 0; XYZ[1] < YMax; ++XYZ[1]) {
        for (XYZ[0] = 0; XYZ[0] < m_WorkItemsRange[0]; ++XYZ[0]) {
          id<Dims> ID;
          for (int I = 0; I < Dims; ++I) {
            ID[I] = XYZ[I];
          }
          m_HostKernel(ID);
        }
      }
    }
  }

  template <bool STask = SingleTask, int Dims = Dimensions>
  typename std::enable_if<
      (STask == false) && (Dims > 0 && Dims < 4) &&
          std::is_same<RangeType, range<Dimensions>>::value &&
          (std::is_same<KernelArgType, item<Dimensions, false>>::value ||
           std::is_same<KernelArgType, item<Dimensions, true>>::value),
      void>::type
  runOnHost() {
    const size_t ZMax = (Dims > 2) ? m_WorkItemsRange[2] : 1;
    const size_t YMax = (Dims > 1) ? m_WorkItemsRange[1] : 1;
    size_t XYZ[3];
    for (XYZ[2] = 0; XYZ[2] < ZMax; ++XYZ[2]) {
      for (XYZ[1] = 0; XYZ[1] < YMax; ++XYZ[1]) {
        for (XYZ[0] = 0; XYZ[0] < m_WorkItemsRange[0]; ++XYZ[0]) {
          id<Dims> ID;
          range<Dims> Range;
          for (int I = 0; I < Dims; ++I) {
            ID[I] = XYZ[I];
            Range[I] = m_WorkItemsRange[I];
          }
          item<Dims, false> Item =
              detail::Builder::createItem<Dims, false>(Range, ID);
          m_HostKernel(Item);
        }
      }
    }
  }

  template <bool STask = SingleTask, int Dims = Dimensions>
  typename std::enable_if<
      (STask == false) && (Dims > 0 && Dims < 4) &&
          std::is_same<RangeType, nd_range<Dimensions>>::value,
      void>::type
  runOnHost() {
    // TODO add offset logic

    const id<3> GlobalSize{
        m_WorkItemsRange.get_global_range()[0],
        ((Dims > 1) ? m_WorkItemsRange.get_global_range()[1] : 1),
        ((Dims > 2) ? m_WorkItemsRange.get_global_range()[2] : 1)};
    const id<3> LocalSize{
        m_WorkItemsRange.get_local_range()[0],
        ((Dims > 1) ? m_WorkItemsRange.get_local_range()[1] : 1),
        ((Dims > 2) ? m_WorkItemsRange.get_local_range()[2] : 1)};
    id<3> GroupSize;
    for (int I = 0; I < 3; ++I) {
      GroupSize[I] = GlobalSize[I] / LocalSize[I];
    }

    size_t GlobalXYZ[3];
    for (GlobalXYZ[2] = 0; GlobalXYZ[2] < GroupSize[2]; ++GlobalXYZ[2]) {
      for (GlobalXYZ[1] = 0; GlobalXYZ[1] < GroupSize[1]; ++GlobalXYZ[1]) {
        for (GlobalXYZ[0] = 0; GlobalXYZ[0] < GroupSize[0]; ++GlobalXYZ[0]) {
          id<Dims> ID;
          for (int I = 0; I < Dims; ++I) {
            ID[I] = GlobalXYZ[I];
          }
          group<Dims> Group = detail::Builder::createGroup<Dims>(
              m_WorkItemsRange.get_global_range(),
              m_WorkItemsRange.get_local_range(), ID);
          size_t LocalXYZ[3];
          for (LocalXYZ[2] = 0; LocalXYZ[2] < LocalSize[2]; ++LocalXYZ[2]) {
            for (LocalXYZ[1] = 0; LocalXYZ[1] < LocalSize[1]; ++LocalXYZ[1]) {
              for (LocalXYZ[0] = 0; LocalXYZ[0] < LocalSize[0]; ++LocalXYZ[0]) {
                id<Dims> GlobalID;
                id<Dims> LocalID;
                for (int I = 0; I < Dims; ++I) {
                  GlobalID[I] = GlobalXYZ[I] * LocalSize[I] + LocalXYZ[I];
                  LocalID[I] = LocalXYZ[I];
                }
                const item<Dims, true> GlobalItem =
                    detail::Builder::createItem<Dims, true>(
                        m_WorkItemsRange.get_global_range(), GlobalID,
                        m_WorkItemsRange.get_offset());
                const item<Dims, false> LocalItem =
                    detail::Builder::createItem<Dims, false>(
                        m_WorkItemsRange.get_local_range(), LocalID);
                nd_item<Dims> NDItem = detail::Builder::createNDItem<Dims>(
                    GlobalItem, LocalItem, Group);
                m_HostKernel(NDItem);
              }
            }
          }
        }
      }
    }
  }

  void executeKernel(std::vector<cl::sycl::event> DepEvents,
                     EventImplPtr Event);

  void enqueueImp(std::vector<cl::sycl::event> DepEvents,
                  EventImplPtr Event) override {
    executeKernel(std::move(DepEvents), std::move(Event));
  }

  template <typename R = RangeType>
  typename std::enable_if<std::is_same<R, range<Dimensions>>::value,
                          cl_event>::type
  runEnqueueNDRangeKernel(cl_command_queue &EnvQueue, cl_kernel &Kernel,
                          std::vector<cl_event> CLEvents);

  template <typename R = RangeType>
  typename std::enable_if<std::is_same<R, nd_range<Dimensions>>::value,
                          cl_event>::type
  runEnqueueNDRangeKernel(cl_command_queue &EnvQueue, cl_kernel &Kernel,
                          std::vector<cl_event> CLEvents);

  std::string m_KernelName;
  const unsigned int m_KernelArgsNum;
  const detail::kernel_param_desc_t *m_KernelArgs;
  RangeType m_WorkItemsRange;
  id<Dimensions> m_WorkItemsOffset;
  KernelType m_HostKernel;
  cl_kernel m_ClKernel;
};

template <typename T, int Dim> class FillCommand : public Command {
public:
  FillCommand(BufferReqPtr Buf, T Pattern, QueueImplPtr Queue, range<Dim> Range,
              id<Dim> Offset)
      : Command(Command::FILL, std::move(Queue)), m_Buf(std::move(Buf)),
        m_Pattern(std::move(Pattern)), m_Offset(std::move(Offset)),
        m_Range(std::move(Range)) {}

  access::mode getAccessModeType() const { return m_Buf->getAccessModeType(); }
  void printDot(std::ostream &Stream) const override;
  void print(std::ostream &Stream) const override;
  void dump() const override { print(std::cout); }

private:
  void enqueueImp(std::vector<cl::sycl::event> DepEvents, EventImplPtr Event) {
    assert(nullptr != m_Buf && "Buf is nullptr");
    m_Buf->fill(m_Queue, std::move(DepEvents), std::move(Event), &m_Pattern,
                sizeof(T), Dim, &m_Offset[0], &m_Range[0]);
  }
  BufferReqPtr m_Buf = nullptr;
  T m_Pattern;
  id<Dim> m_Offset;
  range<Dim> m_Range;
};

template <int DimSrc, int DimDest> class CopyCommand : public Command {
public:
  CopyCommand(BufferReqPtr BufSrc, BufferReqPtr BufDest, QueueImplPtr Queue,
              range<DimSrc> SrcRange, id<DimSrc> SrcOffset,
              id<DimDest> DestOffset, size_t SizeTySrc, size_t SizeSrc,
              range<DimSrc> BuffSrcRange)
      : Command(Command::COPY, std::move(Queue)), m_BufSrc(std::move(BufSrc)),
        m_BufDest(std::move(BufDest)), m_SrcRange(std::move(SrcRange)),
        m_SrcOffset(std::move(SrcOffset)), m_DestOffset(std::move(DestOffset)),
        m_SizeTySrc(SizeTySrc), m_SizeSrc(SizeSrc),
        m_BuffSrcRange(BuffSrcRange) {}

  access::mode getAccessModeType() const {
    return m_BufDest->getAccessModeType();
  }
  void printDot(std::ostream &Stream) const override;
  void print(std::ostream &Stream) const override;
  void dump() const override { print(std::cout); }

private:
  void enqueueImp(std::vector<cl::sycl::event> DepEvents, EventImplPtr Event) {
    assert(nullptr != m_BufSrc && "m_BufSrc is nullptr");
    assert(nullptr != m_BufDest && "m_BufDest is nullptr");
    m_BufDest->copy(m_Queue, std::move(DepEvents), std::move(Event), m_BufSrc,
                    DimSrc, &m_SrcRange[0], &m_SrcOffset[0], &m_DestOffset[0],
                    m_SizeTySrc, m_SizeSrc, &m_BuffSrcRange[0]);
  }
  BufferReqPtr m_BufSrc = nullptr;
  BufferReqPtr m_BufDest = nullptr;
  range<DimSrc> m_SrcRange;
  id<DimSrc> m_SrcOffset;
  id<DimDest> m_DestOffset;
  size_t m_SizeTySrc;
  size_t m_SizeSrc;
  range<DimSrc> m_BuffSrcRange;
};

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
