//===- InternalEventCommon.h - Common internal event basics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides event types, and class/operator declaration macros.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_INTERNALEVENTCOMMON_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_INTERNALEVENTCOMMON_H

#include "omp-tools.h"

#include <cassert>
#include <string>

namespace omptest {

namespace internal {
/// Enum values are used for comparison of observed and asserted events
/// List is based on OpenMP 5.2 specification, table 19.2 (page 447)
enum class EventTy {
  None,                     // not part of OpenMP spec, used for implementation
  AssertionSyncPoint,       // not part of OpenMP spec, used for implementation
  AssertionSuspend,         // not part of OpenMP spec, used for implementation
  BufferRecord,             // not part of OpenMP spec, used for implementation
  BufferRecordDeallocation, // not part of OpenMP spec, used for implementation
  ThreadBegin,
  ThreadEnd,
  ParallelBegin,
  ParallelEnd,
  Work,
  Dispatch,
  TaskCreate,     // TODO: Implement
  Dependences,    // TODO: Implement
  TaskDependence, // TODO: Implement
  TaskSchedule,   // TODO: Implement
  ImplicitTask,   // TODO: Implement
  Masked,         // TODO: Implement
  SyncRegion,
  MutexAcquire, // TODO: Implement
  Mutex,        // TODO: Implement
  NestLock,     // TODO: Implement
  Flush,        // TODO: Implement
  Cancel,       // TODO: Implement
  DeviceInitialize,
  DeviceFinalize,
  DeviceLoad,
  DeviceUnload,
  BufferRequest,
  BufferComplete,
  TargetDataOp,
  TargetDataOpEmi,
  Target,
  TargetEmi,
  TargetSubmit,
  TargetSubmitEmi,
  ControlTool
};

/// Base event class
/// Offers default CTOR, DTOR and CTOR which assigns the actual event type.
struct InternalEvent {
  InternalEvent() : Type(EventTy::None) {}
  InternalEvent(EventTy T) : Type(T) {}
  virtual ~InternalEvent() = default;

  virtual bool equals(const InternalEvent *o) const {
    assert(false && "Base class implementation");
    return false;
  };

  virtual std::string toString() const {
    std::string S{"InternalEvent: Type="};
    S.append(std::to_string((uint32_t)Type));
    return S;
  }

  /// Identifying event type
  EventTy Type;
};

/// Specialize EventType member for each derived internal event type.
/// Effectively selecting an event type as initialization value.
template <typename EventType> struct EventTypeOf;

/// Actual definition macro for EventTypeOf.
#define event_type_trait(EvTy)                                                 \
  template <> struct EventTypeOf<EvTy> {                                       \
    static constexpr EventTy Value = EventTy::EvTy;                            \
  };

/// CRTP (Curiously Recurring Template Pattern) intermediate class
/// Adding a new event type can be achieved by inheriting from an EventBase
/// template instantiation of the new class' name, like this:
/// struct NewEventType : public EventBase<NewEventType>
template <typename Derived> class EventBase : public InternalEvent {
public:
  static constexpr EventTy EventType = EventTypeOf<Derived>::Value;
  EventBase() : InternalEvent(EventType) {}
  virtual ~EventBase() = default;

  /// Equals method to cast and dispatch to the specific class operator==
  virtual bool equals(const InternalEvent *o) const override {
    // Note: When the if-condition evaluates to true, the event types are
    // trivially identical. Otherwise, a cast to the Derived pointer would have
    // been impossible.
    if (const auto Other = dynamic_cast<const Derived *>(o))
      return operator==(*static_cast<const Derived *>(this), *Other);
    return false;
  }

  /// Basic toString method, which may be overridden with own implementations.
  virtual std::string toString() const override {
    std::string S{"EventBase: Type="};
    S.append(std::to_string((uint32_t)Type));
    return S;
  }
};

} // namespace internal

} // namespace omptest

#endif
