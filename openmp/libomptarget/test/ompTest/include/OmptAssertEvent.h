#ifndef OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H
#define OPENMP_LIBOMPTARGET_TEST_OMPTEST_OMPTASSERTEVENT_H

#include "InternalEvent.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

namespace omptest{

enum class AssertState { pass, fail };

struct OmptAssertEvent {
  static OmptAssertEvent ThreadBegin(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ThreadBegin());
  }

  static OmptAssertEvent ThreadEnd(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ThreadEnd());
  }

  static OmptAssertEvent ParallelBegin(int NumThreads,
                                       const std::string &Name) {
    auto EName = getName(Name);
    std::cout << "Creating new ParallelBegin Event (" << EName << ')'
              << std::endl;
    return OmptAssertEvent(EName, new internal::ParallelBegin(NumThreads));
  }

  static OmptAssertEvent ParallelEnd(const std::string &Name) {
    auto EName = getName(Name);
    std::cout << "Creating new ParallelEnd Event (" << EName << ')'
              << std::endl;
    return OmptAssertEvent(EName, new internal::ParallelEnd());
  }

  static OmptAssertEvent TaskCreate(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TaskCreate());
  }

  static OmptAssertEvent TaskSchedule(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TaskSchedule());
  }

  static OmptAssertEvent ImplicitTask(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ImplicitTask());
  }

  static OmptAssertEvent Target(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::Target());
  }

  static OmptAssertEvent TargetEmi(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetEmi());
  }

  /// Create a DataAlloc Event
  static OmptAssertEvent TargetDataOp(const std::string &Name) {
    std::cout << "Creating a new TargetDataOp event." << std::endl;
    return OmptAssertEvent(Name, new internal::TargetDataOp());
  }

  static OmptAssertEvent TargetDataOpEmi(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetDataOpEmi());
  }

  static OmptAssertEvent TargetSubmit(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetSubmit());
  }

  static OmptAssertEvent TargetSubmitEmi(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::TargetSubmitEmi());
  }

  static OmptAssertEvent ControlTool(std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::ControlTool());
  }

  static OmptAssertEvent DeviceInitialize(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceInitialize());
  }

  static OmptAssertEvent DeviceFinalize(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceFinalize());
  }

  static OmptAssertEvent DeviceLoad(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceLoad());
  }

  static OmptAssertEvent DeviceUnload(const std::string &Name) {
    auto EName = getName(Name);
    return OmptAssertEvent(EName, new internal::DeviceUnload());
  }

  /// Allow move construction (due to std::unique_ptr)
  OmptAssertEvent(OmptAssertEvent &&o) = default;
  OmptAssertEvent &operator=(OmptAssertEvent &&o) = default;

  std::string getEventName() const { return Name; }

  /// Make events comparable
  friend bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B);

private:
  OmptAssertEvent(const std::string &Name, internal::InternalEvent *IE)
      : Name(Name), TheEvent(IE) {}
  OmptAssertEvent(const OmptAssertEvent &o) = delete;

  static std::string getName(const std::string &Name) {
    std::string EName = Name;
    if (EName.empty())
      EName = "Auto Generated";
    return EName;
  }

  std::string Name;
  std::unique_ptr<internal::InternalEvent> TheEvent;
};

bool operator==(const OmptAssertEvent &A, const OmptAssertEvent &B) {
  assert(A.TheEvent.get() != nullptr && "A is valid");
  assert(B.TheEvent.get() != nullptr && "B is valid");

  return A.TheEvent->getType() == B.TheEvent->getType() &&
         A.TheEvent->equals(B.TheEvent.get());
}

} // namespace omptest

#endif
