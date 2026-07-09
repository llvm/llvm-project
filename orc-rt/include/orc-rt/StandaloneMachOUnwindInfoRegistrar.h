//===----------- StandaloneMachOUnwindInfoRegistrar.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone registration service for MachO unwind info via libunwind's
// unw_find_dynamic_unwind_sections mechanism.
//
// Note: Should not be used together with MachO-Platform, which provides its
// own unwind-info registration.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_STANDALONEMACHOUNWINDINFOREGISTRAR_H
#define ORC_RT_STANDALONEMACHOUNWINDINFOREGISTRAR_H

#include "orc-rt/Error.h"
#include "orc-rt/ExecutorAddress.h"
#include "orc-rt/SimpleSymbolTable.h"
#include "orc-rt/move_only_function.h"
#include "orc-rt/sps-ci/StandaloneMachOUnwindInfoRegistrarSPSCI.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace orc_rt {

/// Standalone MachO unwind-info registration via libunwind's
/// unw_find_dynamic_unwind_sections mechanism.
///
/// MachO-Platform provides unwind-info registration as part of its broader
/// MachO support, and should be preferred. Use this class only for MachO
/// executors that deliberately omit the full platform.
class StandaloneMachOUnwindInfoRegistrar {
  // Unit-test access to the private UnwindInfoMap. The fixture is defined in
  // orc-rt/unittests/StandaloneMachOUnwindInfoRegistrarTest.cpp.
  friend class UnwindInfoMapTest;

public:
  /// Represents a registration with libunwind that will enable registration
  /// of unwind info. Destruction will trigger a release of the registration,
  /// so clients must keep this alive throughout the lifetime of their
  /// Session.
  class Registration {
    friend class StandaloneMachOUnwindInfoRegistrar;

  public:
    Registration() = default;
    Registration(Registration &&Other);
    Registration &operator=(Registration &&Other);
    ~Registration();

  private:
    Registration(bool Active);
    bool Active = false;
  };

  static Expected<Registration>
  enable(SimpleSymbolTable &ST,
         SimpleSymbolTable::MutatorFn AddInterface =
             sps_ci::addStandaloneMachOUnwindInfoRegistrar);

  static Error registerSections(std::vector<ExecutorAddrRange> CodeRanges,
                                ExecutorAddr DSOBase,
                                ExecutorAddrRange DWARFEHFrame,
                                ExecutorAddrRange CompactUnwind);
  static Error deregisterSections(std::vector<ExecutorAddrRange> CodeRanges);

private:
  /// Describes the unwind-info sections associated with one or more code
  /// ranges. The struct layout is identical to libunwind's
  /// unw_dynamic_unwind_sections class so that the libunwind callback can
  /// populate its caller with a direct assignment.
  ///
  /// This struct must be kept in sync with libunwind's
  /// unw_dynamic_unwind_sections; if the two ever drift, the libunwind
  /// callback will silently corrupt unwind info.
  struct DynamicUnwindSections {
    uintptr_t DSOBase;
    uintptr_t DWARFSection;
    size_t DWARFSectionLength;
    uintptr_t CompactUnwindSection;
    size_t CompactUnwindSectionLength;
  };

  /// Interval map from code-address ranges to unwind-info section
  /// descriptors. Implementation detail; befriended for unit testing only.
  class UnwindInfoMap {
  public:
    /// Register the given code ranges with the given section info. Empty
    /// ranges are silently ignored. Overlapping ranges (with each other or
    /// with already-registered ranges) cause the call to fail; earlier
    /// successful inserts in this call remain registered.
    Error registerRanges(const std::vector<ExecutorAddrRange> &CodeRanges,
                         const DynamicUnwindSections &Info);

    /// Deregister the given code ranges. Returns an error if any range isn't
    /// found; earlier successful erasures in the failing call are not rolled
    /// back.
    Error deregisterRanges(const std::vector<ExecutorAddrRange> &CodeRanges);

    /// Look up DynamicUnwindSections for an address. Returns std::nullopt if
    /// no registered range contains the address.
    std::optional<DynamicUnwindSections> lookup(uintptr_t Addr) const;

  private:
    struct Entry {
      DynamicUnwindSections Info;
      uintptr_t End;
    };

    mutable std::mutex M;
    std::map<uintptr_t, Entry> Ranges;
  };

  // libunwind plumbing -- defined in StandaloneMachOUnwindInfoRegistrar.cpp.
  static UnwindInfoMap &unwindInfoMap();
  static int findUnwindInfoSections(uintptr_t Addr,
                                    DynamicUnwindSections *Info);
  static Error registerWithLibunwind();
  static void deregisterWithLibunwind();
};

} // namespace orc_rt

#endif // ORC_RT_STANDALONEMACHOUNWINDINFOREGISTRAR_H
