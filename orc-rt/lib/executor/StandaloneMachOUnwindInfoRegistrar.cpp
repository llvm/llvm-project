//===- StandaloneMachOUnwindInfoRegistrar.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone registration service for MachO unwind info.
//
// Note: Should not be used together with MachO-Platform, which provides its
// own unwind-info registration.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/StandaloneMachOUnwindInfoRegistrar.h"
#include "orc-rt/Compiler.h"

#include <mutex>

// --- From libunwind/src/libunwind_ext.h ---
struct unw_dynamic_unwind_sections;

using namespace orc_rt;

namespace {

typedef int (*unw_find_dynamic_unwind_sections)(
    uintptr_t addr, struct unw_dynamic_unwind_sections *info);

extern "C" int __unw_add_find_dynamic_unwind_sections(
    unw_find_dynamic_unwind_sections find_dynamic_unwind_sections)
    ORC_RT_WEAK_IMPORT;

extern "C" int __unw_remove_find_dynamic_unwind_sections(
    unw_find_dynamic_unwind_sections find_dynamic_unwind_sections)
    ORC_RT_WEAK_IMPORT;
// --- end libunwind decls ---

std::mutex LibunwindRegistrationMtx;
size_t LibunwindRegistrationCount;

} // anonymous namespace

namespace orc_rt {

// === UnwindInfoMap method definitions ===

Error StandaloneMachOUnwindInfoRegistrar::UnwindInfoMap::registerRanges(
    const std::vector<ExecutorAddrRange> &CodeRanges,
    const DynamicUnwindSections &Info) {
  std::scoped_lock<std::mutex> Lock(M);
  for (auto &CodeRange : CodeRanges) {
    // TODO: We're assuming safe conversion from ExecutorAddr to uintptr_t
    //       here. In the future we should either check, or make sure that
    //       invalid values aren't deserializable (by switching the wire
    //       representation to match the pointer size).
    if (CodeRange.Start.getValue() > std::numeric_limits<uintptr_t>::max() ||
        CodeRange.End.getValue() > std::numeric_limits<uintptr_t>::max())
      return make_error<StringError>(
          "Invalid code-range for unwind-info registration");

    uintptr_t Start = CodeRange.Start.getValue();
    uintptr_t End = CodeRange.End.getValue();

    // Ignore empty ranges.
    // FIXME: Should we hard-error on these instead?
    if (Start == End)
      continue;

    // Check for overlap with neighboring ranges (including any added earlier
    // in this call, since each successful insertion is visible to subsequent
    // iterations).
    auto MakeRangeOverlapError = [] {
      return make_error<StringError>(
          "Code-range for unwind-info registration overlaps an existing range");
    };
    auto I = Ranges.upper_bound(Start);
    if (I != Ranges.end() && I->first < End)
      return MakeRangeOverlapError();

    if (I != Ranges.begin()) {
      auto PrevI = std::prev(I);
      if (PrevI->second.End > Start)
        return MakeRangeOverlapError();
    }

    Ranges.emplace_hint(I, Start, Entry{Info, End});
  }

  return Error::success();
}

Error StandaloneMachOUnwindInfoRegistrar::UnwindInfoMap::deregisterRanges(
    const std::vector<ExecutorAddrRange> &CodeRanges) {
  std::scoped_lock<std::mutex> Lock(M);
  for (auto &CodeRange : CodeRanges) {
    auto I = Ranges.find(CodeRange.Start.getValue());
    if (I == Ranges.end())
      return make_error<StringError>(
          "No unwind-info sections registered for range");
    Ranges.erase(I);
  }
  return Error::success();
}

std::optional<StandaloneMachOUnwindInfoRegistrar::DynamicUnwindSections>
StandaloneMachOUnwindInfoRegistrar::UnwindInfoMap::lookup(
    uintptr_t Addr) const {
  std::scoped_lock<std::mutex> Lock(M);
  auto I = Ranges.upper_bound(Addr);
  if (I == Ranges.begin())
    return std::nullopt;
  --I;
  if (Addr >= I->second.End)
    return std::nullopt;
  return I->second.Info;
}

// === Registrar libunwind plumbing ===

StandaloneMachOUnwindInfoRegistrar::UnwindInfoMap &
StandaloneMachOUnwindInfoRegistrar::unwindInfoMap() {
  static UnwindInfoMap Map;
  return Map;
}

int StandaloneMachOUnwindInfoRegistrar::findUnwindInfoSections(
    uintptr_t Addr, DynamicUnwindSections *Info) {
  auto S = unwindInfoMap().lookup(Addr);
  if (!S)
    return 0;
  *Info = *S;
  return 1;
}

Error StandaloneMachOUnwindInfoRegistrar::registerWithLibunwind() {
  // Check whether the necessary libunwind APIs are available, as they were
  // only introduced in macOS 15.
  if (!__unw_add_find_dynamic_unwind_sections ||
      !__unw_remove_find_dynamic_unwind_sections)
    return make_error<StringError>(
        "libunwind unwind-info registration APIs not available");

  std::scoped_lock<std::mutex> Lock(LibunwindRegistrationMtx);

  // If we've already registered then just bump the ref count and exit.
  if (LibunwindRegistrationCount > 0) {
    ++LibunwindRegistrationCount;
    return Error::success();
  }

  // Try to register.
  // findUnwindInfoSections has the same call ABI as libunwind expects, since
  // DynamicUnwindSections is laid out identically to
  // unw_dynamic_unwind_sections, so the reinterpret_cast here should be safe
  // in practice.
  if (__unw_add_find_dynamic_unwind_sections(
          reinterpret_cast<unw_find_dynamic_unwind_sections>(
              findUnwindInfoSections)) != 0)
    return make_error<StringError>(
        "libunwind find-dynamic-unwind-sections registration failed");

  // If we succeeded, bump the ref count.
  ++LibunwindRegistrationCount;
  return Error::success();
}

void StandaloneMachOUnwindInfoRegistrar::deregisterWithLibunwind() {
  std::scoped_lock<std::mutex> Lock(LibunwindRegistrationMtx);
  if (--LibunwindRegistrationCount == 0)
    __unw_remove_find_dynamic_unwind_sections(
        reinterpret_cast<unw_find_dynamic_unwind_sections>(
            findUnwindInfoSections));
}

// === Public API ===

StandaloneMachOUnwindInfoRegistrar::Registration::Registration(bool Active)
    : Active(Active) {}

StandaloneMachOUnwindInfoRegistrar::Registration::Registration(
    Registration &&Other)
    : Active(Other.Active) {
  Other.Active = false;
}

StandaloneMachOUnwindInfoRegistrar::Registration &
StandaloneMachOUnwindInfoRegistrar::Registration::operator=(
    Registration &&Other) {
  if (Active)
    deregisterWithLibunwind();
  Active = Other.Active;
  Other.Active = false;
  return *this;
}

StandaloneMachOUnwindInfoRegistrar::Registration::~Registration() {
  if (Active)
    deregisterWithLibunwind();
}

Expected<StandaloneMachOUnwindInfoRegistrar::Registration>
StandaloneMachOUnwindInfoRegistrar::enable(
    SimpleSymbolTable &ST, SimpleSymbolTable::MutatorFn AddInterface) {

  if (auto Err = registerWithLibunwind())
    return std::move(Err);

  Registration R(true); // Create an active registration object.

  if (auto Err = AddInterface(ST))
    return std::move(Err);

  return std::move(R);
}

Error StandaloneMachOUnwindInfoRegistrar::registerSections(
    std::vector<ExecutorAddrRange> CodeRanges, ExecutorAddr DSOBase,
    ExecutorAddrRange DWARFEHFrame, ExecutorAddrRange CompactUnwind) {
  DynamicUnwindSections Info{
      DSOBase.getValue(),   DWARFEHFrame.Start.getValue(),
      DWARFEHFrame.size(),  CompactUnwind.Start.getValue(),
      CompactUnwind.size(),
  };
  return unwindInfoMap().registerRanges(CodeRanges, Info);
}

Error StandaloneMachOUnwindInfoRegistrar::deregisterSections(
    std::vector<ExecutorAddrRange> CodeRanges) {
  return unwindInfoMap().deregisterRanges(CodeRanges);
}

} // namespace orc_rt
