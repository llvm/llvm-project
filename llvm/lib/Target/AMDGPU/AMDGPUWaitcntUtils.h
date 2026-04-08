//===- AMDGPUWaitcntUtils.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/TargetParser.h"

namespace llvm {

namespace AMDGPU {

enum InstCounterType {
  LOAD_CNT = 0, // VMcnt prior to gfx12.
  DS_CNT,       // LKGMcnt prior to gfx12.
  EXP_CNT,      //
  STORE_CNT,    // VScnt in gfx10/gfx11.
  NUM_NORMAL_INST_CNTS,
  SAMPLE_CNT = NUM_NORMAL_INST_CNTS, // gfx12+ only.
  BVH_CNT,                           // gfx12+ only.
  KM_CNT,                            // gfx12+ only.
  X_CNT,                             // gfx1250.
  ASYNC_CNT,                         // gfx1250.
  NUM_EXTENDED_INST_CNTS,
  VA_VDST = NUM_EXTENDED_INST_CNTS, // gfx12+ expert mode only.
  VM_VSRC,                          // gfx12+ expert mode only.
  NUM_EXPERT_INST_CNTS,
  NUM_INST_CNTS = NUM_EXPERT_INST_CNTS
};

StringLiteral getInstCounterName(InstCounterType T);

// Return an iterator over all counters between LOAD_CNT (the first counter)
// and \c MaxCounter (exclusive, default value yields an enumeration over
// all counters).
iota_range<InstCounterType>
inst_counter_types(InstCounterType MaxCounter = NUM_INST_CNTS);

} // namespace AMDGPU

template <> struct enum_iteration_traits<AMDGPU::InstCounterType> {
  static constexpr bool is_iterable = true;
};

namespace AMDGPU {

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
class Waitcnt {
  std::array<unsigned, NUM_INST_CNTS> Cnt;

public:
  unsigned get(InstCounterType T) const { return Cnt[T]; }
  void set(InstCounterType T, unsigned Val) { Cnt[T] = Val; }

  Waitcnt() { fill(Cnt, ~0u); }
  // Pre-gfx12 constructor.
  Waitcnt(unsigned VmCnt, unsigned ExpCnt, unsigned LgkmCnt, unsigned VsCnt)
      : Waitcnt() {
    Cnt[LOAD_CNT] = VmCnt;
    Cnt[EXP_CNT] = ExpCnt;
    Cnt[DS_CNT] = LgkmCnt;
    Cnt[STORE_CNT] = VsCnt;
  }

  // gfx12+ constructor.
  Waitcnt(unsigned LoadCnt, unsigned ExpCnt, unsigned DsCnt, unsigned StoreCnt,
          unsigned SampleCnt, unsigned BvhCnt, unsigned KmCnt, unsigned XCnt,
          unsigned AsyncCnt, unsigned VaVdst, unsigned VmVsrc)
      : Waitcnt() {
    Cnt[LOAD_CNT] = LoadCnt;
    Cnt[DS_CNT] = DsCnt;
    Cnt[EXP_CNT] = ExpCnt;
    Cnt[STORE_CNT] = StoreCnt;
    Cnt[SAMPLE_CNT] = SampleCnt;
    Cnt[BVH_CNT] = BvhCnt;
    Cnt[KM_CNT] = KmCnt;
    Cnt[X_CNT] = XCnt;
    Cnt[ASYNC_CNT] = AsyncCnt;
    Cnt[VA_VDST] = VaVdst;
    Cnt[VM_VSRC] = VmVsrc;
  }

  bool hasWait() const {
    return any_of(Cnt, [](unsigned Val) { return Val != ~0u; });
  }

  bool hasWaitExceptStoreCnt() const {
    for (InstCounterType T : inst_counter_types()) {
      if (T == STORE_CNT)
        continue;
      if (Cnt[T] != ~0u)
        return true;
    }
    return false;
  }

  bool hasWaitStoreCnt() const { return Cnt[STORE_CNT] != ~0u; }

  bool hasWaitDepctr() const {
    return Cnt[VA_VDST] != ~0u || Cnt[VM_VSRC] != ~0u;
  }

  Waitcnt combined(const Waitcnt &Other) const {
    // Does the right thing provided self and Other are either both pre-gfx12
    // or both gfx12+.
    Waitcnt Wait;
    for (InstCounterType T : inst_counter_types())
      Wait.Cnt[T] = std::min(Cnt[T], Other.Cnt[T]);
    return Wait;
  }

  void print(raw_ostream &OS) const {
    ListSeparator LS;
    for (InstCounterType T : inst_counter_types())
      OS << LS << getInstCounterName(T) << ": " << Cnt[T];
    if (LS.unused())
      OS << "none";
    OS << '\n';
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  friend raw_ostream &operator<<(raw_ostream &OS, const AMDGPU::Waitcnt &Wait) {
    Wait.print(OS);
    return OS;
  }
};

Waitcnt decodeWaitcnt(const IsaVersion &Version, unsigned Encoded);

unsigned encodeWaitcnt(const IsaVersion &Version, const Waitcnt &Decoded);

// The following are only meaningful on targets that support
// S_WAIT_LOADCNT_DSCNT and S_WAIT_STORECNT_DSCNT.

/// \returns Decoded Waitcnt structure from given \p LoadcntDscnt for given
/// isa \p Version.
Waitcnt decodeLoadcntDscnt(const IsaVersion &Version, unsigned LoadcntDscnt);

/// \returns Decoded Waitcnt structure from given \p StorecntDscnt for given
/// isa \p Version.
Waitcnt decodeStorecntDscnt(const IsaVersion &Version, unsigned StorecntDscnt);

/// \returns \p Loadcnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_LOADCNT_DSCNT for given isa
/// \p Version.
unsigned encodeLoadcntDscnt(const IsaVersion &Version, const Waitcnt &Decoded);

/// \returns \p Storecnt and \p Dscnt components of \p Decoded  encoded as an
/// immediate that can be used with S_WAIT_STORECNT_DSCNT for given isa
/// \p Version.
unsigned encodeStorecntDscnt(const IsaVersion &Version, const Waitcnt &Decoded);

} // namespace AMDGPU

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUWAITCNTUTILS_H
