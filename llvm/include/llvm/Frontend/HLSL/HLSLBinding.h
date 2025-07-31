//===- HLSLBinding.h - Representation for resource bindings in HLSL -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains objects to represent resource bindings.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLBINDING_H
#define LLVM_FRONTEND_HLSL_HLSLBINDING_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace hlsl {

/// BindingInfo represents the ranges of bindings and free space for each
/// `dxil::ResourceClass`. This can represent HLSL-level bindings as well as
/// bindings described in root signatures, and can be used for analysis of
/// overlapping or missing bindings as well as for finding space for implicit
/// bindings.
///
/// As an example, given these resource bindings:
///
/// RWBuffer<float> A[10] : register(u3);
/// RWBuffer<float> B[] : register(u5, space2)
///
/// The binding info for UAV bindings should look like this:
///
/// UAVSpaces {
///   ResClass = ResourceClass::UAV,
///   Spaces = {
///     { Space = 0u, FreeRanges = {{ 0u, 2u }, { 13u, ~0u }} },
///     { Space = 2u, FreeRanges = {{ 0u, 4u }} }
///   }
/// }
class BindingInfo {
public:
  struct BindingRange {
    uint32_t LowerBound;
    uint32_t UpperBound;
    BindingRange(uint32_t LB, uint32_t UB) : LowerBound(LB), UpperBound(UB) {}
  };

  struct RegisterSpace {
    uint32_t Space;
    SmallVector<BindingRange> FreeRanges;
    RegisterSpace(uint32_t Space) : Space(Space) {
      FreeRanges.emplace_back(0, ~0u);
    }
    // Size == -1 means unbounded array
    LLVM_ABI std::optional<uint32_t> findAvailableBinding(int32_t Size);
  };

  struct BindingSpaces {
    dxil::ResourceClass RC;
    llvm::SmallVector<RegisterSpace> Spaces;
    BindingSpaces(dxil::ResourceClass RC) : RC(RC) {}
    LLVM_ABI RegisterSpace &getOrInsertSpace(uint32_t Space);
  };

private:
  BindingSpaces SRVSpaces{dxil::ResourceClass::SRV};
  BindingSpaces UAVSpaces{dxil::ResourceClass::UAV};
  BindingSpaces CBufferSpaces{dxil::ResourceClass::CBuffer};
  BindingSpaces SamplerSpaces{dxil::ResourceClass::Sampler};

public:
  BindingSpaces &getBindingSpaces(dxil::ResourceClass RC) {
    switch (RC) {
    case dxil::ResourceClass::SRV:
      return SRVSpaces;
    case dxil::ResourceClass::UAV:
      return UAVSpaces;
    case dxil::ResourceClass::CBuffer:
      return CBufferSpaces;
    case dxil::ResourceClass::Sampler:
      return SamplerSpaces;
    }

    llvm_unreachable("Invalid resource class");
  }
  const BindingSpaces &getBindingSpaces(dxil::ResourceClass RC) const {
    return const_cast<BindingInfo *>(this)->getBindingSpaces(RC);
  }

  // Size == -1 means unbounded array
  LLVM_ABI std::optional<uint32_t>
  findAvailableBinding(dxil::ResourceClass RC, uint32_t Space, int32_t Size);

  friend class BindingInfoBuilder;
};

/// Builder class for creating a /c BindingInfo.
class BindingInfoBuilder {
public:
  struct Binding {
    dxil::ResourceClass RC;
    uint32_t Space;
    uint32_t LowerBound;
    uint32_t UpperBound;
    const void *Cookie;

    Binding(dxil::ResourceClass RC, uint32_t Space, uint32_t LowerBound,
            uint32_t UpperBound, const void *Cookie)
        : RC(RC), Space(Space), LowerBound(LowerBound), UpperBound(UpperBound),
          Cookie(Cookie) {}

    bool isUnbounded() const { return UpperBound == ~0U; }

    bool operator==(const Binding &RHS) const {
      return std::tie(RC, Space, LowerBound, UpperBound, Cookie) ==
             std::tie(RHS.RC, RHS.Space, RHS.LowerBound, RHS.UpperBound,
                      RHS.Cookie);
    }
    bool operator!=(const Binding &RHS) const { return !(*this == RHS); }

    bool operator<(const Binding &RHS) const {
      return std::tie(RC, Space, LowerBound) <
             std::tie(RHS.RC, RHS.Space, RHS.LowerBound);
    }
  };

private:
  SmallVector<Binding> Bindings;

public:
  void trackBinding(dxil::ResourceClass RC, uint32_t Space, uint32_t LowerBound,
                    uint32_t UpperBound, const void *Cookie) {
    Bindings.emplace_back(RC, Space, LowerBound, UpperBound, Cookie);
  }
  /// Calculate the binding info - \c ReportOverlap will be called once for each
  /// overlapping binding.
  BindingInfo calculateBindingInfo(
      llvm::function_ref<void(const BindingInfoBuilder &Builder,
                              const Binding &Overlapping)>
          ReportOverlap);

  /// Calculate the binding info - \c HasOverlap will be set to indicate whether
  /// there are any overlapping bindings.
  BindingInfo calculateBindingInfo(bool &HasOverlap) {
    HasOverlap = false;
    return calculateBindingInfo(
        [&HasOverlap](auto, auto) { HasOverlap = true; });
  }

  /// For use in the \c ReportOverlap callback of \c calculateBindingInfo -
  /// finds a binding that the \c ReportedBinding overlaps with.
  const Binding &findOverlapping(const Binding &ReportedBinding) const;
};

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLBINDING_H
