//===- PointeeTypeMap.h - Typed-pointer reconstruction ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_POINTEETYPEMAP_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_POINTEETYPEMAP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace llvm {
namespace metal {

// ── Pointee Type Map ─────────────────────────────────────────────────────
//
// AIR GPU JIT requires typed pointers in bitcode (POINTER records with
// pointee type), but LLVM 19+ only has opaque pointers (ptr addrspace(N)).
//
// This side table tracks what each pointer "actually points to" as the IR
// passes transform it. Passes that need typed pointer info read/write this
// map. The custom bitcode writer consumes it to emit typed POINTER records.
//
// Example flow:
//
// LLVM Module (opaque ptrs):
// %p = getelementptr float, ptr addrspace(1) %buf, i32 %idx
// %v = load float, ptr addrspace(1) %p
//
// PointeeTypeMap after InferTypedPointersPass:
// %buf → float (inferred from GEP source type)
// %p → float (inferred from load type)
//
// Custom bitcode writer:
// %buf emitted as float addrspace(1)* (POINTER record with pointee=float)
// %p emitted as float addrspace(1)*
//
// Keeping the type info in a side table (rather than a bespoke typed-pointer
// IR) lets the LLVM Module stay valid with opaque pointers throughout the
// pipeline; only the bitcode writer needs the resolved pointee types.
//
// ── Key rules from AIR GPU JIT ─────────────────────────────────────────
//
// 1. ALL device pointers (addrspace 1) must be typed in bitcode
// 2. When MMA intrinsics are present, ALL device ptrs must be float*
// (narrow loads now stay narrow; the per-load alias contract emitted by
// AIRAliasAnnotate carries the disambiguation the JIT needs)
// 3. TG pointers (addrspace 3) must be typed (usually float*)
// 4. i1* crashes GPU JIT - remap to i8*
// 5. Constant buffer ptrs (addrspace 2) follow scalar packing rules
//
// ── Two-stage design: Analysis vs Pass (READ THIS BEFORE EDITING) ─────────
//
// The pointee-type logic deliberately lives in TWO places, and they overlap.
// This is intentional, but fragile - keep them in sync.
//
// * PointeeTypeAnalysis::run (PointeeTypeMap.cpp) is a cached LLVM Analysis.
// It builds the map from scratch and MUST be fully self-contained: the pass
// manager may invalidate and recompute it at any point, so every
// AIR-specific override (MMA collapse, async-copy event_t, i1->i8) has to
// be reproducible here with no outside state.
//
// * InferTypedPointersPass::run (InferTypedPointers.cpp) is a Transform Pass.
// It calls getResult<PointeeTypeAnalysis>() to get the map, then REFINES it
// after doing IR mutations the analysis can't do (e.g. the
// ptrtoint+inttoptr atomic fixup). Because those mutations create new SSA
// values, the MMA/async overrides must be re-applied here too.
//
// Net effect: the MMA + async-copy override blocks are near-duplicated across
// the two files. If you change one (e.g. add a new MMA intrinsic variant),
// change BOTH. The shared intrinsic-name constants below exist so at least
// those can't silently diverge.

// Shared MMA intrinsic names, used by both PointeeTypeAnalysis and
// InferTypedPointersPass. Keep additions here, not copy-pasted per file.
namespace mma_intrinsics {
inline constexpr const char *kLoad =
    "air.simdgroup_matrix_8x8_load.v64f32.p3f32";
inline constexpr const char *kStore =
    "air.simdgroup_matrix_8x8_store.v64f32.p3f32";
inline constexpr const char *kLoadDev =
    "air.simdgroup_matrix_8x8_load.v64f32.p1f32";
inline constexpr const char *kStoreDev =
    "air.simdgroup_matrix_8x8_store.v64f32.p1f32";
inline constexpr const char *kPrefix = "air.simdgroup_matrix_8x8_";
} // namespace mma_intrinsics

class PointeeTypeMap {
public:
  // Set the pointee type for a pointer value.
  void set(llvm::Value *ptr, llvm::Type *pointeeTy) { map[ptr] = pointeeTy; }

  // Get the pointee type, or nullptr if unknown.
  llvm::Type *get(llvm::Value *ptr) const {
    auto it = map.find(ptr);
    return it != map.end() ? it->second : nullptr;
  }

  // Check if a pointer has a known pointee type.
  bool has(llvm::Value *ptr) const { return map.count(ptr); }

  // Remove a pointer's entry (used when erasing instructions).
  void remove(llvm::Value *ptr) { map.erase(ptr); }

  // Iterate all entries.
  auto begin() const { return map.begin(); }
  auto end() const { return map.end(); }
  size_t size() const { return map.size(); }

  // ── Inference helpers ────────────────────────────────────────────────

  // Infer pointee type from how a pointer is used (loads, stores, GEPs).
  // Returns nullptr if no usage gives a clear type.
  static llvm::Type *inferFromUsage(llvm::Value *Ptr);
  static llvm::Type *
  inferFromUsage(llvm::Value *Ptr,
                 llvm::SmallPtrSetImpl<llvm::Value *> &Visited);

  // Apply the "MMA present → all device ptrs are float*" rule.
  void collapseDevicePointersToFloat(llvm::Module &M);

  // Apply the "i1* → i8*" rule.
  void remapI1ToI8(llvm::Module &M);

private:
  llvm::DenseMap<llvm::Value *, llvm::Type *> map;
};

// Compute the pointee-type side table for a module. This is a pure function of
// the module (no analysis-manager state), so callers that aren't running under
// the new-PM analysis machinery -- e.g. the legacy metallib writer pass -- can
// call it directly.
PointeeTypeMap buildPointeeTypeMap(llvm::Module &M);

inline constexpr llvm::StringRef kEventTypeName = "event_t";
llvm::StructType *getOrCreateEventType(llvm::LLVMContext &Ctx);

struct PointeeTypeAnalysis : llvm::AnalysisInfoMixin<PointeeTypeAnalysis> {
  using Result = PointeeTypeMap;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static llvm::AnalysisKey Key;
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRWRITER_POINTEETYPEMAP_H
