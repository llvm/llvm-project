//===----- DebugInfo.h - analysis and lowering for Debug info -*- C++ -*- -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Analyze and downgrade debug info metadata to match DXIL (LLVM 3.7).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
#define LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

class Module;
class Metadata;
class Value;

namespace dxil {

class DXILDebugInfoMap {
public:
  using MDMap = DenseMap<const Metadata *, const Metadata *>;
  using VMap = DenseMap<const Value *, const Value *>;
  using VSet = DenseSet<const Value *>;

  /// Enumerate extra metadata when Key is encountered in ValueEnumerator.
  MDMap MDExtra;

  /// Completely replace one metadata with another in ValueEnumerator.
  MDMap MDReplace;

  /// Replace the name of one value with that of another.
  VMap VRename;

  /// Remove a value from the output.
  VSet VRemove;
};

namespace DXILDebugInfoPass {

DXILDebugInfoMap run(Module &M);

} // namespace DXILDebugInfoPass
} // namespace dxil
} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
