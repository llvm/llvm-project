//===- ObjCClassHierarchy.h - ObjC class layout facts -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for collecting and resolving Objective-C class layout facts used by
// ivar offset constification.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OBJCCLASSHIERARCHY_H
#define LLVM_IR_OBJCCLASSHIERARCHY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"

namespace llvm {

class GlobalVariable;
class Module;

class ObjCClassHierarchy {
public:
  // A class whose ivar offsets and class_ro_t fields have been resolved.
  struct ResolvedClass {
    GlobalVariable *ROGV;
    uint32_t InstanceStart;
    uint32_t InstanceSize;
  };

  // Construct the resolved hierarchy from module IR. If ImportSummary is
  // provided, use pre-resolved layout from the thin-link (ThinLTO path);
  // otherwise resolve locally from the module alone (FullLTO path).
  explicit ObjCClassHierarchy(const Module &M,
                              const ModuleSummaryIndex *ImportSummary);

  // Emit per-module ObjC class summary records during module summary analysis.
  static void exportToSummary(const Module &M, ModuleSummaryIndex &Index);
  // Pre-slide the hierarchy graph materialized in this map; unreachable classes are erased.
  static void resolveHierarchy(DenseMap<GlobalValue::GUID, ObjCClassInfo> &Classes);

  ArrayRef<ResolvedClass> getResolvedClasses() const { return Resolved; }
  bool empty() const { return Resolved.empty(); }

private:
  SmallVector<ResolvedClass> Resolved;
};

} // end namespace llvm

#endif // LLVM_IR_OBJCCLASSHIERARCHY_H
