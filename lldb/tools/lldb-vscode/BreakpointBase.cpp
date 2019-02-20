//===-- BreakpointBase.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BreakpointBase.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb_vscode;

BreakpointBase::BreakpointBase(const llvm::json::Object &obj)
    : condition(GetString(obj, "condition")),
      hitCondition(GetString(obj, "hitCondition")),
      logMessage(GetString(obj, "logMessage")) {}

void BreakpointBase::SetCondition() { bp.SetCondition(condition.c_str()); }

void BreakpointBase::SetHitCondition() {  
  uint64_t hitCount = 0;
  if (llvm::to_integer(hitCondition, hitCount))
    bp.SetIgnoreCount(hitCount - 1);
}

void BreakpointBase::UpdateBreakpoint(const BreakpointBase &request_bp) {
  if (condition != request_bp.condition) {
    condition = request_bp.condition;
    SetCondition();
  }
  if (hitCondition != request_bp.hitCondition) {
    hitCondition = request_bp.hitCondition;
    SetHitCondition();
  }
}
