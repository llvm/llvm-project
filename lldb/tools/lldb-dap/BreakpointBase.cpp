//===-- BreakpointBase.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BreakpointBase.h"
#include "JSONUtils.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb_dap;

BreakpointBase::BreakpointBase(DAP &d, const llvm::json::Object &obj)
    : m_dap(d),
      m_condition(std::string(GetString(obj, "condition").value_or(""))),
      m_hit_condition(
          std::string(GetString(obj, "hitCondition").value_or(""))) {}

void BreakpointBase::UpdateBreakpoint(const BreakpointBase &request_bp) {
  if (m_condition != request_bp.m_condition) {
    m_condition = request_bp.m_condition;
    SetCondition();
  }
  if (m_hit_condition != request_bp.m_hit_condition) {
    m_hit_condition = request_bp.m_hit_condition;
    SetHitCondition();
  }
}
