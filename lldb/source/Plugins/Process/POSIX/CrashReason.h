//===-- CrashReason.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CrashReason_H_
#define liblldb_CrashReason_H_

#include "lldb/lldb-types.h"

#include <csignal>

#include <string>

enum class CrashReason {
  eInvalidCrashReason,

  // SIGSEGV crash reasons.
  eInvalidAddress,
  ePrivilegedAddress,
  eBoundViolation,
  eAsyncTagCheckFault,
  eSyncTagCheckFault,

  // SIGILL crash reasons.
  eIllegalOpcode,
  eIllegalOperand,
  eIllegalAddressingMode,
  eIllegalTrap,
  ePrivilegedOpcode,
  ePrivilegedRegister,
  eCoprocessorError,
  eInternalStackError,

  // SIGBUS crash reasons,
  eIllegalAlignment,
  eIllegalAddress,
  eHardwareError,

  // SIGFPE crash reasons,
  eIntegerDivideByZero,
  eIntegerOverflow,
  eFloatDivideByZero,
  eFloatOverflow,
  eFloatUnderflow,
  eFloatInexactResult,
  eFloatInvalidOperation,
  eFloatSubscriptRange
};

std::string GetCrashReasonString(CrashReason reason, lldb::addr_t fault_addr);
std::string GetCrashReasonString(CrashReason reason, const siginfo_t &info);

const char *CrashReasonAsString(CrashReason reason);

CrashReason GetCrashReason(const siginfo_t &info);

#endif // #ifndef liblldb_CrashReason_H_
