//===-- CrashReason.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CrashReason_H_
#define liblldb_CrashReason_H_

#include "lldb/lldb-types.h"

#include <signal.h>

#include <string>

enum class CrashReason {
  eInvalidCrashReason,

  // SIGSEGV crash reasons.
  eInvalidAddress,
  ePrivilegedAddress,

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

const char *CrashReasonAsString(CrashReason reason);

CrashReason GetCrashReason(const siginfo_t &info);

#endif // #ifndef liblldb_CrashReason_H_
