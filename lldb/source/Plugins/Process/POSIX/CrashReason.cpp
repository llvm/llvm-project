//===-- CrashReason.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashReason.h"

#include "llvm/Support/raw_ostream.h"

#include <sstream>

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

static void AppendFaultAddr(std::string &str, lldb::addr_t addr) {
  std::stringstream ss;
  ss << " (fault address: 0x" << std::hex << addr << ")";
  str += ss.str();
}

static void AppendBounds(std::string &str, lldb::addr_t lower_bound,
                         lldb::addr_t upper_bound, lldb::addr_t addr) {
  llvm::raw_string_ostream stream(str);
  if ((unsigned long)addr < lower_bound)
    stream << ": lower bound violation ";
  else
    stream << ": upper bound violation ";
  stream << "(fault address: 0x";
  stream.write_hex(addr);
  stream << ", lower bound: 0x";
  stream.write_hex(lower_bound);
  stream << ", upper bound: 0x";
  stream.write_hex(upper_bound);
  stream << ")";
  stream.flush();
}

static CrashReason GetCrashReasonForSIGSEGV(int code) {
  switch (code) {
#ifdef SI_KERNEL
  case SI_KERNEL:
    // Some platforms will occasionally send nonstandard spurious SI_KERNEL
    // codes. One way to get this is via unaligned SIMD loads.
    return CrashReason::eInvalidAddress; // for lack of anything better
#endif
  case SEGV_MAPERR:
    return CrashReason::eInvalidAddress;
  case SEGV_ACCERR:
    return CrashReason::ePrivilegedAddress;
#ifndef SEGV_BNDERR
#define SEGV_BNDERR 3
#endif
  case SEGV_BNDERR:
    return CrashReason::eBoundViolation;
#ifdef __linux__
#ifndef SEGV_MTEAERR
#define SEGV_MTEAERR 8
#endif
  case SEGV_MTEAERR:
    return CrashReason::eAsyncTagCheckFault;
#ifndef SEGV_MTESERR
#define SEGV_MTESERR 9
#endif
  case SEGV_MTESERR:
    return CrashReason::eSyncTagCheckFault;
#endif // __linux__
  }

  return CrashReason::eInvalidCrashReason;
}

static CrashReason GetCrashReasonForSIGILL(int code) {
  switch (code) {
  case ILL_ILLOPC:
    return CrashReason::eIllegalOpcode;
  case ILL_ILLOPN:
    return CrashReason::eIllegalOperand;
  case ILL_ILLADR:
    return CrashReason::eIllegalAddressingMode;
  case ILL_ILLTRP:
    return CrashReason::eIllegalTrap;
  case ILL_PRVOPC:
    return CrashReason::ePrivilegedOpcode;
  case ILL_PRVREG:
    return CrashReason::ePrivilegedRegister;
  case ILL_COPROC:
    return CrashReason::eCoprocessorError;
  case ILL_BADSTK:
    return CrashReason::eInternalStackError;
  }

  return CrashReason::eInvalidCrashReason;
}

static CrashReason GetCrashReasonForSIGFPE(int code) {
  switch (code) {
  case FPE_INTDIV:
    return CrashReason::eIntegerDivideByZero;
  case FPE_INTOVF:
    return CrashReason::eIntegerOverflow;
  case FPE_FLTDIV:
    return CrashReason::eFloatDivideByZero;
  case FPE_FLTOVF:
    return CrashReason::eFloatOverflow;
  case FPE_FLTUND:
    return CrashReason::eFloatUnderflow;
  case FPE_FLTRES:
    return CrashReason::eFloatInexactResult;
  case FPE_FLTINV:
    return CrashReason::eFloatInvalidOperation;
  case FPE_FLTSUB:
    return CrashReason::eFloatSubscriptRange;
  }

  return CrashReason::eInvalidCrashReason;
}

static CrashReason GetCrashReasonForSIGBUS(int code) {
  switch (code) {
  case BUS_ADRALN:
    return CrashReason::eIllegalAlignment;
  case BUS_ADRERR:
    return CrashReason::eIllegalAddress;
  case BUS_OBJERR:
    return CrashReason::eHardwareError;
  }

  return CrashReason::eInvalidCrashReason;
}

static std::string GetCrashReasonString(CrashReason reason,
                                        lldb::addr_t fault_addr) {
  std::string str;

  switch (reason) {
  default:
    str = "unknown crash reason";
    break;

  case CrashReason::eInvalidAddress:
    str = "signal SIGSEGV: invalid address";
    AppendFaultAddr(str, fault_addr);
    break;
  case CrashReason::ePrivilegedAddress:
    str = "signal SIGSEGV: address access protected";
    AppendFaultAddr(str, fault_addr);
    break;
  case CrashReason::eBoundViolation:
    str = "signal SIGSEGV: bound violation";
    break;
  case CrashReason::eAsyncTagCheckFault:
    str = "signal SIGSEGV: async tag check fault";
    break;
  case CrashReason::eSyncTagCheckFault:
    str = "signal SIGSEGV: sync tag check fault";
    AppendFaultAddr(str, fault_addr);
    break;
  case CrashReason::eIllegalOpcode:
    str = "signal SIGILL: illegal instruction";
    break;
  case CrashReason::eIllegalOperand:
    str = "signal SIGILL: illegal instruction operand";
    break;
  case CrashReason::eIllegalAddressingMode:
    str = "signal SIGILL: illegal addressing mode";
    break;
  case CrashReason::eIllegalTrap:
    str = "signal SIGILL: illegal trap";
    break;
  case CrashReason::ePrivilegedOpcode:
    str = "signal SIGILL: privileged instruction";
    break;
  case CrashReason::ePrivilegedRegister:
    str = "signal SIGILL: privileged register";
    break;
  case CrashReason::eCoprocessorError:
    str = "signal SIGILL: coprocessor error";
    break;
  case CrashReason::eInternalStackError:
    str = "signal SIGILL: internal stack error";
    break;
  case CrashReason::eIllegalAlignment:
    str = "signal SIGBUS: illegal alignment";
    break;
  case CrashReason::eIllegalAddress:
    str = "signal SIGBUS: illegal address";
    break;
  case CrashReason::eHardwareError:
    str = "signal SIGBUS: hardware error";
    break;
  case CrashReason::eIntegerDivideByZero:
    str = "signal SIGFPE: integer divide by zero";
    break;
  case CrashReason::eIntegerOverflow:
    str = "signal SIGFPE: integer overflow";
    break;
  case CrashReason::eFloatDivideByZero:
    str = "signal SIGFPE: floating point divide by zero";
    break;
  case CrashReason::eFloatOverflow:
    str = "signal SIGFPE: floating point overflow";
    break;
  case CrashReason::eFloatUnderflow:
    str = "signal SIGFPE: floating point underflow";
    break;
  case CrashReason::eFloatInexactResult:
    str = "signal SIGFPE: inexact floating point result";
    break;
  case CrashReason::eFloatInvalidOperation:
    str = "signal SIGFPE: invalid floating point operation";
    break;
  case CrashReason::eFloatSubscriptRange:
    str = "signal SIGFPE: invalid floating point subscript range";
    break;
  }

  return str;
}

static CrashReason GetCrashReason(int signo, int code) {
  switch (signo) {
  case SIGSEGV:
    return GetCrashReasonForSIGSEGV(code);
  case SIGBUS:
    return GetCrashReasonForSIGBUS(code);
  case SIGFPE:
    return GetCrashReasonForSIGFPE(code);
  case SIGILL:
    return GetCrashReasonForSIGILL(code);
  }

  assert(false && "unexpected signal");
  return CrashReason::eInvalidCrashReason;
}

static std::string GetCrashReasonString(int signo, int code, lldb::addr_t addr,
                                        std::optional<lldb::addr_t> lower,
                                        std::optional<lldb::addr_t> upper) {
  CrashReason reason = GetCrashReason(signo, code);

  if (lower && upper) {
    std::string str;
    if (reason == CrashReason::eBoundViolation) {
      str = "signal SIGSEGV";
      AppendBounds(str, *lower, *upper, addr);
      return str;
    }
  }

  return GetCrashReasonString(reason, addr);
}

std::string GetCrashReasonString(const siginfo_t &info) {
#if defined(si_lower) && defined(si_upper)
  std::optional<lldb::addr_t> lower =
      reinterpret_cast<lldb::addr_t>(info.si_lower);
  std::optional<lldb::addr_t> upper =
      reinterpret_cast<lldb::addr_t>(info.si_upper);
#else
  std::optional<lldb::addr_t> lower;
  std::optional<lldb::addr_t> upper;
#endif
  return GetCrashReasonString(info.si_signo, info.si_code,
                              reinterpret_cast<uintptr_t>(info.si_addr), lower,
                              upper);
}
