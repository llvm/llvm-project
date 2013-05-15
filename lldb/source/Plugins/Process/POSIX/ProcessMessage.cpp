//===-- ProcessMessage.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessMessage.h"

using namespace lldb_private;

const char *
ProcessMessage::GetCrashReasonString(CrashReason reason)
{
    const char *str = NULL;

    switch (reason)
    {
    default:
        assert(false && "invalid CrashReason");
        break;

    case eInvalidAddress:
        str = "invalid address";
        break;
    case ePrivilegedAddress:
        str = "address access protected";
        break;
    case eIllegalOpcode:
        str = "illegal instruction";
        break;
    case eIllegalOperand:
        str = "illegal instruction operand";
        break;
    case eIllegalAddressingMode:
        str = "illegal addressing mode";
        break;
    case eIllegalTrap:
        str = "illegal trap";
        break;
    case ePrivilegedOpcode:
        str = "privileged instruction";
        break;
    case ePrivilegedRegister:
        str = "privileged register";
        break;
    case eCoprocessorError:
        str = "coprocessor error";
        break;
    case eInternalStackError:
        str = "internal stack error";
        break;
    case eIllegalAlignment:
        str = "illegal alignment";
        break;
    case eIllegalAddress:
        str = "illegal address";
        break;
    case eHardwareError:
        str = "hardware error";
        break;
    case eIntegerDivideByZero:
        str = "integer divide by zero";
        break;
    case eIntegerOverflow:
        str = "integer overflow";
        break;
    case eFloatDivideByZero:
        str = "floating point divide by zero";
        break;
    case eFloatOverflow:
        str = "floating point overflow";
        break;
    case eFloatUnderflow:
        str = "floating point underflow";
        break;
    case eFloatInexactResult:
        str = "inexact floating point result";
        break;
    case eFloatInvalidOperation:
        str = "invalid floating point operation";
        break;
    case eFloatSubscriptRange:
        str = "invalid floating point subscript range";
        break;
    }

    return str;
}

const char *
ProcessMessage::PrintCrashReason(CrashReason reason)
{
#ifdef LLDB_CONFIGURATION_BUILDANDINTEGRATION
    // Just return the code in asci for integration builds.
    chcar str[8];
    sprintf(str, "%d", reason);
#else
    const char *str = NULL;

    switch (reason)
    {
        case eInvalidCrashReason:
            str = "eInvalidCrashReason";
            break;

        // SIGSEGV crash reasons.
        case eInvalidAddress:
            str = "eInvalidAddress";
            break;
        case ePrivilegedAddress:
            str = "ePrivilegedAddress";
            break;

        // SIGILL crash reasons.
        case eIllegalOpcode:
            str = "eIllegalOpcode";
            break;
        case eIllegalOperand:
            str = "eIllegalOperand";
            break;
        case eIllegalAddressingMode:
            str = "eIllegalAddressingMode";
            break;
        case eIllegalTrap:
            str = "eIllegalTrap";
            break;
        case ePrivilegedOpcode:
            str = "ePrivilegedOpcode";
            break;
        case ePrivilegedRegister:
            str = "ePrivilegedRegister";
            break;
        case eCoprocessorError:
            str = "eCoprocessorError";
            break;
        case eInternalStackError:
            str = "eInternalStackError";
            break;

        // SIGBUS crash reasons:
        case eIllegalAlignment:
            str = "eIllegalAlignment";
            break;
        case eIllegalAddress:
            str = "eIllegalAddress";
            break;
        case eHardwareError:
            str = "eHardwareError";
            break;

        // SIGFPE crash reasons:
        case eIntegerDivideByZero:
            str = "eIntegerDivideByZero";
            break;
        case eIntegerOverflow:
            str = "eIntegerOverflow";
            break;
        case eFloatDivideByZero:
            str = "eFloatDivideByZero";
            break;
        case eFloatOverflow:
            str = "eFloatOverflow";
            break;
        case eFloatUnderflow:
            str = "eFloatUnderflow";
            break;
        case eFloatInexactResult:
            str = "eFloatInexactResult";
            break;
        case eFloatInvalidOperation:
            str = "eFloatInvalidOperation";
            break;
        case eFloatSubscriptRange:
            str = "eFloatSubscriptRange";
            break;
    }
#endif

    return str;
}

const char *
ProcessMessage::PrintCrashReason() const
{
    return PrintCrashReason(m_crash_reason);
}

const char *
ProcessMessage::PrintKind(Kind kind)
{
#ifdef LLDB_CONFIGURATION_BUILDANDINTEGRATION
    // Just return the code in asci for integration builds.
    chcar str[8];
    sprintf(str, "%d", reason);
#else
    const char *str = NULL;

    switch (kind)
    {
    case eInvalidMessage:
        str = "eInvalidMessage";
        break;
    case eExitMessage:
        str = "eExitMessage";
        break;
    case eLimboMessage:
        str = "eLimboMessage";
        break;
    case eSignalMessage:
        str = "eSignalMessage";
        break;
    case eSignalDeliveredMessage:
        str = "eSignalDeliveredMessage";
        break;
    case eTraceMessage:
        str = "eTraceMessage";
        break;
    case eBreakpointMessage:
        str = "eBreakpointMessage";
        break;
    case eWatchpointMessage:
        str = "eWatchpointMessage";
        break;
    case eCrashMessage:
        str = "eCrashMessage";
        break;
    case eNewThreadMessage:
        str = "eNewThreadMessage";
        break;
    }
#endif

    return str;
}

const char *
ProcessMessage::PrintKind() const
{
    return PrintKind(m_kind);
}
