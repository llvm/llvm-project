//===-- llvm/CallingConv.h - LLVM Calling Conventions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines LLVM's set of calling conventions. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CALLINGCONV_H
#define LLVM_CALLINGCONV_H

namespace llvm {

/// CallingConv Namespace - This namespace contains an enum with a value for
/// the well-known calling conventions.
///
namespace CallingConv {
  /// A set of enums which specify the assigned numeric values for known llvm 
  /// calling conventions.
  /// @brief LLVM Calling Convention Representation
  enum ID {
    /// C - The default llvm calling convention, compatible with C.  This
    /// convention is the only calling convention that supports varargs calls.
    /// As with typical C calling conventions, the callee/caller have to 
    /// tolerate certain amounts of prototype mismatch.
    C = 0,
    
    // Generic LLVM calling conventions.  None of these calling conventions
    // support varargs calls, and all assume that the caller and callee
    // prototype exactly match.

    /// Fast - This calling convention attempts to make calls as fast as 
    /// possible (e.g. by passing things in registers).
    Fast = 8,

    // Cold - This calling convention attempts to make code in the caller as
    // efficient as possible under the assumption that the call is not commonly
    // executed.  As such, these calls often preserve all registers so that the
    // call does not break any live ranges in the caller side.
    Cold = 9,

    // Target - This is the start of the target-specific calling conventions,
    // e.g. fastcall and thiscall on X86.
    FirstTargetCC = 64,

    /// X86_StdCall - stdcall is the calling conventions mostly used by the
    /// Win32 API. It is basically the same as the C convention with the
    /// difference in that the callee is responsible for popping the arguments
    /// from the stack.
    X86_StdCall = 64,

    /// X86_FastCall - 'fast' analog of X86_StdCall. Passes first two arguments
    /// in ECX:EDX registers, others - via stack. Callee is responsible for
    /// stack cleaning.
    X86_FastCall = 65
  };
} // End CallingConv namespace

} // End llvm namespace

#endif
