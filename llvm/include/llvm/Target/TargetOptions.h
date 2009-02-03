//===-- llvm/Target/TargetOptions.h - Target Options ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines command line option flags that are shared across various
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPTIONS_H
#define LLVM_TARGET_TARGETOPTIONS_H

namespace llvm {
  /// PrintMachineCode - This flag is enabled when the -print-machineinstrs
  /// option is specified on the command line, and should enable debugging
  /// output from the code generator.
  extern bool PrintMachineCode;

  /// NoFramePointerElim - This flag is enabled when the -disable-fp-elim is
  /// specified on the command line.  If the target supports the frame pointer
  /// elimination optimization, this option should disable it.
  extern bool NoFramePointerElim;

  /// NoExcessFPPrecision - This flag is enabled when the
  /// -disable-excess-fp-precision flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is allowed to produce
  /// results that are "more precise" than IEEE allows.  This includes use of
  /// FMA-like operations and use of the X86 FP registers without rounding all
  /// over the place.
  extern bool NoExcessFPPrecision;

  /// UnsafeFPMath - This flag is enabled when the
  /// -enable-unsafe-fp-math flag is specified on the command line.  When
  /// this flag is off (the default), the code generator is not allowed to
  /// produce results that are "less precise" than IEEE allows.  This includes
  /// use of X86 instructions like FSIN and FCOS instead of libcalls.
  /// UnsafeFPMath implies FiniteOnlyFPMath.
  extern bool UnsafeFPMath;

  /// FiniteOnlyFPMath - This returns true when the -enable-finite-only-fp-math
  /// option is specified on the command line. If this returns false (default),
  /// the code generator is not allowed to assume that FP arithmetic arguments
  /// and results are never NaNs or +-Infs.
  extern bool FiniteOnlyFPMathOption;
  extern bool FiniteOnlyFPMath();
  
  /// HonorSignDependentRoundingFPMath - This returns true when the
  /// -enable-sign-dependent-rounding-fp-math is specified.  If this returns
  /// false (the default), the code generator is allowed to assume that the
  /// rounding behavior is the default (round-to-zero for all floating point to
  /// integer conversions, and round-to-nearest for all other arithmetic
  /// truncations).  If this is enabled (set to true), the code generator must
  /// assume that the rounding mode may dynamically change.
  extern bool HonorSignDependentRoundingFPMathOption;
  extern bool HonorSignDependentRoundingFPMath();
  
  /// UseSoftFloat - This flag is enabled when the -soft-float flag is specified
  /// on the command line.  When this flag is on, the code generator will
  /// generate libcalls to the software floating point library instead of
  /// target FP instructions.
  extern bool UseSoftFloat;

  /// NoZerosInBSS - By default some codegens place zero-initialized data to
  /// .bss section. This flag disables such behaviour (necessary, e.g. for
  /// crt*.o compiling).
  extern bool NoZerosInBSS;
  
  /// ExceptionHandling - This flag indicates that exception information should
  /// be emitted.
  extern bool ExceptionHandling;

  /// UnwindTablesMandatory - This flag indicates that unwind tables should
  /// be emitted for all functions.
  extern bool UnwindTablesMandatory;

  /// PerformTailCallOpt - This flag is enabled when -tailcallopt is specified
  /// on the commandline. When the flag is on, the target will perform tail call
  /// optimization (pop the caller's stack) providing it supports it.
  extern bool PerformTailCallOpt;

  /// StackAlignment - Override default stack alignment for target.
  extern unsigned StackAlignment;

  /// RealignStack - This flag indicates, whether stack should be automatically
  /// realigned, if needed.
  extern bool RealignStack;

  /// VerboseAsm - When this flag is set, the asm printer prints additional
  /// comments to asm directives.
  extern bool VerboseAsm;

  /// DisableJumpTables - This flag indicates jump tables should not be 
  /// generated.
  extern bool DisableJumpTables;

  /// FastISel - This flag enables fast-path instruction selection
  /// which trades away generated code quality in favor of reducing
  /// compile time.
  extern bool EnableFastISel;
  
  /// StrongPHIElim - This flag enables more aggressive PHI elimination
  /// wth earlier copy coalescing.
  extern bool StrongPHIElim;

  /// DisableRedZone - This flag disables use of the "Red Zone" on
  /// targets which would otherwise have one.
  extern bool DisableRedZone;

} // End llvm namespace

#endif
