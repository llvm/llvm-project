//===-- ARMSubtarget.h - Define Subtarget for the ARM ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ARM specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSUBTARGET_H
#define ARMSUBTARGET_H

#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/ADT/Triple.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "ARMGenSubtargetInfo.inc"

namespace llvm {
class GlobalValue;
class StringRef;

class ARMSubtarget : public ARMGenSubtargetInfo {
protected:
  enum ARMProcFamilyEnum {
    Others, CortexA8, CortexA9, CortexA15, Swift
  };

  /// ARMProcFamily - ARM processor family: Cortex-A8, Cortex-A9, and others.
  ARMProcFamilyEnum ARMProcFamily;

  /// HasV4TOps, HasV5TOps, HasV5TEOps, HasV6Ops, HasV6T2Ops, HasV7Ops -
  /// Specify whether target support specific ARM ISA variants.
  bool HasV4TOps;
  bool HasV5TOps;
  bool HasV5TEOps;
  bool HasV6Ops;
  bool HasV6T2Ops;
  bool HasV7Ops;

  /// HasVFPv2, HasVFPv3, HasVFPv4, HasNEON - Specify what
  /// floating point ISAs are supported.
  bool HasVFPv2;
  bool HasVFPv3;
  bool HasVFPv4;
  bool HasNEON;

  /// UseNEONForSinglePrecisionFP - if the NEONFP attribute has been
  /// specified. Use the method useNEONForSinglePrecisionFP() to
  /// determine if NEON should actually be used.
  bool UseNEONForSinglePrecisionFP;

  /// UseMulOps - True if non-microcoded fused integer multiply-add and
  /// multiply-subtract instructions should be used.
  bool UseMulOps;

  /// SlowFPVMLx - If the VFP2 / NEON instructions are available, indicates
  /// whether the FP VML[AS] instructions are slow (if so, don't use them).
  bool SlowFPVMLx;

  /// HasVMLxForwarding - If true, NEON has special multiplier accumulator
  /// forwarding to allow mul + mla being issued back to back.
  bool HasVMLxForwarding;

  /// SlowFPBrcc - True if floating point compare + branch is slow.
  bool SlowFPBrcc;

  /// InThumbMode - True if compiling for Thumb, false for ARM.
  bool InThumbMode;

  /// HasThumb2 - True if Thumb2 instructions are supported.
  bool HasThumb2;

  /// IsMClass - True if the subtarget belongs to the 'M' profile of CPUs -
  /// v6m, v7m for example.
  bool IsMClass;

  /// NoARM - True if subtarget does not support ARM mode execution.
  bool NoARM;

  /// PostRAScheduler - True if using post-register-allocation scheduler.
  bool PostRAScheduler;

  /// IsR9Reserved - True if R9 is a not available as general purpose register.
  bool IsR9Reserved;

  /// UseMovt - True if MOVT / MOVW pairs are used for materialization of 32-bit
  /// imms (including global addresses).
  bool UseMovt;

  /// SupportsTailCall - True if the OS supports tail call. The dynamic linker
  /// must be able to synthesize call stubs for interworking between ARM and
  /// Thumb.
  bool SupportsTailCall;

  /// HasFP16 - True if subtarget supports half-precision FP (We support VFP+HF
  /// only so far)
  bool HasFP16;

  /// HasD16 - True if subtarget is limited to 16 double precision
  /// FP registers for VFPv3.
  bool HasD16;

  /// HasHardwareDivide - True if subtarget supports [su]div
  bool HasHardwareDivide;

  /// HasHardwareDivideInARM - True if subtarget supports [su]div in ARM mode
  bool HasHardwareDivideInARM;

  /// HasT2ExtractPack - True if subtarget supports thumb2 extract/pack
  /// instructions.
  bool HasT2ExtractPack;

  /// HasDataBarrier - True if the subtarget supports DMB / DSB data barrier
  /// instructions.
  bool HasDataBarrier;

  /// Pref32BitThumb - If true, codegen would prefer 32-bit Thumb instructions
  /// over 16-bit ones.
  bool Pref32BitThumb;

  /// AvoidCPSRPartialUpdate - If true, codegen would avoid using instructions
  /// that partially update CPSR and add false dependency on the previous
  /// CPSR setting instruction.
  bool AvoidCPSRPartialUpdate;

  /// HasRAS - Some processors perform return stack prediction. CodeGen should
  /// avoid issue "normal" call instructions to callees which do not return.
  bool HasRAS;

  /// HasMPExtension - True if the subtarget supports Multiprocessing
  /// extension (ARMv7 only).
  bool HasMPExtension;

  /// FPOnlySP - If true, the floating point unit only supports single
  /// precision.
  bool FPOnlySP;

  /// AllowsUnalignedMem - If true, the subtarget allows unaligned memory
  /// accesses for some types.  For details, see
  /// ARMTargetLowering::allowsUnalignedMemoryAccesses().
  bool AllowsUnalignedMem;

  /// Thumb2DSP - If true, the subtarget supports the v7 DSP (saturating arith
  /// and such) instructions in Thumb2 code.
  bool Thumb2DSP;

  /// stackAlignment - The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// CPUString - String name of used CPU.
  std::string CPUString;

  /// TargetTriple - What processor and OS we're targeting.
  Triple TargetTriple;

  /// SchedModel - Processor specific instruction costs.
  const MCSchedModel *SchedModel;

  /// Selected instruction itineraries (one entry per itinerary class.)
  InstrItineraryData InstrItins;

 public:
  enum {
    isELF, isDarwin
  } TargetType;

  enum {
    ARM_ABI_APCS,
    ARM_ABI_AAPCS // ARM EABI
  } TargetABI;

  /// This constructor initializes the data members to match that
  /// of the specified triple.
  ///
  ARMSubtarget(const std::string &TT, const std::string &CPU,
               const std::string &FS);

  /// getMaxInlineSizeThreshold - Returns the maximum memset / memcpy size
  /// that still makes it profitable to inline the call.
  unsigned getMaxInlineSizeThreshold() const {
    // FIXME: For now, we don't lower memcpy's to loads / stores for Thumb1.
    // Change this once Thumb1 ldmia / stmia support is added.
    return isThumb1Only() ? 0 : 64;
  }
  /// ParseSubtargetFeatures - Parses features string setting specified
  /// subtarget options.  Definition of function is auto generated by tblgen.
  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  void computeIssueWidth();

  bool hasV4TOps()  const { return HasV4TOps;  }
  bool hasV5TOps()  const { return HasV5TOps;  }
  bool hasV5TEOps() const { return HasV5TEOps; }
  bool hasV6Ops()   const { return HasV6Ops;   }
  bool hasV6T2Ops() const { return HasV6T2Ops; }
  bool hasV7Ops()   const { return HasV7Ops;  }

  bool isCortexA8() const { return ARMProcFamily == CortexA8; }
  bool isCortexA9() const { return ARMProcFamily == CortexA9; }
  bool isCortexA15() const { return ARMProcFamily == CortexA15; }
  bool isSwift()    const { return ARMProcFamily == Swift; }
  bool isCortexM3() const { return CPUString == "cortex-m3"; }
  bool isLikeA9() const { return isCortexA9() || isCortexA15(); }

  bool hasARMOps() const { return !NoARM; }

  bool hasVFP2() const { return HasVFPv2; }
  bool hasVFP3() const { return HasVFPv3; }
  bool hasVFP4() const { return HasVFPv4; }
  bool hasNEON() const { return HasNEON;  }
  bool useNEONForSinglePrecisionFP() const {
    return hasNEON() && UseNEONForSinglePrecisionFP; }

  bool hasDivide() const { return HasHardwareDivide; }
  bool hasDivideInARMMode() const { return HasHardwareDivideInARM; }
  bool hasT2ExtractPack() const { return HasT2ExtractPack; }
  bool hasDataBarrier() const { return HasDataBarrier; }
  bool useMulOps() const { return UseMulOps; }
  bool useFPVMLx() const { return !SlowFPVMLx; }
  bool hasVMLxForwarding() const { return HasVMLxForwarding; }
  bool isFPBrccSlow() const { return SlowFPBrcc; }
  bool isFPOnlySP() const { return FPOnlySP; }
  bool prefers32BitThumb() const { return Pref32BitThumb; }
  bool avoidCPSRPartialUpdate() const { return AvoidCPSRPartialUpdate; }
  bool hasRAS() const { return HasRAS; }
  bool hasMPExtension() const { return HasMPExtension; }
  bool hasThumb2DSP() const { return Thumb2DSP; }

  bool hasFP16() const { return HasFP16; }
  bool hasD16() const { return HasD16; }

  const Triple &getTargetTriple() const { return TargetTriple; }

  bool isTargetIOS() const { return TargetTriple.getOS() == Triple::IOS; }
  bool isTargetDarwin() const { return TargetTriple.isOSDarwin(); }
  bool isTargetNaCl() const {
    return TargetTriple.getOS() == Triple::NativeClient;
  }
  bool isTargetELF() const { return !isTargetDarwin(); }

  bool isAPCS_ABI() const { return TargetABI == ARM_ABI_APCS; }
  bool isAAPCS_ABI() const { return TargetABI == ARM_ABI_AAPCS; }

  bool isThumb() const { return InThumbMode; }
  bool isThumb1Only() const { return InThumbMode && !HasThumb2; }
  bool isThumb2() const { return InThumbMode && HasThumb2; }
  bool hasThumb2() const { return HasThumb2; }
  bool isMClass() const { return IsMClass; }
  bool isARClass() const { return !IsMClass; }

  bool isR9Reserved() const { return IsR9Reserved; }

  bool useMovt() const { return UseMovt && hasV6T2Ops(); }
  bool supportsTailCall() const { return SupportsTailCall; }

  bool allowsUnalignedMem() const { return AllowsUnalignedMem; }

  const std::string & getCPUString() const { return CPUString; }

  unsigned getMispredictionPenalty() const;

  /// enablePostRAScheduler - True at 'More' optimization.
  bool enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                             TargetSubtargetInfo::AntiDepBreakMode& Mode,
                             RegClassVector& CriticalPathRCs) const;

  /// getInstrItins - Return the instruction itineraies based on subtarget
  /// selection.
  const InstrItineraryData &getInstrItineraryData() const { return InstrItins; }

  /// getStackAlignment - Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// GVIsIndirectSymbol - true if the GV will be accessed via an indirect
  /// symbol.
  bool GVIsIndirectSymbol(const GlobalValue *GV, Reloc::Model RelocM) const;
};
} // End llvm namespace

#endif  // ARMSUBTARGET_H
